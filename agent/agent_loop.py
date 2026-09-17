"""
The agent loop: repeatedly calls the LLM with tool access to execute_sql,
until it produces a final structured diagnosis or hits the call limit.
"""
import re
from groq import BadRequestError

import json
import os

from dotenv import load_dotenv
from groq import Groq

from sql_tool import execute_sql
from system_prompt import SYSTEM_PROMPT

load_dotenv("../.env")

MODEL = "openai/gpt-oss-120b"
MAX_TOOL_CALLS = 8

client = Groq(api_key=os.environ["GROQ_API_KEY"])

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "execute_sql",
            "description": "Execute a read-only SQL SELECT query against the training-log database.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A SQL SELECT statement.",
                    }
                },
                "required": ["query"],
            },
        },
    }
]


def run_diagnosis(run_id: str, verbose: bool = True) -> dict:
    """
    Runs the agent loop for a single run_id.
    Returns a dict with the final diagnosis plus the full query history and transcript.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Investigate run_id = '{run_id}' and give your diagnosis."},
    ]

    query_history = []
    tool_call_count = 0

    while tool_call_count < MAX_TOOL_CALLS:
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
            )
        except BadRequestError as e:
            # Some models occasionally emit their final answer as a fake tool call
            # (e.g. calling a nonexistent "json" tool) instead of plain text.
            # Recover the diagnosis directly from the failed_generation payload.
            try:
                err_json = e.response.json()
            except Exception:
                err_json = {}

            failed_gen = err_json.get("error", {}).get("failed_generation", "")
            diagnosis = None
            try:
                parsed = json.loads(failed_gen)
                args = parsed.get("arguments")
                if isinstance(args, str):
                    args = json.loads(args)
                if isinstance(args, dict):
                    diagnosis = args
            except (json.JSONDecodeError, TypeError, AttributeError):
                pass

            if diagnosis:
                diagnosis["run_id"] = run_id
                diagnosis["query_history"] = query_history
                diagnosis["tool_calls_used"] = tool_call_count
                return diagnosis

            raise  # couldn't recover, surface the real error

        msg = response.choices[0].message

        if msg.tool_calls:
            # Build a minimal, clean assistant message instead of dumping the whole
            # SDK response object (which includes fields the API itself rejects).
            messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ],
            })

            for tool_call in msg.tool_calls:
                args = json.loads(tool_call.function.arguments)
                query = args.get("query", "")
                tool_call_count += 1

                if verbose:
                    print(f"  [{tool_call_count}] Query: {query}")

                result = execute_sql(query)
                query_history.append({"query": query, "result": result})

                if verbose:
                    if result["success"]:
                        print(f"      -> {len(result['rows'])} rows")
                    else:
                        print(f"      -> ERROR: {result['error']}")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result),
                })

            continue

        # No tool call: agent thinks it's done. Try to parse the final JSON diagnosis.
        content = msg.content.strip()
        try:
            diagnosis = json.loads(content)
            diagnosis["run_id"] = run_id
            diagnosis["query_history"] = query_history
            diagnosis["tool_calls_used"] = tool_call_count
            return diagnosis
        except json.JSONDecodeError:
            # Agent responded with non-JSON text; nudge it to comply.
            messages.append({"role": "assistant", "content": content})
            messages.append({
                "role": "user",
                "content": "Please respond with ONLY the JSON diagnosis object, no other text.",
            })

    # Hit the call limit without a clean final diagnosis
    # Hit the call limit — force one last text-only answer using whatever evidence exists,
    # instead of giving up outright.
    messages.append({
        "role": "user",
        "content": "You've used all available queries. Based on everything you've seen so far, "
                    "give your best final diagnosis now, in the exact JSON format specified. "
                    "Do not call any tool.",
    })
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tool_choice="none",
        )
        content = response.choices[0].message.content.strip()
        diagnosis = json.loads(content)
        diagnosis["run_id"] = run_id
        diagnosis["query_history"] = query_history
        diagnosis["tool_calls_used"] = tool_call_count
        diagnosis["forced_final_answer"] = True
        return diagnosis
    except (json.JSONDecodeError, AttributeError, Exception):
        return {
            "run_id": run_id,
            "problem_detected": None,
            "diagnosis": "UNRESOLVED_CALL_LIMIT",
            "evidence": "Agent hit the tool call limit and could not produce a valid final diagnosis.",
            "confidence": "low",
            "query_history": query_history,
            "tool_calls_used": tool_call_count,
        }


if __name__ == "__main__":
    import sys

    run_id = sys.argv[1] if len(sys.argv) > 1 else "healthy_01"
    print(f"Investigating: {run_id}\n")

    result = run_diagnosis(run_id)

    print("\n--- Final Diagnosis ---")
    print(json.dumps(result, indent=2))