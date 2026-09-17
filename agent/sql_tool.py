"""
The agent's only tool: execute_sql.
Validates that queries are read-only SELECT statements, runs them against
the training_logs.db SQLite database, and returns results (or a clean
error message so the agent can self-correct).
"""

import re
import sqlite3

DB_PATH = "../database/training_logs.db"

# Anything that isn't a SELECT is rejected outright — this is the safety guardrail.
FORBIDDEN_KEYWORDS = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE|ATTACH|PRAGMA|VACUUM)\b",
    re.IGNORECASE,
)


def is_safe_select(query: str) -> tuple[bool, str]:
    """Returns (is_safe, reason_if_not)."""
    stripped = query.strip().rstrip(";").strip()

    if not stripped:
        return False, "Empty query."

    if not re.match(r"^\s*SELECT\b", stripped, re.IGNORECASE):
        return False, "Only SELECT statements are allowed."

    if FORBIDDEN_KEYWORDS.search(stripped):
        return False, "Query contains a forbidden keyword (only read-only SELECTs are permitted)."

    # Reject multiple statements stacked with semicolons (e.g. "SELECT 1; DROP TABLE runs")
    if ";" in stripped:
        return False, "Multiple statements are not allowed."

    return True, ""


def execute_sql(query: str) -> dict:
    """
    Executes a validated read-only SQL query.
    Returns a dict with either {"success": True, "columns": [...], "rows": [...]}
    or {"success": False, "error": "..."} so the agent can see what went wrong.
    """
    safe, reason = is_safe_select(query)
    if not safe:
        return {"success": False, "error": f"Query rejected: {reason}"}

    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute(query)
        rows = cur.fetchall()
        columns = [desc[0] for desc in cur.description] if cur.description else []
        conn.close()
        return {"success": True, "columns": columns, "rows": rows}
    except sqlite3.Error as e:
        return {"success": False, "error": f"SQL error: {str(e)}"}


if __name__ == "__main__":
    # Quick manual test
    result = execute_sql("SELECT run_id, label FROM runs LIMIT 3")
    print(result)

    bad_result = execute_sql("DROP TABLE runs")
    print(bad_result)