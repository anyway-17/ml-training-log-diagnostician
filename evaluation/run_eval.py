"""
Evaluation harness: runs the agent against every run in the dataset,
saves full transcripts, and computes detection/diagnosis metrics
against ground_truth.json.

Usage:
    python run_eval.py
"""

import json
import os
import sys
import time

sys.path.append("../agent")
from agent_loop import run_diagnosis

GROUND_TRUTH_PATH = "../database/ground_truth.json"
RAW_DIR = "../data_generation/raw"
TRACES_DIR = "../results/traces"
TABLES_DIR = "../results/tables"

SLEEP_BETWEEN_RUNS = 12  # seconds, to stay under Groq free-tier rate limits


def get_all_run_ids():
    run_ids = []
    for fname in sorted(os.listdir(RAW_DIR)):
        if fname.endswith(".json") and fname != "test_run.json":
            run_ids.append(fname.replace(".json", ""))
    return run_ids


def run_full_evaluation():
    with open(GROUND_TRUTH_PATH, "r") as f:
        ground_truth = json.load(f)

    run_ids = get_all_run_ids()
    os.makedirs(TRACES_DIR, exist_ok=True)
    os.makedirs(TABLES_DIR, exist_ok=True)

    results = []

    for i, run_id in enumerate(run_ids, 1):
        print(f"\n=== [{i}/{len(run_ids)}] {run_id} ===")

        diagnosis = run_diagnosis(run_id, verbose=True)
        truth = ground_truth[run_id]

        record = {
            "run_id": run_id,
            "true_label": truth["label"],
            "true_problem_type": truth["injected_problem_type"],
            "predicted_problem_detected": diagnosis.get("problem_detected"),
            "predicted_diagnosis": diagnosis.get("diagnosis"),
            "confidence": diagnosis.get("confidence"),
            "evidence": diagnosis.get("evidence"),
            "tool_calls_used": diagnosis.get("tool_calls_used"),
        }
        results.append(record)

        # Save full transcript for this run
        with open(os.path.join(TRACES_DIR, f"{run_id}.json"), "w") as f:
            json.dump(diagnosis, f, indent=2)

        if i < len(run_ids):
            time.sleep(SLEEP_BETWEEN_RUNS)

    with open(os.path.join(TABLES_DIR, "raw_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved {len(results)} traces to {TRACES_DIR}/")
    print(f"Saved raw results to {TABLES_DIR}/raw_results.json")

    return results


if __name__ == "__main__":
    run_full_evaluation()