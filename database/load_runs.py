"""
Loads all generated run JSON files (from data_generation/raw/) into the SQLite database,
using the schema defined in schema.sql.

Usage:
    python load_runs.py
"""

import json
import os
import sqlite3

RAW_DIR = "../data_generation/raw"
DB_PATH = "training_logs.db"
SCHEMA_PATH = "schema.sql"


def create_schema(conn):
    with open(SCHEMA_PATH, "r") as f:
        conn.executescript(f.read())


def load_run(conn, record: dict):
    cur = conn.cursor()

    # Strip ground-truth keys out of the config before storing — the agent must never
    # see these; they only exist because WE know the answer in advance.
    clean_config = {
        k: v for k, v in record["config"].items()
        if k not in ("label", "injected_problem_type", "run_id")
    }

    cur.execute(
        """
        INSERT OR REPLACE INTO runs (run_id, config_json, test_acc, created_at)
        VALUES (?, ?, ?, ?)
        """,
        (
            record["run_id"],
            json.dumps(clean_config),
            record.get("test_acc"),
            record["created_at"],
        ),
    )

    cur.execute("DELETE FROM epochs WHERE run_id = ?", (record["run_id"],))

    for e in record["epochs"]:
        cur.execute(
            """
            INSERT INTO epochs (run_id, epoch, train_loss, val_loss, train_acc, val_acc, grad_norm, learning_rate)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record["run_id"], e["epoch"], e["train_loss"], e["val_loss"],
                e["train_acc"], e["val_acc"], e["grad_norm"], e["learning_rate"],
            ),
        )


def main():
    conn = sqlite3.connect(DB_PATH)
    create_schema(conn)

    loaded = 0
    skipped = []

    for fname in sorted(os.listdir(RAW_DIR)):
        if not fname.endswith(".json") or fname == "test_run.json":
            continue
        path = os.path.join(RAW_DIR, fname)
        with open(path, "r") as f:
            record = json.load(f)

        try:
            load_run(conn, record)
            loaded += 1
        except KeyError as e:
            skipped.append((fname, str(e)))

    conn.commit()
    conn.close()

    print(f"Loaded {loaded} runs into {DB_PATH}")
    if skipped:
        print(f"Skipped {len(skipped)} files due to missing fields:")
        for fname, err in skipped:
            print(f"  {fname}: missing {err}")

    build_ground_truth_file()

def build_ground_truth_file():
    """
    Writes ground_truth.json — a run_id -> {label, injected_problem_type} lookup,
    read directly from the raw JSON files. Used ONLY by the evaluation harness in
    Phase 4, never exposed to the agent's SQL tool.
    """
    ground_truth = {}
    for fname in sorted(os.listdir(RAW_DIR)):
        if not fname.endswith(".json") or fname == "test_run.json":
            continue
        with open(os.path.join(RAW_DIR, fname), "r") as f:
            record = json.load(f)
        ground_truth[record["run_id"]] = {
            "label": record["label"],
            "injected_problem_type": record["injected_problem_type"],
        }

    with open("ground_truth.json", "w") as f:
        json.dump(ground_truth, f, indent=2)
    print(f"Wrote ground_truth.json with {len(ground_truth)} entries")

if __name__ == "__main__":
    main()
    