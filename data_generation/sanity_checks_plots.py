"""
Quick sanity-check plots for generated runs.
Produces one PNG per failure category (all runs of that type overlaid),
plus one for healthy runs, so you can visually confirm each injected
failure actually looks different from a healthy run.

Usage:
    python sanity_check_plots.py
"""

import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt

RAW_DIR = "raw"
OUT_DIR = "sanity_check_plots"

os.makedirs(OUT_DIR, exist_ok=True)

# Group run files by injected_problem_type (or "healthy")
groups = defaultdict(list)

for fname in sorted(os.listdir(RAW_DIR)):
    if not fname.endswith(".json") or fname == "test_run.json":
        continue
    with open(os.path.join(RAW_DIR, fname)) as f:
        record = json.load(f)
    key = record["injected_problem_type"] if record["label"] == "problem" else "healthy"
    groups[key].append(record)


def plot_group(problem_type, records):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle(f"{problem_type} ({len(records)} runs)", fontsize=14)

    for record in records:
        epochs = [e["epoch"] for e in record["epochs"]]
        train_loss = [e["train_loss"] for e in record["epochs"]]
        train_acc = [e["train_acc"] for e in record["epochs"]]
        val_acc = [e["val_acc"] for e in record["epochs"]]
        grad_norm = [e["grad_norm"] for e in record["epochs"]]

        line, = axes[0].plot(epochs, train_loss, marker="o", label=record["run_id"])
        color = line.get_color()

        # Same color per run: solid = val_acc, dashed = train_acc, so the gap is visible
        axes[1].plot(epochs, val_acc, marker="o", linestyle="-", color=color, label=f"{record['run_id']} (val)")
        axes[1].plot(epochs, train_acc, marker="x", linestyle="--", color=color, alpha=0.6, label=f"{record['run_id']} (train)")

        axes[2].plot(epochs, grad_norm, marker="o", label=record["run_id"])

    axes[0].set_title("Train Loss")
    axes[0].set_xlabel("Epoch")
    axes[1].set_title("Train Acc (dashed) vs Val Acc (solid)")
    axes[1].set_xlabel("Epoch")
    axes[2].set_title("Grad Norm")
    axes[2].set_xlabel("Epoch")

    for ax in axes:
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, f"{problem_type}.png")
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"Saved {out_path}")


for problem_type, records in groups.items():
    plot_group(problem_type, records)

print(f"\nAll plots saved to {OUT_DIR}/")