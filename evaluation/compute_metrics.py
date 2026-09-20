"""
Computes evaluation metrics from raw_results.json (produced by run_eval.py)
against the ground truth labels, and writes a markdown results table.

Usage:
    python compute_metrics.py
"""

import json
import os
from collections import defaultdict

RESULTS_PATH = "../results/tables/raw_results.json"
OUTPUT_PATH = "../results/tables/metrics_summary.md"

ALL_PROBLEM_TYPES = [
    "lr_too_high", "lr_too_low", "overfitting",
    "vanishing_gradients", "label_noise", "frozen_layer",
]


def load_results():
    with open(RESULTS_PATH, "r") as f:
        return json.load(f)


def compute_detection_metrics(results):
    """Precision/recall/false-positive-rate on the binary healthy-vs-problem question."""
    tp = fp = tn = fn = 0

    for r in results:
        true_is_problem = r["true_label"] == "problem"
        pred_is_problem = r["predicted_problem_detected"] is True

        if true_is_problem and pred_is_problem:
            tp += 1
        elif not true_is_problem and pred_is_problem:
            fp += 1
        elif not true_is_problem and not pred_is_problem:
            tn += 1
        elif true_is_problem and not pred_is_problem:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    recall = tp / (tp + fn) if (tp + fn) > 0 else None
    fpr = fp / (fp + tn) if (fp + tn) > 0 else None

    return {
        "true_positives": tp, "false_positives": fp,
        "true_negatives": tn, "false_negatives": fn,
        "precision": precision, "recall": recall,
        "false_positive_rate": fpr,
    }


def compute_diagnosis_accuracy(results):
    """
    For runs that ARE problems, was the specific diagnosis correct?
    Strict match = exact problem_type string match.
    """
    problem_runs = [r for r in results if r["true_label"] == "problem"]
    if not problem_runs:
        return {"strict_accuracy": None, "n_evaluated": 0}

    correct = sum(1 for r in problem_runs if r["predicted_diagnosis"] == r["true_problem_type"])
    return {
        "strict_accuracy": correct / len(problem_runs),
        "n_correct": correct,
        "n_evaluated": len(problem_runs),
    }


def compute_per_type_breakdown(results):
    """Accuracy broken down by each injected problem type, plus healthy."""
    breakdown = defaultdict(lambda: {"total": 0, "correct": 0})

    for r in results:
        key = r["true_problem_type"]  # "none" for healthy, else the failure type
        breakdown[key]["total"] += 1

        if key == "none":
            # "Correct" for healthy means NOT flagged as a problem
            if r["predicted_problem_detected"] is False:
                breakdown[key]["correct"] += 1
        else:
            if r["predicted_diagnosis"] == key:
                breakdown[key]["correct"] += 1

    return dict(breakdown)


def compute_efficiency(results):
    calls = [r["tool_calls_used"] for r in results if r["tool_calls_used"] is not None]
    return {
        "avg_tool_calls": sum(calls) / len(calls) if calls else None,
        "min_tool_calls": min(calls) if calls else None,
        "max_tool_calls": max(calls) if calls else None,
    }


def format_pct(x):
    return f"{x*100:.1f}%" if x is not None else "N/A"


def write_markdown_report(results, detection, diagnosis, breakdown, efficiency):
    lines = []
    lines.append("# Evaluation Results\n")
    lines.append(f"Total runs evaluated: **{len(results)}**\n")

    lines.append("## Detection (healthy vs. problem)\n")
    lines.append(f"- Precision: **{format_pct(detection['precision'])}**")
    lines.append(f"- Recall: **{format_pct(detection['recall'])}**")
    lines.append(f"- False positive rate: **{format_pct(detection['false_positive_rate'])}**")
    lines.append(f"- Confusion: TP={detection['true_positives']}, FP={detection['false_positives']}, "
                 f"TN={detection['true_negatives']}, FN={detection['false_negatives']}\n")

    lines.append("## Diagnosis accuracy (for correctly-flagged problem runs' specific type)\n")
    lines.append(f"- Strict accuracy: **{format_pct(diagnosis['strict_accuracy'])}** "
                 f"({diagnosis['n_correct']}/{diagnosis['n_evaluated']})\n")

    lines.append("## Breakdown by problem type\n")
    lines.append("| Type | Correct | Total | Accuracy |")
    lines.append("|---|---|---|---|")
    for key in ["none"] + ALL_PROBLEM_TYPES:
        if key in breakdown:
            b = breakdown[key]
            acc = b["correct"] / b["total"] if b["total"] > 0 else None
            label = "healthy" if key == "none" else key
            lines.append(f"| {label} | {b['correct']} | {b['total']} | {format_pct(acc)} |")
    lines.append("")

    lines.append("## Efficiency\n")
    lines.append(f"- Avg tool calls per run: **{efficiency['avg_tool_calls']:.1f}**")
    lines.append(f"- Range: {efficiency['min_tool_calls']}–{efficiency['max_tool_calls']}\n")

    lines.append("## Per-run results\n")
    lines.append("| run_id | true label | true type | detected? | diagnosis | correct? | confidence | calls |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in results:
        if r["true_label"] == "healthy":
            correct = r["predicted_problem_detected"] is False
        else:
            correct = r["predicted_diagnosis"] == r["true_problem_type"]
        mark = "✅" if correct else "❌"
        lines.append(
            f"| {r['run_id']} | {r['true_label']} | {r['true_problem_type']} | "
            f"{r['predicted_problem_detected']} | {r['predicted_diagnosis']} | {mark} | "
            f"{r['confidence']} | {r['tool_calls_used']} |"
        )

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Wrote {OUTPUT_PATH}")


def main():
    results = load_results()
    detection = compute_detection_metrics(results)
    diagnosis = compute_diagnosis_accuracy(results)
    breakdown = compute_per_type_breakdown(results)
    efficiency = compute_efficiency(results)

    write_markdown_report(results, detection, diagnosis, breakdown, efficiency)

    # Also print a quick summary to console
    print(f"\nDetection: precision={format_pct(detection['precision'])}, "
          f"recall={format_pct(detection['recall'])}, "
          f"FPR={format_pct(detection['false_positive_rate'])}")
    print(f"Diagnosis accuracy: {format_pct(diagnosis['strict_accuracy'])}")
    print(f"Avg tool calls: {efficiency['avg_tool_calls']:.1f}")


if __name__ == "__main__":
    main()