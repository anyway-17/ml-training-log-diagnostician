"""
Generates the full labeled run set for the training-log diagnostician project.

Runs train_baseline.py as a subprocess for each configured run, so every run
gets its own clean Python process (avoids state leaking between runs).

Usage:
    python generate_runs.py --stage healthy   # generates the 5 healthy runs only
    python generate_runs.py --stage failures  # generates all failure-mode runs
    python generate_runs.py --stage all       # generates everything
"""

import argparse
import subprocess
import sys

EPOCHS = 10  # full epoch count for real runs (vs. the 2-epoch test_run)

# ---------------------------------------------------------------------------
# Run configs
# ---------------------------------------------------------------------------

HEALTHY_RUNS = [
    {"run_id": "healthy_01", "seed": 1},
    {"run_id": "healthy_02", "seed": 2},
    {"run_id": "healthy_03", "seed": 3},
    {"run_id": "healthy_04", "seed": 4},
    {"run_id": "healthy_05", "seed": 5},
]

FAILURE_RUNS = [
    # --- LR too high (10-50x) ---
    {"run_id": "lr_high_01", "seed": 1, "lr": 0.03, "injected_problem_type": "lr_too_high"},
    {"run_id": "lr_high_02", "seed": 2, "lr": 0.05, "injected_problem_type": "lr_too_high"},
    {"run_id": "lr_high_03", "seed": 3, "lr": 0.08, "injected_problem_type": "lr_too_high"},

    # --- LR too low (50-100x smaller) ---
    {"run_id": "lr_low_01", "seed": 1, "lr": 0.00001, "injected_problem_type": "lr_too_low"},
    {"run_id": "lr_low_02", "seed": 2, "lr": 0.000005, "injected_problem_type": "lr_too_low"},
    {"run_id": "lr_low_03", "seed": 3, "lr": 0.00002, "injected_problem_type": "lr_too_low"},

    # --- Overfitting (shrink training set, no weight decay) ---
    {"run_id": "overfit_01", "seed": 1, "epochs": 30, "train_subset_frac": 0.03, "dropout_rate": 0.0, "injected_problem_type": "overfitting"},
    {"run_id": "overfit_02", "seed": 2, "epochs": 30, "train_subset_frac": 0.02, "dropout_rate": 0.0, "injected_problem_type": "overfitting"},
    {"run_id": "overfit_03", "seed": 3, "epochs": 30, "train_subset_frac": 0.03, "dropout_rate": 0.0, "injected_problem_type": "overfitting"},

    # --- Vanishing gradients (bad init) ---
    {"run_id": "vanish_grad_01", "seed": 1, "init_scheme": "bad", "injected_problem_type": "vanishing_gradients"},
    {"run_id": "vanish_grad_02", "seed": 2, "init_scheme": "bad", "injected_problem_type": "vanishing_gradients"},
    {"run_id": "vanish_grad_03", "seed": 3, "init_scheme": "bad", "injected_problem_type": "vanishing_gradients"},

    # --- Label noise ---
    {"run_id": "label_noise_01", "seed": 1, "label_noise_frac": 0.3, "injected_problem_type": "label_noise"},
    {"run_id": "label_noise_02", "seed": 2, "label_noise_frac": 0.4, "injected_problem_type": "label_noise"},
    {"run_id": "label_noise_03", "seed": 3, "label_noise_frac": 0.5, "injected_problem_type": "label_noise"},

        # --- Frozen/misconfigured layer ---
    {"run_id": "frozen_layer_01", "seed": 1, "freeze_conv1": True, "freeze_conv2": True, "injected_problem_type": "frozen_layer"},
    {"run_id": "frozen_layer_02", "seed": 2, "freeze_conv1": True, "freeze_conv2": True, "injected_problem_type": "frozen_layer"},
    {"run_id": "frozen_layer_03", "seed": 3, "freeze_conv1": True, "freeze_conv2": True, "injected_problem_type": "frozen_layer"},
]

EDGE_CASE_RUNS = [
    # --- Overlapping failures ---
    {"run_id": "overlap_lr_noise_01", "seed": 1, "lr": 0.03, "label_noise_frac": 0.3,
     "injected_problem_type": "lr_too_high+label_noise"},
    {"run_id": "overlap_frozen_lownoise_01", "seed": 2, "freeze_conv1": True, "freeze_conv2": True,
     "label_noise_frac": 0.15, "injected_problem_type": "frozen_layer+label_noise"},

    # --- Borderline/mild ---
    {"run_id": "borderline_lr_mild_01", "seed": 3, "lr": 0.005,
     "injected_problem_type": "lr_too_high_mild"},  # only 5x, vs 10-50x for the "real" lr_too_high runs
    {"run_id": "borderline_noise_mild_01", "seed": 4, "label_noise_frac": 0.1,
     "injected_problem_type": "label_noise_mild"},  # 10% vs 30-50% for the "real" label_noise runs
]

def build_command(cfg: dict) -> list:
    epochs = cfg.get("epochs", EPOCHS)
    cmd = [
        sys.executable, "train_baseline.py",
        "--run_id", cfg["run_id"],
        "--epochs", str(epochs),
        "--seed", str(cfg["seed"]),
    ]

    label = "healthy" if cfg.get("injected_problem_type", "none") == "none" else "problem"
    cmd += ["--label", label]
    cmd += ["--injected_problem_type", cfg.get("injected_problem_type", "none")]

    if "lr" in cfg:
        cmd += ["--lr", str(cfg["lr"])]
    if "train_subset_frac" in cfg:
        cmd += ["--train_subset_frac", str(cfg["train_subset_frac"])]
    if "label_noise_frac" in cfg:
        cmd += ["--label_noise_frac", str(cfg["label_noise_frac"])]
    if "init_scheme" in cfg:
        cmd += ["--init_scheme", cfg["init_scheme"]]
    if cfg.get("freeze_conv1"):
        cmd += ["--freeze_conv1"]
    if cfg.get("freeze_conv2"):
        cmd += ["--freeze_conv2"]
    if "dropout_rate" in cfg:
        cmd += ["--dropout_rate", str(cfg["dropout_rate"])]

    return cmd


def run_all(configs: list):
    for i, cfg in enumerate(configs, 1):
        cmd = build_command(cfg)
        print(f"\n=== [{i}/{len(configs)}] Running {cfg['run_id']} ===")
        print(" ".join(cmd))
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"!!! {cfg['run_id']} FAILED (exit code {result.returncode}) — stopping.")
            sys.exit(1)
    print(f"\nDone. {len(configs)} runs completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["healthy", "failures", "edge_cases", "all"], default="healthy")
    parser.add_argument("--filter", type=str, default=None,
                         help="Only run configs whose run_id contains this substring (e.g. 'overfit')")
    args = parser.parse_args()

    if args.stage == "healthy":
        configs = HEALTHY_RUNS
    elif args.stage == "failures":
        configs = FAILURE_RUNS
    elif args.stage == "edge_cases":
        configs = EDGE_CASE_RUNS
    else:
        configs = HEALTHY_RUNS + FAILURE_RUNS
    

    if args.filter:
        configs = [cfg for cfg in configs if args.filter in cfg["run_id"]]
        if not configs:
            print(f"No runs matched filter '{args.filter}'")
            sys.exit(1)

    run_all(configs)