# scripts/collect_metrics.py
# Collects the per-run/per-stage metrics JSON files already written across every
# outputs/<task>/**/seed_*/ run directory into one long-format Parquet table, tagged by
# task/seed/variant/subset_id/model. The table is task-level (variant-independent) and
# covers every regime found on disk (n-subset runs and the full-dataset run); filter on its
# `variant` column to select one. Per-seed untrained-init comparison rows (Chebyshev / sine /
# random, from scripts/eval_sine_baseline.py) are the `qkan_baseline_ideal`,
# `qkan_sine_baseline_ideal` and `qkan_baseline_random_ideal` models.
import argparse
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.resolve()))
from src.utils import workspace
from src.utils.reporting import compute_run_statistics


def main(args):
    df = compute_run_statistics(args.task)

    out_path = workspace.get_config(args.task, seed=0)["metrics_table_path"]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_parquet(out_path, index=False)

    n_seeds = df["seed"].nunique() if len(df) else 0
    print(f"Collected {len(df)} metric rows across {n_seeds} seed(s) -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collect per-run/per-model metrics JSON files into one Parquet table."
    )
    parser.add_argument("--task", type=str, required=True, choices=["top", "quark-gluon"])
    main(parser.parse_args())
