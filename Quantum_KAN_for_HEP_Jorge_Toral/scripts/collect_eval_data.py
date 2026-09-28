# scripts/collect_eval_data.py
# Collects the per-run/per-stage saved test-set arrays (true / probs / binary .npy) and
# training histories (loss / AUC curves JSON) across every outputs/<task>/**/seed_*/ run
# directory into one long-format Parquet table, one row per (run, model), tagged by
# task/seed/variant/subset_id/model -- the eval-data counterpart of collect_metrics.py.
# Arrays and curves are stored as list columns; untrained stages (baselines, RF) have
# no history columns filled.
import argparse
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.resolve()))
from src.utils import workspace
from src.utils.reporting import collect_eval_data


def main(args):
    df = collect_eval_data(args.task)

    out_path = workspace.get_config(args.task, seed=0)["eval_table_path"]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_parquet(out_path, index=False)

    n_seeds = df["seed"].nunique() if len(df) else 0
    print(f"Collected {len(df)} eval rows across {n_seeds} seed(s) -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collect per-run/per-model eval arrays and training histories into one Parquet table."
    )
    parser.add_argument("--task", type=str, required=True, choices=["top", "quark-gluon"])
    main(parser.parse_args())
