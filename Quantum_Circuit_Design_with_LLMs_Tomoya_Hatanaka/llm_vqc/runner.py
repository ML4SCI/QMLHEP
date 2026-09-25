"""Config-driven run scaffolding.

Phase 0 scope only: load a RunConfig, create a timestamped run directory,
and write its manifest. Search drivers, tasks, and the evaluation harness
are Phase 1+ (LLM-VQC_MASTER_PLAN.md Section 9) and are not implemented
here.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from llm_vqc.config import load_config
from llm_vqc.manifest import create_run_dir, write_manifest

DEFAULT_RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"


def run_from_config(config_path: str | Path, runs_dir: str | Path = DEFAULT_RUNS_DIR) -> Path:
    """Load config_path, create a run directory, write its manifest.

    Returns the run directory path.
    """
    config = load_config(config_path)
    run_dir = create_run_dir(runs_dir, config.name)
    write_manifest(run_dir, config)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a manifested run directory from a config.")
    parser.add_argument("config", help="Path to a YAML RunConfig file.")
    parser.add_argument(
        "--runs-dir",
        default=str(DEFAULT_RUNS_DIR),
        help="Base directory under which run directories are created.",
    )
    args = parser.parse_args()

    run_dir = run_from_config(args.config, runs_dir=args.runs_dir)
    print(f"Run directory created: {run_dir}")
    print(f"Manifest written: {run_dir / 'manifest.json'}")


if __name__ == "__main__":
    sys.exit(main())
