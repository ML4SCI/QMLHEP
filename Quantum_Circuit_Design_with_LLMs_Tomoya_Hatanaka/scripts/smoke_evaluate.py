#!/usr/bin/env python
"""Phase 2 smoke workflow: exercise the full evaluation harness on a
handful of circuits, with a resumable result store.

This is NOT a search algorithm. Circuits are drawn from Phase 1's
`sample_random_ir` purely as a convenient circuit *source* to exercise
the IR -> compiler -> training -> validation pipeline end to end; no
search logic (selection, mutation, LLM guidance) lives here. That is
explicitly out of Phase 2 scope.

Usage:
    python scripts/smoke_evaluate.py configs/t1_smoke.yaml
    # run again with the same config: resumes, skips completed work
    python scripts/smoke_evaluate.py configs/t1_smoke.yaml
    # same run_id + config as t1_smoke.yaml -> also resumes
    python scripts/smoke_evaluate.py configs/resume_smoke.yaml
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm_vqc.config import load_config  # noqa: E402
from llm_vqc.evaluation.harness import evaluate_candidate  # noqa: E402
from llm_vqc.evaluation.seeds import train_seed_for_circuit  # noqa: E402
from llm_vqc.evaluation.store import IncompatibleResumeError, ResultStore  # noqa: E402
from llm_vqc.evaluation.training import TrainingConfig  # noqa: E402
from llm_vqc.ir.budget import BudgetLedger  # noqa: E402
from llm_vqc.ir.canonicalize import structural_hash  # noqa: E402
from llm_vqc.ir.sampler import sample_random_ir  # noqa: E402
from llm_vqc.tasks import TASK_REGISTRY  # noqa: E402

DEFAULT_RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"
_REPO_ROOT = Path(__file__).resolve().parent.parent


def _git_sha() -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(_REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _git_dirty() -> bool | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(_REPO_ROOT), "status", "--porcelain"],
            capture_output=True, text=True, check=True,
        )
        return bool(result.stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Path to a smoke-test YAML config")
    parser.add_argument("--runs-dir", default=str(DEFAULT_RUNS_DIR))
    parser.add_argument(
        "--allow-incompatible",
        action="store_true",
        help="Override IncompatibleResumeError (explicit, non-silent opt-in)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    params = config.params
    task_name = params["task"]
    training_params = params["training"]
    n_circuits = params["n_circuits"]

    task = TASK_REGISTRY[task_name]()
    train_val = task.build(seed=config.seed)
    training_config = TrainingConfig(**training_params)

    run_dir = Path(args.runs_dir) / config.name
    run_dir.mkdir(parents=True, exist_ok=True)
    db_path = run_dir / "results.sqlite"

    repro_fields = {
        "task": task_name,
        "seed": config.seed,
        "training": training_config.model_dump(),
    }

    try:
        store = ResultStore.open_or_create(
            db_path,
            run_id=config.name,
            config_json=config.model_dump_json(),
            config_reproducibility_fields=repro_fields,
            git_sha=_git_sha(),
            created_at=datetime.now(timezone.utc).isoformat(),
            allow_incompatible=args.allow_incompatible,
        )
    except IncompatibleResumeError as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        sys.exit(1)

    ledger = BudgetLedger()
    rng = np.random.default_rng(config.seed)
    git_sha = _git_sha()
    git_dirty = _git_dirty()

    skipped = 0
    evaluated = 0
    for i in range(n_circuits):
        ir = sample_random_ir(rng)
        ir_hash = structural_hash(ir)
        proposal_id = f"{config.name}-smoke-{i}"

        train_seed = train_seed_for_circuit(config.seed, ir_hash)
        already_done = store.get_cached(task_name, ir_hash, train_seed) is not None
        result = evaluate_candidate(
            ir,
            task_name=task_name,
            run_seed=config.seed,
            train_val=train_val,
            training_config=training_config,
            proposal_id=proposal_id,
            ledger=ledger,
            cache=store,
            git_sha=git_sha,
            git_dirty=git_dirty,
        )
        if already_done:
            skipped += 1
        else:
            evaluated += 1
        print(
            f"[{i}] hash={ir_hash[:10]} val_outcome={result.validation_outcome.value} "
            f"train_outcome={result.training_outcome.value} "
            f"val_metric={result.val_metric_value} cache_hit={result.cache_hit}"
        )

    print(f"\nDone. Evaluated this run: {evaluated}, resumed/skipped: {skipped}")
    print(f"Ledger: {ledger.summary()}")
    print(f"Store total rows: {store.count_evaluations()}")
    store.close()


if __name__ == "__main__":
    main()
