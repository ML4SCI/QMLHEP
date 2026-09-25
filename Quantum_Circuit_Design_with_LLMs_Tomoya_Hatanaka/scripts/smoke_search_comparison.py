#!/usr/bin/env python
"""Stage 7 smoke comparison: run every implemented search arm through the
real `SearchRunner`, on the same task/budget/training config, at a fixed
seed -- including one deliberate interruption-and-resume of one arm's
run, using real stored results (SQLite), and exercising each arm's final
candidate selection.

*** THIS SCRIPT IS FOR SYSTEM VALIDATION ONLY. ***
Its output must never be reported as scientific evidence of one arm
outperforming another (small budget, a single seed, fast/reduced
training). See LLM-VQC_MASTER_PLAN.md Section 6.3 for the actual
research-scale protocol and `scripts/pilot_experiment.py` (Stage 8) for
the first real pilot run.

Test-set evaluation is deliberately never invoked here -- this script
imports nothing from `llm_vqc.evaluation.final_test`; only the
validation-only search loop and its final candidate selection are
exercised. Final-test scoring, if ever wanted for these circuits, is a
separate, explicit step outside this script.

Usage:
    python scripts/smoke_search_comparison.py configs/search_smoke.yaml
"""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm_vqc.config import load_config  # noqa: E402
from llm_vqc.evaluation.store import ResultStore  # noqa: E402
from llm_vqc.evaluation.training import TrainingConfig  # noqa: E402
from llm_vqc.llm.provider import MockLLMProvider  # noqa: E402
from llm_vqc.search.arms.evolutionary_arm import (  # noqa: E402
    EvolutionaryArm,
    EvolutionaryArmConfig,
)
from llm_vqc.search.arms.greedy_arm import GreedyArm  # noqa: E402
from llm_vqc.search.arms.llm_evo_arm import LLMEvoArm  # noqa: E402
from llm_vqc.search.arms.llm_iter_arm import LLMIterArm  # noqa: E402
from llm_vqc.search.arms.random_arm import RandomArm  # noqa: E402
from llm_vqc.search.runner import SearchRunner  # noqa: E402
from llm_vqc.tasks import TASK_REGISTRY  # noqa: E402

DEFAULT_RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"
_REPO_ROOT = Path(__file__).resolve().parent.parent
_TASK_DESCRIPTIONS = {
    "T1": "T1 1D Gaussian-peak regression (predict the peak location)",
    "T2": "T2 electron-vs-photon-fallback digits 3-vs-8 binary classification",
}


def _git_sha() -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(_REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _make_arm(
    name: str, lower_is_better: bool, task_description: str, store, run_id: str, seed: int
):
    if name == "random":
        return RandomArm(lower_is_better=lower_is_better)
    if name == "greedy":
        return GreedyArm(lower_is_better=lower_is_better, k=3)
    if name == "evolutionary":
        return EvolutionaryArm(
            lower_is_better=lower_is_better, config=EvolutionaryArmConfig(mu=3, lambda_=3)
        )
    if name == "llm_iter":
        return LLMIterArm(
            lower_is_better=lower_is_better,
            task_description=task_description,
            provider=MockLLMProvider(seed=seed),
            result_store=store,
            run_id=run_id,
            budget=None,  # LLM_API_BUDGET_USD unset -> mock provider only, no paid calls
        )
    if name == "llm_evo":
        return LLMEvoArm(
            lower_is_better=lower_is_better,
            task_description=task_description,
            provider=MockLLMProvider(seed=seed),
            result_store=store,
            run_id=run_id,
            budget=None,
        )
    raise ValueError(f"unknown arm {name!r}")


def main() -> None:
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    config = load_config(sys.argv[1])
    params = config.params
    task_name = params["task"]
    budget = params["budget"]
    arm_names = params["arms"]
    interrupt_arm = params.get("interrupt_and_resume_arm")
    training_config = TrainingConfig(**params["training"])

    task = TASK_REGISTRY[task_name]()
    train_val = task.build(seed=config.seed)
    task_description = _TASK_DESCRIPTIONS.get(task_name, task_name)
    git_sha = _git_sha()

    run_dir = DEFAULT_RUNS_DIR / config.name
    run_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("STAGE 7 SMOKE COMPARISON -- SYSTEM VALIDATION ONLY.")
    print("NOT SCIENTIFIC EVIDENCE. Do not cite these numbers as a result.")
    print("=" * 70)
    print(f"task={task_name} budget={budget} seed={config.seed} arms={arm_names}")
    print(f"git_sha={git_sha}\n")

    results = {}
    for arm_name in arm_names:
        run_id = f"{config.name}_{arm_name}"
        db_path = run_dir / f"{arm_name}.sqlite"
        repro_fields = {
            "task": task_name, "arm": arm_name, "budget": budget,
            "seed": config.seed, "training": training_config.model_dump(),
        }

        lower_is_better = train_val.spec.lower_is_better

        if arm_name == interrupt_arm:
            half = max(1, budget // 2)
            print(f"--- {arm_name}: running {half}/{budget} candidates, simulating a crash ---")
            store = ResultStore.open_or_create(
                db_path, run_id=run_id, config_json=config.model_dump_json(),
                config_reproducibility_fields=repro_fields, git_sha=git_sha,
                created_at=datetime.now(timezone.utc).isoformat(),
            )
            arm = _make_arm(arm_name, lower_is_better, task_description, store, run_id, config.seed)
            SearchRunner(
                arm, task_name, train_val, training_config,
                budget_limit=half, run_seed=config.seed, result_store=store, run_id=run_id,
            ).run()
            store.close()
            print(f"--- {arm_name}: 'exited'. Reopening store, resuming to budget={budget} ---")

        store = ResultStore.open_or_create(
            db_path, run_id=run_id, config_json=config.model_dump_json(),
            config_reproducibility_fields=repro_fields, git_sha=git_sha,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        arm = _make_arm(arm_name, lower_is_better, task_description, store, run_id, config.seed)
        runner = SearchRunner(
            arm, task_name, train_val, training_config,
            budget_limit=budget, run_seed=config.seed, result_store=store, run_id=run_id,
        )
        result = runner.run()
        results[arm_name] = result
        print(f"[{arm_name}] ledger={result.ledger_summary}")
        print(
            f"[{arm_name}] selected_hash={(result.selected_structural_hash or '')[:12]} "
            f"selected_val_metric={result.selected_val_metric_value}"
        )
        store.close()
        print()

    print("=" * 70)
    print("Summary (best-so-far VALIDATION metric of the selected circuit; "
          "lower is better for T1's RMSE):")
    for arm_name, result in results.items():
        print(f"  {arm_name:14s} val_metric={result.selected_val_metric_value}")
    print("\nReminder: this is a smoke run for system validation only.")


if __name__ == "__main__":
    main()
