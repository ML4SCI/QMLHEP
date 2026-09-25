#!/usr/bin/env python
"""Stage 8: the first real pilot experiment (LLM-VQC_MASTER_PLAN.md
Section 9, Phase 6): all arms, T1, budget B=25, 3 seeds.

**Scope note (governance decision G4-A):** `LLM_API_BUDGET_USD` is unset
in this environment, so per the controlling cost policy no paid LLM API
call is authorized. This pilot therefore runs the three arms that can be
legitimately executed with real, non-fabricated results: `random`,
`evolutionary`, `greedy`. `llm_iter`/`llm_evo` were fully implemented and
validated with `MockLLMProvider` (Stages 5 and 7) but are NOT included
here -- running them with a mock and reporting the numbers as pilot
results would misrepresent a seeded random-circuit generator as "LLM
guidance," which the research-integrity constraints ("do not fabricate
unavailable API/dataset/experiment results") forbid.

**Protocol, applied identically to all three arms:**
- One frozen data split for the entire pilot (`task.build(seed=0)` /
  `task.build_test(seed=0)`) -- shared across every arm and every
  repetition seed, so the split itself is never a confound.
- 3 repetition seeds (0, 1, 2) per arm, each controlling only the
  search-RNG and per-candidate training-init streams (never the data
  split) -- matching Section 6.4's "training noise... deterministic
  training seed" and Section 11's three-independent-streams requirement.
- Budget B=25 candidate evaluations per run, full master-plan training
  pipeline (`TrainingConfig()` defaults: AdamW, lr=0.05, 20 epochs, batch
  16 -- not the reduced smoke settings).
- All 9 (arm x seed) runs share ONE SQLite result store file, so an
  identical (structural_hash, train_seed) proposed by two different arms
  or seeds is a genuine cache hit ("global caching", Section 6.4).
- Search selects the final candidate on VALIDATION only. The quarantined
  final-test evaluation (`llm_vqc.evaluation.final_test.evaluate_on_test`)
  is run exactly once per run, only after that run's search has
  completely finished, on the exact already-trained weights (never
  retrained) -- selection-then-test order is never reversed.

This script is the search + final-test EXECUTION step. Run
`scripts/analyze_pilot.py` afterward for the statistical analysis
(anytime curves, Kruskal-Wallis, Mann-Whitney U + Holm, Cliff's delta).
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm_vqc.evaluation.final_test import evaluate_on_test  # noqa: E402
from llm_vqc.evaluation.store import ResultStore  # noqa: E402
from llm_vqc.evaluation.training import TrainingConfig  # noqa: E402
from llm_vqc.ir.schema import CircuitIR  # noqa: E402
from llm_vqc.search.arms.evolutionary_arm import (  # noqa: E402
    EvolutionaryArm,
    EvolutionaryArmConfig,
)
from llm_vqc.search.arms.greedy_arm import GreedyArm  # noqa: E402
from llm_vqc.search.arms.random_arm import RandomArm  # noqa: E402
from llm_vqc.search.runner import SearchRunner  # noqa: E402
from llm_vqc.tasks.t1_gaussian import T1GaussianPeakTask  # noqa: E402

TASK_NAME = "T1"
DATA_SPLIT_SEED = 0  # one frozen split for the whole pilot
BUDGET = 25
SEEDS = (0, 1, 2)
ARMS = ("random", "evolutionary", "greedy")

_REPO_ROOT = Path(__file__).resolve().parent.parent
RUN_DIR = _REPO_ROOT / "runs" / "pilot_t1"
DB_PATH = RUN_DIR / "results.sqlite"
SUMMARY_PATH = RUN_DIR / "pilot_summary.json"


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


def _make_arm(name: str, lower_is_better: bool):
    if name == "random":
        return RandomArm(lower_is_better=lower_is_better)
    if name == "greedy":
        return GreedyArm(lower_is_better=lower_is_better, k=5)
    if name == "evolutionary":
        return EvolutionaryArm(
            lower_is_better=lower_is_better, config=EvolutionaryArmConfig(mu=5, lambda_=5)
        )
    raise ValueError(f"unknown non-LLM arm {name!r}")


def main() -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    task = T1GaussianPeakTask()
    train_val = task.build(seed=DATA_SPLIT_SEED)
    test_split = task.build_test(seed=DATA_SPLIT_SEED)
    training_config = TrainingConfig()  # master-plan defaults: AdamW, lr=0.05, 20 epochs, batch 16
    git_sha = _git_sha()
    git_dirty = _git_dirty()

    print("=" * 70)
    print("STAGE 8 PILOT EXPERIMENT -- REAL RESULTS (random/evolutionary/greedy only)")
    print(f"task={TASK_NAME} budget={BUDGET} seeds={SEEDS} arms={ARMS}")
    print(f"git_sha={git_sha} git_dirty={git_dirty}")
    print("=" * 70)

    all_runs = []
    for arm_name in ARMS:
        for seed in SEEDS:
            run_id = f"{arm_name}_s{seed}"
            store = ResultStore.open_or_create(
                DB_PATH, run_id=run_id, config_json="{}",
                config_reproducibility_fields={
                    "task": TASK_NAME, "arm": arm_name, "budget": BUDGET,
                    "data_split_seed": DATA_SPLIT_SEED, "repetition_seed": seed,
                    "training": training_config.model_dump(),
                },
                git_sha=git_sha, created_at=datetime.now(timezone.utc).isoformat(),
            )
            arm = _make_arm(arm_name, train_val.spec.lower_is_better)
            runner = SearchRunner(
                arm, TASK_NAME, train_val, training_config,
                budget_limit=BUDGET, run_seed=seed, result_store=store, run_id=run_id,
                git_sha=git_sha, git_dirty=git_dirty,
            )
            search_result = runner.run()
            print(
                f"[{run_id}] ledger={search_result.ledger_summary} "
                f"selected_val={search_result.selected_val_metric_value}"
            )

            test_result = None
            selected_hash = search_result.selected_structural_hash
            selected_seed = search_result.selected_train_seed
            if selected_hash is not None:
                weights = store.get_trained_weights(TASK_NAME, selected_hash, selected_seed)
                cached_eval = store.get_cached(TASK_NAME, selected_hash, selected_seed)
                has_ir_json = cached_eval is not None and cached_eval.circuit_canonical_json
                if weights is not None and has_ir_json:
                    ir = CircuitIR.model_validate_json(cached_eval.circuit_canonical_json)
                    test_result = evaluate_on_test(
                        ir, weights["classical_state"], test_split, train_val.spec, selected_seed
                    )
                    print(
                        f"[{run_id}] FINAL TEST {test_result.test_metric_name}="
                        f"{test_result.test_metric_value}"
                    )

            all_runs.append(
                {
                    "run_id": run_id,
                    "arm": arm_name,
                    "seed": seed,
                    "ledger_summary": search_result.ledger_summary,
                    "selected_structural_hash": search_result.selected_structural_hash,
                    "selected_val_metric_name": search_result.selected_val_metric_name,
                    "selected_val_metric_value": search_result.selected_val_metric_value,
                    "test_metric_name": test_result.test_metric_name if test_result else None,
                    "test_metric_value": test_result.test_metric_value if test_result else None,
                }
            )
            store.close()

    SUMMARY_PATH.write_text(
        json.dumps(
            {
                "git_sha": git_sha,
                "git_dirty": git_dirty,
                "task": TASK_NAME,
                "budget": BUDGET,
                "seeds": list(SEEDS),
                "arms": list(ARMS),
                "data_split_seed": DATA_SPLIT_SEED,
                "training_config": training_config.model_dump(),
                "runs": all_runs,
            },
            indent=2,
        )
    )
    print(f"\nWrote machine-readable summary to {SUMMARY_PATH}")
    print("Run scripts/analyze_pilot.py next for statistical analysis.")


if __name__ == "__main__":
    main()
