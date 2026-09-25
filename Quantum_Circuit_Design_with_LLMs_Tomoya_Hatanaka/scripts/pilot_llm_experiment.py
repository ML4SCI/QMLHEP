#!/usr/bin/env python
"""LLM-arm pilot experiment with a REAL OpenAI provider — completes the
Stage 8 pilot (master plan Phase 6: T1, B=25, 3 seeds) for the arms that
required a human-approved API budget.

Identical conditions to the already-run non-LLM pilot
(`scripts/pilot_experiment.py`): same frozen data split (seed 0), same
budget B=25, same full `TrainingConfig()` (20 epochs), same repetition
seeds {0,1,2}, and the SAME SQLite store (`runs/pilot_t1/results.sqlite`)
so identical (structural_hash, train_seed) pairs across arms are genuine
cache hits and the comparison is budget-matched by construction.

Arms: llm_iter closed-loop, llm_iter open-loop (ablation A5), llm_evo.

Cost controls (human-approved for this run):
- `LLM_API_BUDGET_USD` = 10.00 (hard dollar cap, run-specific override
  via the sanctioned `from_env()` mechanism);
- conservative $0.02/call charged against the cap (OpenAI reports no
  dollar cost; the estimate is deliberately ~20x the likely true cost);
- global real-call cap 500 + per-run cap 100 (nested
  `CallCountLimitedProvider`s);
- max 2 repair retries per malformed response (master plan Phase 5).

A run interrupted by a cap or API error keeps all persisted results and
is resumable later (durable ledger + arm-state checkpoints); the budget
never resets. Secrets: the API key enters only via the process
environment and is never printed or written anywhere.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm_vqc.evaluation.final_test import evaluate_on_test  # noqa: E402
from llm_vqc.evaluation.harness import evaluate_candidate  # noqa: E402
from llm_vqc.evaluation.seeds import train_seed_for_circuit  # noqa: E402
from llm_vqc.evaluation.store import ResultStore  # noqa: E402
from llm_vqc.evaluation.training import TrainingConfig  # noqa: E402
from llm_vqc.ir.budget import BudgetLedger  # noqa: E402
from llm_vqc.ir.schema import CircuitIR  # noqa: E402
from llm_vqc.llm.budget import LLMApiBudget, LLMBudgetExceededError  # noqa: E402
from llm_vqc.llm.openai_provider import (  # noqa: E402
    CallBudgetExceededError,
    CallCountLimitedProvider,
    OpenAIProvider,
)
from llm_vqc.search.arms.llm_evo_arm import LLMEvoArm  # noqa: E402
from llm_vqc.search.arms.llm_iter_arm import LLMIterArm  # noqa: E402
from llm_vqc.search.feedback import SearchFeedback  # noqa: E402
from llm_vqc.tasks.t1_gaussian import T1GaussianPeakTask  # noqa: E402

TASK_NAME = "T1"
DATA_SPLIT_SEED = 0  # same frozen split as the non-LLM pilot
BUDGET = 25
SEEDS = (0, 1, 2)
MAX_REPAIR_ATTEMPTS = 2
COST_ESTIMATE_PER_CALL_USD = 0.02
DOLLAR_CAP_USD = "10.00"
GLOBAL_CALL_CAP = 500
PER_RUN_CALL_CAP = 100
WALL_CLOCK_LIMIT_S = 90 * 60  # generous overall safety deadline
TEMPERATURE = 0.7  # master plan section 6.2 fixed value
TASK_DESCRIPTION = "T1 1D Gaussian-peak regression (predict the peak location)"

_REPO_ROOT = Path(__file__).resolve().parent.parent
RUN_DIR = _REPO_ROOT / "runs" / "pilot_t1"
DB_PATH = RUN_DIR / "results.sqlite"
SUMMARY_PATH = RUN_DIR / "pilot_llm_summary.json"

ARM_CONDITIONS = ("llm_iter_closed", "llm_iter_open", "llm_evo")


def _git_sha() -> str | None:
    try:
        r = subprocess.run(
            ["git", "-C", str(_REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return r.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _make_arm(condition, lower_is_better, provider, budget, store, run_id):
    common = dict(
        lower_is_better=lower_is_better,
        task_description=TASK_DESCRIPTION,
        provider=provider,
        result_store=store,
        run_id=run_id,
        temperature=TEMPERATURE,
        max_repair_attempts=MAX_REPAIR_ATTEMPTS,
        budget=budget,
        cost_estimate_per_call_usd=COST_ESTIMATE_PER_CALL_USD,
    )
    if condition == "llm_iter_closed":
        return LLMIterArm(open_loop=False, **common)
    if condition == "llm_iter_open":
        return LLMIterArm(open_loop=True, **common)
    if condition == "llm_evo":
        return LLMEvoArm(**common)
    raise ValueError(condition)


def _run_search(arm, train_val, training_config, run_seed, store, run_id, deadline):
    ledger = BudgetLedger.from_store(store, run_id)
    raw_state = store.load_run_state(run_id)
    state = arm.deserialize_state(raw_state) if raw_state is not None else arm.initialize(run_seed)
    if raw_state is None:
        store.save_run_state(run_id, arm.serialize_state(state))

    proposal_index = ledger.num_proposed
    stop_reason = None
    while not ledger.is_exhausted(BUDGET):
        if time.monotonic() >= deadline:
            stop_reason = "wall_clock_deadline"
            break
        try:
            raw_proposal = arm.propose(state)
            result = evaluate_candidate(
                raw_proposal, TASK_NAME, run_seed, train_val, training_config,
                f"{run_id}:{proposal_index}", ledger,
                cache=store, proposal_event_store=store, run_id=run_id,
            )
            feedback = SearchFeedback.from_evaluation_result(result)
            state = arm.update_state(state, raw_proposal, feedback)
            store.save_run_state(run_id, arm.serialize_state(state))
            proposal_index += 1
        except (CallBudgetExceededError, LLMBudgetExceededError) as exc:
            stop_reason = f"budget_cap: {exc}"
            break
        except Exception as exc:  # API/network errors: preserve completed results, stop cleanly
            stop_reason = f"error: {type(exc).__name__}: {exc}"
            break

    best_hash = arm.select_final(state)
    train_seed = val_name = val_value = None
    if best_hash is not None:
        train_seed = train_seed_for_circuit(run_seed, best_hash)
        cached = store.get_cached(TASK_NAME, best_hash, train_seed)
        if cached is not None:
            val_name, val_value = cached.val_metric_name, cached.val_metric_value
    return {
        "run_id": run_id,
        "ledger_summary": ledger.summary(),
        "selected_structural_hash": best_hash,
        "selected_train_seed": train_seed,
        "selected_val_metric_name": val_name,
        "selected_val_metric_value": val_value,
        "stop_reason": stop_reason,
    }


def main() -> None:
    t_start = time.monotonic()
    deadline = t_start + WALL_CLOCK_LIMIT_S

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        print("BLOCKED: no OPENAI_API_KEY in process environment.")
        sys.exit(1)
    model = os.environ.get("OPENAI_MODEL", "gpt-5.4-mini").strip()

    os.environ["LLM_API_BUDGET_USD"] = DOLLAR_CAP_USD
    budget = LLMApiBudget.from_env()
    assert budget is not None

    # 6.5s pacing for the org's 10-requests/minute limit on this model,
    # plus provider-internal 429 retries with backoff.
    real = OpenAIProvider(
        api_key=api_key, model=model, max_output_tokens=700, min_seconds_between_calls=6.5
    )
    global_capped = CallCountLimitedProvider(real, max_calls=GLOBAL_CALL_CAP)

    task = T1GaussianPeakTask()
    train_val = task.build(seed=DATA_SPLIT_SEED)
    test_split = task.build_test(seed=DATA_SPLIT_SEED)
    training_config = TrainingConfig()  # full pilot settings: 20 epochs etc.
    git_sha = _git_sha()

    print(f"model={model} dollar_cap=${budget.cap_usd} global_call_cap={GLOBAL_CALL_CAP} "
          f"budget_per_run={BUDGET} seeds={SEEDS} arms={ARM_CONDITIONS}")

    all_runs = []
    for condition in ARM_CONDITIONS:
        for seed in SEEDS:
            run_id = f"{condition}_s{seed}"
            if time.monotonic() >= deadline:
                all_runs.append({"run_id": run_id, "stop_reason": "not_started_deadline"})
                continue
            store = ResultStore.open_or_create(
                DB_PATH, run_id=run_id, config_json="{}",
                config_reproducibility_fields={
                    "task": TASK_NAME, "arm": condition, "budget": BUDGET,
                    "data_split_seed": DATA_SPLIT_SEED, "repetition_seed": seed,
                    "model": model, "temperature": TEMPERATURE,
                    "training": training_config.model_dump(),
                },
                git_sha=git_sha, created_at=datetime.now(timezone.utc).isoformat(),
            )
            per_run_capped = CallCountLimitedProvider(global_capped, max_calls=PER_RUN_CALL_CAP)
            arm = _make_arm(
                condition, train_val.spec.lower_is_better, per_run_capped, budget, store, run_id
            )
            print(f"--- {run_id} ---")
            result = _run_search(arm, train_val, training_config, seed, store, run_id, deadline)
            result["real_calls_this_run"] = per_run_capped.calls_made
            print(f"[{run_id}] ledger={result['ledger_summary']} "
                  f"val={result['selected_val_metric_value']} stop={result['stop_reason']} "
                  f"calls={per_run_capped.calls_made} global={global_capped.calls_made} "
                  f"spent=${budget.spent_usd:.2f}")

            selected_hash = result["selected_structural_hash"]
            selected_seed = result["selected_train_seed"]
            if selected_hash is not None:
                weights = store.get_trained_weights(TASK_NAME, selected_hash, selected_seed)
                cached = store.get_cached(TASK_NAME, selected_hash, selected_seed)
                if weights is not None and cached is not None and cached.circuit_canonical_json:
                    ir = CircuitIR.model_validate_json(cached.circuit_canonical_json)
                    test_result = evaluate_on_test(
                        ir, weights["classical_state"], test_split, train_val.spec, selected_seed
                    )
                    result["test_metric_name"] = test_result.test_metric_name
                    result["test_metric_value"] = test_result.test_metric_value
                    print(f"[{run_id}] FINAL TEST rmse={test_result.test_metric_value}")

            llm_calls = list(store.iter_llm_calls(run_id))
            result["llm_call_count"] = len(llm_calls)
            result["input_tokens"] = sum(c.input_tokens for c in llm_calls)
            result["output_tokens"] = sum(c.output_tokens for c in llm_calls)
            result["mean_latency_seconds"] = (
                sum(c.latency_seconds for c in llm_calls) / len(llm_calls) if llm_calls else None
            )
            all_runs.append(result)
            store.close()

    elapsed = time.monotonic() - t_start
    SUMMARY_PATH.write_text(json.dumps({
        "git_sha": git_sha,
        "model": model,
        "provider": "OpenAIProvider (real)",
        "task": TASK_NAME, "budget": BUDGET, "seeds": list(SEEDS),
        "arms": list(ARM_CONDITIONS),
        "data_split_seed": DATA_SPLIT_SEED,
        "temperature": TEMPERATURE,
        "max_repair_attempts": MAX_REPAIR_ATTEMPTS,
        "dollar_cap_usd": budget.cap_usd,
        "dollar_cap_spent_estimated_usd": budget.spent_usd,
        "cost_estimate_per_call_usd": COST_ESTIMATE_PER_CALL_USD,
        "global_real_calls_made": global_capped.calls_made,
        "global_call_cap": GLOBAL_CALL_CAP,
        "elapsed_seconds": elapsed,
        "training_config": training_config.model_dump(),
        "runs": all_runs,
    }, indent=2))
    print(f"\nelapsed={elapsed:.0f}s global_calls={global_capped.calls_made}/{GLOBAL_CALL_CAP} "
          f"spent_estimated=${budget.spent_usd:.2f}/${budget.cap_usd}")
    print(f"Wrote {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
