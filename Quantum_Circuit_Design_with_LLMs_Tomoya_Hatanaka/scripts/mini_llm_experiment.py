#!/usr/bin/env python
"""Time-boxed mini LAQS-Bench experiment with a REAL OpenAI provider.

Reuses the already-validated CircuitIR / validator / PennyLane compiler /
harness / test-quarantine / SQLite store / durable budget / random /
evolutionary / greedy / LLMIterArm (open- and closed-loop) unchanged.

**Deliberate adaptation for this run only:** a thin custom loop
(`_run_arm_with_deadline`) replaces `SearchRunner.run()`'s internal loop
so a wall-clock deadline and a real-provider exception (call-count cap,
network/rate-limit error) can interrupt an arm cleanly mid-run without
losing already-persisted results -- `SearchRunner` itself is unmodified;
this script does not redesign it, it wraps the same primitives
(`evaluate_candidate`, `BudgetLedger`, arm `propose`/`update_state`/
`select_final`) with a deadline check the existing class does not have.

Security: the API key is read from the clipboard by the caller (shell
wrapper) into `OPENAI_API_KEY`; this script never prints it, never
writes it anywhere, and only ever passes it to the OpenAI SDK client
constructor in-process.
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
from llm_vqc.llm.budget import LLMApiBudget  # noqa: E402
from llm_vqc.llm.openai_provider import (  # noqa: E402
    CallBudgetExceededError,
    CallCountLimitedProvider,
    OpenAIProvider,
)
from llm_vqc.search.arms.evolutionary_arm import (  # noqa: E402
    EvolutionaryArm,
    EvolutionaryArmConfig,
)
from llm_vqc.search.arms.greedy_arm import GreedyArm  # noqa: E402
from llm_vqc.search.arms.llm_iter_arm import LLMIterArm  # noqa: E402
from llm_vqc.search.arms.random_arm import RandomArm  # noqa: E402
from llm_vqc.search.feedback import SearchFeedback  # noqa: E402
from llm_vqc.tasks.t1_gaussian import T1GaussianPeakTask  # noqa: E402

TASK_NAME = "T1"
DATA_SPLIT_SEED = 0
SEARCH_SEED = 0
BUDGET_PER_ARM = 2
MAX_REAL_CALLS = 12
MAX_REPAIR_ATTEMPTS = 1
# Conservative per-call charge against the dollar cap. OpenAI's chat
# completions API never returns an actual dollar cost, so the driver
# charges this estimate instead (see LLMDriver); 5 cents/call is far above
# the true cost of a ~500-token mini-model call, so the cap can only be
# hit early, never silently overrun. Without a nonzero value here the
# dollar cap would accumulate zero spend and be decorative.
COST_ESTIMATE_PER_CALL_USD = 0.05
WALL_CLOCK_LIMIT_S = 300
TARGET_S = 240
TASK_DESCRIPTION = "T1 1D Gaussian-peak regression (predict the peak location)"

_REPO_ROOT = Path(__file__).resolve().parent.parent
RUN_DIR = _REPO_ROOT / "runs" / "mini_llm_experiment"
DB_PATH = RUN_DIR / "results.sqlite"
SUMMARY_PATH = RUN_DIR / "mini_summary.json"
REPORT_PATH = RUN_DIR / "mini_report.md"


def _git_sha() -> str | None:
    try:
        r = subprocess.run(
            ["git", "-C", str(_REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return r.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _run_arm_with_deadline(
    arm, task_name, train_val, training_config, budget_limit, run_seed, store, run_id, deadline
):
    ledger = BudgetLedger.from_store(store, run_id)
    raw_state = store.load_run_state(run_id)
    if raw_state is not None:
        state = arm.deserialize_state(raw_state)
    else:
        state = arm.initialize(run_seed)
        store.save_run_state(run_id, arm.serialize_state(state))

    proposal_index = ledger.num_proposed
    stop_reason = None
    while not ledger.is_exhausted(budget_limit):
        if time.monotonic() >= deadline:
            stop_reason = "wall_clock_deadline"
            break
        try:
            proposal_id = f"{run_id}:{proposal_index}"
            raw_proposal = arm.propose(state)
            result = evaluate_candidate(
                raw_proposal, task_name, run_seed, train_val, training_config, proposal_id,
                ledger, cache=store, proposal_event_store=store, run_id=run_id,
            )
            feedback = SearchFeedback.from_evaluation_result(result)
            state = arm.update_state(state, raw_proposal, feedback)
            store.save_run_state(run_id, arm.serialize_state(state))
            proposal_index += 1
        except CallBudgetExceededError as exc:
            stop_reason = f"call_budget_exceeded: {exc}"
            break
        except Exception as exc:  # real network/API errors must not crash the whole experiment
            stop_reason = f"error: {type(exc).__name__}: {exc}"
            break

    best_hash = arm.select_final(state)
    train_seed = None
    val_name = None
    val_value = None
    if best_hash is not None:
        train_seed = train_seed_for_circuit(run_seed, best_hash)
        cached = store.get_cached(task_name, best_hash, train_seed)
        if cached is not None:
            val_name = cached.val_metric_name
            val_value = cached.val_metric_value
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
    deadline = t_start + TARGET_S
    hard_deadline = t_start + WALL_CLOCK_LIMIT_S

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        print("BLOCKED: no OPENAI_API_KEY in process environment (clipboard credential unusable).")
        sys.exit(1)

    model = os.environ.get("OPENAI_MODEL", "gpt-5.4-mini").strip()
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    # Run-specific override only: an explicit, nonzero cap for this one
    # process, via the existing sanctioned from_env() mechanism -- not a
    # deletion or weakening of the cap check itself.
    os.environ["LLM_API_BUDGET_USD"] = "2.00"
    budget = LLMApiBudget.from_env()
    assert budget is not None

    real_provider = OpenAIProvider(api_key=api_key, model=model, max_output_tokens=500)
    shared_provider = CallCountLimitedProvider(real_provider, max_calls=MAX_REAL_CALLS)

    task = T1GaussianPeakTask()
    train_val = task.build(seed=DATA_SPLIT_SEED)
    test_split = task.build_test(seed=DATA_SPLIT_SEED)
    training_config = TrainingConfig(epochs=1, batch_size=16)
    git_sha = _git_sha()

    store = ResultStore.open_or_create(
        DB_PATH, run_id="mini", config_json="{}",
        config_reproducibility_fields={"mini": True, "budget": BUDGET_PER_ARM},
        git_sha=git_sha, created_at=datetime.now(timezone.utc).isoformat(),
        allow_incompatible=True,
    )

    lower_is_better = train_val.spec.lower_is_better
    evo_config = EvolutionaryArmConfig(mu=2, lambda_=2)
    arm_specs = [
        ("random", RandomArm(lower_is_better=lower_is_better)),
        ("evolutionary", EvolutionaryArm(lower_is_better=lower_is_better, config=evo_config)),
        ("greedy", GreedyArm(lower_is_better=lower_is_better, k=2)),
        (
            "llm_iter_open",
            LLMIterArm(
                lower_is_better=lower_is_better, task_description=TASK_DESCRIPTION,
                provider=shared_provider, result_store=store, run_id="llm_iter_open",
                budget=budget, max_repair_attempts=MAX_REPAIR_ATTEMPTS, open_loop=True,
                cost_estimate_per_call_usd=COST_ESTIMATE_PER_CALL_USD,
            ),
        ),
        (
            "llm_iter_closed",
            LLMIterArm(
                lower_is_better=lower_is_better, task_description=TASK_DESCRIPTION,
                provider=shared_provider, result_store=store, run_id="llm_iter_closed",
                budget=budget, max_repair_attempts=MAX_REPAIR_ATTEMPTS, open_loop=False,
                cost_estimate_per_call_usd=COST_ESTIMATE_PER_CALL_USD,
            ),
        ),
    ]
    omitted = ["llm_evo (omitted: kept combined real-call count safely under the 12-call cap)"]

    print(f"model={model} max_real_calls={MAX_REAL_CALLS} budget_per_arm={BUDGET_PER_ARM}")

    run_results = []
    for arm_name, arm in arm_specs:
        if time.monotonic() >= deadline:
            run_results.append({"run_id": arm_name, "stop_reason": "not_started_deadline"})
            continue
        print(f"--- running {arm_name} ---")
        result = _run_arm_with_deadline(
            arm, TASK_NAME, train_val, training_config, BUDGET_PER_ARM,
            SEARCH_SEED, store, arm_name, min(deadline, hard_deadline),
        )
        print(f"[{arm_name}] {result}")

        test_result = None
        selected_hash = result["selected_structural_hash"]
        selected_seed = result["selected_train_seed"]
        if selected_hash is not None and time.monotonic() < hard_deadline:
            weights = store.get_trained_weights(TASK_NAME, selected_hash, selected_seed)
            cached_eval = store.get_cached(TASK_NAME, selected_hash, selected_seed)
            has_ir_json = cached_eval is not None and cached_eval.circuit_canonical_json
            if weights is not None and has_ir_json:
                ir = CircuitIR.model_validate_json(cached_eval.circuit_canonical_json)
                test_result = evaluate_on_test(
                    ir, weights["classical_state"], test_split, train_val.spec, selected_seed
                )
                print(
                    f"[{arm_name}] FINAL TEST {test_result.test_metric_name}="
                    f"{test_result.test_metric_value}"
                )
                result["test_metric_name"] = test_result.test_metric_name
                result["test_metric_value"] = test_result.test_metric_value
        run_results.append(result)

    elapsed = time.monotonic() - t_start
    real_calls_made = shared_provider.calls_made

    llm_calls = list(store.iter_llm_calls("llm_iter_open")) + list(
        store.iter_llm_calls("llm_iter_closed")
    )
    llm_call_records = [
        {
            "run_id": c.proposal_id,
            "model": c.model,
            "input_tokens": c.input_tokens,
            "output_tokens": c.output_tokens,
            "estimated_cost_usd": c.estimated_cost_usd,
            "latency_seconds": c.latency_seconds,
            "validation_errors": c.validation_errors,
        }
        for c in llm_calls
    ]

    summary = {
        "git_sha": git_sha,
        "model": model,
        "provider": "OpenAIProvider (real)",
        "real_calls_made": real_calls_made,
        "dollar_cap_usd": budget.cap_usd,
        "dollar_cap_spent_estimated_usd": budget.spent_usd,
        "cost_estimate_per_call_usd": COST_ESTIMATE_PER_CALL_USD,
        "max_real_calls": MAX_REAL_CALLS,
        "elapsed_seconds": elapsed,
        "budget_per_arm": BUDGET_PER_ARM,
        "arms": [name for name, _ in arm_specs],
        "omitted_arms": omitted,
        "runs": run_results,
        "llm_calls": llm_call_records,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    store.close()

    print(f"\nelapsed={elapsed:.1f}s real_calls={real_calls_made}/{MAX_REAL_CALLS}")
    print(f"Wrote {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
