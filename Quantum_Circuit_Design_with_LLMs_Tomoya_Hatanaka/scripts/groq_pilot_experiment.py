#!/usr/bin/env python
"""Groq free-tier pilot (provider: groq, model openai/gpt-oss-20b).

A NEW, ISOLATED experiment namespace: store `runs/groq_pilot_t1/` (smoke:
`runs/groq_smoke/`), run_ids prefixed `groq_gpt_oss_20b_`. Never touches
the OpenAI stores (`runs/pilot_t1/`, `runs/mini_llm_experiment/`), which
are preserved as separate preliminary evidence. Provider+model are part
of every run's reproducibility-compat hash, so resuming under a different
provider/model raises `IncompatibleResumeError`.

Predeclared pilot config (recorded before inspecting any results):
T1, frozen split seed 0; arms random / evolutionary / greedy /
llm open-loop / llm closed-loop / llm_evo; B=10 per arm; seeds {0,1,2};
full fixed TrainingConfig (20 epochs); validation-only feedback;
protected final test once per run after search ends.

Token plan: ~110 LLM budget-consuming calls x ~(400 in + 512 out max,
reasoning included in output usage) << the 180,000-token internal cap
(itself >=20k under the documented 200k/day free limit). The
`FreeTierController` stops BEFORE any request that would cross a cap;
runs then checkpoint durably and resume later under the same
provider/model only.

Usage:
  GROQ_API_KEY=... python scripts/groq_pilot_experiment.py smoke   # B=2, seed 0, LLM arms
  GROQ_API_KEY=... python scripts/groq_pilot_experiment.py pilot   # full predeclared config
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
from llm_vqc.llm.groq_provider import (  # noqa: E402
    GROQ_MODEL,
    FreeTierController,
    FreeTierExhaustedError,
    GroqProvider,
)
from llm_vqc.search.arms.evolutionary_arm import (  # noqa: E402
    EvolutionaryArm,
    EvolutionaryArmConfig,
)
from llm_vqc.search.arms.greedy_arm import GreedyArm  # noqa: E402
from llm_vqc.search.arms.llm_evo_arm import LLMEvoArm  # noqa: E402
from llm_vqc.search.arms.llm_iter_arm import LLMIterArm  # noqa: E402
from llm_vqc.search.arms.random_arm import RandomArm  # noqa: E402
from llm_vqc.search.feedback import SearchFeedback  # noqa: E402
from llm_vqc.tasks.t1_gaussian import T1GaussianPeakTask  # noqa: E402

TASK_NAME = "T1"
DATA_SPLIT_SEED = 0
TEMPERATURE = 0.2  # provider enforces its own low-temp decoding
HISTORY_WINDOW = 4  # compact closed-loop feedback (token-limited free tier)
PREFIX = "groq_gpt_oss_20b"
_REPO_ROOT = Path(__file__).resolve().parent.parent


def _git_sha():
    try:
        r = subprocess.run(["git", "-C", str(_REPO_ROOT), "rev-parse", "HEAD"],
                           capture_output=True, text=True, check=True)
        return r.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


# Complete replacement system prompt for this provider condition (fixed
# across seeds, fully recorded in every LLMCallRecord's system_prompt).
# Replaces (not appends to) the shared template because the shared one
# advertises the `"all"` wires shorthand, which this provider's strict
# schema deliberately does not accept (array-of-int only). Content policy
# is unchanged: task description + output format only, no extra hints.
GROQ_SYSTEM_PROMPT = """\
You design variational quantum circuits for the task: {task}.
Respond with ONLY one JSON object.
CRITICAL: every "wires" field is an array of separate integers. For 3 qubits
write "wires": [0, 1, 2] -- three separate numbers, no quotes."""


# Repeated in every user turn: at low reasoning effort the model follows
# the most recent instruction far more reliably than the system prompt
# (verified live). Pure format instruction, fixed across seeds, recorded.
GROQ_USER_SUFFIX = (
    "\nPropose it as JSON: n_qubits (2-10); encoding {type:\"angle\", gate:\"RX/RY/RZ\", "
    "wires:[integers], reupload:0}; layers, each either rot {type:\"rot\", "
    "gates:[\"RX/RY/RZ/H\"], wires:[integers]} or entangle {type:\"entangle\", "
    "pattern:\"ring/line/star/all_to_all/pairs/none\", "
    "gate:\"CNOT/CZ/CRZ\", wires:[integers], center:int-or-null, pairs:[[c,t],...]-or-null}; "
    "measurements {observable:\"X/Y/Z\", wires:[integers]}. "
    "Remember: wires like [0, 1, 2], never \"012\"."
)


def _make_arm(condition, lower, provider, budget, store, run_id):
    llm_common = dict(
        lower_is_better=lower,
        task_description="T1 1D Gaussian-peak regression (predict the peak location)",
        provider=provider, result_store=store, run_id=run_id,
        temperature=TEMPERATURE, max_repair_attempts=0,  # strict schema; no free repairs
        budget=budget, cost_estimate_per_call_usd=0.0,  # free plan: token caps, not dollars
        user_prompt_suffix=GROQ_USER_SUFFIX,
    )
    if condition == "random":
        return RandomArm(lower_is_better=lower)
    if condition == "evolutionary":
        return EvolutionaryArm(lower_is_better=lower, config=EvolutionaryArmConfig(mu=3, lambda_=3))
    if condition == "greedy":
        return GreedyArm(lower_is_better=lower, k=3)
    if condition == "llm_open":
        arm = LLMIterArm(open_loop=True, **llm_common)
    elif condition == "llm_closed":
        arm = LLMIterArm(open_loop=False, history_window=HISTORY_WINDOW, **llm_common)
    elif condition == "llm_evo":
        arm = LLMEvoArm(**llm_common)
    else:
        raise ValueError(condition)
    arm.system_prompt = GROQ_SYSTEM_PROMPT.format(task=llm_common["task_description"])
    return arm


def _run(arm, train_val, tc, budget_limit, seed, store, run_id):
    ledger = BudgetLedger.from_store(store, run_id)
    raw = store.load_run_state(run_id)
    state = arm.deserialize_state(raw) if raw is not None else arm.initialize(seed)
    if raw is None:
        store.save_run_state(run_id, arm.serialize_state(state))
    i = ledger.num_proposed
    stop = None
    max_proposals = budget_limit * 3  # invalid proposals are budget-free; cap total attempts
    while not ledger.is_exhausted(budget_limit):
        if i >= max_proposals:
            stop = f"proposal_cap: {i} proposals without exhausting budget"
            break
        try:
            proposal = arm.propose(state)
            result = evaluate_candidate(
                proposal, TASK_NAME, seed, train_val, tc, f"{run_id}:{i}", ledger,
                cache=store, proposal_event_store=store, run_id=run_id)
            state = arm.update_state(state, proposal, SearchFeedback.from_evaluation_result(result))
            store.save_run_state(run_id, arm.serialize_state(state))
            i += 1
        except FreeTierExhaustedError as exc:
            stop = f"free_tier_cap: {exc}"
            break
        except Exception as exc:
            stop = f"error: {type(exc).__name__}: {exc}"
            break
    best = arm.select_final(state)
    ts = vn = vv = None
    if best is not None:
        ts = train_seed_for_circuit(seed, best)
        cached = store.get_cached(TASK_NAME, best, ts)
        if cached is not None:
            vn, vv = cached.val_metric_name, cached.val_metric_value
    return {"run_id": run_id, "ledger_summary": ledger.summary(),
            "selected_structural_hash": best, "selected_train_seed": ts,
            "selected_val_metric_name": vn, "selected_val_metric_value": vv,
            "stop_reason": stop}


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "pilot"
    assert mode in ("smoke", "pilot")
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        print("BLOCKED: no GROQ_API_KEY in process environment.")
        sys.exit(1)

    if mode == "smoke":
        run_dir, budget_limit, seeds = _REPO_ROOT / "runs" / "groq_smoke", 2, (0,)
        conditions = ("llm_open", "llm_closed", "llm_evo")
        tc = TrainingConfig(epochs=2)
        label = "SMOKE (system validation only -- not evidence)"
    else:
        run_dir, budget_limit, seeds = _REPO_ROOT / "runs" / "groq_pilot_t1", 10, (0, 1, 2)
        conditions = ("random", "evolutionary", "greedy", "llm_open", "llm_closed", "llm_evo")
        tc = TrainingConfig()  # full fixed 20-epoch pilot training protocol
        label = "PILOT (free-tier, B=10, predeclared before results)"

    run_dir.mkdir(parents=True, exist_ok=True)
    db = run_dir / "results.sqlite"
    controller = FreeTierController(max_requests=250)
    provider = GroqProvider(api_key=api_key, controller=controller)
    os.environ.setdefault("LLM_API_BUDGET_USD", "1.00")  # required nonzero cap; $0 is spent
    dollar_budget = LLMApiBudget.from_env()

    task = T1GaussianPeakTask()
    train_val = task.build(seed=DATA_SPLIT_SEED)
    test_split = task.build_test(seed=DATA_SPLIT_SEED)
    git_sha = _git_sha()
    t0 = time.monotonic()
    print(f"{label}: model={GROQ_MODEL} B={budget_limit} seeds={seeds} arms={conditions}")

    runs = []
    for condition in conditions:
        for seed in seeds:
            run_id = f"{PREFIX}_{condition}_s{seed}"
            store = ResultStore.open_or_create(
                db, run_id=run_id, config_json="{}",
                config_reproducibility_fields={
                    "provider": "groq", "model": GROQ_MODEL, "task": TASK_NAME,
                    "arm": condition, "budget": budget_limit, "mode": mode,
                    "data_split_seed": DATA_SPLIT_SEED, "repetition_seed": seed,
                    "temperature": TEMPERATURE, "history_window": HISTORY_WINDOW,
                    "training": tc.model_dump()},
                git_sha=git_sha, created_at=datetime.now(timezone.utc).isoformat())
            arm = _make_arm(condition, train_val.spec.lower_is_better,
                            provider, dollar_budget, store, run_id)
            r = _run(arm, train_val, tc, budget_limit, seed, store, run_id)
            print(f"[{run_id}] {r['ledger_summary']} val={r['selected_val_metric_value']} "
                  f"stop={r['stop_reason']} tok={controller.total_tokens} "
                  f"req={controller.requests_made}")
            if r["selected_structural_hash"] is not None:
                w = store.get_trained_weights(TASK_NAME, r["selected_structural_hash"],
                                              r["selected_train_seed"])
                c = store.get_cached(TASK_NAME, r["selected_structural_hash"],
                                     r["selected_train_seed"])
                if w is not None and c is not None and c.circuit_canonical_json:
                    ir = CircuitIR.model_validate_json(c.circuit_canonical_json)
                    tr = evaluate_on_test(ir, w["classical_state"], test_split,
                                          train_val.spec, r["selected_train_seed"])
                    r["test_metric_name"] = tr.test_metric_name
                    r["test_metric_value"] = tr.test_metric_value
                    print(f"[{run_id}] FINAL TEST rmse={tr.test_metric_value}")
                    if c.circuit_cost is not None:
                        r["selected_circuit_cost"] = c.circuit_cost.model_dump()
            calls = list(store.iter_llm_calls(run_id))
            r["llm_call_count"] = len(calls)
            r["input_tokens"] = sum(x.input_tokens for x in calls)
            r["output_tokens"] = sum(x.output_tokens for x in calls)
            r["mean_latency_seconds"] = (
                sum(x.latency_seconds for x in calls) / len(calls) if calls else None)
            runs.append(r)
            store.close()
            if r["stop_reason"] and "free_tier_cap" in r["stop_reason"]:
                print("Free-tier cap reached -- checkpointing and stopping the matrix.")
                break
        else:
            continue
        break

    summary_path = run_dir / f"groq_{mode}_summary.json"
    summary_path.write_text(json.dumps({
        "provider": "groq", "model": GROQ_MODEL, "mode": mode, "label": label,
        "git_sha": git_sha, "task": TASK_NAME, "budget": budget_limit,
        "seeds": list(seeds), "arms": list(conditions),
        "temperature": TEMPERATURE, "history_window": HISTORY_WINDOW,
        "reasoning_effort": provider.reasoning_effort,
        "max_output_tokens": provider.max_output_tokens,
        "free_tier": {"max_total_tokens": controller.max_total_tokens,
                      "max_requests": controller.max_requests,
                      "tokens_used": controller.total_tokens,
                      "requests_made": controller.requests_made},
        "elapsed_seconds": time.monotonic() - t0,
        "training_config": tc.model_dump(), "runs": runs}, indent=2))
    print(f"tokens={controller.total_tokens}/{controller.max_total_tokens} "
          f"requests={controller.requests_made}/{controller.max_requests}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
