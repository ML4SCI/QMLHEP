"""The four frozen search methods, run at an arbitrary `Condition`.

Method logic is a literal transcription of the reference study
(`neutral_v5`): Random = B independent uniform draws; Greedy = B/2 random
warm starts + B/2 one-local-mutation refinements; LLM-Open = B semantic
proposals generated before any score is seen; LLM-Closed = B/2 semantic
warm starts + B/2 free-form redesigns of the current best using validation
feedback. Selection and feedback use validation only.

RNG stream offsets are the reference offsets, so at the reference
condition every non-API arm reproduces the previous study bit-for-bit,
and across conditions the architecture draws stay paired wherever the
space is unchanged.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from llm_vqc.experiments.qae_robustness import prompts as P
from llm_vqc.experiments.qae_robustness.conditions import (
    Condition,
    arch_distance,
    evaluate_architecture,
    mutate_one_choice,
    parse_candidate,
    sample_neutral_arch,
    space_for,
    train_seed,
)
from llm_vqc.experiments.qae_tfim.pilot import architecture_key
from llm_vqc.llm.budget import LLMApiBudget
from llm_vqc.llm.records import LLMCallRecord

# Frozen RNG stream offsets (identical to the reference study).
RANDOM_STREAM = 62_000
GREEDY_STREAM = 93_000
CLOSED_FALLBACK_STREAM = 86_000
WARM_DEFICIT_STREAM = 95_000
OPEN_DEFICIT_STREAM = 999_999

# Frozen conservative pre-call cost estimates (USD).
POOL_CALL_COST_ESTIMATE_USD = 0.01
WARM_CALL_COST_ESTIMATE_USD = 0.01
REFINE_CALL_COST_ESTIMATE_USD = 0.005

STRATEGIES = ("local_adjustment", "topology_redesign", "rotation_redesign",
              "global_redesign")

ROW_EXTRA_DEFAULTS = {
    "fallback": False, "invalid_errors": "", "edit_distance": np.nan, "strategy": "",
}


# ------------------------------------------------------------ LLM layer ----

def build_provider(condition: Condition, requested_candidates: int):
    from llm_vqc.llm.openai_provider import OpenAIProvider

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY must be set for API arms")
    return OpenAIProvider(
        api_key=api_key,
        model=condition.model,
        max_output_tokens=P.max_output_tokens(condition, requested_candidates),
    )


def complete_json(
    provider,
    budget: LLMApiBudget,
    user_prompt: str,
    proposal_id: str,
    cost_estimate: float,
) -> tuple[dict | None, list[LLMCallRecord]]:
    """Frozen call + JSON-repair-retry behaviour (identical to the reference)."""
    records: list[LLMCallRecord] = []
    prompt = user_prompt
    for attempt in range(P.MAX_REPAIR_ATTEMPTS + 1):
        budget.check_can_afford(cost_estimate)
        response = provider.complete(P.SYSTEM_PROMPT, prompt, P.TEMPERATURE)
        actual = response.estimated_cost_usd
        budget.record_spend(actual if actual is not None else cost_estimate)
        raw = response.raw_text.strip()
        if raw.startswith("```"):
            raw = raw.strip("`")
            raw = raw[raw.find("{"):] if "{" in raw else raw
        parsed: dict | None
        errors: list[str] = []
        try:
            data = json.loads(raw)
            parsed = data if isinstance(data, dict) else None
            if parsed is None:
                errors = ["parsed JSON is not an object"]
        except json.JSONDecodeError as exc:
            parsed = None
            errors = [f"response is not valid JSON: {exc}"]
        records.append(
            LLMCallRecord(
                proposal_id=proposal_id,
                call_index=attempt,
                system_prompt=P.SYSTEM_PROMPT,
                user_prompt=prompt,
                model=response.model,
                temperature=P.TEMPERATURE,
                raw_response=response.raw_text,
                parsed_proposal=parsed,
                validation_errors=errors,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                estimated_cost_usd=response.estimated_cost_usd,
                latency_seconds=response.latency_seconds,
            )
        )
        if parsed is not None:
            return parsed, records
        prompt = (
            user_prompt
            + "\n\nYour previous response could not be parsed as JSON: "
            + "; ".join(errors)
            + "\nRespond again with ONLY the JSON object."
        )
    return None, records


def record_to_dict(record: LLMCallRecord) -> dict:
    return json.loads(record.model_dump_json())


@dataclass
class LLMPool:
    candidates: dict[str, list[dict]]
    call_records: list[LLMCallRecord] = field(default_factory=list)
    fallback_names: list[str] = field(default_factory=list)
    rejected: list[dict] = field(default_factory=list)


def _absorb(
    parsed_obj: dict | None,
    condition: Condition,
    wanted: int,
    candidates: dict[str, list[dict]],
    seen: set[str],
    rejected: list[dict],
    prefix: str,
) -> None:
    """Frozen candidate-absorption policy: validate, drop duplicates, cap."""
    if parsed_obj is None:
        return
    space = space_for(condition)
    for i, entry in enumerate(parsed_obj.get("candidates", [])[: wanted * 2]):
        architecture, errors = parse_candidate(entry, space)
        if architecture is None:
            rejected.append({"index": i, "errors": errors, "entry": entry})
            continue
        key = architecture_key(architecture)
        if key in seen:
            rejected.append({"index": i, "errors": ["duplicate"], "entry": entry})
            continue
        if len(candidates) >= wanted:
            break
        seen.add(key)
        name = str(entry.get("name", f"{prefix}_{i}"))[:48] or f"{prefix}_{i}"
        while name in candidates:
            name += "_"
        candidates[name] = architecture


def _generate_batch(
    condition: Condition,
    budget: LLMApiBudget,
    provider,
    first_prompt: str,
    wanted: int,
    call_id: str,
    cost_estimate: float,
    deficit_rng: np.random.Generator,
    fallback_prefix: str,
    name_prefix: str,
) -> tuple[dict[str, list[dict]], list[str], list[dict], list[LLMCallRecord]]:
    """One batch call + bounded capacity repair + flagged random deficit fill."""
    space = space_for(condition)
    candidates: dict[str, list[dict]] = {}
    seen: set[str] = set()
    fallback_names: list[str] = []
    rejected: list[dict] = []
    records: list[LLMCallRecord] = []

    parsed, call_records = complete_json(
        provider, budget, first_prompt, call_id, cost_estimate
    )
    records.extend(call_records)
    _absorb(parsed, condition, wanted, candidates, seen, rejected, name_prefix)

    repair_round = 0
    while len(candidates) < wanted and rejected and repair_round < P.MAX_REPAIR_ATTEMPTS:
        repair_round += 1
        deficit = wanted - len(candidates)
        complaint = "\n".join(
            f"- {json.dumps(r['entry'], separators=(',', ':'))[:400]} rejected: "
            + "; ".join(r["errors"])[:200]
            for r in rejected[-deficit:]
        )
        parsed, call_records = complete_json(
            provider, budget, P.repair_prompt(condition, complaint, deficit),
            f"{call_id}_repair{repair_round}", cost_estimate,
        )
        records.extend(call_records)
        _absorb(parsed, condition, wanted, candidates, seen, rejected, name_prefix)

    while len(candidates) < wanted:
        architecture = sample_neutral_arch(deficit_rng, space)
        if architecture_key(architecture) in seen:
            continue
        seen.add(architecture_key(architecture))
        name = f"{fallback_prefix}_{len(fallback_names)}"
        candidates[name] = architecture
        fallback_names.append(name)

    return candidates, fallback_names, rejected, records


def generate_open_pool(condition: Condition, destination: Path) -> LLMPool:
    """LLM-Open: one batch of B candidates, produced before any score."""
    pool_path = destination / "llm_open_pool.json"
    if pool_path.exists():
        stored = json.loads(pool_path.read_text())
        return LLMPool(
            candidates=dict(stored["candidates"]),
            fallback_names=stored.get("fallback_names", []),
            rejected=stored.get("rejected", []),
        )

    budget = LLMApiBudget.from_env()
    if budget is None:
        raise RuntimeError("LLM_API_BUDGET_USD is not configured; refusing paid API calls")
    provider = build_provider(condition, condition.budget)
    calls_dir = destination / "llm_calls"
    calls_dir.mkdir(parents=True, exist_ok=True)

    candidates, fallback_names, rejected, records = _generate_batch(
        condition, budget, provider, P.open_loop_prompt(condition), condition.budget,
        "open_pool", POOL_CALL_COST_ESTIMATE_USD,
        np.random.default_rng(OPEN_DEFICIT_STREAM), "fallback_random", "api",
    )

    for i, record in enumerate(records):
        (calls_dir / f"open_pool_call{i}.json").write_text(
            json.dumps(record_to_dict(record), indent=2)
        )
    pool_path.write_text(json.dumps({
        "candidates": candidates,
        "fallback_names": fallback_names,
        "rejected": rejected,
        "model_snapshots": sorted({r.model for r in records}),
        "total_input_tokens": sum(r.input_tokens for r in records),
        "total_output_tokens": sum(r.output_tokens for r in records),
        "n_calls": len(records),
        "max_output_tokens": P.max_output_tokens(condition, condition.budget),
    }, indent=2))
    return LLMPool(candidates=candidates, call_records=records,
                   fallback_names=fallback_names, rejected=rejected)


# ----------------------------------------------------------------- arms ----

def _row(seed: int, method: str, candidate: str, order: int, phase: str,
         result, **extra) -> dict:
    row = {"seed": seed, "method": method, "candidate": candidate,
           "order": order, "phase": phase, **ROW_EXTRA_DEFAULTS}
    row.update(extra)
    row.update(result.__dict__)
    return row


def run_random_seed(condition: Condition, seed: int, train, val, test) -> list[dict]:
    """B independent uniform candidates."""
    space = space_for(condition)
    rng = np.random.default_rng(RANDOM_STREAM + seed)
    rows: list[dict] = []
    seen: set[str] = set()
    order = 0
    while order < condition.budget:
        architecture = sample_neutral_arch(rng, space)
        key = architecture_key(architecture)
        if key in seen:
            continue
        seen.add(key)
        order += 1
        result = evaluate_architecture(
            architecture, train, val, test,
            seed=train_seed(100 + seed, architecture), n=condition.n_qubits,
        )
        rows.append(_row(seed, "Random", f"R{order}", order, "explore", result))
    return rows


def run_greedy_seed(condition: Condition, seed: int, train, val, test) -> list[dict]:
    """B/2 random warm starts + B/2 one-local-mutation refinements."""
    space = space_for(condition)
    n_warm = condition.n_warm
    rng = np.random.default_rng(GREEDY_STREAM + seed)
    rows: list[dict] = []
    seen: set[str] = set()

    incumbent = None
    incumbent_result = None
    for order in range(1, n_warm + 1):
        while True:
            architecture = sample_neutral_arch(rng, space)
            if architecture_key(architecture) not in seen:
                break
        seen.add(architecture_key(architecture))
        result = evaluate_architecture(
            architecture, train, val, test,
            seed=train_seed(100 + seed, architecture), n=condition.n_qubits,
        )
        rows.append(_row(seed, "Greedy", f"G{order}_warm", order, "explore", result))
        if incumbent_result is None or result.val_loss < incumbent_result.val_loss:
            incumbent, incumbent_result = architecture, result

    for order in range(n_warm + 1, condition.budget + 1):
        candidate = None
        fallback = False
        for _ in range(200):
            mutated = mutate_one_choice(incumbent, rng, space)
            if architecture_key(mutated) not in seen:
                candidate = mutated
                break
        if candidate is None:
            while True:
                candidate = sample_neutral_arch(rng, space)
                if architecture_key(candidate) not in seen:
                    break
            fallback = True
        distance = arch_distance(incumbent, candidate)
        seen.add(architecture_key(candidate))
        result = evaluate_architecture(
            candidate, train, val, test,
            seed=train_seed(100 + seed, candidate), n=condition.n_qubits,
        )
        accepted = result.val_loss < incumbent_result.val_loss
        if accepted:
            incumbent, incumbent_result = candidate, result
        rows.append(_row(
            seed, "Greedy", f"G{order}_{'acc' if accepted else 'rej'}", order,
            "refine", result, fallback=fallback, edit_distance=distance,
        ))
    return rows


def run_open_seed(condition: Condition, seed: int, pool: LLMPool,
                  train, val, test) -> list[dict]:
    """B semantic proposals generated before any score is seen."""
    rows: list[dict] = []
    for i, (name, architecture) in enumerate(pool.candidates.items()):
        result = evaluate_architecture(
            architecture, train, val, test,
            seed=train_seed(100 + seed, architecture), n=condition.n_qubits,
        )
        rows.append(_row(seed, "LLM-Open", name, i + 1, "explore", result,
                         fallback=name in pool.fallback_names))
    return rows


def run_closed_seed(condition: Condition, seed: int, destination: Path,
                    train, val, test) -> list[dict]:
    """B/2 semantic warm starts + B/2 free-form redesigns of the current
    best, conditioned on the incumbent and its VALIDATION fidelity."""
    store = destination / f"llm_closed_seed{seed}.json"
    if store.exists():
        return json.loads(store.read_text())["rows"]

    space = space_for(condition)
    n_warm = condition.n_warm
    budget = LLMApiBudget.from_env()
    if budget is None:
        raise RuntimeError("LLM_API_BUDGET_USD is not configured; refusing paid API calls")
    calls_dir = destination / "llm_calls"
    calls_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    architectures_by_order: list[list[dict]] = []
    seen: set[str] = set()
    fallback_rng = np.random.default_rng(CLOSED_FALLBACK_STREAM + seed)
    all_records: list[LLMCallRecord] = []
    failure_log: list[dict] = []

    warm_provider = build_provider(condition, n_warm)
    warm, warm_fallbacks, warm_rejected, warm_records = _generate_batch(
        condition, budget, warm_provider, P.warmstart_prompt(condition), n_warm,
        f"warm_s{seed}", WARM_CALL_COST_ESTIMATE_USD,
        np.random.default_rng(WARM_DEFICIT_STREAM + seed), "warm_fallback", "warm",
    )
    all_records.extend(warm_records)

    incumbent = None
    incumbent_result = None
    for i, (name, architecture) in enumerate(warm.items()):
        seen.add(architecture_key(architecture))
        architectures_by_order.append(architecture)
        result = evaluate_architecture(
            architecture, train, val, test,
            seed=train_seed(100 + seed, architecture), n=condition.n_qubits,
        )
        rows.append(_row(seed, "LLM-Closed", name, i + 1, "explore", result,
                         fallback=name in warm_fallbacks))
        if incumbent_result is None or result.val_loss < incumbent_result.val_loss:
            incumbent, incumbent_result = architecture, result

    refine_provider = build_provider(condition, 1)
    for proposal in range(n_warm, condition.budget):
        parsed, records = complete_json(
            refine_provider, budget,
            P.redesign_prompt(condition, incumbent, incumbent_result.val_fid),
            f"redesign_s{seed}_p{proposal}", REFINE_CALL_COST_ESTIMATE_USD,
        )
        all_records.extend(records)
        architecture = None
        entry = None
        errors: list[str] = []
        if parsed is not None:
            entries = parsed.get("candidates", [])
            entry = entries[0] if entries else None
            if entry is not None:
                architecture, errors = parse_candidate(entry, space)
            else:
                errors = ["empty candidates"]
        failure = ""
        if architecture is None:
            failure = "invalid: " + "; ".join(errors)[:160]
        else:
            key = architecture_key(architecture)
            if key == architecture_key(incumbent):
                failure = "duplicate_of_incumbent"
            elif key in seen:
                failure = "duplicate_of_other"
        fallback = bool(failure)
        if fallback:
            failure_log.append({"proposal": proposal, "reason": failure})
            while True:
                architecture = sample_neutral_arch(fallback_rng, space)
                if architecture_key(architecture) not in seen:
                    break
        distance = arch_distance(incumbent, architecture)
        seen.add(architecture_key(architecture))
        architectures_by_order.append(architecture)
        result = evaluate_architecture(
            architecture, train, val, test,
            seed=train_seed(100 + seed, architecture), n=condition.n_qubits,
        )
        accepted = result.val_loss < incumbent_result.val_loss
        if accepted:
            incumbent, incumbent_result = architecture, result
        strategy = ""
        if entry is not None and not fallback:
            raw_strategy = str(entry.get("strategy", ""))
            strategy = raw_strategy if raw_strategy in STRATEGIES else f"other:{raw_strategy[:24]}"
        rows.append(_row(
            seed, "LLM-Closed",
            (str(entry.get("name", f"p{proposal}"))[:48]
             if entry is not None and not fallback else f"fallback_p{proposal}"),
            proposal + 1, "refine", result,
            fallback=fallback, invalid_errors=failure, edit_distance=distance,
            strategy=strategy,
        ))

    for i, record in enumerate(all_records):
        (calls_dir / f"closed_s{seed}_call{i}.json").write_text(
            json.dumps(record_to_dict(record), indent=2)
        )
    store.write_text(json.dumps({
        "rows": rows,
        "architectures": architectures_by_order,
        "warm_rejected": warm_rejected,
        "refine_failures": failure_log,
        "model_snapshots": sorted({r.model for r in all_records}),
        "total_input_tokens": sum(r.input_tokens for r in all_records),
        "total_output_tokens": sum(r.output_tokens for r in all_records),
        "budget_spent_usd_estimate": budget.spent_usd,
        "n_calls": len(all_records),
    }, indent=2))
    return rows
