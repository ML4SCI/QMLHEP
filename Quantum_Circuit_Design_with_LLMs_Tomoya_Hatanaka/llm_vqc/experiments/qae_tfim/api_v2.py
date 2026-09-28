"""QAE-TFIM v2: version-pinned API verification of the semantic-prior pilot.

Extends the frozen pilot protocol (`llm_vqc.experiments.qae_tfim.pilot`) with:

- **LLM-API-open**: one real, version-pinned OpenAI chat-completion call that
  replays the exact frozen prompt from ``docs/research/LLM_QAE_SKILL.md`` and
  returns a pool of 8 capacity-checked candidates. The pool is generated once,
  stored with full provenance, and reused across all verification seeds.
- **LLM-API-closed**: per-seed sequential proposals with validation-only
  feedback (never protected-test information).
- **Evolutionary**: mutation-based search under the identical B=8 budget.
- **Reference-QAE**: one fixed, hand-designed textbook hardware-efficient
  encoder (brickwork nearest-neighbour entanglement), no search.
- **Random / RY-random / LLM-chat-frozen**: identical to the pilot (same rng
  streams, so pilot numbers reproduce exactly).

Cost policy: real API calls happen only when ``LLM_API_BUDGET_USD`` is set to
an explicit nonzero cap (see ``llm_vqc.llm.budget``). Every call is charged a
conservative pre-call estimate against the cap. All raw requests/responses,
token counts, and the pinned model snapshot id are stored under the output
directory. The run is resumable: stored pool/closed-loop artifacts are reused
instead of re-calling the API.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon

from llm_vqc.experiments.qae_tfim.pilot import (
    BUDGET,
    DIRECTED_PAIRS,
    SEMANTIC_CANDIDATES,
    VERIFY_SEEDS,
    _mk,
    _train_seed,
    architecture_key,
    evaluate_architecture,
    sample_random_arch,
    state_sets,
    verify_capacity,
)
from llm_vqc.llm.budget import LLMApiBudget
from llm_vqc.llm.records import LLMCallRecord

TEMPERATURE = 0.7
MAX_REPAIR_ATTEMPTS = 2
POOL_CALL_COST_ESTIMATE_USD = 0.01
CLOSED_CALL_COST_ESTIMATE_USD = 0.005

# Hand-designed textbook reference: hardware-efficient RY-RZ-RY layers with
# brickwork nearest-neighbour entanglement. Deliberately NOT informed by the
# latent/trash split (no trash-to-latent funnel); it controls for "a sensible
# generic QAE ansatz" rather than task-aware topology reasoning.
REFERENCE_QAE = _mk(
    [["Y"] * 4, ["Z"] * 4, ["Y"] * 4],
    [(0, 1), (2, 3)],
    [(1, 2), (3, 2)],
)

SYSTEM_PROMPT = (
    "You are a quantum-autoencoder circuit architect. Your job is architecture "
    "proposal, not numerical optimization. Respect the exact resource budget. "
    "Use the physical structure in the task description to propose circuit "
    "hypotheses. Return circuit structure only."
)

TASK_CARD = (
    "Compress ground states of the 4-qubit open-chain transverse-field Ising "
    "Hamiltonian H = -sum_i Z_i Z_{i+1} - h sum_i X_i, h in [0.2, 2.0], into "
    "latent qubits q0,q1 while trash qubits q2,q3 should end in |00>. "
    "The Hamiltonian is real in the computational basis, so its ground-state "
    "amplitudes can be chosen real. Nearest-neighbor correlations are important. "
    "Use exactly three 4-qubit rotation layers (12 trainable rotations total) and "
    "exactly four CNOTs: two after rotation layer 1 and two after rotation layer 2. "
    "Rotation axes may be X, Y, or Z."
)

JSON_SCHEMA_CARD = (
    "Return ONLY a single JSON object, no prose and no markdown fences, with this schema:\n"
    '{"candidates": [{"name": "<short-id>", '
    '"axes": [["Y","Y","Y","Y"],["Y","Y","Y","Y"],["Y","Y","Y","Y"]], '
    '"cnots_after_layer1": [[3,2],[1,0]], "cnots_after_layer2": [[2,1],[3,1]]}, ...]}\n'
    "axes is 3 layers x 4 qubits, each entry X, Y, or Z. Each CNOT is "
    "[control, target] with control != target, qubits 0-3."
)

OPEN_LOOP_PROMPT = (
    TASK_CARD
    + "\nOutput 8 distinct candidates. Prefer hypotheses that (1) preserve a "
    "real-valued representation when useful and (2) route correlations/information "
    "from the trash qubits toward the latent qubits. Do not change the number of "
    "qubits, trainable parameters, CNOTs, or optimization budget.\n"
    + JSON_SCHEMA_CARD
)


def closed_loop_prompt(history: list[dict]) -> str:
    lines = [TASK_CARD, ""]
    if history:
        lines.append("Feedback on candidates evaluated so far (validation only):")
        for entry in history:
            lines.append(
                f"- candidate {json.dumps(entry['candidate'], separators=(',', ':'))} "
                f"validation trash fidelity: {entry['val_fid']:.4f} "
                f"duplicate: {str(entry['duplicate']).lower()}"
            )
        lines.append("")
    lines.append(
        "Propose ONE new candidate with exactly one clearly stated structural "
        "hypothesis in its name. Do not repeat an already-evaluated candidate. "
        "Do not use or request test-set information."
    )
    lines.append(JSON_SCHEMA_CARD + "\nThe candidates array must contain exactly 1 candidate.")
    return "\n".join(lines)


def parse_candidate(entry: dict) -> tuple[list[dict] | None, list[str]]:
    """Turn one JSON candidate into an architecture, or explain why not."""
    errors: list[str] = []
    if not isinstance(entry, dict):
        return None, ["candidate is not an object"]
    axes = entry.get("axes")
    l1 = entry.get("cnots_after_layer1")
    l2 = entry.get("cnots_after_layer2")
    if (
        not isinstance(axes, list)
        or len(axes) != 3
        or any(not isinstance(layer, list) or len(layer) != 4 for layer in axes)
        or any(a not in ("X", "Y", "Z") for layer in axes for a in layer)
    ):
        errors.append("axes must be 3 layers x 4 entries from X/Y/Z")
    for label, cnots in (("cnots_after_layer1", l1), ("cnots_after_layer2", l2)):
        if (
            not isinstance(cnots, list)
            or len(cnots) != 2
            or any(
                not isinstance(p, list)
                or len(p) != 2
                or any(not isinstance(w, int) or not 0 <= w <= 3 for w in p)
                or p[0] == p[1]
                for p in cnots
            )
        ):
            errors.append(f"{label} must be 2 [control,target] pairs on qubits 0-3")
    if errors:
        return None, errors
    architecture = _mk(
        [[str(a) for a in layer] for layer in axes],
        [(int(p[0]), int(p[1])) for p in l1],
        [(int(p[0]), int(p[1])) for p in l2],
    )
    try:
        verify_capacity(architecture)
    except ValueError as exc:
        return None, [str(exc)]
    return architecture, []


@dataclass
class LLMPool:
    """A named, capacity-checked pool with the provenance that produced it."""

    candidates: dict[str, list[dict]]
    call_records: list[LLMCallRecord] = field(default_factory=list)
    fallback_names: list[str] = field(default_factory=list)


def _record_to_dict(record: LLMCallRecord) -> dict:
    return json.loads(record.model_dump_json())


def _build_provider():
    """Construct the real OpenAI provider; only called when a budget cap exists."""
    from llm_vqc.llm.openai_provider import OpenAIProvider

    api_key = os.environ.get("OPENAI_API_KEY", "")
    model = os.environ.get("OPENAI_MODEL", "")
    if not api_key or not model:
        raise RuntimeError("OPENAI_API_KEY and OPENAI_MODEL must be set for API arms")
    return OpenAIProvider(api_key=api_key, model=model, max_output_tokens=4000)


def _complete_json(
    provider,
    budget: LLMApiBudget,
    user_prompt: str,
    proposal_id: str,
    cost_estimate: float,
) -> tuple[dict | None, list[LLMCallRecord]]:
    """One provider call plus bounded JSON repair retries, fully recorded."""
    records: list[LLMCallRecord] = []
    prompt = user_prompt
    for attempt in range(MAX_REPAIR_ATTEMPTS + 1):
        budget.check_can_afford(cost_estimate)
        response = provider.complete(SYSTEM_PROMPT, prompt, TEMPERATURE)
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
                system_prompt=SYSTEM_PROMPT,
                user_prompt=prompt,
                model=response.model,
                temperature=TEMPERATURE,
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


def generate_open_pool(destination: Path) -> LLMPool:
    """Generate (or reload) the API open-loop pool of 8 candidates."""
    pool_path = destination / "llm_open_pool.json"
    if pool_path.exists():
        stored = json.loads(pool_path.read_text())
        return LLMPool(
            candidates={k: v for k, v in stored["candidates"].items()},
            fallback_names=stored.get("fallback_names", []),
        )

    budget = LLMApiBudget.from_env()
    if budget is None:
        raise RuntimeError(
            "LLM_API_BUDGET_USD is not configured; refusing to make paid API calls"
        )
    provider = _build_provider()
    calls_dir = destination / "llm_calls"
    calls_dir.mkdir(parents=True, exist_ok=True)

    candidates: dict[str, list[dict]] = {}
    seen: set[str] = set()
    all_records: list[LLMCallRecord] = []
    fallback_names: list[str] = []

    parsed, records = _complete_json(
        provider, budget, OPEN_LOOP_PROMPT, "open_pool", POOL_CALL_COST_ESTIMATE_USD
    )
    all_records.extend(records)
    rejected: list[dict] = []
    if parsed is not None:
        for i, entry in enumerate(parsed.get("candidates", [])[: BUDGET * 2]):
            architecture, errors = parse_candidate(entry)
            if architecture is None:
                rejected.append({"index": i, "errors": errors, "entry": entry})
                continue
            key = architecture_key(architecture)
            if key in seen:
                rejected.append({"index": i, "errors": ["duplicate"], "entry": entry})
                continue
            if len(candidates) >= BUDGET:
                break
            seen.add(key)
            name = str(entry.get("name", f"api_{i}"))[:48] or f"api_{i}"
            while name in candidates:
                name += "_"
            candidates[name] = architecture

    # Honest deficit handling: if the API pool is short after retries, fill the
    # remaining slots with Random samples and flag them (they are then part of
    # the LLM arm's budget accounting, not silently dropped).
    deficit_rng = np.random.default_rng(999_999)
    while len(candidates) < BUDGET:
        architecture = sample_random_arch(deficit_rng)
        key = architecture_key(architecture)
        if key in seen:
            continue
        seen.add(key)
        name = f"fallback_random_{len(fallback_names)}"
        candidates[name] = architecture
        fallback_names.append(name)

    for i, record in enumerate(all_records):
        (calls_dir / f"open_pool_call{i}.json").write_text(
            json.dumps(_record_to_dict(record), indent=2)
        )
    pool_path.write_text(
        json.dumps(
            {
                "candidates": candidates,
                "fallback_names": fallback_names,
                "rejected": rejected,
                "model_snapshots": sorted({r.model for r in all_records}),
                "total_input_tokens": sum(r.input_tokens for r in all_records),
                "total_output_tokens": sum(r.output_tokens for r in all_records),
                "budget_spent_usd_estimate": budget.spent_usd,
                "n_calls": len(all_records),
            },
            indent=2,
        )
    )
    return LLMPool(candidates=candidates, call_records=all_records, fallback_names=fallback_names)


def mutate_arch(architecture: list[dict], rng: np.random.Generator) -> list[dict]:
    """One structural mutation: change one rotation axis or one CNOT pair."""
    rotations = [op for op in architecture if op["type"] == "R"]
    cnots = [op for op in architecture if op["type"] == "CX"]
    axes = [[rotations[layer * 4 + q]["axis"] for q in range(4)] for layer in range(3)]
    pairs = [(op["c"], op["t"]) for op in cnots]
    if rng.random() < 0.5:
        layer = int(rng.integers(3))
        qubit = int(rng.integers(4))
        options = [a for a in ("X", "Y", "Z") if a != axes[layer][qubit]]
        axes[layer][qubit] = str(rng.choice(options))
    else:
        index = int(rng.integers(4))
        remaining = [p for p in DIRECTED_PAIRS if p not in pairs]
        pairs[index] = remaining[int(rng.integers(len(remaining)))]
    mutated = _mk(axes, pairs[:2], pairs[2:])
    verify_capacity(mutated)
    return mutated


def evolutionary_candidates(seed: int) -> list[tuple[str, list[dict]]]:
    """(1+lambda)-style search: 4 random parents, then 4 mutations of the two
    parents with the best validation loss. Exactly BUDGET evaluations."""
    rng = np.random.default_rng(40_000 + seed)
    out: list[tuple[str, list[dict]]] = []
    seen: set[str] = set()
    while len(out) < 4:
        architecture = sample_random_arch(rng)
        key = architecture_key(architecture)
        if key in seen:
            continue
        seen.add(key)
        out.append((f"E_init{len(out) + 1}", architecture))
    return out


def evolutionary_offspring(
    parents: list[list[dict]], seed: int, seen: set[str]
) -> list[tuple[str, list[dict]]]:
    rng = np.random.default_rng(41_000 + seed)
    children: list[tuple[str, list[dict]]] = []
    attempts = 0
    while len(children) < 4 and attempts < 200:
        attempts += 1
        parent = parents[len(children) % len(parents)]
        child = mutate_arch(parent, rng)
        key = architecture_key(child)
        if key in seen:
            continue
        seen.add(key)
        children.append((f"E_child{len(children) + 1}", child))
    while len(children) < 4:
        child = sample_random_arch(rng)
        key = architecture_key(child)
        if key in seen:
            continue
        seen.add(key)
        children.append((f"E_child{len(children) + 1}", child))
    return children


def run_closed_loop_seed(
    seed: int,
    destination: Path,
    train: np.ndarray,
    val: np.ndarray,
    test: np.ndarray,
) -> list[dict]:
    """Sequential API proposals with validation-only feedback. Resumable."""
    store = destination / f"llm_closed_seed{seed}.json"
    if store.exists():
        return json.loads(store.read_text())["rows"]

    budget = LLMApiBudget.from_env()
    if budget is None:
        raise RuntimeError(
            "LLM_API_BUDGET_USD is not configured; refusing to make paid API calls"
        )
    provider = _build_provider()
    calls_dir = destination / "llm_calls"
    calls_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    history: list[dict] = []
    seen: set[str] = set()
    fallback_rng = np.random.default_rng(50_000 + seed)
    all_records: list[LLMCallRecord] = []

    for proposal in range(BUDGET):
        parsed, records = _complete_json(
            provider,
            budget,
            closed_loop_prompt(history),
            f"closed_s{seed}_p{proposal}",
            CLOSED_CALL_COST_ESTIMATE_USD,
        )
        all_records.extend(records)
        architecture = None
        entry = None
        errors: list[str] = []
        if parsed is not None:
            entries = parsed.get("candidates", [])
            entry = entries[0] if entries else None
            if entry is not None:
                architecture, errors = parse_candidate(entry)
            else:
                errors = ["empty candidates"]
        duplicate = False
        fallback = False
        if architecture is not None:
            key = architecture_key(architecture)
            duplicate = key in seen
        if architecture is None or duplicate:
            # Honest budget accounting: an invalid or duplicate proposal still
            # consumes one of the 8 evaluations, filled by a flagged random.
            while True:
                architecture = sample_random_arch(fallback_rng)
                if architecture_key(architecture) not in seen:
                    break
            fallback = True
        key = architecture_key(architecture)
        seen.add(key)
        result = evaluate_architecture(
            architecture, train, val, test, seed=_train_seed(100 + seed, architecture)
        )
        candidate_json = entry if (entry is not None and not fallback) else {
            "name": f"fallback_random_p{proposal}", "note": "fallback"
        }
        history.append(
            {"candidate": candidate_json, "val_fid": result.val_fid, "duplicate": duplicate}
        )
        rows.append(
            {
                "seed": seed,
                "method": "LLM-API-closed",
                "candidate": (
                    str(entry.get("name", f"p{proposal}"))[:48]
                    if entry is not None and not fallback
                    else f"fallback_p{proposal}"
                ),
                "order": proposal + 1,
                "fallback": fallback,
                "invalid_errors": "; ".join(errors),
                **result.__dict__,
            }
        )

    for i, record in enumerate(all_records):
        (calls_dir / f"closed_s{seed}_call{i}.json").write_text(
            json.dumps(_record_to_dict(record), indent=2)
        )
    store.write_text(
        json.dumps(
            {
                "rows": rows,
                "model_snapshots": sorted({r.model for r in all_records}),
                "total_input_tokens": sum(r.input_tokens for r in all_records),
                "total_output_tokens": sum(r.output_tokens for r in all_records),
                "budget_spent_usd_estimate": budget.spent_usd,
                "n_calls": len(all_records),
            },
            indent=2,
        )
    )
    return rows


def run_seed_deterministic(seed: int, pool: LLMPool | None) -> list[dict]:
    """All non-network arms for one seed (plus the stored API open pool)."""
    train, val, test = state_sets(seed)
    rows: list[dict] = []

    def add(method: str, name: str, architecture: list[dict], order: int, fallback: bool = False):
        result = evaluate_architecture(
            architecture, train, val, test, seed=_train_seed(100 + seed, architecture)
        )
        rows.append(
            {
                "seed": seed,
                "method": method,
                "candidate": name,
                "order": order,
                "fallback": fallback,
                "invalid_errors": "",
                **result.__dict__,
            }
        )
        return result

    for i, (name, architecture) in enumerate(SEMANTIC_CANDIDATES.items()):
        add("LLM-chat-frozen", name, architecture, i + 1)

    if pool is not None:
        for i, (name, architecture) in enumerate(pool.candidates.items()):
            add("LLM-API-open", name, architecture, i + 1, fallback=name in pool.fallback_names)

    rng = np.random.default_rng(20_000 + seed)
    for method, ry_only in (("Random", False), ("RY-random", True)):
        seen: set[str] = set()
        proposal = 0
        while proposal < BUDGET:
            architecture = sample_random_arch(rng, ry_only=ry_only)
            key = architecture_key(architecture)
            if key in seen:
                continue
            seen.add(key)
            proposal += 1
            add(method, f"R{proposal}", architecture, proposal)

    # Evolutionary: 4 random parents, then 4 mutations of the top-2 by val loss.
    parents = evolutionary_candidates(seed)
    seen_evo = {architecture_key(a) for _, a in parents}
    parent_results = []
    for i, (name, architecture) in enumerate(parents):
        result = add("Evolutionary", name, architecture, i + 1)
        parent_results.append((result.val_loss, architecture))
    parent_results.sort(key=lambda item: item[0])
    top_parents = [architecture for _, architecture in parent_results[:2]]
    for i, (name, architecture) in enumerate(
        evolutionary_offspring(top_parents, seed, seen_evo)
    ):
        add("Evolutionary", name, architecture, 5 + i)

    add("Reference-QAE", "textbook_brickwork", REFERENCE_QAE, 1)
    return rows


def analyze(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    search_methods = [m for m in frame["method"].unique() if m != "Reference-QAE"]
    selected = (
        frame[frame["method"].isin(search_methods)]
        .sort_values("val_loss")
        .groupby(["seed", "method"], as_index=False)
        .first()
    )
    reference = frame[frame["method"] == "Reference-QAE"]
    selected = pd.concat([selected, reference], ignore_index=True)
    pivot = selected.pivot(index="seed", columns="method", values="test_fid")
    stats: dict[str, dict] = {}
    for llm_method in ("LLM-API-open", "LLM-API-closed", "LLM-chat-frozen"):
        if llm_method not in pivot:
            continue
        for other in ("Random", "RY-random", "Evolutionary", "Reference-QAE"):
            if other not in pivot:
                continue
            diff = (pivot[llm_method] - pivot[other]).dropna()
            if len(diff) < 6:
                continue
            if np.allclose(diff, 0):
                continue
            exact = wilcoxon(diff, alternative="two-sided", method="exact")
            sign = binomtest(int((diff > 0).sum()), len(diff), 0.5, alternative="two-sided")
            stats[f"{llm_method}_vs_{other}"] = {
                "mean_paired_fidelity_gain": float(diff.mean()),
                "median_paired_fidelity_gain": float(diff.median()),
                "llm_wins": int((diff > 0).sum()),
                "n": int(len(diff)),
                "wilcoxon_exact_two_sided_p": float(exact.pvalue),
                "sign_test_two_sided_p": float(sign.pvalue),
            }
    return selected, stats


def anytime_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Best-so-far (by validation) protected-test fidelity vs candidates used."""
    rows = []
    for (seed, method), group in frame.groupby(["seed", "method"]):
        if method == "Reference-QAE":
            continue
        group = group.sort_values("order")
        best_val = float("inf")
        best_test = float("nan")
        for _, row in group.iterrows():
            if row["val_loss"] < best_val:
                best_val = row["val_loss"]
                best_test = row["test_fid"]
            rows.append(
                {
                    "seed": seed,
                    "method": method,
                    "candidates_used": int(row["order"]),
                    "best_so_far_test_fid": best_test,
                }
            )
    anytime = pd.DataFrame(rows)
    return (
        anytime.groupby(["method", "candidates_used"], as_index=False)[
            "best_so_far_test_fid"
        ].mean()
    )


def main(
    output_dir: str = "outputs/qae_tfim_api_v2",
    *,
    with_api: bool = True,
    closed_loop: bool = True,
    seeds: tuple[int, ...] = VERIFY_SEEDS,
) -> None:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    pool = generate_open_pool(destination) if with_api else None

    rows: list[dict] = []
    for seed in seeds:
        print(f"seed {seed}: deterministic arms")
        rows.extend(run_seed_deterministic(seed, pool))
    if with_api and closed_loop:
        for seed in seeds:
            print(f"seed {seed}: closed-loop API arm")
            train, val, test = state_sets(seed)
            rows.extend(run_closed_loop_seed(seed, destination, train, val, test))

    frame = pd.DataFrame(rows)
    selected, stats = analyze(frame)
    frame.to_csv(destination / "candidate_results.csv", index=False)
    selected.to_csv(destination / "selected_results.csv", index=False)
    anytime_table(frame).to_csv(destination / "anytime_mean.csv", index=False)
    (destination / "paired_stats.json").write_text(json.dumps(stats, indent=2))
    print(selected.groupby("method")["test_fid"].agg(["mean", "median", "std"]))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
