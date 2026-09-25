"""QAE-TFIM v3: neutral-space four-method benchmark (pre-registered).

Protocol: docs/research/QAE_PROTOCOL_V3.md (frozen before any
protected-test inspection). Methods: Random, Greedy, LLM-Open, LLM-Closed
— a 2x2 over (semantics) x (validation feedback).

Neutral circuit space: an ORDERED sequence of exactly 16 gates — 12
single-qubit rotations (axis in {RX,RY,RZ} free, qubit free) and 4 CNOTs
(control != target, repeats allowed), ordering completely free. The rigid
layered layout of v1/v2 is dropped.

The trainer, data streams, seeds, selection rule, and budget B=8 are
identical to v1/v2 (imported from `pilot`). Real API calls require an
explicit nonzero LLM_API_BUDGET_USD cap; every call is stored with full
provenance and the runs are resumable from stored artifacts.
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
    VERIFY_SEEDS,
    _train_seed,
    architecture_key,
    evaluate_architecture,
    state_sets,
)
from llm_vqc.llm.budget import LLMApiBudget
from llm_vqc.llm.records import LLMCallRecord

TEMPERATURE = 0.7
MAX_REPAIR_ATTEMPTS = 2
POOL_CALL_COST_ESTIMATE_USD = 0.01
CLOSED_CALL_COST_ESTIMATE_USD = 0.005

N_ROTATIONS = 12
N_CNOTS = 4
N_GATES = N_ROTATIONS + N_CNOTS
WIRES = (0, 1, 2, 3)
AXES = ("X", "Y", "Z")
DIRECTED_PAIRS = tuple((c, t) for c in WIRES for t in WIRES if c != t)

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
    "amplitudes can be chosen real. Interactions are nearest-neighbor on the "
    "open chain q0-q1-q2-q3, so nearest-neighbor correlations are important. "
    "Resource contract (exact, non-negotiable): the encoder is an ordered "
    "sequence of exactly 16 gates - exactly 12 single-qubit rotations, each "
    "RX, RY, or RZ on any qubit (no per-axis quota), and exactly 4 CNOTs, "
    "each with any control and any different target (the same pair may repeat "
    "at different positions). You choose the gate ORDER freely. You do not "
    "choose rotation angles: a shared numerical optimizer trains all 12 "
    "angles identically for every method."
)

JSON_SCHEMA_CARD = (
    "Return ONLY a single JSON object, no prose and no markdown fences, with this schema:\n"
    '{"candidates": [{"name": "<short-id>", "gates": ['
    '{"g": "RY", "q": 0}, {"g": "CNOT", "c": 3, "t": 2}, ...]}]}\n'
    'Each gates list holds exactly 16 entries in execution order: a rotation is '
    '{"g": "RX"|"RY"|"RZ", "q": 0-3} and a CNOT is {"g": "CNOT", "c": 0-3, "t": 0-3, c != t}. '
    "Exactly 12 rotations and exactly 4 CNOTs per candidate."
)

OPEN_LOOP_PROMPT = (
    TASK_CARD
    + "\nOutput 8 distinct candidates. Prefer hypotheses that (1) preserve a "
    "real-valued representation when useful and (2) route correlations/information "
    "from the trash qubits toward the latent qubits.\n"
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


# ---------------------------------------------------------------- space ----

def verify_neutral_capacity(architecture: list[dict]) -> None:
    if len(architecture) != N_GATES:
        raise ValueError(f"expected {N_GATES} gates, got {len(architecture)}")
    rotations = sum(op["type"] == "R" for op in architecture)
    cnots = sum(op["type"] == "CX" for op in architecture)
    if rotations != N_ROTATIONS or cnots != N_CNOTS:
        raise ValueError(f"capacity mismatch: rotations={rotations}, CNOTs={cnots}")
    for op in architecture:
        if op["type"] == "R":
            if op["axis"] not in AXES or op["q"] not in WIRES:
                raise ValueError(f"invalid rotation {op}")
        else:
            if (op["c"], op["t"]) not in DIRECTED_PAIRS:
                raise ValueError(f"invalid CNOT {op}")


def sample_neutral_arch(rng: np.random.Generator) -> list[dict]:
    """Uniform draw from the neutral space: free axes/qubits/pairs/order."""
    ops: list[dict] = [
        {"type": "R", "axis": str(rng.choice(AXES)), "q": int(rng.choice(WIRES))}
        for _ in range(N_ROTATIONS)
    ]
    for _ in range(N_CNOTS):
        control, target = DIRECTED_PAIRS[int(rng.integers(len(DIRECTED_PAIRS)))]
        ops.append({"type": "CX", "c": control, "t": target})
    order = rng.permutation(N_GATES)
    architecture = [ops[i] for i in order]
    verify_neutral_capacity(architecture)
    return architecture


def parse_candidate(entry: dict) -> tuple[list[dict] | None, list[str]]:
    """Turn one JSON candidate (neutral schema) into an architecture."""
    if not isinstance(entry, dict):
        return None, ["candidate is not an object"]
    gates = entry.get("gates")
    if not isinstance(gates, list) or len(gates) != N_GATES:
        return None, [f"gates must be a list of exactly {N_GATES} entries"]
    architecture: list[dict] = []
    errors: list[str] = []
    for i, g in enumerate(gates):
        if not isinstance(g, dict):
            errors.append(f"gate {i} is not an object")
            continue
        kind = g.get("g")
        if kind in ("RX", "RY", "RZ"):
            q = g.get("q")
            if not isinstance(q, int) or q not in WIRES:
                errors.append(f"gate {i}: rotation qubit must be 0-3")
                continue
            architecture.append({"type": "R", "axis": kind[1], "q": q})
        elif kind == "CNOT":
            c, t = g.get("c"), g.get("t")
            if not isinstance(c, int) or not isinstance(t, int) or (c, t) not in DIRECTED_PAIRS:
                errors.append(f"gate {i}: CNOT needs control != target in 0-3")
                continue
            architecture.append({"type": "CX", "c": c, "t": t})
        else:
            errors.append(f"gate {i}: g must be RX, RY, RZ, or CNOT")
    if errors:
        return None, errors
    try:
        verify_neutral_capacity(architecture)
    except ValueError as exc:
        return None, [str(exc)]
    return architecture, []


def mutate_one_choice(architecture: list[dict], rng: np.random.Generator) -> list[dict]:
    """Exactly one structural change (Greedy move set; see protocol v3)."""
    arch = [dict(op) for op in architecture]
    rotation_idx = [i for i, op in enumerate(arch) if op["type"] == "R"]
    cnot_idx = [i for i, op in enumerate(arch) if op["type"] == "CX"]
    move = int(rng.integers(5))
    if move == 0:  # one rotation axis
        i = rotation_idx[int(rng.integers(len(rotation_idx)))]
        arch[i]["axis"] = str(rng.choice([a for a in AXES if a != arch[i]["axis"]]))
    elif move == 1:  # one rotation qubit
        i = rotation_idx[int(rng.integers(len(rotation_idx)))]
        arch[i]["q"] = int(rng.choice([w for w in WIRES if w != arch[i]["q"]]))
    elif move == 2:  # one CNOT control
        i = cnot_idx[int(rng.integers(len(cnot_idx)))]
        options = [w for w in WIRES if w != arch[i]["c"] and w != arch[i]["t"]]
        arch[i]["c"] = int(rng.choice(options))
    elif move == 3:  # one CNOT target
        i = cnot_idx[int(rng.integers(len(cnot_idx)))]
        options = [w for w in WIRES if w != arch[i]["t"] and w != arch[i]["c"]]
        arch[i]["t"] = int(rng.choice(options))
    else:  # swap two gate positions
        i, j = rng.choice(N_GATES, size=2, replace=False)
        arch[int(i)], arch[int(j)] = arch[int(j)], arch[int(i)]
    verify_neutral_capacity(arch)
    return arch


# ------------------------------------------------------------- LLM layer ----

def _build_provider():
    from llm_vqc.llm.openai_provider import OpenAIProvider

    api_key = os.environ.get("OPENAI_API_KEY", "")
    model = os.environ.get("OPENAI_MODEL", "")
    if not api_key or not model:
        raise RuntimeError("OPENAI_API_KEY and OPENAI_MODEL must be set for API arms")
    return OpenAIProvider(api_key=api_key, model=model, max_output_tokens=6000)


def _complete_json(
    provider,
    budget: LLMApiBudget,
    user_prompt: str,
    proposal_id: str,
    cost_estimate: float,
) -> tuple[dict | None, list[LLMCallRecord]]:
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


def _record_to_dict(record: LLMCallRecord) -> dict:
    return json.loads(record.model_dump_json())


@dataclass
class LLMPool:
    candidates: dict[str, list[dict]]
    call_records: list[LLMCallRecord] = field(default_factory=list)
    fallback_names: list[str] = field(default_factory=list)
    rejected: list[dict] = field(default_factory=list)


def generate_open_pool(destination: Path) -> LLMPool:
    """Generate (or reload) the LLM-Open pool of 8 neutral candidates."""
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
    provider = _build_provider()
    calls_dir = destination / "llm_calls"
    calls_dir.mkdir(parents=True, exist_ok=True)

    candidates: dict[str, list[dict]] = {}
    seen: set[str] = set()
    fallback_names: list[str] = []
    rejected: list[dict] = []
    records: list[LLMCallRecord] = []

    def absorb(parsed_obj: dict | None) -> list[dict]:
        """Validate one response's candidates; return the newly rejected ones."""
        newly_rejected: list[dict] = []
        if parsed_obj is None:
            return newly_rejected
        for i, entry in enumerate(parsed_obj.get("candidates", [])[: BUDGET * 2]):
            architecture, errors = parse_candidate(entry)
            if architecture is None:
                newly_rejected.append({"index": i, "errors": errors, "entry": entry})
                continue
            key = architecture_key(architecture)
            if key in seen:
                newly_rejected.append({"index": i, "errors": ["duplicate"], "entry": entry})
                continue
            if len(candidates) >= BUDGET:
                break
            seen.add(key)
            name = str(entry.get("name", f"api_{i}"))[:48] or f"api_{i}"
            while name in candidates:
                name += "_"
            candidates[name] = architecture
        return newly_rejected

    parsed, call_records = _complete_json(
        provider, budget, OPEN_LOOP_PROMPT, "open_pool", POOL_CALL_COST_ESTIMATE_USD
    )
    records.extend(call_records)
    rejected.extend(absorb(parsed))

    # Pre-inspection amendment 1: bounded capacity repair. Up to 2 extra
    # calls request replacements for rejected entries, quoting the errors.
    repair_round = 0
    while len(candidates) < BUDGET and rejected and repair_round < MAX_REPAIR_ATTEMPTS:
        repair_round += 1
        deficit = BUDGET - len(candidates)
        complaint_lines = [
            f"- {json.dumps(r['entry'], separators=(',', ':'))[:400]} rejected: "
            + "; ".join(r["errors"])[:200]
            for r in rejected[-deficit:]
        ]
        repair_prompt = (
            TASK_CARD
            + "\nYour earlier candidates below were REJECTED for violating the exact "
            "resource contract (exactly 12 rotations and exactly 4 CNOTs, 16 gates):\n"
            + "\n".join(complaint_lines)
            + f"\nOutput exactly {deficit} NEW distinct candidates that satisfy the "
            "contract exactly. Count the gates before answering.\n"
            + JSON_SCHEMA_CARD
        )
        parsed, call_records = _complete_json(
            provider, budget, repair_prompt, f"open_pool_repair{repair_round}",
            POOL_CALL_COST_ESTIMATE_USD,
        )
        records.extend(call_records)
        rejected.extend(absorb(parsed))

    deficit_rng = np.random.default_rng(999_999)
    while len(candidates) < BUDGET:
        architecture = sample_neutral_arch(deficit_rng)
        key = architecture_key(architecture)
        if key in seen:
            continue
        seen.add(key)
        name = f"fallback_random_{len(fallback_names)}"
        candidates[name] = architecture
        fallback_names.append(name)

    for i, record in enumerate(records):
        (calls_dir / f"open_pool_call{i}.json").write_text(
            json.dumps(_record_to_dict(record), indent=2)
        )
    pool_path.write_text(
        json.dumps(
            {
                "candidates": candidates,
                "fallback_names": fallback_names,
                "rejected": rejected,
                "model_snapshots": sorted({r.model for r in records}),
                "total_input_tokens": sum(r.input_tokens for r in records),
                "total_output_tokens": sum(r.output_tokens for r in records),
                "budget_spent_usd_estimate": budget.spent_usd,
                "n_calls": len(records),
            },
            indent=2,
        )
    )
    return LLMPool(candidates=candidates, call_records=records,
                   fallback_names=fallback_names, rejected=rejected)


# ----------------------------------------------------------------- arms ----

def run_random_seed(seed: int, train, val, test) -> list[dict]:
    rng = np.random.default_rng(60_000 + seed)
    rows: list[dict] = []
    seen: set[str] = set()
    order = 0
    while order < BUDGET:
        architecture = sample_neutral_arch(rng)
        key = architecture_key(architecture)
        if key in seen:
            continue
        seen.add(key)
        order += 1
        result = evaluate_architecture(
            architecture, train, val, test, seed=_train_seed(100 + seed, architecture)
        )
        rows.append({"seed": seed, "method": "Random", "candidate": f"R{order}",
                     "order": order, "fallback": False, "invalid_errors": "",
                     **result.__dict__})
    return rows


def run_greedy_seed(seed: int, train, val, test) -> list[dict]:
    rng = np.random.default_rng(70_000 + seed)
    rows: list[dict] = []
    seen: set[str] = set()

    incumbent = sample_neutral_arch(rng)
    seen.add(architecture_key(incumbent))
    incumbent_result = evaluate_architecture(
        incumbent, train, val, test, seed=_train_seed(100 + seed, incumbent)
    )
    rows.append({"seed": seed, "method": "Greedy", "candidate": "G1_init",
                 "order": 1, "fallback": False, "invalid_errors": "",
                 **incumbent_result.__dict__})

    for order in range(2, BUDGET + 1):
        candidate = None
        fallback = False
        for _ in range(200):
            mutated = mutate_one_choice(incumbent, rng)
            if architecture_key(mutated) not in seen:
                candidate = mutated
                break
        if candidate is None:  # pathological: fall back to a fresh random draw
            while True:
                candidate = sample_neutral_arch(rng)
                if architecture_key(candidate) not in seen:
                    break
            fallback = True
        seen.add(architecture_key(candidate))
        result = evaluate_architecture(
            candidate, train, val, test, seed=_train_seed(100 + seed, candidate)
        )
        accepted = result.val_loss < incumbent_result.val_loss
        if accepted:
            incumbent, incumbent_result = candidate, result
        rows.append({"seed": seed, "method": "Greedy",
                     "candidate": f"G{order}_{'acc' if accepted else 'rej'}",
                     "order": order, "fallback": fallback, "invalid_errors": "",
                     **result.__dict__})
    return rows


def run_open_seed(seed: int, pool: LLMPool, train, val, test) -> list[dict]:
    rows: list[dict] = []
    for i, (name, architecture) in enumerate(pool.candidates.items()):
        result = evaluate_architecture(
            architecture, train, val, test, seed=_train_seed(100 + seed, architecture)
        )
        rows.append({"seed": seed, "method": "LLM-Open", "candidate": name,
                     "order": i + 1, "fallback": name in pool.fallback_names,
                     "invalid_errors": "", **result.__dict__})
    return rows


def run_closed_seed(seed: int, destination: Path, train, val, test) -> list[dict]:
    store = destination / f"llm_closed_seed{seed}.json"
    if store.exists():
        return json.loads(store.read_text())["rows"]

    budget = LLMApiBudget.from_env()
    if budget is None:
        raise RuntimeError("LLM_API_BUDGET_USD is not configured; refusing paid API calls")
    provider = _build_provider()
    calls_dir = destination / "llm_calls"
    calls_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    history: list[dict] = []
    seen: set[str] = set()
    fallback_rng = np.random.default_rng(80_000 + seed)
    all_records: list[LLMCallRecord] = []

    for proposal in range(BUDGET):
        parsed, records = _complete_json(
            provider, budget, closed_loop_prompt(history),
            f"closed_s{seed}_p{proposal}", CLOSED_CALL_COST_ESTIMATE_USD,
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
        duplicate = architecture is not None and architecture_key(architecture) in seen
        fallback = False
        if architecture is None or duplicate:
            while True:
                architecture = sample_neutral_arch(fallback_rng)
                if architecture_key(architecture) not in seen:
                    break
            fallback = True
        seen.add(architecture_key(architecture))
        result = evaluate_architecture(
            architecture, train, val, test, seed=_train_seed(100 + seed, architecture)
        )
        candidate_json = entry if (entry is not None and not fallback) else {
            "name": f"fallback_random_p{proposal}", "note": "fallback"
        }
        history.append({"candidate": candidate_json, "val_fid": result.val_fid,
                        "duplicate": duplicate})
        rows.append({"seed": seed, "method": "LLM-Closed",
                     "candidate": (str(entry.get("name", f"p{proposal}"))[:48]
                                   if entry is not None and not fallback
                                   else f"fallback_p{proposal}"),
                     "order": proposal + 1, "fallback": fallback,
                     "invalid_errors": "; ".join(errors), **result.__dict__})

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


# ------------------------------------------------------------- analysis ----

METHODS = ("Random", "Greedy", "LLM-Open", "LLM-Closed")


def analyze(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    selected = (
        frame.sort_values("val_loss")
        .groupby(["seed", "method"], as_index=False)
        .first()
    )
    pivot = selected.pivot(index="seed", columns="method", values="test_fid")
    rng = np.random.default_rng(0)
    stats: dict[str, dict] = {}
    pairs = [(a, b) for i, a in enumerate(METHODS) for b in METHODS[i + 1:]]
    for a, b in pairs:
        if a not in pivot or b not in pivot:
            continue
        diff = (pivot[b] - pivot[a]).dropna()
        if len(diff) < 6 or np.allclose(diff, 0):
            continue
        exact = wilcoxon(diff, alternative="two-sided", method="exact")
        sign = binomtest(int((diff > 0).sum()), len(diff), 0.5, alternative="two-sided")
        boots = np.array([
            rng.choice(diff.values, size=len(diff), replace=True).mean()
            for _ in range(10_000)
        ])
        sd = diff.std(ddof=1)
        stats[f"{b}_minus_{a}"] = {
            "mean_paired_gain": float(diff.mean()),
            "median_paired_gain": float(diff.median()),
            "bootstrap_95ci": [float(np.percentile(boots, 2.5)),
                               float(np.percentile(boots, 97.5))],
            "cohens_dz": float(diff.mean() / sd) if sd > 0 else float("inf"),
            "wins_b": int((diff > 0).sum()),
            "n": int(len(diff)),
            "wilcoxon_exact_two_sided_p": float(exact.pvalue),
            "sign_test_two_sided_p": float(sign.pvalue),
        }
    return selected, stats


def anytime_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (seed, method), group in frame.groupby(["seed", "method"]):
        group = group.sort_values("order")
        best_val = float("inf")
        best_test = float("nan")
        for _, row in group.iterrows():
            if row["val_loss"] < best_val:
                best_val = row["val_loss"]
                best_test = row["test_fid"]
            rows.append({"seed": seed, "method": method,
                         "candidates_used": int(row["order"]),
                         "best_so_far_test_fid": best_test})
    anytime = pd.DataFrame(rows)
    return anytime.groupby(["method", "candidates_used"], as_index=False)[
        "best_so_far_test_fid"
    ].mean()


def proposal_quality_table(frame: pd.DataFrame, pool: LLMPool | None) -> dict:
    out: dict[str, dict] = {}
    for method in METHODS:
        sub = frame[frame["method"] == method]
        out[method] = {
            "evaluations": int(len(sub)),
            "fallback_evaluations": int(sub["fallback"].sum()),
        }
    if pool is not None:
        out["LLM-Open"]["pool_rejected_entries"] = len(pool.rejected)
        out["LLM-Open"]["pool_fallbacks"] = len(pool.fallback_names)
    return out


def main(
    output_dir: str = "outputs/qae_tfim_neutral_v3",
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
        print(f"seed {seed}: Random / Greedy", flush=True)
        train, val, test = state_sets(seed)
        rows.extend(run_random_seed(seed, train, val, test))
        rows.extend(run_greedy_seed(seed, train, val, test))
        if pool is not None:
            rows.extend(run_open_seed(seed, pool, train, val, test))
    if with_api and closed_loop:
        for seed in seeds:
            print(f"seed {seed}: LLM-Closed", flush=True)
            train, val, test = state_sets(seed)
            rows.extend(run_closed_seed(seed, destination, train, val, test))

    frame = pd.DataFrame(rows)
    selected, stats = analyze(frame)
    frame.to_csv(destination / "candidate_results.csv", index=False)
    selected.to_csv(destination / "selected_results.csv", index=False)
    anytime_table(frame).to_csv(destination / "anytime_mean.csv", index=False)
    (destination / "paired_stats.json").write_text(json.dumps(stats, indent=2))
    (destination / "proposal_quality.json").write_text(
        json.dumps(proposal_quality_table(frame, pool), indent=2)
    )
    print(selected.groupby("method")["test_fid"].agg(["mean", "median", "std"]))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
