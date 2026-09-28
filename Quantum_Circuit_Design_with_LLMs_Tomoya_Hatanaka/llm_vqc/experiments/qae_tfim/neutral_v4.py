"""QAE-TFIM v4: multi-start four-method benchmark (pre-registered).

Protocol: docs/research/QAE_PROTOCOL_V4.md (frozen before any v4
protected-test inspection). v4 keeps the v3 neutral space, task, trainer,
paired seeds, and B=8 budget, and changes ONLY the adaptive methods'
allocation policy to a 4+4 exploration/refinement split:

- Random: 8 independent draws (fresh v4 rng streams — an independent
  replication, not a replay of the v3 stream).
- Greedy: 4 independent random warm starts, then 4 one-change local
  refinements of the best warm start (accept only if validation improves).
- LLM-Open: one batch call for 8 candidates, no feedback (fresh v4 pool).
- LLM-Closed: one batch call for 4 no-feedback semantic warm starts per
  seed, then 4 sequential feedback-informed proposals.

v3 (`neutral_v3.py`, `outputs/qae_tfim_neutral_v3/`) is preserved
unchanged as the single-start diagnostic experiment.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon

from llm_vqc.experiments.qae_tfim.neutral_v3 import (
    JSON_SCHEMA_CARD,
    TASK_CARD,
    LLMPool,
    _build_provider,
    _complete_json,
    _record_to_dict,
    closed_loop_prompt,
    generate_open_pool,
    mutate_one_choice,
    parse_candidate,
    sample_neutral_arch,
)
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

N_WARM = 4  # evaluations in the exploration phase of each adaptive method
WARM_CALL_COST_ESTIMATE_USD = 0.01
CLOSED_CALL_COST_ESTIMATE_USD = 0.005
MAX_REPAIR_ATTEMPTS = 2

WARMSTART_PROMPT = (
    TASK_CARD
    + f"\nOutput {N_WARM} distinct candidates. Aim for structural DIVERSITY "
    "across the candidates (different orderings, wirings, and axis patterns), "
    "while preferring hypotheses that (1) preserve a real-valued representation "
    "when useful and (2) route correlations/information from the trash qubits "
    "toward the latent qubits.\n"
    + JSON_SCHEMA_CARD
)


def run_random_seed(seed: int, train, val, test) -> list[dict]:
    """8 independent uniform draws; fresh v4 stream (61_000+seed)."""
    rng = np.random.default_rng(61_000 + seed)
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
                     "order": order, "phase": "explore", "fallback": False,
                     "invalid_errors": "", **result.__dict__})
    return rows


def run_greedy44_seed(seed: int, train, val, test) -> list[dict]:
    """Phase 1: 4 random warm starts. Phase 2: 4 one-change refinements of
    the best warm start, accepted only on validation improvement."""
    rng = np.random.default_rng(91_000 + seed)
    rows: list[dict] = []
    seen: set[str] = set()

    incumbent = None
    incumbent_result = None
    for order in range(1, N_WARM + 1):
        while True:
            architecture = sample_neutral_arch(rng)
            if architecture_key(architecture) not in seen:
                break
        seen.add(architecture_key(architecture))
        result = evaluate_architecture(
            architecture, train, val, test, seed=_train_seed(100 + seed, architecture)
        )
        rows.append({"seed": seed, "method": "Greedy", "candidate": f"G{order}_warm",
                     "order": order, "phase": "explore", "fallback": False,
                     "invalid_errors": "", **result.__dict__})
        if incumbent_result is None or result.val_loss < incumbent_result.val_loss:
            incumbent, incumbent_result = architecture, result

    for order in range(N_WARM + 1, BUDGET + 1):
        candidate = None
        fallback = False
        for _ in range(200):
            mutated = mutate_one_choice(incumbent, rng)
            if architecture_key(mutated) not in seen:
                candidate = mutated
                break
        if candidate is None:
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
                     "order": order, "phase": "refine", "fallback": fallback,
                     "invalid_errors": "", **result.__dict__})
    return rows


def run_open_seed(seed: int, pool: LLMPool, train, val, test) -> list[dict]:
    rows: list[dict] = []
    for i, (name, architecture) in enumerate(pool.candidates.items()):
        result = evaluate_architecture(
            architecture, train, val, test, seed=_train_seed(100 + seed, architecture)
        )
        rows.append({"seed": seed, "method": "LLM-Open", "candidate": name,
                     "order": i + 1, "phase": "explore",
                     "fallback": name in pool.fallback_names,
                     "invalid_errors": "", **result.__dict__})
    return rows


def _generate_warm_batch(
    seed: int, destination: Path, budget: LLMApiBudget, provider
) -> tuple[dict[str, list[dict]], list[str], list[dict], list[LLMCallRecord]]:
    """One batch call (plus bounded capacity repair) for 4 diverse
    no-feedback candidates. Deficits after repair are filled with flagged
    random draws."""
    candidates: dict[str, list[dict]] = {}
    seen: set[str] = set()
    fallback_names: list[str] = []
    rejected: list[dict] = []
    records: list[LLMCallRecord] = []

    def absorb(parsed_obj: dict | None) -> None:
        if parsed_obj is None:
            return
        for i, entry in enumerate(parsed_obj.get("candidates", [])[: N_WARM * 2]):
            architecture, errors = parse_candidate(entry)
            if architecture is None:
                rejected.append({"index": i, "errors": errors, "entry": entry})
                continue
            key = architecture_key(architecture)
            if key in seen:
                rejected.append({"index": i, "errors": ["duplicate"], "entry": entry})
                continue
            if len(candidates) >= N_WARM:
                break
            seen.add(key)
            name = str(entry.get("name", f"warm_{i}"))[:48] or f"warm_{i}"
            while name in candidates:
                name += "_"
            candidates[name] = architecture

    parsed, call_records = _complete_json(
        provider, budget, WARMSTART_PROMPT, f"warm_s{seed}", WARM_CALL_COST_ESTIMATE_USD
    )
    records.extend(call_records)
    absorb(parsed)

    repair_round = 0
    while len(candidates) < N_WARM and rejected and repair_round < MAX_REPAIR_ATTEMPTS:
        repair_round += 1
        deficit = N_WARM - len(candidates)
        complaint = "\n".join(
            f"- {json.dumps(r['entry'], separators=(',', ':'))[:400]} rejected: "
            + "; ".join(r["errors"])[:200]
            for r in rejected[-deficit:]
        )
        repair_prompt = (
            TASK_CARD
            + "\nYour earlier candidates below were REJECTED for violating the exact "
            "resource contract (exactly 12 rotations and exactly 4 CNOTs, 16 gates):\n"
            + complaint
            + f"\nOutput exactly {deficit} NEW distinct candidates that satisfy the "
            "contract exactly. Count the gates before answering.\n"
            + JSON_SCHEMA_CARD
        )
        parsed, call_records = _complete_json(
            provider, budget, repair_prompt, f"warm_s{seed}_repair{repair_round}",
            WARM_CALL_COST_ESTIMATE_USD,
        )
        records.extend(call_records)
        absorb(parsed)

    deficit_rng = np.random.default_rng(95_000 + seed)
    while len(candidates) < N_WARM:
        architecture = sample_neutral_arch(deficit_rng)
        if architecture_key(architecture) in seen:
            continue
        seen.add(architecture_key(architecture))
        name = f"warm_fallback_{len(fallback_names)}"
        candidates[name] = architecture
        fallback_names.append(name)

    return candidates, fallback_names, rejected, records


def run_closed44_seed(seed: int, destination: Path, train, val, test) -> list[dict]:
    """Phase 1: batch of 4 semantic warm starts (no feedback). Phase 2:
    4 sequential feedback-informed proposals. Resumable per seed."""
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
    architectures_by_order: list[list[dict]] = []
    history: list[dict] = []
    seen: set[str] = set()
    fallback_rng = np.random.default_rng(85_000 + seed)
    all_records: list[LLMCallRecord] = []

    warm, warm_fallbacks, warm_rejected, warm_records = _generate_warm_batch(
        seed, destination, budget, provider
    )
    all_records.extend(warm_records)
    for i, (name, architecture) in enumerate(warm.items()):
        seen.add(architecture_key(architecture))
        architectures_by_order.append(architecture)
        result = evaluate_architecture(
            architecture, train, val, test, seed=_train_seed(100 + seed, architecture)
        )
        history.append({"candidate": {"name": name}, "val_fid": result.val_fid,
                        "duplicate": False})
        rows.append({"seed": seed, "method": "LLM-Closed", "candidate": name,
                     "order": i + 1, "phase": "explore",
                     "fallback": name in warm_fallbacks,
                     "invalid_errors": "", **result.__dict__})

    for proposal in range(N_WARM, BUDGET):
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
        architectures_by_order.append(architecture)
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
                     "order": proposal + 1, "phase": "refine", "fallback": fallback,
                     "invalid_errors": "; ".join(errors), **result.__dict__})

    for i, record in enumerate(all_records):
        (calls_dir / f"closed_s{seed}_call{i}.json").write_text(
            json.dumps(_record_to_dict(record), indent=2)
        )
    store.write_text(
        json.dumps(
            {
                "rows": rows,
                "architectures": architectures_by_order,
                "warm_rejected": warm_rejected,
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


def refinement_gains(frame: pd.DataFrame) -> dict:
    """Selected result minus the best (by validation) of the first 4
    warm-start candidates, for the adaptive methods."""
    out: dict[str, dict] = {}
    rng = np.random.default_rng(1)
    for method in ("Greedy", "LLM-Closed"):
        per_seed_val, per_seed_test = [], []
        for _seed, group in frame[frame["method"] == method].groupby("seed"):
            group = group.sort_values("order")
            warm = group[group["order"] <= N_WARM]
            warm_best = warm.loc[warm["val_loss"].idxmin()]
            final_best = group.loc[group["val_loss"].idxmin()]
            per_seed_val.append(float(final_best["val_fid"] - warm_best["val_fid"]))
            per_seed_test.append(float(final_best["test_fid"] - warm_best["test_fid"]))
        test_arr = np.array(per_seed_test)
        boots = np.array([
            rng.choice(test_arr, size=len(test_arr), replace=True).mean()
            for _ in range(10_000)
        ])
        nonzero = test_arr[test_arr != 0]
        out[method] = {
            "mean_val_gain": float(np.mean(per_seed_val)),
            "mean_test_gain": float(np.mean(per_seed_test)),
            "median_test_gain": float(np.median(per_seed_test)),
            "test_gain_bootstrap_95ci": [float(np.percentile(boots, 2.5)),
                                         float(np.percentile(boots, 97.5))],
            "seeds_improved": int((test_arr > 0).sum()),
            "seeds_unchanged": int((test_arr == 0).sum()),
            "n": len(per_seed_test),
            "wilcoxon_p_nonzero_diffs": (
                float(wilcoxon(nonzero, alternative="two-sided").pvalue)
                if len(nonzero) >= 6 else None
            ),
        }
    return out


def _arch_distance(a: list[dict], b: list[dict]) -> float:
    same = sum(1 for x, y in zip(a, b, strict=True) if x == y)
    return 1.0 - same / len(a)


def _mean_pairwise_distance(archs: list[list[dict]]) -> float | None:
    pairs = [(i, j) for i in range(len(archs)) for j in range(i + 1, len(archs))]
    if not pairs:
        return None
    return float(np.mean([_arch_distance(archs[i], archs[j]) for i, j in pairs]))


def diversity_stats(destination: Path, pool: LLMPool) -> dict:
    """Unique counts and mean pairwise structural distance (fraction of
    differing gate slots) for LLM-Open (first/last 4 of the pool) and
    LLM-Closed (warm batch vs refinement proposals, averaged over seeds)."""
    out: dict[str, dict] = {}
    pool_archs = list(pool.candidates.values())
    out["LLM-Open"] = {
        "first4_mean_pairwise_distance": _mean_pairwise_distance(pool_archs[:4]),
        "last4_mean_pairwise_distance": _mean_pairwise_distance(pool_archs[4:]),
        "unique_of_8": len({architecture_key(a) for a in pool_archs}),
    }
    warm_d, refine_d, fallbacks = [], [], 0
    for seed in VERIFY_SEEDS:
        stored = json.loads((destination / f"llm_closed_seed{seed}.json").read_text())
        archs = stored.get("architectures", [])
        fallbacks += sum(1 for r in stored["rows"] if r["fallback"])
        if len(archs) == BUDGET:
            w = _mean_pairwise_distance(archs[:N_WARM])
            r = _mean_pairwise_distance(archs[N_WARM:])
            if w is not None:
                warm_d.append(w)
            if r is not None:
                refine_d.append(r)
    out["LLM-Closed"] = {
        "warm4_mean_pairwise_distance": float(np.mean(warm_d)) if warm_d else None,
        "refine4_mean_pairwise_distance": float(np.mean(refine_d)) if refine_d else None,
        "fallback_evaluations_total": fallbacks,
    }
    return out


def proposal_quality(frame: pd.DataFrame, pool: LLMPool) -> dict:
    out: dict[str, dict] = {}
    for method in METHODS:
        sub = frame[frame["method"] == method]
        entry = {
            "evaluations": int(len(sub)),
            "fallback_evaluations": int(sub["fallback"].sum()),
        }
        if method in ("Greedy", "LLM-Closed"):
            refine = sub[sub["phase"] == "refine"]
            entry["refine_fallbacks"] = int(refine["fallback"].sum())
        out[method] = entry
    out["LLM-Open"]["pool_rejected_entries"] = len(pool.rejected)
    out["LLM-Open"]["pool_fallbacks"] = len(pool.fallback_names)
    return out


def main(
    output_dir: str = "outputs/qae_tfim_neutral_v4",
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
        print(f"seed {seed}: Random / Greedy(4+4)", flush=True)
        train, val, test = state_sets(seed)
        rows.extend(run_random_seed(seed, train, val, test))
        rows.extend(run_greedy44_seed(seed, train, val, test))
        if pool is not None:
            rows.extend(run_open_seed(seed, pool, train, val, test))
    if with_api and closed_loop:
        for seed in seeds:
            print(f"seed {seed}: LLM-Closed(4+4)", flush=True)
            train, val, test = state_sets(seed)
            rows.extend(run_closed44_seed(seed, destination, train, val, test))

    frame = pd.DataFrame(rows)
    selected, stats = analyze(frame)
    frame.to_csv(destination / "candidate_results.csv", index=False)
    selected.to_csv(destination / "selected_results.csv", index=False)
    anytime_table(frame).to_csv(destination / "anytime_mean.csv", index=False)
    (destination / "paired_stats.json").write_text(json.dumps(stats, indent=2))
    (destination / "refinement_gains.json").write_text(
        json.dumps(refinement_gains(frame), indent=2)
    )
    quality = proposal_quality(frame, pool) if pool is not None else {}
    if pool is not None and closed_loop:
        quality["diversity"] = diversity_stats(destination, pool)
    (destination / "proposal_quality.json").write_text(json.dumps(quality, indent=2))
    print(selected.groupby("method")["test_fid"].agg(["mean", "median", "std"]))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
