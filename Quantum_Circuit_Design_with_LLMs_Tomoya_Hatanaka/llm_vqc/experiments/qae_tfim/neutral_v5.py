"""QAE-TFIM v5: incumbent-based free-form LLM redesign (pre-registered).

Protocol: docs/research/QAE_PROTOCOL_V5.md (frozen before any v5
protected-test inspection). v3/v4 are preserved unchanged.

v5 changes ONLY the LLM-Closed refinement step: it receives the CURRENT
BEST architecture and its validation trash fidelity and may redesign the
architecture freely within the fixed 12R+4CX capacity (one change, several
coordinated changes, or a complete redesign). Greedy remains deliberately
restricted to one structural change per step. Random and LLM-Open keep
their designs with fresh v5 draws/pool.
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
    generate_open_pool,
    mutate_one_choice,
    parse_candidate,
    sample_neutral_arch,
)
from llm_vqc.experiments.qae_tfim.neutral_v4 import (
    N_WARM,
    _arch_distance,
    _generate_warm_batch,
    _mean_pairwise_distance,
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

REFINE_CALL_COST_ESTIMATE_USD = 0.005
STRATEGIES = ("local_adjustment", "topology_redesign", "rotation_redesign",
              "global_redesign")


def _arch_to_gate_list(architecture: list[dict]) -> list[dict]:
    out = []
    for op in architecture:
        if op["type"] == "R":
            out.append({"g": "R" + op["axis"], "q": op["q"]})
        else:
            out.append({"g": "CNOT", "c": op["c"], "t": op["t"]})
    return out


def redesign_prompt(incumbent: list[dict], val_fid: float) -> str:
    """The v5 free-form redesign prompt (versioned here; see protocol §9)."""
    gate_list = json.dumps(_arch_to_gate_list(incumbent), separators=(",", ":"))
    return (
        "You are improving the currently best quantum-autoencoder architecture.\n"
        + TASK_CARD
        + f"\n\nCurrent best architecture (execution order): {gate_list}\n"
        f"Current validation trash fidelity: {val_fid:.4f}\n\n"
        "You may redesign the architecture freely. Your next proposal may modify "
        "one gate choice, several coordinated choices, or the entire architecture, "
        "provided the exact resource budget is preserved: exactly 12 single-qubit "
        "rotations (each RX, RY, or RZ), exactly 4 CNOTs, exactly 16 gates total, "
        "four qubits, arbitrary valid ordering, arbitrary valid CNOT control/target. "
        "You may retain the incumbent conceptually, modify it locally, or redesign "
        "it substantially. Do not propose numerical rotation angles - a shared "
        "optimizer trains all 12 angles. Do not repeat the incumbent or any "
        "previously evaluated architecture exactly.\n"
        "Use the physical structure of the task - the real-valued ground states, "
        "nearest-neighbor Ising interactions, and latent/trash roles - to propose "
        "an architecture you expect to outperform the incumbent.\n"
        "Return one complete new architecture.\n"
        + JSON_SCHEMA_CARD
        + "\nThe candidates array must contain exactly 1 candidate. In addition to "
        '"name" and "gates", include: "strategy" (one of "local_adjustment", '
        '"topology_redesign", "rotation_redesign", "global_redesign"), '
        '"rationale" (one short sentence), "preserved" (short phrase: what you '
        'kept from the incumbent), "changed" (short phrase: what you changed). '
        "This metadata is recorded for analysis only."
    )


# ----------------------------------------------------------------- arms ----

def run_random_seed(seed: int, train, val, test) -> list[dict]:
    rng = np.random.default_rng(62_000 + seed)
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
                     "invalid_errors": "", "edit_distance": np.nan,
                     "strategy": "", **result.__dict__})
    return rows


def run_greedy_seed(seed: int, train, val, test) -> list[dict]:
    """Unchanged 4+4 design; fresh v5 stream (93_000+seed)."""
    rng = np.random.default_rng(93_000 + seed)
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
                     "invalid_errors": "", "edit_distance": np.nan,
                     "strategy": "", **result.__dict__})
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
        distance = _arch_distance(incumbent, candidate)
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
                     "invalid_errors": "", "edit_distance": distance,
                     "strategy": "", **result.__dict__})
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
                     "invalid_errors": "", "edit_distance": np.nan,
                     "strategy": "", **result.__dict__})
    return rows


def run_closed_seed(seed: int, destination: Path, train, val, test) -> list[dict]:
    """v5 LLM-Closed: 4 semantic warm starts -> best incumbent -> 4
    free-form redesign proposals conditioned only on the current best."""
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
    seen: set[str] = set()
    fallback_rng = np.random.default_rng(86_000 + seed)
    all_records: list[LLMCallRecord] = []
    failure_log: list[dict] = []

    warm, warm_fallbacks, warm_rejected, warm_records = _generate_warm_batch(
        seed, destination, budget, provider
    )
    all_records.extend(warm_records)

    incumbent = None
    incumbent_result = None
    for i, (name, architecture) in enumerate(warm.items()):
        seen.add(architecture_key(architecture))
        architectures_by_order.append(architecture)
        result = evaluate_architecture(
            architecture, train, val, test, seed=_train_seed(100 + seed, architecture)
        )
        rows.append({"seed": seed, "method": "LLM-Closed", "candidate": name,
                     "order": i + 1, "phase": "explore",
                     "fallback": name in warm_fallbacks,
                     "invalid_errors": "", "edit_distance": np.nan,
                     "strategy": "", **result.__dict__})
        if incumbent_result is None or result.val_loss < incumbent_result.val_loss:
            incumbent, incumbent_result = architecture, result

    for proposal in range(N_WARM, BUDGET):
        parsed, records = _complete_json(
            provider, budget, redesign_prompt(incumbent, incumbent_result.val_fid),
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
                architecture, errors = parse_candidate(entry)
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
                architecture = sample_neutral_arch(fallback_rng)
                if architecture_key(architecture) not in seen:
                    break
        distance = _arch_distance(incumbent, architecture)
        seen.add(architecture_key(architecture))
        architectures_by_order.append(architecture)
        result = evaluate_architecture(
            architecture, train, val, test, seed=_train_seed(100 + seed, architecture)
        )
        accepted = result.val_loss < incumbent_result.val_loss
        if accepted:
            incumbent, incumbent_result = architecture, result
        strategy = ""
        if entry is not None and not fallback:
            raw_strategy = str(entry.get("strategy", ""))
            strategy = raw_strategy if raw_strategy in STRATEGIES else f"other:{raw_strategy[:24]}"
        rows.append({"seed": seed, "method": "LLM-Closed",
                     "candidate": (str(entry.get("name", f"p{proposal}"))[:48]
                                   if entry is not None and not fallback
                                   else f"fallback_p{proposal}"),
                     "order": proposal + 1, "phase": "refine", "fallback": fallback,
                     "invalid_errors": failure, "edit_distance": distance,
                     "strategy": strategy, **result.__dict__})

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
                "refine_failures": failure_log,
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
    out: dict[str, dict] = {}
    rng = np.random.default_rng(1)
    for method in ("Greedy", "LLM-Closed"):
        per_seed_val, per_seed_test, accepted_counts = [], [], []
        for _seed, group in frame[frame["method"] == method].groupby("seed"):
            group = group.sort_values("order")
            warm = group[group["order"] <= N_WARM]
            warm_best = warm.loc[warm["val_loss"].idxmin()]
            final_best = group.loc[group["val_loss"].idxmin()]
            per_seed_val.append(float(final_best["val_fid"] - warm_best["val_fid"]))
            per_seed_test.append(float(final_best["test_fid"] - warm_best["test_fid"]))
            refine = group[group["phase"] == "refine"]
            accepted_counts.append(int(refine["candidate"].str.endswith("_acc").sum())
                                   if method == "Greedy" else np.nan)
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


def redesign_diagnostics(frame: pd.DataFrame, destination: Path) -> dict:
    """Edit-distance, acceptance-vs-locality, and strategy statistics for
    the v5 LLM-Closed refinement proposals (pre-declared)."""
    refine = frame[(frame["method"] == "LLM-Closed")
                   & (frame["phase"] == "refine")].copy()
    real = refine[~refine["fallback"]]
    # acceptance: a refine row was accepted iff its val_loss beats every
    # earlier candidate in its seed (matching the incumbent update rule).
    def accepted_flags(group: pd.DataFrame) -> pd.Series:
        best = float("inf")
        flags = []
        for _, row in group.sort_values("order").iterrows():
            flags.append(row["val_loss"] < best if row["phase"] == "refine" else False)
            best = min(best, row["val_loss"])
        return pd.Series(flags, index=group.sort_values("order").index)

    flags = []
    for _seed, group in frame[frame["method"] == "LLM-Closed"].groupby("seed"):
        flags.append(accepted_flags(group))
    flags = pd.concat(flags)
    refine_flags = flags.loc[real.index]

    local = real[real["edit_distance"] <= 0.25]
    global_ = real[real["edit_distance"] > 0.25]
    strategy_stats = {}
    for strategy, group in real.groupby("strategy"):
        if not strategy:
            continue
        g_flags = refine_flags.loc[group.index]
        strategy_stats[strategy] = {
            "proposals": int(len(group)),
            "accepted": int(g_flags.sum()),
            "mean_edit_distance": float(group["edit_distance"].mean()),
        }
    diversity = {}
    warm_d, refine_d = [], []
    for seed in VERIFY_SEEDS:
        stored = json.loads((destination / f"llm_closed_seed{seed}.json").read_text())
        archs = stored.get("architectures", [])
        if len(archs) == BUDGET:
            w = _mean_pairwise_distance(archs[:N_WARM])
            r = _mean_pairwise_distance(archs[N_WARM:])
            if w is not None:
                warm_d.append(w)
            if r is not None:
                refine_d.append(r)
    diversity = {
        "warm4_mean_pairwise_distance": float(np.mean(warm_d)) if warm_d else None,
        "refine4_mean_pairwise_distance": float(np.mean(refine_d)) if refine_d else None,
    }
    return {
        "n_refine_proposals": int(len(refine)),
        "n_llm_proposals": int(len(real)),
        "n_fallbacks": int(refine["fallback"].sum()),
        "edit_distance_mean": float(real["edit_distance"].mean()) if len(real) else None,
        "edit_distance_median": float(real["edit_distance"].median()) if len(real) else None,
        "edit_distance_min": float(real["edit_distance"].min()) if len(real) else None,
        "edit_distance_max": float(real["edit_distance"].max()) if len(real) else None,
        "changed_slots_mean": float((real["edit_distance"] * 16).mean()) if len(real) else None,
        "accepted_total": int(refine_flags.sum()),
        "local_proposals_le_4_slots": {
            "n": int(len(local)),
            "accepted": int(refine_flags.loc[local.index].sum()) if len(local) else 0,
        },
        "global_proposals_gt_4_slots": {
            "n": int(len(global_)),
            "accepted": int(refine_flags.loc[global_.index].sum()) if len(global_) else 0,
        },
        "strategy_stats": strategy_stats,
        "diversity": diversity,
        "_accepted_note": ("accepted = validation loss strictly better than "
                           "all prior candidates in the seed"),
        "_local_note": "local = edit distance <= 0.25 (<= 4 of 16 slots changed)",
    }


def proposal_quality(frame: pd.DataFrame, pool: LLMPool, destination: Path) -> dict:
    out: dict[str, dict] = {}
    for method in METHODS:
        sub = frame[frame["method"] == method]
        entry = {
            "evaluations": int(len(sub)),
            "fallback_evaluations": int(sub["fallback"].sum()),
        }
        out[method] = entry
    out["LLM-Open"]["pool_rejected_entries"] = len(pool.rejected)
    out["LLM-Open"]["pool_fallbacks"] = len(pool.fallback_names)
    reasons: dict[str, int] = {}
    for seed in VERIFY_SEEDS:
        stored = json.loads((destination / f"llm_closed_seed{seed}.json").read_text())
        for f in stored.get("refine_failures", []):
            key = f["reason"].split(":")[0]
            reasons[key] = reasons.get(key, 0) + 1
    out["LLM-Closed"]["refine_failure_reasons"] = reasons
    return out


def main(
    output_dir: str = "outputs/qae_tfim_neutral_v5",
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
            print(f"seed {seed}: LLM-Closed (free-form redesign)", flush=True)
            train, val, test = state_sets(seed)
            rows.extend(run_closed_seed(seed, destination, train, val, test))

    frame = pd.DataFrame(rows)
    selected, stats = analyze(frame)
    frame.to_csv(destination / "candidate_results.csv", index=False)
    selected.to_csv(destination / "selected_results.csv", index=False)
    anytime_table(frame).to_csv(destination / "anytime_mean.csv", index=False)
    (destination / "paired_stats.json").write_text(json.dumps(stats, indent=2))
    (destination / "refinement_gains.json").write_text(
        json.dumps(refinement_gains(frame), indent=2)
    )
    if with_api and closed_loop:
        (destination / "redesign_diagnostics.json").write_text(
            json.dumps(redesign_diagnostics(frame, destination), indent=2)
        )
        (destination / "proposal_quality.json").write_text(
            json.dumps(proposal_quality(frame, pool, destination), indent=2)
        )
    print(selected.groupby("method")["test_fid"].agg(["mean", "median", "std"]))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
