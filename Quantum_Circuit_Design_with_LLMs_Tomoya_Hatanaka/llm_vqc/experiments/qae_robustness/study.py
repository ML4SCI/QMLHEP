"""Run and analyse one condition of the QAE robustness study.

Analysis conventions are the reference study's, unchanged: selection by
validation loss, held-out test trash fidelity read once, all seed points
kept, bootstrap 95% CIs (10,000 resamples, seed 0), exact two-sided
Wilcoxon, two-sided sign test, Cohen's dz, win counts.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon

from llm_vqc.experiments.qae_robustness import arms
from llm_vqc.experiments.qae_robustness import manifest as M
from llm_vqc.experiments.qae_robustness.conditions import (
    CONDITIONS_BY_KEY,
    REFERENCE,
    VERIFY_SEEDS,
    Condition,
    mean_pairwise_distance,
    state_sets,
)

METHODS = ("Random", "Greedy", "LLM-Open", "LLM-Closed")
REPORTED_CONTRASTS = (
    ("LLM-Open", "LLM-Closed"),   # Closed - Open
    ("Random", "LLM-Open"),       # Open - Random
    ("Random", "LLM-Closed"),     # Closed - Random
    ("Random", "Greedy"),         # Greedy - Random
)
REFERENCE_OUTPUT = Path("outputs/qae_tfim_neutral_v5")
STUDY_ROOT = Path("outputs/qae_robustness")


def _boot_ci(values: np.ndarray, rng: np.random.Generator) -> list[float]:
    boots = np.array([
        rng.choice(values, size=len(values), replace=True).mean()
        for _ in range(10_000)
    ])
    return [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]


def paired_stats(pivot: pd.DataFrame) -> dict:
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
        sd = diff.std(ddof=1)
        stats[f"{b}_minus_{a}"] = {
            "mean_paired_gain": float(diff.mean()),
            "median_paired_gain": float(diff.median()),
            "bootstrap_95ci": _boot_ci(diff.values, rng),
            "cohens_dz": float(diff.mean() / sd) if sd > 0 else float("inf"),
            "wins_b": int((diff > 0).sum()),
            "n": int(len(diff)),
            "wilcoxon_exact_two_sided_p": float(exact.pvalue),
            "sign_test_two_sided_p": float(sign.pvalue),
        }
    return stats


def analyze(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    selected = (
        frame.sort_values("val_loss")
        .groupby(["seed", "method"], as_index=False)
        .first()
    )
    pivot = selected.pivot(index="seed", columns="method", values="test_fid")
    return selected, paired_stats(pivot)


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


def refinement_gains(frame: pd.DataFrame, n_warm: int) -> dict:
    out: dict[str, dict] = {}
    rng = np.random.default_rng(1)
    for method in ("Greedy", "LLM-Closed"):
        sub = frame[frame["method"] == method]
        if sub.empty:
            continue
        per_seed_val, per_seed_test = [], []
        for _seed, group in sub.groupby("seed"):
            group = group.sort_values("order")
            warm = group[group["order"] <= n_warm]
            warm_best = warm.loc[warm["val_loss"].idxmin()]
            final_best = group.loc[group["val_loss"].idxmin()]
            per_seed_val.append(float(final_best["val_fid"] - warm_best["val_fid"]))
            per_seed_test.append(float(final_best["test_fid"] - warm_best["test_fid"]))
        test_arr = np.array(per_seed_test)
        nonzero = test_arr[test_arr != 0]
        out[method] = {
            "mean_val_gain": float(np.mean(per_seed_val)),
            "mean_test_gain": float(np.mean(test_arr)),
            "median_test_gain": float(np.median(test_arr)),
            "test_gain_bootstrap_95ci": _boot_ci(test_arr, rng),
            "seeds_improved": int((test_arr > 0).sum()),
            "seeds_unchanged": int((test_arr == 0).sum()),
            "n": int(len(test_arr)),
            "wilcoxon_p_nonzero_diffs": (
                float(wilcoxon(nonzero, alternative="two-sided").pvalue)
                if len(nonzero) >= 6 else None
            ),
        }
    return out


def proposal_quality(frame: pd.DataFrame, destination: Path,
                     pool_path: Path | None) -> dict:
    out: dict[str, dict] = {}
    for method in METHODS:
        sub = frame[frame["method"] == method]
        if sub.empty:
            continue
        out[method] = {
            "evaluations": int(len(sub)),
            "fallback_evaluations": int(sub["fallback"].sum()),
        }
    if pool_path is not None and pool_path.exists():
        stored = json.loads(pool_path.read_text())
        out.setdefault("LLM-Open", {})
        out["LLM-Open"]["pool_rejected_entries"] = len(stored.get("rejected", []))
        out["LLM-Open"]["pool_fallbacks"] = len(stored.get("fallback_names", []))
    reasons: dict[str, int] = {}
    warm_rejected = 0
    for seed in VERIFY_SEEDS:
        store = destination / f"llm_closed_seed{seed}.json"
        if not store.exists():
            continue
        stored = json.loads(store.read_text())
        warm_rejected += len(stored.get("warm_rejected", []))
        for f in stored.get("refine_failures", []):
            key = f["reason"].split(":")[0]
            reasons[key] = reasons.get(key, 0) + 1
    if "LLM-Closed" in out:
        out["LLM-Closed"]["refine_failure_reasons"] = reasons
        out["LLM-Closed"]["warm_rejected_entries"] = warm_rejected
    return out


def redesign_diagnostics(frame: pd.DataFrame, destination: Path,
                         n_warm: int, n_gates: int) -> dict:
    refine = frame[(frame["method"] == "LLM-Closed")
                   & (frame["phase"] == "refine")].copy()
    if refine.empty:
        return {}
    real = refine[~refine["fallback"]]

    flags = []
    for _seed, group in frame[frame["method"] == "LLM-Closed"].groupby("seed"):
        group = group.sort_values("order")
        best = float("inf")
        values = []
        for _, row in group.iterrows():
            values.append(row["val_loss"] < best if row["phase"] == "refine" else False)
            best = min(best, row["val_loss"])
        flags.append(pd.Series(values, index=group.index))
    flags = pd.concat(flags)
    refine_flags = flags.loc[real.index]

    threshold = 4 / n_gates
    local = real[real["edit_distance"] <= threshold]
    global_ = real[real["edit_distance"] > threshold]
    strategy_stats = {}
    for strategy, group in real.groupby("strategy"):
        if not strategy:
            continue
        strategy_stats[strategy] = {
            "proposals": int(len(group)),
            "accepted": int(refine_flags.loc[group.index].sum()),
            "mean_edit_distance": float(group["edit_distance"].mean()),
        }
    warm_d, refine_d = [], []
    for seed in VERIFY_SEEDS:
        store = destination / f"llm_closed_seed{seed}.json"
        if not store.exists():
            continue
        archs = json.loads(store.read_text()).get("architectures", [])
        w = mean_pairwise_distance(archs[:n_warm])
        r = mean_pairwise_distance(archs[n_warm:])
        if w is not None:
            warm_d.append(w)
        if r is not None:
            refine_d.append(r)
    return {
        "n_refine_proposals": int(len(refine)),
        "n_llm_proposals": int(len(real)),
        "n_fallbacks": int(refine["fallback"].sum()),
        "edit_distance_mean": float(real["edit_distance"].mean()) if len(real) else None,
        "edit_distance_median": float(real["edit_distance"].median()) if len(real) else None,
        "changed_slots_mean": (float((real["edit_distance"] * n_gates).mean())
                               if len(real) else None),
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
        "diversity": {
            "warm_mean_pairwise_distance": float(np.mean(warm_d)) if warm_d else None,
            "refine_mean_pairwise_distance": float(np.mean(refine_d)) if refine_d else None,
        },
    }


def api_usage(destination: Path) -> dict:
    calls, input_tokens, output_tokens = 0, 0, 0
    models: set[str] = set()
    calls_dir = destination / "llm_calls"
    if calls_dir.exists():
        for path in sorted(calls_dir.glob("*.json")):
            record = json.loads(path.read_text())
            calls += 1
            input_tokens += int(record.get("input_tokens") or 0)
            output_tokens += int(record.get("output_tokens") or 0)
            if record.get("model"):
                models.add(record["model"])
    return {"n_calls": calls, "input_tokens": input_tokens,
            "output_tokens": output_tokens, "model_snapshots": sorted(models)}


# ------------------------------------------------------------- driver ------

def load_reference_frame() -> pd.DataFrame:
    """The previous study's committed candidate table (the reference cell).

    Reused rather than re-run: `tests/test_qae_robustness_reference.py`
    proves the code in this package reproduces those rows bit-for-bit.
    """
    return pd.read_csv(REFERENCE_OUTPUT / "candidate_results.csv")


def run_condition(
    condition: Condition,
    *,
    with_api: bool = True,
    closed_loop: bool = True,
    seeds: tuple[int, ...] = VERIFY_SEEDS,
    root: Path = STUDY_ROOT,
) -> Path:
    destination = root / condition.key
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "manifest.json").write_text(
        json.dumps({"manifest": M.condition_manifest(condition),
                    "violations": M.verify(condition)}, indent=2)
    )

    if condition.key == REFERENCE.key:
        frame = load_reference_frame()
        pool_path = REFERENCE_OUTPUT / "llm_open_pool.json"
        closed_dir = REFERENCE_OUTPUT
    else:
        pool = arms.generate_open_pool(condition, destination) if with_api else None
        rows: list[dict] = []
        for seed in seeds:
            print(f"[{condition.key}] seed {seed}: Random / Greedy", flush=True)
            train, val, test = state_sets(seed, condition.family, condition.n_qubits)
            rows.extend(arms.run_random_seed(condition, seed, train, val, test))
            rows.extend(arms.run_greedy_seed(condition, seed, train, val, test))
            if pool is not None:
                rows.extend(arms.run_open_seed(condition, seed, pool, train, val, test))
        if with_api and closed_loop:
            for seed in seeds:
                print(f"[{condition.key}] seed {seed}: LLM-Closed", flush=True)
                train, val, test = state_sets(seed, condition.family, condition.n_qubits)
                rows.extend(arms.run_closed_seed(condition, seed, destination,
                                                 train, val, test))
        frame = pd.DataFrame(rows)
        pool_path = destination / "llm_open_pool.json"
        closed_dir = destination

    selected, stats = analyze(frame)
    frame.to_csv(destination / "candidate_results.csv", index=False)
    selected.to_csv(destination / "selected_results.csv", index=False)
    anytime_table(frame).to_csv(destination / "anytime_mean.csv", index=False)
    (destination / "paired_stats.json").write_text(json.dumps(stats, indent=2))
    (destination / "refinement_gains.json").write_text(
        json.dumps(refinement_gains(frame, condition.n_warm), indent=2)
    )
    (destination / "proposal_quality.json").write_text(
        json.dumps(proposal_quality(frame, closed_dir, pool_path), indent=2)
    )
    diagnostics = redesign_diagnostics(
        frame, closed_dir, condition.n_warm, condition.n_qubits * 4
    )
    (destination / "redesign_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2)
    )
    (destination / "api_usage.json").write_text(
        json.dumps(api_usage(closed_dir), indent=2)
    )
    summary = selected.groupby("method")["test_fid"].agg(["mean", "median", "std", "count"])
    print(f"\n=== {condition.key} ({condition.label}) ===")
    print(summary)
    print(json.dumps({k: {kk: round(vv, 5) if isinstance(vv, float) else vv
                          for kk, vv in v.items() if kk in
                          ("mean_paired_gain", "wins_b", "n",
                           "wilcoxon_exact_two_sided_p", "cohens_dz")}
                      for k, v in stats.items()}, indent=2))
    return destination


def main(keys: list[str], *, with_api: bool = True, root: Path = STUDY_ROOT) -> None:
    for key in keys:
        run_condition(CONDITIONS_BY_KEY[key], with_api=with_api, root=root)


# ---------------------------------------------------------- aggregation -----

def condition_summary(condition: Condition, root: Path = STUDY_ROOT) -> dict:
    """Everything the cross-condition table and the deck need for one cell."""
    destination = root / condition.key
    selected_path = destination / "selected_results.csv"
    if not selected_path.exists():
        return {"key": condition.key, "status": "missing"}
    selected = pd.read_csv(selected_path)
    stats = json.loads((destination / "paired_stats.json").read_text())
    quality = json.loads((destination / "proposal_quality.json").read_text())
    usage = json.loads((destination / "api_usage.json").read_text())

    per_method = {}
    complete = True
    for method in METHODS:
        values = selected[selected["method"] == method]["test_fid"].to_numpy()
        if len(values) != len(VERIFY_SEEDS):
            complete = False
        rng = np.random.default_rng(0)
        per_method[method] = {
            "mean": float(values.mean()) if len(values) else None,
            "median": float(np.median(values)) if len(values) else None,
            "sd": float(values.std(ddof=1)) if len(values) > 1 else None,
            "n_seeds": int(len(values)),
            "bootstrap_95ci": _boot_ci(values, rng) if len(values) else None,
        }
    contrasts = {
        f"{b}_minus_{a}": stats.get(f"{b}_minus_{a}")
        for a, b in REPORTED_CONTRASTS
    }
    llm_evaluations = sum(
        quality.get(m, {}).get("evaluations", 0) for m in ("LLM-Open", "LLM-Closed")
    )
    llm_fallbacks = sum(
        quality.get(m, {}).get("fallback_evaluations", 0)
        for m in ("LLM-Open", "LLM-Closed")
    )
    return {
        "key": condition.key,
        "factor": condition.factor,
        "label": condition.label,
        "factors": condition.factors,
        "changed_factors": M.C.changed_factors(condition),
        "status": "complete" if complete else "preliminary",
        "n_seeds": int(selected["seed"].nunique()),
        "per_method": per_method,
        "contrasts": contrasts,
        "llm_proposal_validity": {
            "evaluations": llm_evaluations,
            "flagged_random_fallbacks": llm_fallbacks,
            "valid_fraction": (round(1 - llm_fallbacks / llm_evaluations, 4)
                               if llm_evaluations else None),
        },
        "api_usage": usage,
        "manifest_violations": M.verify(condition),
    }


def summarize_all(root: Path = STUDY_ROOT) -> dict:
    from llm_vqc.experiments.qae_robustness.conditions import CONDITIONS

    summaries = {c.key: condition_summary(c, root) for c in CONDITIONS}
    reference = summaries[REFERENCE.key]
    for key, entry in summaries.items():
        if key == REFERENCE.key or entry.get("status") == "missing":
            continue
        entry["vs_reference"] = {
            name: {
                "reference_mean_paired_gain": (
                    reference["contrasts"].get(name) or {}).get("mean_paired_gain"),
                "condition_mean_paired_gain": (
                    entry["contrasts"].get(name) or {}).get("mean_paired_gain"),
                "sign_preserved": _sign_preserved(
                    (reference["contrasts"].get(name) or {}).get("mean_paired_gain"),
                    (entry["contrasts"].get(name) or {}).get("mean_paired_gain"),
                ),
                "significant_at_0_05": (
                    ((entry["contrasts"].get(name) or {})
                     .get("wilcoxon_exact_two_sided_p", 1.0) or 1.0) < 0.05),
            }
            for name in (f"{b}_minus_{a}" for a, b in REPORTED_CONTRASTS)
        }
    total = {"n_calls": 0, "input_tokens": 0, "output_tokens": 0, "models": set()}
    for entry in summaries.values():
        usage = entry.get("api_usage") or {}
        total["n_calls"] += usage.get("n_calls", 0)
        total["input_tokens"] += usage.get("input_tokens", 0)
        total["output_tokens"] += usage.get("output_tokens", 0)
        total["models"].update(usage.get("model_snapshots", []))
    total["models"] = sorted(total["models"])
    return {"reference_key": REFERENCE.key, "conditions": summaries,
            "api_usage_total": total}


def _sign_preserved(reference_value, condition_value) -> bool | None:
    if reference_value is None or condition_value is None:
        return None
    return bool(np.sign(reference_value) == np.sign(condition_value))


def summary_table(summary: dict) -> pd.DataFrame:
    rows = []
    for key, entry in summary["conditions"].items():
        if entry.get("status") == "missing":
            continue
        row = {"condition": key, "label": entry["label"],
               "factor": entry["factor"], "status": entry["status"],
               "n_qubits": entry["factors"]["n_qubits"],
               "family": entry["factors"]["family"],
               "budget": entry["factors"]["budget"],
               "model": entry["factors"]["model"],
               "seeds": entry["n_seeds"]}
        for method in METHODS:
            row[f"{method}_mean"] = entry["per_method"][method]["mean"]
        for a, b in REPORTED_CONTRASTS:
            name = f"{b}_minus_{a}"
            contrast = entry["contrasts"].get(name) or {}
            row[f"{name}_gain"] = contrast.get("mean_paired_gain")
            row[f"{name}_ci_lo"] = (contrast.get("bootstrap_95ci") or [None, None])[0]
            row[f"{name}_ci_hi"] = (contrast.get("bootstrap_95ci") or [None, None])[1]
            row[f"{name}_p"] = contrast.get("wilcoxon_exact_two_sided_p")
            row[f"{name}_wins"] = contrast.get("wins_b")
        row["llm_valid_fraction"] = entry["llm_proposal_validity"]["valid_fraction"]
        row["api_calls"] = (entry.get("api_usage") or {}).get("n_calls")
        rows.append(row)
    return pd.DataFrame(rows)
