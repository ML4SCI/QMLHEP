"""Validation-only runner for the minimum-budget-to-target study.

Two invariants this module exists to enforce, both from
`docs/research/QAE_BUDGET_TARGET_PROTOCOL.md`:

* **No candidate is ever evaluated on the test set.** The frozen trainer is
  handed an *empty* test array, so no test state is contracted with a
  trained circuit; the resulting non-finite test columns are asserted to be
  non-finite and then dropped, and the written tables carry a
  validation-only column allowlist.
* **Each configured `B` is a separate policy.** Nothing here concatenates,
  truncates or re-labels a run at another budget.

Everything else -- methods, prompts, schema, repair policy, trainer, data
streams, RNG offsets, selection rule -- is imported unchanged from
`qae_robustness`.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

from llm_vqc.experiments.qae_budget_targets.conditions import TargetCell
from llm_vqc.experiments.qae_robustness import arms
from llm_vqc.experiments.qae_robustness import manifest as M
from llm_vqc.experiments.qae_robustness.conditions import (
    VERIFY_SEEDS,
    Condition,
    canonical_json,
    state_sets,
)

STUDY_ROOT = Path("outputs/qae_budget_targets_v2_20260908")

# The exact column allowlist written to disk. No test column exists here,
# and `write_tables` refuses to write any column outside this list.
SAFE_COLUMNS = ("seed", "method", "candidate", "order", "phase", "fallback",
                "invalid_errors", "edit_distance", "strategy",
                "val_loss", "val_fid", "train_loss", "train_fid")
FORBIDDEN_SUBSTRING = "test"

# Official list prices, read from https://developers.openai.com/api/docs/pricing
# on 2026-09-08 (standard processing tier). Used only for the reported cost
# reconstruction; the hard cap is enforced separately by `LLMApiBudget`
# against its own conservative pre-call estimates.
PRICE_USD_PER_1M = {
    "gpt-5.4-mini-2026-03-17": {"input": 0.75, "output": 4.50},
    "gpt-4.1-mini-2025-04-14": {"input": 0.40, "output": 1.60},
}
PRICE_SOURCE = ("https://developers.openai.com/api/docs/pricing "
                "(standard tier, read 2026-09-08)")


def validation_only_state_sets(seed: int, family: str, n: int):
    """Train/validation states from the frozen data function, plus an EMPTY
    test array.

    The frozen `state_sets` is called unchanged so the Hamiltonian-parameter
    streams stay paired with every previous study. Its test states are
    discarded immediately and are never handed to a candidate evaluation;
    the empty array below is what the trainer actually sees, which makes the
    "no test evaluation" guarantee structural rather than a convention.
    """
    train, val, _discarded_test = state_sets(seed, family, n)
    del _discarded_test
    return train, val, np.zeros((0, 2**n), dtype=complex)


def assert_no_test_values(rows: list[dict]) -> None:
    """Proof obligation: every recorded test quantity is non-finite."""
    for row in rows:
        for key, value in row.items():
            if FORBIDDEN_SUBSTRING in key and isinstance(value, float):
                if math.isfinite(value):
                    raise AssertionError(
                        f"runner produced a finite test value {key}={value}; "
                        "the test set must never be evaluated here"
                    )


def project(rows: list[dict]) -> list[dict]:
    """Drop every column outside the validation-only allowlist."""
    return [{k: row.get(k, "") for k in SAFE_COLUMNS} for row in rows]


def select(rows: list[dict]) -> list[dict]:
    """One row per (seed, method): the lowest validation loss, ties broken by
    the earlier candidate order -- the frozen selection rule, on validation."""
    best: dict[tuple[int, str], dict] = {}
    for row in rows:
        key = (int(row["seed"]), row["method"])
        current = best.get(key)
        if current is None or (float(row["val_loss"]), int(row["order"])) < (
            float(current["val_loss"]), int(current["order"])
        ):
            best[key] = row
    return [best[key] for key in sorted(best, key=lambda k: (k[1], k[0]))]


def write_csv(path: Path, rows: list[dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SAFE_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cost_record(destination: Path, condition: Condition) -> dict:
    """Reconstruct calls, repairs, tokens and USD from the stored call logs.

    Kept separate from the enforced cap: the cap is checked before each
    request against a conservative estimate, while this is the after-the-fact
    accounting at official list prices.
    """
    calls_dir = destination / "llm_calls"
    calls = repairs = input_tokens = output_tokens = 0
    models: set[str] = set()
    if calls_dir.exists():
        for path in sorted(calls_dir.glob("*.json")):
            record = json.loads(path.read_text())
            calls += 1
            # Two distinct kinds of extra call, both counted here: a bounded
            # CAPACITY repair (its proposal id carries a `_repairN` suffix)
            # and a JSON re-ask of the same prompt (call_index > 0).
            if ("repair" in str(record.get("proposal_id") or "")
                    or int(record.get("call_index") or 0) > 0):
                repairs += 1
            input_tokens += int(record.get("input_tokens") or 0)
            output_tokens += int(record.get("output_tokens") or 0)
            if record.get("model"):
                models.add(record["model"])
    price = PRICE_USD_PER_1M.get(condition.model)
    usd = None
    if price is not None:
        usd = round(input_tokens / 1e6 * price["input"]
                    + output_tokens / 1e6 * price["output"], 6)
    return {
        "n_calls": calls,
        "n_repair_or_retry_calls": repairs,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "model_snapshots": sorted(models),
        "declared_model": condition.model,
        "list_price_usd_per_1m": price,
        "price_source": PRICE_SOURCE,
        "estimated_cost_usd_at_list_price": usd,
    }


def validity_record(rows: list[dict], destination: Path,
                    cell: TargetCell) -> dict:
    """invalid / repaired / fallback, kept distinct.

    Random fallbacks consume candidate budget and stay in the primary
    analysis; the fallback-excluding count below is an auxiliary,
    non-causal diagnostic only.
    """
    out: dict[str, dict] = {}
    for method in cell.methods:
        subset = [r for r in rows if r["method"] == method]
        if not subset:
            continue
        out[method] = {
            "evaluated_candidates": len(subset),
            "flagged_random_fallbacks": sum(bool(r["fallback"]) for r in subset),
            "candidates_with_recorded_invalid_errors":
                sum(bool(r["invalid_errors"]) for r in subset),
        }
    pool_path = destination / "llm_open_pool.json"
    if cell.runs_open and pool_path.exists():
        pool = json.loads(pool_path.read_text())
        out.setdefault("LLM-Open", {})
        out["LLM-Open"]["pool_rejected_entries"] = len(pool.get("rejected", []))
        out["LLM-Open"]["pool_fallback_entries"] = len(pool.get("fallback_names", []))
    if cell.runs_closed:
        reasons: dict[str, int] = {}
        warm_rejected = 0
        for seed in VERIFY_SEEDS:
            store = destination / f"llm_closed_seed{seed}.json"
            if not store.exists():
                continue
            stored = json.loads(store.read_text())
            warm_rejected += len(stored.get("warm_rejected", []))
            for failure in stored.get("refine_failures", []):
                reason = failure["reason"].split(":")[0]
                reasons[reason] = reasons.get(reason, 0) + 1
        out.setdefault("LLM-Closed", {})
        out["LLM-Closed"]["redesign_failure_reasons"] = reasons
        out["LLM-Closed"]["warm_rejected_entries"] = warm_rejected
    out["note"] = (
        "Random fallbacks consume candidate budget and are INCLUDED in the "
        "primary analysis. Recorded invalid-error history and a final "
        "fallback flag are distinct quantities."
    )
    return out


def completed_seeds(cell: TargetCell, root: Path = STUDY_ROOT) -> list[int]:
    """Seeds whose paid LLM-Closed work is already on disk (resume state)."""
    destination = root / cell.key
    return [s for s in VERIFY_SEEDS
            if (destination / f"llm_closed_seed{s}.json").exists()]


def missing_artifacts(cell: TargetCell, seeds: tuple[int, ...],
                      root: Path) -> list[str]:
    """Paid artefacts this cell would have to buy if run right now."""
    destination = root / cell.key
    missing = []
    if cell.runs_open and not (destination / "llm_open_pool.json").exists():
        missing.append("llm_open_pool.json")
    if cell.runs_closed:
        missing += [f"llm_closed_seed{s}.json" for s in seeds
                    if not (destination / f"llm_closed_seed{s}.json").exists()]
    return missing


class _NoApi:
    """Stand-in provider that makes a paid call structurally impossible."""

    def __init__(self, *args, **kwargs) -> None:
        raise RuntimeError(
            "reuse-only mode attempted an API call; this is a bug, because "
            "every paid artefact was checked to be on disk beforehand")


def run_cell(
    cell: TargetCell,
    *,
    with_api: bool = True,
    reuse_only: bool = False,
    seeds: tuple[int, ...] = VERIFY_SEEDS,
    root: Path = STUDY_ROOT,
) -> Path:
    """Run one budget-target cell and write its validation-only artefacts.

    Resumable: the LLM-Open pool and each LLM-Closed seed store are reloaded
    from disk when present, keyed by (condition, seed), so a resumed run
    never pays twice for a result whose configured budget, model, prompt,
    seed, initialisation and data are all identical.
    """
    condition = cell.condition
    destination = root / cell.key
    destination.mkdir(parents=True, exist_ok=True)
    started = time.time()

    if reuse_only:
        # Regenerate the tables and the record from artefacts already paid
        # for, without the possibility of buying anything new.
        missing = missing_artifacts(cell, seeds, root)
        if missing:
            raise RuntimeError(
                f"{cell.key}: reuse-only mode needs these paid artefacts on "
                f"disk, and they are absent: {missing}")
        arms.build_provider = _NoApi

    violations = M.verify(condition, cell.anchor)
    (destination / "manifest.json").write_text(json.dumps({
        "cell": cell.describe(),
        "manifest": M.condition_manifest(condition, cell.anchor),
        "violations": violations,
        "test_policy": "no candidate is evaluated on the test set in this study",
        "column_allowlist": list(SAFE_COLUMNS),
    }, indent=2) + "\n")
    if violations:
        raise RuntimeError(f"{cell.key} manifest violations: {violations}")

    pool = None
    if cell.runs_open:
        if not with_api:
            raise RuntimeError("LLM-Open needs the API; rerun without --no-api")
        pool = arms.generate_open_pool(condition, destination)

    rows: list[dict] = []
    for seed in seeds:
        train, val, empty_test = validation_only_state_sets(
            seed, condition.family, condition.n_qubits
        )
        if "Random" in cell.methods:
            rows.extend(arms.run_random_seed(condition, seed, train, val, empty_test))
        if "Greedy" in cell.methods:
            rows.extend(arms.run_greedy_seed(condition, seed, train, val, empty_test))
        if pool is not None:
            rows.extend(arms.run_open_seed(condition, seed, pool, train, val, empty_test))
        print(f"[{cell.key}] seed {seed}: non-API arms done", flush=True)

    if cell.runs_closed:
        if not with_api:
            raise RuntimeError("LLM-Closed needs the API; rerun without --no-api")
        for seed in seeds:
            train, val, empty_test = validation_only_state_sets(
                seed, condition.family, condition.n_qubits
            )
            rows.extend(arms.run_closed_seed(
                condition, seed, destination, train, val, empty_test
            ))
            print(f"[{cell.key}] seed {seed}: LLM-Closed done", flush=True)

    assert_no_test_values(rows)
    candidates = project(rows)
    selected = select(candidates)
    hashes = {
        "candidate_results.csv": write_csv(
            destination / "candidate_results.csv", candidates),
        "selected_results.csv": write_csv(
            destination / "selected_results.csv", selected),
    }
    costs = cost_record(destination, condition)
    (destination / "run_record.json").write_text(json.dumps({
        "cell": cell.describe(),
        "seeds": list(seeds),
        "evaluated_candidates_total": len(candidates),
        "evaluated_candidates_per_seed": condition.budget,
        "wall_clock_seconds_this_invocation": round(time.time() - started, 1),
        "wall_clock_note": (
            "Time for THIS invocation only. A cell that was interrupted and "
            "resumed, or regenerated in reuse-only mode, reloads its stored "
            "seeds from disk, so this figure is an operational number and not "
            "a benchmark of a full cold run."),
        "reuse_only_regeneration": reuse_only,
        "api_usage": costs,
        "generation_validity": validity_record(candidates, destination, cell),
        "output_sha256": hashes,
        "test_evaluations": 0,
        "manifest_anchor": cell.anchor.key,
        "factors_sha": hashlib.sha256(
            canonical_json(condition.factors).encode()).hexdigest()[:16],
    }, indent=2) + "\n")
    print(f"[{cell.key}] wrote {len(candidates)} validation-only rows "
          f"({costs['n_calls']} API calls, "
          f"${costs['estimated_cost_usd_at_list_price']} at list price)")
    return destination
