"""Retrospective, validation-only budget audit; never invokes a model or trainer.

Each configured budget is a separate search policy. A prefix of a B=16 run
is NOT a separately executed B=8 run. Test columns are dropped at ingestion
and are never used for threshold selection, curves, or summary statistics.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import shutil
import statistics
from collections import defaultdict
from pathlib import Path

CONDITIONS = {"budget_b4": 4, "reference": 8, "budget_b16": 16,
              "qubits_n6": 8, "qubits_n8": 8, "hamiltonian_xxz": 8,
              "model_alt": 8}
METHODS = ("Random", "Greedy", "LLM-Open", "LLM-Closed")
TARGETS = (0.95, 0.99)
SEEDS = set(range(12))
REQUIRED = 10
SAFE_COLUMNS = ("seed", "method", "candidate", "order", "phase", "fallback",
                "invalid_errors", "edit_distance", "strategy", "val_loss", "val_fid")


def running_best(values: list[float]) -> list[float]:
    result = []
    best = -math.inf
    for value in values:
        if not math.isfinite(value) or not -1e-12 <= value <= 1 + 1e-12:
            raise ValueError(f"Invalid validation fidelity: {value}")
        best = max(best, value)
        result.append(best)
    return result


def first_hit(values: list[float], target: float) -> int | None:
    return next((i + 1 for i, value in enumerate(values) if value >= target), None)


def write_csv(path: Path, rows: list[dict], fields=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        if not rows:
            raise ValueError(f"Cannot infer columns for empty table: {path}")
        fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_validation(path: Path, provenance: list[dict]) -> list[dict]:
    raw = path.read_bytes()
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8")))
    mandatory = {"seed", "method", "order", "val_fid"}
    if not mandatory.issubset(reader.fieldnames or []):
        raise ValueError(f"Missing validation columns: {path}")
    # Explicit allowlist: no test data is retained, calculated, or exported.
    rows = [{key: row.get(key, "") for key in SAFE_COLUMNS} for row in reader]
    provenance.append({"source_path": str(path), "rows": len(rows),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "source_git_blob_sha": hashlib.sha1(
            f"blob {len(raw)}\0".encode() + raw).hexdigest(),
        "excluded_columns": [c for c in reader.fieldnames if c not in SAFE_COLUMNS]})
    return rows


def audit_condition(key: str, budget: int, candidates: list[dict], selected: list[dict],
                    methods: tuple[str, ...] = METHODS):
    """`methods` is explicit so a boundary probe that deliberately ran a
    single method is audited against what it actually executed, instead of
    being reported as incomplete. Default is the full four-method set, so
    every historical condition is audited exactly as before."""
    groups = defaultdict(list)
    chosen = {}
    for row in candidates:
        groups[(row["method"], int(row["seed"]))].append(row)
    for row in selected:
        identifier = (row["method"], int(row["seed"]))
        if identifier in chosen:
            raise ValueError(f"Duplicate selected row: {key}/{identifier}")
        chosen[identifier] = row
    expected = {(method, seed) for method in methods for seed in SEEDS}
    if set(groups) != expected or set(chosen) != expected:
        missing = sorted(expected - set(groups))
        extra = sorted(set(groups) - expected)
        raise ValueError(
            f"Incomplete or unexpected method/seed set: {key}; "
            f"missing={missing} unexpected={extra}")
    curves, endpoints, hits, prefix_summary, validity = [], [], [], [], []
    for method in methods:
        traces, valid_maxima, records = [], [], []
        for seed in sorted(SEEDS):
            rows = sorted(groups[(method, seed)], key=lambda row: int(row["order"]))
            if [int(row["order"]) for row in rows] != list(range(1, budget + 1)):
                raise ValueError(f"Missing, duplicate, or noncontiguous candidates: {key}/{method}/{seed}")
            trace = running_best([float(row["val_fid"]) for row in rows])
            endpoint = float(chosen[(method, seed)]["val_fid"])
            if not math.isfinite(endpoint) or abs(trace[-1] - endpoint) > 1e-10:
                raise ValueError(f"Selected value is not validation maximum: {key}/{method}/{seed}")
            traces.append(trace)
            valid = [float(row["val_fid"]) for row in rows if row["fallback"].lower() != "true"]
            valid_maxima.append(max(valid) if valid else -math.inf)
            records.extend(rows)
            for target in TARGETS:
                k = first_hit(trace, target)
                hits.append({"condition": key, "configured_budget": budget,
                    "method": method, "seed": seed, "target": target,
                    "first_hit_in_this_trace": k, "right_censored": k is None,
                    "censor_at": budget if k is None else ""})
        for k in range(1, budget + 1):
            vals = [trace[k - 1] for trace in traces]
            curves.append({"condition": key, "configured_budget": budget,
                "method": method, "candidates_used": k,
                "mean_best_validation": statistics.mean(vals),
                "median_best_validation": statistics.median(vals),
                "min_best_validation": min(vals), "max_best_validation": max(vals),
                "success_095": sum(value >= 0.95 for value in vals),
                "success_099": sum(value >= 0.99 for value in vals), "n_seeds": 12})
        vals = [trace[-1] for trace in traces]
        for target in TARGETS:
            counts = [sum(trace[k] >= target for trace in traces) for k in range(budget)]
            k10 = next((k + 1 for k, count in enumerate(counts) if count >= REQUIRED), None)
            endpoints.append({"condition": key, "configured_budget": budget,
                "method": method, "target": target, "successes": counts[-1],
                "required": REQUIRED, "n_seeds": 12, "passes": counts[-1] >= REQUIRED,
                "mean_validation": statistics.mean(vals),
                "tenth_largest_validation": sorted(vals, reverse=True)[9],
                "successes_without_fallback_candidates_diagnostic_only": sum(
                    value >= target for value in valid_maxima)})
            prefix_summary.append({"condition": key, "configured_budget": budget,
                "method": method, "target": target, "first_k_with_10_of_12_in_this_trace": k10,
                "interpretation": "within-run prefix only; not an independently executed budget"})
        validity.append({"condition": key, "method": method, "evaluated_candidates": len(records),
            "fallback_candidates": sum(row["fallback"].lower() == "true" for row in records),
            "candidates_with_recorded_invalid_errors": sum(bool(row["invalid_errors"]) for row in records),
            "note": "Invalid-error history and final random fallback are distinct; no candidate removed from primary analysis."})
    return curves, endpoints, hits, prefix_summary, validity


def self_test() -> None:
    assert running_best([0.8, 0.7, 0.96, 0.9]) == [0.8, 0.8, 0.96, 0.96]
    assert first_hit([0.949999999, 0.95], 0.95) == 2
    assert first_hit([0.98, 0.989], 0.99) is None
    for bad in (float("nan"), float("inf"), -0.01, 1.01):
        try:
            running_best([bad])
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid fidelity was accepted")
    candidates = []
    for method in METHODS:
        for seed in sorted(SEEDS):
            for order, val in enumerate((0.80, 0.96 if seed < 10 else 0.94), 1):
                row = {column: "" for column in SAFE_COLUMNS}
                row.update(seed=str(seed), method=method, order=str(order),
                           val_fid=str(val), fallback="True" if seed == 0 else "False")
                candidates.append(row)
    selected = [row.copy() for row in candidates if row["order"] == "2"]
    result = audit_condition("synthetic", 2, candidates, selected)
    assert all(row["successes"] == (10 if row["target"] == 0.95 else 0) for row in result[1])
    assert all(row["first_k_with_10_of_12_in_this_trace"] == (2 if row["target"] == 0.95 else None) for row in result[3])
    try:
        audit_condition("missing", 2, candidates[:-1], selected)
    except ValueError:
        pass
    else:
        raise AssertionError("Missing candidate was accepted")
    print("Self-tests passed: threshold equality, censoring, fallback inclusion, finite values, completeness.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/qae_robustness"))
    parser.add_argument("--out", type=Path, default=Path("outputs/qae_budget_targets_20260908"))
    parser.add_argument("--source-sha", default="e846c27fc21d06aa20e026590fcc06429f8bcfb4")
    parser.add_argument(
        "--extra-condition", action="append", default=[], metavar="KEY:B:METHODS:ROOT",
        help=("Audit an additional condition that lives outside --root, e.g. "
              "'target_xxz_b10:10:LLM-Closed:outputs/qae_budget_targets_v2_20260908'. "
              "METHODS is '|'-separated or 'all'. Historical conditions are "
              "unaffected, so the default invocation stays byte-identical."))
    args = parser.parse_args()
    self_test()
    out = args.out
    if out.resolve() == args.root.resolve() or args.root.resolve() in out.resolve().parents:
        raise SystemExit("Output must be separate from the historical source directory")
    out.mkdir(parents=True, exist_ok=True)
    plan = [(key, budget, METHODS, args.root) for key, budget in CONDITIONS.items()]
    for spec in args.extra_condition:
        key, budget, methods, root = spec.split(":", 3)
        plan.append((key, int(budget),
                     METHODS if methods == "all" else tuple(methods.split("|")),
                     Path(root)))
    provenance, all_curves, all_endpoints, all_hits, all_prefix, all_validity = [], [], [], [], [], []
    for key, budget, methods, root in plan:
        inputs = {}
        for name in ("candidate_results.csv", "selected_results.csv"):
            rows = read_validation(root / key / name, provenance)
            write_csv(out / "data" / key / name, rows, SAFE_COLUMNS)
            inputs[name] = rows
        results = audit_condition(key, budget, inputs["candidate_results.csv"],
                                  inputs["selected_results.csv"], methods)
        for destination, rows in zip((all_curves, all_endpoints, all_hits, all_prefix, all_validity), results):
            destination.extend(rows)
        for name in ("manifest.json", "api_usage.json", "proposal_quality.json",
                     "run_record.json"):
            source = root / key / name
            if source.exists():
                shutil.copyfile(source, out / "data" / key / name)
    for name, rows in (("validation_curves.csv", all_curves), ("endpoint_attainment.csv", all_endpoints),
                       ("per_seed_first_hits.csv", all_hits), ("within_run_prefix_summary.csv", all_prefix),
                       ("proposal_validity.csv", all_validity)):
        write_csv(out / name, rows)
    record = {"source_data_commit": args.source_sha, "execution_commit": os.getenv("GITHUB_SHA"),
        "analysis": "retrospective validation-only", "targets": TARGETS, "required_successes": REQUIRED,
        "seeds": sorted(SEEDS), "new_model_calls": 0, "new_training_runs": 0,
        "new_test_evaluations": 0, "source_files": provenance,
        "warning": "Do not infer monotonicity across independently executed, budget-dependent policies. Historical test columns were discarded and no test statistic was computed."}
    (out / "provenance.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    (out / "summary.json").write_text(json.dumps({"endpoints": all_endpoints, "prefixes": all_prefix,
        "validity": all_validity}, indent=2) + "\n", encoding="utf-8")
    shutil.copyfile(__file__, out / "analyze_budget_targets.py")
    print(f"Validated {sum(r['rows'] for r in provenance if 'candidate_results' in r['source_path'])} candidate rows and {len(all_endpoints)} endpoint/target cells.")
    for row in all_endpoints:
        print(f"{row['condition']:18s} B={row['configured_budget']:2} {row['method']:11s} F>={row['target']:.2f}: {row['successes']:2}/12")
    print(f"Saved validation-only audit to {out}")


if __name__ == "__main__":
    main()
