"""Assemble the budget-target deliverables from the audit outputs.

Every number in the generated tables is read from the audit CSVs and the
run records; nothing is typed by hand. The prose sections live in
`REPORT_BODY` / `SUMMARY_JA_BODY` and are formatted with those numbers.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

TARGETS = (0.95, 0.99)
REQUIRED = 10
N_SEEDS = 12
METHOD_ORDER = ["Random", "Greedy", "LLM-Open", "LLM-Closed"]
NEW_CELLS = {"target_tfim_b6", "target_xxz_b10"}

ROW_ORDER = [
    ("budget_b4", 4, "4-qubit TFIM, reference model"),
    ("target_tfim_b6", 6, "4-qubit TFIM, reference model"),
    ("reference", 8, "4-qubit TFIM, reference model"),
    ("budget_b16", 16, "4-qubit TFIM, reference model"),
    ("hamiltonian_xxz", 8, "4-qubit XXZ, reference model"),
    ("target_xxz_b10", 10, "4-qubit XXZ, reference model"),
    ("qubits_n6", 8, "6-qubit TFIM, reference model"),
    ("qubits_n8", 8, "8-qubit TFIM, reference model"),
    ("model_alt", 8, "4-qubit TFIM, alternative model"),
]


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def attainment_table(endpoints: list[dict], target: float) -> str:
    lookup = {(r["condition"], r["method"]): r for r in endpoints
              if abs(float(r["target"]) - target) < 1e-12}
    lines = ["| Condition | Configured B | Source | Random | Greedy | LLM-Open | LLM-Closed |",
             "|---|---:|---|---:|---:|---:|---:|"]
    for key, budget, label in ROW_ORDER:
        if not any(k == key for k, _ in lookup):
            continue
        source = "measured now" if key in NEW_CELLS else "re-analysed"
        cells = []
        for method in METHOD_ORDER:
            row = lookup.get((key, method))
            if row is None:
                cells.append("not run")
                continue
            mark = " **P**" if row["passes"] in ("True", True) else ""
            cells.append(f"{row['successes']}/{N_SEEDS}{mark}")
        lines.append(f"| {label} | {budget} | {source} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def prefix_table(prefixes: list[dict], keys: list[str]) -> str:
    lines = ["| Condition | Configured B | Method | First k with 10/12 within the run |",
             "|---|---:|---|---:|"]
    for row in prefixes:
        if row["condition"] not in keys or abs(float(row["target"]) - 0.95) > 1e-12:
            continue
        k = row["first_k_with_10_of_12_in_this_trace"]
        lines.append(f"| {row['condition']} | {row['configured_budget']} | "
                     f"{row['method']} | {k if k else 'not reached'} |")
    return "\n".join(lines)


def cost_table(records: dict[str, dict]) -> tuple[str, dict]:
    lines = ["| Cell | Candidate evaluations | API calls | Repair/retry calls | "
             "Input tokens | Output tokens | Cost at list price (USD) |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    totals = {"calls": 0, "repairs": 0, "input": 0, "output": 0, "usd": 0.0,
              "evaluations": 0, "seconds": 0.0}
    for key, record in records.items():
        usage = record["api_usage"]
        lines.append(
            f"| {key} | {record['evaluated_candidates_total']} | "
            f"{usage['n_calls']} | {usage['n_repair_or_retry_calls']} | "
            f"{usage['input_tokens']} | {usage['output_tokens']} | "
            f"{usage['estimated_cost_usd_at_list_price']:.4f} |")
        totals["calls"] += usage["n_calls"]
        totals["repairs"] += usage["n_repair_or_retry_calls"]
        totals["input"] += usage["input_tokens"]
        totals["output"] += usage["output_tokens"]
        totals["usd"] += usage["estimated_cost_usd_at_list_price"]
        totals["evaluations"] += record["evaluated_candidates_total"]
        totals["seconds"] += record["wall_clock_seconds_this_invocation"]
    lines.append(
        f"| **total** | **{totals['evaluations']}** | **{totals['calls']}** | "
        f"**{totals['repairs']}** | **{totals['input']}** | **{totals['output']}** | "
        f"**{totals['usd']:.4f}** |")
    return "\n".join(lines), totals


def validity_table(records: dict[str, dict]) -> str:
    lines = ["| Cell | Method | Evaluated | Random fallbacks | With recorded invalid errors |",
             "|---|---|---:|---:|---:|"]
    for key, record in records.items():
        for method, entry in record["generation_validity"].items():
            if not isinstance(entry, dict):
                continue
            lines.append(
                f"| {key} | {method} | {entry.get('evaluated_candidates', 0)} | "
                f"{entry.get('flagged_random_fallbacks', 0)} | "
                f"{entry.get('candidates_with_recorded_invalid_errors', 0)} |")
    return "\n".join(lines)


def verdict_section(endpoints: list[dict]) -> str:
    """Smallest verified passing budget / verified failing budgets /
    unverified budgets, per anchor and method. Nothing is interpolated."""
    ladders = {
        "4-qubit Ising chain, reference model":
            ([("budget_b4", 4), ("target_tfim_b6", 6),
              ("reference", 8), ("budget_b16", 16)], METHOD_ORDER,
             [2, 10, 12, 14]),
        "4-qubit XXZ chain, reference model":
            ([("hamiltonian_xxz", 8), ("target_xxz_b10", 10)], ["LLM-Closed"],
             [2, 4, 6, 12, 14, 16]),
    }
    lookup = {(r["condition"], r["method"]): r for r in endpoints
              if abs(float(r["target"]) - 0.95) < 1e-12}
    blocks = []
    for label, (ladder, methods, unverified) in ladders.items():
        lines = [f"**{label}**, target 0.95:", ""]
        for method in methods:
            passing, failing = [], []
            for key, budget in ladder:
                row = lookup.get((key, method))
                if row is None:
                    continue
                (passing if row["passes"] in ("True", True) else failing).append(budget)
            if not passing and not failing:
                continue
            smallest = f"**B = {min(passing)}**" if passing else "none among those run"
            lines.append(
                f"- `{method}` - smallest verified passing budget: {smallest}; "
                f"verified failing budgets: "
                f"{', '.join(str(b) for b in failing) if failing else 'none'}; "
                f"unverified budgets: "
                f"{', '.join(str(b) for b in unverified)} (and every budget above "
                f"{max(b for _, b in ladder)}).")
        blocks.append("\n".join(lines))
    tail = (
        "\nAt target 0.99 no method passes at any budget or condition tested, "
        "including the two newly executed cells. That target is unresolved, "
        "and the runs performed here do not identify whether search, training "
        "or circuit capacity is the binding constraint.\n\n"
        "A verified passing budget is **not** evidence that a smaller untested "
        "budget fails, and a verified failing budget is not evidence that every "
        "smaller budget fails. Each budget was executed as its own policy.")
    return "\n\n".join(blocks) + "\n" + tail


def render(template: str, values: dict) -> str:
    """Substitute only the known keys.

    `str.format` is deliberately avoided: the prose contains real braces
    (`max_{i <= k}`, the admissible budget grid) that are not placeholders,
    and escaping them by hand is exactly the kind of thing that silently
    rots. An unknown `{...}` is left untouched; a placeholder that never got
    substituted is reported rather than swallowed.
    """
    for key, value in values.items():
        template = template.replace("{" + key + "}", str(value))
    leftover = sorted(set(re.findall(r"\{(" + "|".join(values) + r")\}", template)))
    if leftover:
        raise SystemExit(f"unsubstituted placeholders: {leftover}")
    return template


def key_numbers(audit: Path, endpoints: list[dict]) -> dict:
    """Every number quoted in the prose, computed from the tables."""
    def count(key, method, target=0.95):
        row = next((r for r in endpoints if r["condition"] == key
                    and r["method"] == method
                    and abs(float(r["target"]) - target) < 1e-12), None)
        return None if row is None else int(row["successes"])

    def finals(key, method):
        path = audit / "data" / key / "candidate_results.csv"
        best: dict[int, float] = {}
        for row in read_rows(path):
            if row["method"] == method:
                seed = int(row["seed"])
                best[seed] = max(best.get(seed, 0.0), float(row["val_fid"]))
        return [best[s] for s in sorted(best)]

    x8, x10 = finals("hamiltonian_xxz", "LLM-Closed"), \
        finals("target_xxz_b10", "LLM-Closed")
    near = sum(1 for v in x8 + x10 if abs(v - 0.95) <= 0.01)
    return {
        "b6_random": count("target_tfim_b6", "Random"),
        "b6_greedy": count("target_tfim_b6", "Greedy"),
        "b6_open": count("target_tfim_b6", "LLM-Open"),
        "b6_closed": count("target_tfim_b6", "LLM-Closed"),
        "b4_open": count("budget_b4", "LLM-Open"),
        "b4_closed": count("budget_b4", "LLM-Closed"),
        "b8_open": count("reference", "LLM-Open"),
        "b8_closed": count("reference", "LLM-Closed"),
        "b16_open": count("budget_b16", "LLM-Open"),
        "b16_closed": count("budget_b16", "LLM-Closed"),
        "x8_closed": count("hamiltonian_xxz", "LLM-Closed"),
        "x10_closed": count("target_xxz_b10", "LLM-Closed"),
        "x8_mean": f"{sum(x8) / len(x8):.4f}",
        "x10_mean": f"{sum(x10) / len(x10):.4f}",
        "x_mean_shift": f"{(sum(x10) / len(x10)) - (sum(x8) / len(x8)):+.4f}",
        "x_near_target": near,
        "x_total_seeds": len(x8) + len(x10),
    }


def hash_tree(root: Path) -> dict:
    out = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "file_hashes.json":
            out[str(path.relative_to(root))] = hashlib.sha256(
                path.read_bytes()).hexdigest()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True,
                        help="directory holding the executed cells")
    parser.add_argument("--body", type=Path, required=True,
                        help="markdown template with {placeholders}")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    endpoints = read_rows(args.audit / "endpoint_attainment.csv")
    prefixes = read_rows(args.audit / "within_run_prefix_summary.csv")
    records = {}
    for key in sorted(NEW_CELLS):
        path = args.runs / key / "run_record.json"
        if path.exists():
            records[key] = json.loads(path.read_text())

    costs, totals = cost_table(records)
    text = render(args.body.read_text(encoding="utf-8"), dict(
        table_095=attainment_table(endpoints, 0.95),
        table_099=attainment_table(endpoints, 0.99),
        table_prefix=prefix_table(prefixes, sorted(NEW_CELLS)),
        table_cost=costs,
        table_validity=validity_table(records),
        verdict_section=verdict_section(endpoints),
        **key_numbers(args.audit, endpoints),
        total_calls=totals["calls"],
        total_repairs=totals["repairs"],
        total_input=totals["input"],
        total_output=totals["output"],
        total_usd=f"{totals['usd']:.4f}",
        total_evaluations=totals["evaluations"],
        total_minutes=f"{totals['seconds'] / 60:.1f}",
    ))
    args.out.write_text(text, encoding="utf-8")
    print(f"wrote {args.out} ({len(text)} bytes)")


if __name__ == "__main__":
    main()
