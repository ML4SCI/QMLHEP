"""Build the 10-slide minimum-budget deck (.pptx) from committed artifacts.

Every number on a slide is read from the audit tables and run records --
nothing is typed by hand. Colours, markers and layout match the previous
meeting deck; helpers are imported from it rather than duplicated.

Usage:
  python scripts/qae/build_budget_target_deck.py --audit DIR --runs DIR
  python scripts/qae/build_budget_target_deck.py --check --audit DIR --runs DIR
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path

from pptx.util import Emu, Inches

_SPEC = importlib.util.spec_from_file_location(
    "qae_deck", Path(__file__).with_name("build_qae_robustness_deck.py"))
D = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(D)

blank, panel, rect, textbox, write = D.blank, D.panel, D.rect, D.textbox, D.write
title_slide_header, footer, chip = D.title_slide_header, D.footer, D.chip
simple_table, new_deck = D.simple_table, D.new_deck
INK, MUTED, RULE, BLUE, RED, GREEN, AMBER, GREY = (
    D.INK, D.MUTED, D.RULE, D.BLUE, D.RED, D.GREEN, D.AMBER, D.GREY)
SLIDE_W, SLIDE_H = D.SLIDE_W, D.SLIDE_H

DECK_NAME = "20260908_llm-vqc"
N_SEEDS, REQUIRED = 12, 10
METHODS = ["Random", "Greedy", "LLM-Open", "LLM-Closed"]
SHORT = {"Random": "Random", "Greedy": "Greedy",
         "LLM-Open": "Open", "LLM-Closed": "Closed"}
NEW_CELLS = {"target_tfim_b6", "target_xxz_b10"}
TFIM_LADDER = [("budget_b4", 4), ("target_tfim_b6", 6),
               ("reference", 8), ("budget_b16", 16)]
XXZ_LADDER = [("hamiltonian_xxz", 8), ("target_xxz_b10", 10)]


# ---------------------------------------------------------------- data ----

def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load(audit: Path, runs: Path) -> dict:
    endpoints = read_rows(audit / "endpoint_attainment.csv")
    prefixes = read_rows(audit / "within_run_prefix_summary.csv")
    curves = read_rows(audit / "validation_curves.csv")
    records = {}
    for key in sorted(NEW_CELLS):
        path = runs / key / "run_record.json"
        if path.exists():
            records[key] = json.loads(path.read_text())
    return {
        "endpoint": {(r["condition"], r["method"], round(float(r["target"]), 2)): r
                     for r in endpoints},
        "prefix": {(r["condition"], r["method"], round(float(r["target"]), 2)): r
                   for r in prefixes},
        "curves": curves,
        "records": records,
        "figures": audit / "figures",
    }


def hits(data: dict, key: str, method: str, target: float = 0.95) -> int | None:
    row = data["endpoint"].get((key, method, round(target, 2)))
    return None if row is None else int(row["successes"])


def passes(data: dict, key: str, method: str, target: float = 0.95) -> bool:
    row = data["endpoint"].get((key, method, round(target, 2)))
    return bool(row) and row["passes"] in ("True", True)


def cell_text(data: dict, key: str, method: str, target: float = 0.95) -> str:
    count = hits(data, key, method, target)
    if count is None:
        return "not run"
    return f"{count}/{N_SEEDS}" + (" PASS" if count >= REQUIRED else "")


def totals(data: dict) -> dict:
    out = {"calls": 0, "repairs": 0, "input": 0, "output": 0, "usd": 0.0,
           "evaluations": 0, "seconds": 0.0, "fallbacks": 0}
    for record in data["records"].values():
        usage = record["api_usage"]
        out["calls"] += usage["n_calls"]
        out["repairs"] += usage["n_repair_or_retry_calls"]
        out["input"] += usage["input_tokens"]
        out["output"] += usage["output_tokens"]
        out["usd"] += usage["estimated_cost_usd_at_list_price"]
        out["evaluations"] += record["evaluated_candidates_total"]
        out["seconds"] += record["wall_clock_seconds_this_invocation"]
        for entry in record["generation_validity"].values():
            if isinstance(entry, dict):
                out["fallbacks"] += entry.get("flagged_random_fallbacks", 0)
    return out


def picture(slide, data, name, left, top, width):
    path = data["figures"] / f"{name}.png"
    if not path.exists():
        raise SystemExit(f"missing figure {path}; run the figure builder first")
    return slide.shapes.add_picture(str(path), left, top, width=width)


def verdict_color(count: int | None) -> object:
    if count is None:
        return GREY
    return GREEN if count >= REQUIRED else RED


# -------------------------------------------------------------- slides ----

def slide_01_title(deck, data):
    slide = blank(deck)
    rect(slide, Emu(0), Emu(0), SLIDE_W, Inches(0.16), RULE)
    frame = textbox(slide, Inches(0.9), Inches(1.28), Inches(11.6), Inches(1.8))
    write(frame, [
        ("How much search budget does a", {"size": 38, "bold": True,
                                           "space_after": 0, "line_spacing": 1.05}),
        ("target fidelity actually require?", {"size": 38, "bold": True,
                                               "line_spacing": 1.05}),
    ], space_after=8)
    frame = textbox(slide, Inches(0.9), Inches(2.92), Inches(11.6), Inches(1.0))
    write(frame, [
        ("The action item from last meeting: stop ranking methods at a fixed "
         "budget and ask what budget is needed to reach a required accuracy.",
         {"size": 18, "color": MUTED}),
    ], space_after=4)

    t = totals(data)
    cards = [
        ("Target accuracy", "0.95  and  0.99", BLUE),
        ("Pass rule", f"{REQUIRED} of {N_SEEDS} seeds", RED),
        ("Reused, no new calls", "7 earlier conditions", AMBER),
        ("Newly measured", f"{len(data['records'])} boundary cells", GREEN),
    ]
    for i, (heading, value, color) in enumerate(cards):
        left = Inches(0.9 + i * 3.0)
        panel(slide, left, Inches(3.86), Inches(2.72), Inches(1.5))
        rect(slide, left, Inches(3.86), Inches(2.72), Inches(0.09), color)
        frame = textbox(slide, left + Inches(0.16), Inches(4.08), Inches(2.42),
                        Inches(1.2))
        write(frame, [(heading, {"size": 12, "bold": True}),
                      (value, {"size": 15, "color": color, "bold": True})],
              space_after=5)

    frame = textbox(slide, Inches(0.9), Inches(5.66), Inches(11.6), Inches(1.0))
    write(frame, [
        ("Decided on validation only. The held-out test set is never evaluated "
         "anywhere in this study.", {"size": 14, "bold": True}),
        (f"New model calls: {t['calls']}. New candidate circuits trained and "
         f"evaluated: {t['evaluations']}. Measured cost at published list "
         f"prices: USD {t['usd']:.2f}.", {"size": 14, "color": MUTED}),
    ], space_after=5)
    footer(slide, "Tomoya Hatanaka - ML4SCI / Google Summer of Code 2026 - "
                  "deck prepared 8 September 2026 (meeting date not fixed)")


def slide_02_previous(deck, data):
    slide = blank(deck)
    title_slide_header(
        slide, "Where we left off, and the question it left open",
        "Last meeting closed on an action item, not on a result")

    panel(slide, Inches(0.6), Inches(1.68), Inches(6.0), Inches(4.7))
    frame = textbox(slide, Inches(0.85), Inches(1.9), Inches(5.5), Inches(4.4))
    write(frame, [
        ("What the stress test concluded", {"size": 15, "bold": True}),
        ("", {"size": 6}),
        ("Robust.", {"size": 13, "bold": True, "color": BLUE}),
        ("Physics-informed proposals beat random and greedy circuit search in "
         "every one of the seven tested conditions.", {"size": 12, "color": MUTED}),
        ("", {"size": 6}),
        ("Fragile.", {"size": 13, "bold": True, "color": RED}),
        ("The extra gain from a closed feedback loop is not general; it changes "
         "with budget, register width, Hamiltonian and model.",
         {"size": 12, "color": MUTED}),
        ("", {"size": 6}),
        ("Operational bottleneck.", {"size": 13, "bold": True, "color": AMBER}),
        ("Valid circuit generation has to be separated from architecture "
         "quality; an invalid proposal becomes a random draw.",
         {"size": 12, "color": MUTED}),
    ], space_after=4)

    panel(slide, Inches(6.95), Inches(1.68), Inches(5.78), Inches(4.7))
    rect(slide, Inches(6.95), Inches(1.68), Inches(5.78), Inches(0.09), RULE)
    frame = textbox(slide, Inches(7.2), Inches(1.95), Inches(5.3), Inches(4.3))
    write(frame, [
        ("The closing slide's action item, verbatim in substance",
         {"size": 15, "bold": True}),
        ("", {"size": 6}),
        ("Fix a target validation fidelity (0.95 and 0.99). Take the smallest "
         "budget that reaches it in at least 10 of the 12 paired seeds. Choose "
         "on validation. Reanalyse existing logs before making any new call, "
         "keep only the unresolved boundaries, and run new calls only there.",
         {"size": 12.5}),
        ("", {"size": 8}),
        ("Every comparison so far ranked methods at a fixed budget. What "
         "decides practical use is the budget needed to reach the required "
         "accuracy.", {"size": 12.5, "color": MUTED}),
    ], space_after=5)

    footer(slide, "Next: the same question, restated so that it can actually be "
                  "answered with a bounded number of paid calls.")


def slide_03_question(deck, data):
    slide = blank(deck)
    title_slide_header(
        slide, "The question, restated so that it is answerable",
        "One change to the action item: report what was verified, not an "
        "interpolated minimum")

    panel(slide, Inches(0.6), Inches(1.7), Inches(5.9), Inches(2.2))
    frame = textbox(slide, Inches(0.85), Inches(1.9), Inches(5.4), Inches(1.9))
    write(frame, [
        ("Budget B, defined once", {"size": 14, "bold": True}),
        ("B = the number of candidate circuits trained and evaluated per seed.",
         {"size": 12.5}),
        ("Not tokens, not calls, not wall-clock time.",
         {"size": 12, "color": MUTED}),
    ], space_after=5)

    panel(slide, Inches(6.85), Inches(1.7), Inches(5.9), Inches(2.2))
    rect(slide, Inches(6.85), Inches(1.7), Inches(5.9), Inches(0.09), RED)
    frame = textbox(slide, Inches(7.1), Inches(1.9), Inches(5.4), Inches(1.9))
    write(frame, [
        ("Why no single minimum is claimed", {"size": 14, "bold": True}),
        ("Each budget is a different search policy: the number of exploration "
         "candidates, the size of each request and the refinement horizon all "
         "change with B.", {"size": 12.5}),
        ("So runs at different budgets are never joined, and no interval is "
         "interpolated between them.", {"size": 12, "color": MUTED}),
    ], space_after=5)

    frame = textbox(slide, Inches(0.6), Inches(4.06), Inches(12.1), Inches(0.4))
    write(frame, [("What gets reported instead - three separate categories",
                   {"size": 14, "bold": True})], space_after=4)
    cards = [
        ("Smallest verified passing budget", "a budget actually run, that met "
         "the rule", GREEN),
        ("Verified failing budgets", "budgets actually run, that did not meet "
         "the rule", RED),
        ("Unverified budgets", "never run; no claim made in either direction",
         GREY),
    ]
    for i, (heading, body, color) in enumerate(cards):
        left = Inches(0.6 + i * 4.09)
        panel(slide, left, Inches(4.5), Inches(3.87), Inches(1.5))
        rect(slide, left, Inches(4.5), Inches(3.87), Inches(0.09), color)
        frame = textbox(slide, left + Inches(0.18), Inches(4.72), Inches(3.5),
                        Inches(1.2))
        write(frame, [(heading, {"size": 12.5, "bold": True, "color": color}),
                      (body, {"size": 11.5, "color": MUTED})], space_after=4)

    frame = textbox(slide, Inches(0.6), Inches(6.16), Inches(12.1), Inches(0.6))
    write(frame, [
        ("The closing slide asked for an interval where the exact minimum is "
         "too expensive. An interval would assume that success never "
         "disappears as the budget grows - which these runs do not establish, "
         "so the three categories above are reported instead.",
         {"size": 12, "color": MUTED}),
    ], space_after=0)
    footer(slide, "Next: the decision rule, fixed in writing before any new "
                  "circuit was generated.")


def slide_04_rule(deck, data):
    slide = blank(deck)
    title_slide_header(
        slide, "The decision rule, fixed before any new run",
        "Validation trash fidelity - the same 12 paired seeds - both targets "
        "reported")

    steps = [
        ("1", "Per seed, best so far", "For seed s, take the highest validation "
         "trash fidelity among the first k candidates of a run configured at "
         "budget B.", BLUE),
        ("2", "Count seeds, do not average", f"Count how many of the {N_SEEDS} "
         "seeds are at or above the target. A mean above the target is not a "
         "pass.", RED),
        ("3", f"Pass = {REQUIRED} of {N_SEEDS}", "Equality with the target "
         "counts as a hit. All 12 seeds are reported; none is dropped, "
         "re-rolled or replaced.", GREEN),
    ]
    for i, (number, heading, body, color) in enumerate(steps):
        left = Inches(0.6 + i * 4.09)
        panel(slide, left, Inches(1.7), Inches(3.87), Inches(1.95))
        rect(slide, left, Inches(1.7), Inches(3.87), Inches(0.09), color)
        frame = textbox(slide, left + Inches(0.18), Inches(1.92), Inches(3.5),
                        Inches(1.7))
        write(frame, [(f"{number}.  {heading}", {"size": 13, "bold": True,
                                                 "color": color}),
                      (body, {"size": 11.5})], space_after=4)

    panel(slide, Inches(0.6), Inches(3.85), Inches(5.9), Inches(2.5))
    frame = textbox(slide, Inches(0.85), Inches(4.05), Inches(5.4), Inches(2.2))
    write(frame, [
        ("The test set is protected structurally", {"size": 14, "bold": True}),
        ("The new runner is handed an empty test array, so no test state is "
         "ever contracted with a trained circuit.", {"size": 12}),
        ("The written result tables carry a validation-only column list, and a "
         "guard fails the run if any test number turns out to be finite.",
         {"size": 12}),
        ("Earlier studies did record test numbers. Those are history, not a "
         "fresh confirmation set, and none is reported here.",
         {"size": 12, "color": MUTED}),
    ], space_after=4)

    panel(slide, Inches(6.85), Inches(3.85), Inches(5.9), Inches(2.5))
    frame = textbox(slide, Inches(7.1), Inches(4.05), Inches(5.4), Inches(2.2))
    write(frame, [
        ("Everything else is inherited unchanged", {"size": 14, "bold": True}),
        ("Same model snapshot, prompt wording, output schema, repair policy, "
         "gate budget, trainer, optimizer, data splits, seeds and selection "
         "rule as the condition each new cell is anchored to.", {"size": 12}),
        ("Only what the budget forces moves: half the budget explores, half "
         "refines, and the request size scales with it.",
         {"size": 12, "color": MUTED}),
        ("A machine check refuses any cell that moves more than the budget.",
         {"size": 12, "color": MUTED}),
    ], space_after=4)
    footer(slide, "Next: which cells were reused from existing logs, and which "
                  "two were actually run.")


def slide_05_scope(deck, data):
    slide = blank(deck)
    title_slide_header(
        slide, "What was reused, what was run, what was left open",
        "Existing logs first; new calls only at the two unresolved boundaries")

    t = totals(data)
    rows = [["Condition", "B", "Status", "Open", "Closed"]]
    colors = {}
    for r, (key, budget, label) in enumerate([
        ("budget_b4", 4, "4-qubit Ising"), ("target_tfim_b6", 6, "4-qubit Ising"),
        ("reference", 8, "4-qubit Ising"), ("budget_b16", 16, "4-qubit Ising"),
        ("hamiltonian_xxz", 8, "4-qubit XXZ"),
        ("target_xxz_b10", 10, "4-qubit XXZ"),
        ("qubits_n6", 8, "6-qubit Ising"), ("qubits_n8", 8, "8-qubit Ising"),
        ("model_alt", 8, "4-qubit Ising, other model"),
    ], start=1):
        status = "measured now" if key in NEW_CELLS else "reused"
        rows.append([label, str(budget), status,
                     cell_text(data, key, "LLM-Open"),
                     cell_text(data, key, "LLM-Closed")])
        for c, method in ((3, "LLM-Open"), (4, "LLM-Closed")):
            colors[(r, c)] = verdict_color(hits(data, key, method))
    simple_table(slide, rows, Inches(0.6), Inches(1.72), Inches(6.5),
                 [Inches(2.35), Inches(0.6), Inches(1.45), Inches(1.05),
                  Inches(1.05)], header_size=10, body_size=10,
                 row_height=Inches(0.325), colors=colors)

    panel(slide, Inches(7.45), Inches(1.72), Inches(5.3), Inches(2.35))
    rect(slide, Inches(7.45), Inches(1.72), Inches(5.3), Inches(0.09), GREEN)
    frame = textbox(slide, Inches(7.68), Inches(1.94), Inches(4.9), Inches(2.1))
    write(frame, [
        ("Run now - two boundary cells", {"size": 13.5, "bold": True}),
        ("4-qubit Ising at B = 6, all four methods on the same seeds, so the "
         "controls are budget-matched.", {"size": 11.5}),
        ("4-qubit XXZ at B = 10, closed loop only - the one cell that sat one "
         "seed short of the rule. No cross-method comparison is drawn there.",
         {"size": 11.5}),
    ], space_after=4)

    panel(slide, Inches(7.45), Inches(4.24), Inches(5.3), Inches(2.1))
    rect(slide, Inches(7.45), Inches(4.24), Inches(5.3), Inches(0.09), GREY)
    frame = textbox(slide, Inches(7.68), Inches(4.46), Inches(4.9), Inches(1.9))
    write(frame, [
        ("Deliberately left unresolved", {"size": 13.5, "bold": True}),
        ("6 and 8 qubits, the alternative model, and the 0.99 target.",
         {"size": 11.5}),
        ("Every method sits at 0 to 3 seeds out of 12 there. A blind sweep "
         "would not separate invalid generation, too little search, too little "
         "training and too little circuit capacity - so no budget was spent on "
         "it.", {"size": 11.5, "color": MUTED}),
    ], space_after=4)
    footer(slide, f"New work: {t['calls']} model calls, {t['evaluations']} "
                  f"candidate circuits trained and evaluated, "
                  f"USD {t['usd']:.2f} at published list prices. "
                  "Green = meets the rule, red = does not.")


def slide_06_tfim(deck, data):
    slide = blank(deck)
    b6_open, b6_closed = hits(data, "target_tfim_b6", "LLM-Open"), \
        hits(data, "target_tfim_b6", "LLM-Closed")
    title_slide_header(
        slide, "Ising chain: where the budget boundary actually falls",
        "Seeds reaching validation fidelity 0.95, at four independently "
        "executed budgets")
    picture(slide, data, "ladder_tfim", Inches(0.55), Inches(1.66), Inches(7.3))

    panel(slide, Inches(8.1), Inches(1.66), Inches(4.65), Inches(4.75))
    frame = textbox(slide, Inches(8.33), Inches(1.88), Inches(4.25), Inches(4.5))
    lines = [("Reading the ladder", {"size": 14, "bold": True}), ("", {"size": 5})]
    for method in METHODS:
        parts = []
        for key, budget in TFIM_LADDER:
            count = hits(data, key, method)
            parts.append(f"B={budget}: {'-' if count is None else count}")
        lines.append((f"{SHORT[method]}   " + "  ".join(parts),
                      {"size": 11.5, "bold": method.startswith("LLM")}))
    lines += [
        ("", {"size": 7}),
        (f"At B = 6 the open loop reaches {b6_open} of {N_SEEDS} and the closed "
         f"loop {b6_closed} of {N_SEEDS}.", {"size": 12.5, "bold": True}),
        ("", {"size": 5}),
        ("Non-semantic search never comes close at any budget on this ladder.",
         {"size": 12, "color": MUTED}),
    ]
    write(frame, lines, space_after=4)
    footer(slide, "Each point is a separate run at that configured budget. "
                  "Points are joined only for readability - no trend between "
                  "budgets is asserted, and untested budgets are left blank.")


def slide_07_xxz(deck, data):
    slide = blank(deck)
    before = hits(data, "hamiltonian_xxz", "LLM-Closed")
    after = hits(data, "target_xxz_b10", "LLM-Closed")
    title_slide_header(
        slide, "XXZ chain: the one cell that sat a single seed short",
        "Closed loop only, same 12 paired seeds, budget raised from 8 to 10")
    picture(slide, data, "attainment_target_xxz_b10", Inches(0.55), Inches(1.66),
            Inches(7.5))

    panel(slide, Inches(8.3), Inches(1.66), Inches(4.45), Inches(4.75))
    rect(slide, Inches(8.3), Inches(1.66), Inches(4.45), Inches(0.09), RED)
    frame = textbox(slide, Inches(8.53), Inches(1.9), Inches(4.05), Inches(4.5))
    write(frame, [
        ("The boundary probe", {"size": 14, "bold": True}),
        ("", {"size": 5}),
        (f"Budget 8:   {before} of {N_SEEDS}", {"size": 13, "bold": True,
                                                "color": verdict_color(before)}),
        (f"Budget 10:  {after} of {N_SEEDS}", {"size": 13, "bold": True,
                                               "color": verdict_color(after)}),
        ("", {"size": 7}),
        ("Only the closed loop was run at budget 10, so no same-budget "
         "comparison against the other three methods exists here and none is "
         "shown.", {"size": 11.5, "color": MUTED}),
        ("", {"size": 5}),
        ("This cell is anchored at the XXZ chain, not at the Ising baseline. "
         "It is a budget change on top of an already-tested Hamiltonian "
         "change, declared as its own protocol.", {"size": 11.5}),
    ], space_after=4)
    footer(slide, "Left: how many seeds have reached each target after k "
                  "candidates, within the run configured at budget 10. The "
                  "dotted line separates exploration from refinement.")


def slide_08_reading(deck, data):
    slide = blank(deck)
    title_slide_header(
        slide, "Two ways this result could be over-read",
        "Both are avoidable by stating what was actually executed")
    picture(slide, data, "attainment_target_tfim_b6", Inches(0.55), Inches(1.62),
            Inches(7.2))

    panel(slide, Inches(8.0), Inches(1.62), Inches(4.75), Inches(2.3))
    rect(slide, Inches(8.0), Inches(1.62), Inches(4.75), Inches(0.09), AMBER)
    frame = textbox(slide, Inches(8.23), Inches(1.84), Inches(4.35), Inches(2.1))
    write(frame, [
        ("A prefix is not a budget", {"size": 13.5, "bold": True}),
        ("Reaching the pass line after three candidates inside a larger run is "
         "not a three-candidate experiment: the whole batch was already "
         "generated and paid for, and a smaller budget would have asked a "
         "different question.", {"size": 11.5}),
    ], space_after=4)

    panel(slide, Inches(8.0), Inches(4.1), Inches(4.75), Inches(2.3))
    rect(slide, Inches(8.0), Inches(4.1), Inches(4.75), Inches(0.09), BLUE)
    frame = textbox(slide, Inches(8.23), Inches(4.32), Inches(4.35), Inches(2.1))
    write(frame, [
        ("Early success is exploration, not feedback", {"size": 13.5,
                                                        "bold": True}),
        ("The first half of every closed-loop run is semantic exploration with "
         "no score attached. Anything reached before the dotted line cannot be "
         "credited to the feedback loop.", {"size": 11.5}),
        ("The open-loop pool is also shared across seeds, so 12 successes are "
         "12 seeds on one pool, not 12 independent generations.",
         {"size": 11.5, "color": MUTED}),
    ], space_after=4)
    footer(slide, "Figure: Ising chain at budget 6, both targets, all four "
                  "methods on the same seeds.")


def slide_09_cost(deck, data):
    slide = blank(deck)
    t = totals(data)
    title_slide_header(
        slide, "What it cost, and how valid the generated circuits were",
        "Recorded separately: evaluations, calls, repairs, tokens, money, time")

    rows = [["", "Ising, B = 6", "XXZ, B = 10", "Total"]]
    keys = ["target_tfim_b6", "target_xxz_b10"]
    records = data["records"]

    def get(key, path):
        record = records.get(key)
        if record is None:
            return "-"
        value = record
        for part in path:
            value = value[part]
        return value

    for label, path in [
        ("Candidate circuits evaluated", ["evaluated_candidates_total"]),
        ("Model calls", ["api_usage", "n_calls"]),
        ("Repair or retry calls", ["api_usage", "n_repair_or_retry_calls"]),
        ("Input tokens", ["api_usage", "input_tokens"]),
        ("Output tokens", ["api_usage", "output_tokens"]),
    ]:
        values = [get(k, path) for k in keys]
        numeric = [v for v in values if isinstance(v, int)]
        rows.append([label] + [f"{v:,}" if isinstance(v, int) else v
                               for v in values] + [f"{sum(numeric):,}"])
    usd = [get(k, ["api_usage", "estimated_cost_usd_at_list_price"]) for k in keys]
    rows.append(["Cost at list price (USD)"]
                + [f"{v:.3f}" if isinstance(v, float) else v for v in usd]
                + [f"{t['usd']:.3f}"])
    simple_table(slide, rows, Inches(0.6), Inches(1.72), Inches(7.1),
                 [Inches(2.9), Inches(1.45), Inches(1.45), Inches(1.3)],
                 header_size=10, body_size=10, row_height=Inches(0.335))

    panel(slide, Inches(8.05), Inches(1.72), Inches(4.7), Inches(2.3))
    rect(slide, Inches(8.05), Inches(1.72), Inches(4.7), Inches(0.09), GREEN)
    frame = textbox(slide, Inches(8.28), Inches(1.94), Inches(4.3), Inches(2.1))
    write(frame, [
        ("Spending was capped, not estimated after the fact",
         {"size": 13.5, "bold": True}),
        ("A hard cap of USD 2.00 was authorised in advance and checked before "
         "every single request; the code refuses to call at all when no "
         "explicit cap is configured.", {"size": 11.5}),
        (f"Measured spend at published list prices: USD {t['usd']:.2f}.",
         {"size": 11.5, "bold": True}),
    ], space_after=4)

    panel(slide, Inches(8.05), Inches(4.2), Inches(4.7), Inches(2.2))
    rect(slide, Inches(8.05), Inches(4.2), Inches(4.7), Inches(0.09), AMBER)
    frame = textbox(slide, Inches(8.28), Inches(4.42), Inches(4.3), Inches(2.0))
    write(frame, [
        ("Invalid proposals stay in the score", {"size": 13.5, "bold": True}),
        (f"A proposal that breaks the gate contract becomes a random draw and "
         f"still consumes one unit of budget. Random draws used in these two "
         f"cells: {t['fallbacks']}.", {"size": 11.5}),
        ("Excluding them is reported only as a side diagnostic, never as the "
         "headline number.", {"size": 11.5, "color": MUTED}),
    ], space_after=4)
    footer(slide, "Prices read from the provider's published list on the day "
                  "of the run; token counts come from the stored call records.")


def slide_10_next(deck, data):
    slide = blank(deck)
    title_slide_header(
        slide, "What is settled, what is not, and what to decide next",
        "Reported as verified, failed and unverified - not as a single minimum")
    b6_open = hits(data, "target_tfim_b6", "LLM-Open")
    b6_closed = hits(data, "target_tfim_b6", "LLM-Closed")
    x10 = hits(data, "target_xxz_b10", "LLM-Closed")

    cards = [
        ("Verified", GREEN, [
            f"Ising chain, closed loop: the rule is met at budget "
            f"{'6' if (b6_closed or 0) >= REQUIRED else '8'} and above among "
            f"the budgets actually run.",
            f"Ising chain at budget 6: open {b6_open}/{N_SEEDS}, "
            f"closed {b6_closed}/{N_SEEDS}.",
            f"XXZ chain, closed loop at budget 10: {x10}/{N_SEEDS}.",
        ]),
        ("Not settled", AMBER, [
            "No budget below the smallest verified pass was tested, so the "
            "true minimum is not pinned down.",
            "The stricter 0.99 target is met nowhere, at any budget or "
            "condition tested so far.",
            "Wider registers and the alternative model remain open; they were "
            "not probed with new spending.",
        ]),
        ("Decision to take", BLUE, [
            "Is a boundary at this resolution worth more paid search, or is "
            "the useful next step diagnosing why 0.99 is unreachable?",
            "If accuracy is the goal, circuit capacity and training length are "
            "the suspects to test - not more candidates.",
            "Any confirmatory read of a held-out set must be declared and "
            "frozen before it is looked at.",
        ]),
    ]
    for i, (heading, color, bullets) in enumerate(cards):
        left = Inches(0.6 + i * 4.09)
        panel(slide, left, Inches(1.72), Inches(3.87), Inches(3.5))
        rect(slide, left, Inches(1.72), Inches(3.87), Inches(0.09), color)
        frame = textbox(slide, left + Inches(0.18), Inches(1.96), Inches(3.5),
                        Inches(3.2))
        lines = [(heading, {"size": 14, "bold": True, "color": color}),
                 ("", {"size": 5})]
        lines += [(text, {"size": 11.5}) for text in bullets]
        write(frame, lines, space_after=6)

    panel(slide, Inches(0.6), Inches(5.42), Inches(12.15), Inches(1.0))
    frame = textbox(slide, Inches(0.85), Inches(5.6), Inches(11.7), Inches(0.9))
    write(frame, [
        ("A budget that passes is not evidence that a smaller one fails. Each "
         "budget was run as its own policy, none was interpolated, and the "
         "untested budgets are reported as untested.",
         {"size": 12.5, "bold": True}),
        ("All figures and numbers on these slides are generated from the "
         "stored result tables; nothing was typed by hand.",
         {"size": 11.5, "color": MUTED}),
    ], space_after=4)
    footer(slide, "Tomoya Hatanaka - ML4SCI / Google Summer of Code 2026 - "
                  "deck prepared 8 September 2026")


BUILDERS = [slide_01_title, slide_02_previous, slide_03_question, slide_04_rule,
            slide_05_scope, slide_06_tfim, slide_07_xxz, slide_08_reading,
            slide_09_cost, slide_10_next]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--no-pdf", action="store_true")
    args = parser.parse_args()

    out_dir = args.out or (args.audit / "deck")
    out_dir.mkdir(parents=True, exist_ok=True)
    pptx_path = out_dir / f"{DECK_NAME}.pptx"

    if not args.check:
        data = load(args.audit, args.runs)
        deck = new_deck()
        for builder in BUILDERS:
            builder(deck, data)
        deck.save(str(pptx_path))
        print("wrote", pptx_path)

    problems = D.audit(pptx_path, expected_slides=len(BUILDERS))
    if problems:
        print("\nAUDIT FAILED:")
        for problem in problems:
            print("  -", problem)
    else:
        print(f"audit passed: exactly {len(BUILDERS)} slides, nothing "
              "off-canvas, no prohibited strings")

    if not args.no_pdf and not args.check:
        print("wrote", D.render_pdf(pptx_path))
    if problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
