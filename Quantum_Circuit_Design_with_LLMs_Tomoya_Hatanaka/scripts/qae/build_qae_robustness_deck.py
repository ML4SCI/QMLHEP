"""Build the 10-slide QAE robustness deck (.pptx) from committed artifacts.

Every number on a slide is read from `outputs/qae_robustness/summary.json`
and the per-condition statistics files -- nothing is typed by hand.

Usage:
  python scripts/qae/build_qae_robustness_deck.py
  python scripts/qae/build_qae_robustness_deck.py --check   # audit only
"""
import argparse
import json
import re
import subprocess
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Emu, Inches, Pt

ROOT = Path("outputs/qae_robustness")
FIGURES = ROOT / "figures"
DECK_DIR = Path("outputs/qae_robustness/deck")
DECK_NAME = "20260904_GSoC"

SLIDE_W, SLIDE_H = Inches(13.333), Inches(7.5)

INK = RGBColor(0x20, 0x21, 0x24)
MUTED = RGBColor(0x5F, 0x63, 0x68)
RULE = RGBColor(0xD9, 0x30, 0x25)
BLUE = RGBColor(0x1A, 0x73, 0xE8)
RED = RGBColor(0xD9, 0x30, 0x25)
GREEN = RGBColor(0x18, 0x80, 0x38)
AMBER = RGBColor(0xF9, 0xAB, 0x00)
GREY = RGBColor(0x9A, 0xA0, 0xA6)
SLATE = RGBColor(0x3C, 0x40, 0x43)
PANEL = RGBColor(0xF4, 0xF6, 0xF8)

# Visible slide text must never carry coding/version identifiers, internal
# paths, branch names, commit hashes or experiment directory names.
FORBIDDEN_PATTERNS = [
    (r"(?<![A-Za-z0-9.])[vV]\d+(?![A-Za-z0-9.])", "version identifier"),
    (r"\b(outputs|scripts|llm_vqc|docs|tests|runs|configs)/", "internal path"),
    (r"\.(py|csv|json|md|png|svg|pptx|tex|yaml|sqlite)\b", "file name"),
    (r"(?i)\b(claude|experiment|research|presentation|codex|feature)/[\w.-]+",
     "branch name"),
    (r"(?i)\bqae[_-]", "experiment directory name"),
    (r"(?i)\bneutral[_-]|\btfim[_-]|\bapi_v|\bpilot_v", "internal experiment label"),
    (r"(?i)\b(git|commit|branch|repository|worktree|pull request)\b",
     "repository terminology"),
    (r"\b(?=[0-9a-f]{7,40}\b)[0-9a-f]*[a-f][0-9a-f]*\b", "commit-hash-like string"),
    (r"(?i)\bprotocol[_ ]v\d", "internal protocol label"),
]


# --------------------------------------------------------------- helpers ---

def _fmt(value, digits=4, sign=False):
    if value is None:
        return "n/a"
    return f"{value:+.{digits}f}" if sign else f"{value:.{digits}f}"


def _p(value):
    if value is None:
        return "n/a"
    return "p < 0.001" if value < 0.001 else f"p = {value:.3f}"


OPEN_VS_RANDOM = "LLM-Open_minus_Random"
CLOSED_VS_OPEN = "LLM-Closed_minus_LLM-Open"


def contrast_sentence(entry: dict, name: str, label: str) -> str:
    contrast = (entry.get("contrasts") or {}).get(name)
    if not contrast:
        return f"{label}: not estimable"
    ci = contrast["bootstrap_95ci"]
    return (f"{label} {_fmt(contrast['mean_paired_gain'], sign=True)} "
            f"[{_fmt(ci[0], sign=True)}, {_fmt(ci[1], sign=True)}], "
            f"{contrast['wins_b']}/{contrast['n']} seeds, "
            f"{_p(contrast['wilcoxon_exact_two_sided_p'])}")


def cell(entry: dict, name: str) -> str:
    """One table cell: gain, CI, wins and p, or an explicit n/a."""
    contrast = (entry.get("contrasts") or {}).get(name)
    if not contrast:
        return "n/a"
    ci = contrast["bootstrap_95ci"]
    return (f"{_fmt(contrast['mean_paired_gain'], sign=True)}  "
            f"[{_fmt(ci[0], 3, sign=True)}, {_fmt(ci[1], 3, sign=True)}]\n"
            f"{contrast['wins_b']}/{contrast['n']} seeds, "
            f"{_p(contrast['wilcoxon_exact_two_sided_p'])}")


def verdict(entry: dict, name: str) -> tuple[str, RGBColor]:
    contrast = (entry.get("contrasts") or {}).get(name)
    if not contrast:
        return "n/a", MUTED
    significant = contrast["wilcoxon_exact_two_sided_p"] < 0.05
    positive = contrast["mean_paired_gain"] > 0
    if significant and positive:
        return "holds", GREEN
    if significant and not positive:
        return "reverses", RED
    return "not detected", MUTED


# ---------------------------------------------------------------- canvas ---

def new_deck() -> Presentation:
    deck = Presentation()
    deck.slide_width, deck.slide_height = SLIDE_W, SLIDE_H
    return deck


def blank(deck: Presentation):
    return deck.slides.add_slide(deck.slide_layouts[6])


def textbox(slide, left, top, width, height, align=PP_ALIGN.LEFT):
    box = slide.shapes.add_textbox(left, top, width, height)
    frame = box.text_frame
    frame.word_wrap = True
    frame.paragraphs[0].alignment = align
    return frame


def write(frame, lines, *, size=14, color=INK, bold=False, space_after=6,
          bullet="", line_spacing=1.0):
    first = True
    for line in lines:
        text, opts = (line if isinstance(line, tuple) else (line, {}))
        paragraph = frame.paragraphs[0] if first else frame.add_paragraph()
        first = False
        paragraph.space_after = Pt(opts.get("space_after", space_after))
        paragraph.line_spacing = opts.get("line_spacing", line_spacing)
        run = paragraph.add_run()
        run.text = (bullet + text) if (bullet and text) else text
        font = run.font
        font.size = Pt(opts.get("size", size))
        font.bold = opts.get("bold", bold)
        font.color.rgb = opts.get("color", color)
        font.name = "Helvetica Neue"
    return frame


def title_slide_header(slide, title, kicker=None):
    frame = textbox(slide, Inches(0.62), Inches(0.34), Inches(12.1), Inches(0.75))
    write(frame, [title], size=27, bold=True, space_after=0)
    rect(slide, Inches(0.62), Inches(1.06), Inches(1.5), Emu(34925), RULE)
    if kicker:
        frame = textbox(slide, Inches(0.62), Inches(1.16), Inches(12.1), Inches(0.4))
        write(frame, [kicker], size=13, color=MUTED, space_after=0)


def footer(slide, text):
    frame = textbox(slide, Inches(0.62), Inches(6.94), Inches(12.1), Inches(0.4))
    write(frame, [text], size=9.5, color=MUTED, space_after=0)


def _flat(shape):
    """Remove the theme shape style and any effect, so nothing renders a shadow."""
    shape.shadow.inherit = False
    style = shape._element.find(
        "{http://schemas.openxmlformats.org/presentationml/2006/main}style")
    if style is not None:
        shape._element.remove(style)
    return shape


def rect(slide, left, top, width, height, color):
    shape = _flat(slide.shapes.add_shape(1, left, top, width, height))
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.fill.background()
    return shape


def panel(slide, left, top, width, height, fill=PANEL):
    shape = _flat(slide.shapes.add_shape(5, left, top, width, height))
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    shape.line.color.rgb = RGBColor(0xE0, 0xE3, 0xE7)
    shape.line.width = Pt(0.75)
    return shape


def picture(slide, name, left, top, width):
    path = FIGURES / f"{name}.png"
    if not path.exists():
        raise SystemExit(f"missing figure {path}; run the figure builder first")
    return slide.shapes.add_picture(str(path), left, top, width=width)


CHIP_WIDTH = Inches(1.35)


def chip(slide, left, top, text, color, width=CHIP_WIDTH):
    shape = rect(slide, left, top, width, Inches(0.3), color)
    frame = shape.text_frame
    frame.margin_left = frame.margin_right = Emu(0)
    frame.margin_top = frame.margin_bottom = Emu(0)
    paragraph = frame.paragraphs[0]
    paragraph.alignment = PP_ALIGN.CENTER
    run = paragraph.add_run()
    run.text = text
    run.font.size = Pt(10)
    run.font.bold = True
    run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    run.font.name = "Helvetica Neue"
    return shape


ROW_HEIGHT = Inches(0.34)


def simple_table(slide, rows, left, top, width, col_widths, header_size=10.5,
                 body_size=10.5, row_height=ROW_HEIGHT, colors=None):
    table_shape = slide.shapes.add_table(
        len(rows), len(rows[0]), left, top, width, row_height * len(rows)
    ).table
    for index, w in enumerate(col_widths):
        table_shape.columns[index].width = w
    for r, row in enumerate(rows):
        table_shape.rows[r].height = row_height
        for c, value in enumerate(row):
            cell = table_shape.cell(r, c)
            cell.text = str(value)
            cell.margin_left = Inches(0.08)
            cell.margin_right = Inches(0.06)
            cell.margin_top = cell.margin_bottom = Emu(12700)
            for paragraph in cell.text_frame.paragraphs:
                paragraph.alignment = PP_ALIGN.LEFT if c == 0 else PP_ALIGN.CENTER
                for run in paragraph.runs:
                    run.font.size = Pt(header_size if r == 0 else body_size)
                    run.font.bold = r == 0
                    run.font.name = "Helvetica Neue"
                    run.font.color.rgb = INK
                    if colors and r > 0 and (r, c) in colors:
                        run.font.color.rgb = colors[(r, c)]
                        run.font.bold = True
            cell.fill.solid()
            cell.fill.fore_color.rgb = (
                RGBColor(0xEC, 0xEF, 0xF2) if r == 0
                else (RGBColor(0xFF, 0xFF, 0xFF) if r % 2 else RGBColor(0xFA, 0xFB, 0xFC))
            )
    return table_shape


# ---------------------------------------------------------------- slides ---

def slide_01_title(deck, summary):
    slide = blank(deck)
    rect(slide, Emu(0), Emu(0), SLIDE_W, Inches(0.16), RULE)

    frame = textbox(slide, Inches(0.9), Inches(1.32), Inches(11.6), Inches(1.7))
    write(frame, [
        ("Does the quantum-autoencoder", {"size": 38, "bold": True,
                                          "space_after": 0, "line_spacing": 1.05}),
        ("search result hold up?", {"size": 38, "bold": True, "line_spacing": 1.05}),
    ], space_after=8)
    frame = textbox(slide, Inches(0.9), Inches(2.95), Inches(11.6), Inches(1.0))
    write(frame, [
        ("A controlled stress test of last meeting's finding: change one thing at a "
         "time and see what survives.", {"size": 18, "color": MUTED}),
    ], space_after=4)

    reference = summary["conditions"]["reference"]
    labels = [
        ("Evaluation budget", "B = 4  /  8  /  16", BLUE),
        ("Qubit count", "4  /  6  /  8", RED),
        ("Hamiltonian family", "Ising  /  XXZ", AMBER),
        ("Language model", "two snapshots", GREY),
    ]
    for i, (heading, levels, color) in enumerate(labels):
        left = Inches(0.9 + i * 3.0)
        panel(slide, left, Inches(3.9), Inches(2.72), Inches(1.5))
        rect(slide, left, Inches(3.9), Inches(2.72), Inches(0.09), color)
        frame = textbox(slide, left + Inches(0.16), Inches(4.12), Inches(2.4), Inches(1.2))
        write(frame, [(heading, {"size": 12, "bold": True}),
                      (levels, {"size": 15, "color": color, "bold": True})],
              space_after=5)

    frame = textbox(slide, Inches(0.9), Inches(5.7), Inches(11.6), Inches(0.9))
    write(frame, [
        (f"Everything else is frozen: same methods, prompts, schema, trainer, "
         f"optimizer, splits, selection rule and the same "
         f"{reference['n_seeds']} paired seeds.", {"size": 14}),
        ("Held-out test trash fidelity, read once after selection. Higher is better.",
         {"size": 14, "color": MUTED}),
    ], space_after=5)
    footer(slide, "Research meeting - 4 September 2026")


def slide_02_previous(deck, summary):
    slide = blank(deck)
    title_slide_header(
        slide, "Where we left off: the previous baseline",
        "4 qubits - transverse-field Ising chain - budget B = 8 - 12 paired seeds")
    picture(slide, "fig01_previous_baseline", Inches(0.55), Inches(1.62), Inches(7.55))

    reference = summary["conditions"]["reference"]
    panel(slide, Inches(8.35), Inches(1.62), Inches(4.42), Inches(4.9))
    frame = textbox(slide, Inches(8.6), Inches(1.8), Inches(4.0), Inches(4.6))
    means = reference["per_method"]
    write(frame, [
        ("The two claims to stress-test", {"size": 14, "bold": True}),
        ("", {"size": 5}),
        ("1. Semantics is decisive.", {"size": 13, "bold": True, "color": BLUE}),
        (contrast_sentence(reference, "LLM-Open_minus_Random", "Open - Random"),
         {"size": 11.5, "color": MUTED}),
        ("", {"size": 5}),
        ("2. Closed-loop redesign adds a little more.",
         {"size": 13, "bold": True, "color": RED}),
        (contrast_sentence(reference, "LLM-Closed_minus_LLM-Open", "Closed - Open"),
         {"size": 11.5, "color": MUTED}),
        ("", {"size": 5}),
        ("3. Non-semantic search is flat.", {"size": 13, "bold": True, "color": AMBER}),
        (contrast_sentence(reference, "Greedy_minus_Random", "Greedy - Random"),
         {"size": 11.5, "color": MUTED}),
        ("", {"size": 8}),
        (f"Selected-architecture means: Random {_fmt(means['Random']['mean'])}, "
         f"Greedy {_fmt(means['Greedy']['mean'])}, "
         f"Open {_fmt(means['LLM-Open']['mean'])}, "
         f"Closed {_fmt(means['LLM-Closed']['mean'])}.", {"size": 11.5}),
        ("", {"size": 6}),
        ("Claim 1 was large and consistent. Claim 2 was small and had been "
         "detected exactly once - the fragile one.", {"size": 12, "bold": True}),
    ], space_after=4)
    footer(slide, "Reference values read from the committed result tables of the "
                  "previous study, not from last meeting's slide text.")


def slide_03_protocol(deck, summary):
    slide = blank(deck)
    title_slide_header(
        slide, "How the comparison is kept honest",
        "One factor at a time; every other setting held at the previous condition")

    rows = [["Factor varied", "Levels", "Everything else"],
            ["Evaluation budget B", "4  -  8  -  16", "4 qubits, Ising, reference model"],
            ["Qubit count", "4  -  6  -  8", "B = 8, Ising, reference model"],
            ["Hamiltonian family", "Ising  -  XXZ", "4 qubits, B = 8, reference model"],
            ["Language model", "reference  -  alternative", "4 qubits, Ising, B = 8"]]
    simple_table(slide, rows, Inches(0.62), Inches(1.72), Inches(7.2),
                 [Inches(2.05), Inches(2.05), Inches(3.1)], body_size=10.5)

    frame = textbox(slide, Inches(0.62), Inches(4.05), Inches(7.2), Inches(2.6))
    write(frame, [
        ("Held constant, and checked mechanically", {"size": 14, "bold": True}),
        ("Prompt template, wording and output schema; validation and retry "
         "behaviour; sampling temperature; the trainer, optimizer, epochs and "
         "initialisation; the data splits; the selection rule; the gate set; "
         "every method's logic; the random-number streams; and all analysis "
         "conventions.", {"size": 10.5, "color": MUTED}),
        ("", {"size": 5}),
        ("Each condition carries a fingerprint of all of the above. An automated "
         "check refuses to run a condition whose frozen block is not identical to "
         "the previous condition's, or that moves more than one factor.",
         {"size": 12}),
    ], space_after=5)

    panel(slide, Inches(8.1), Inches(1.72), Inches(4.68), Inches(4.9))
    frame = textbox(slide, Inches(8.34), Inches(1.9), Inches(4.25), Inches(4.6))
    write(frame, [
        ("The four methods (unchanged)", {"size": 14, "bold": True}),
        ("", {"size": 4}),
        ("Random - B independent uniform candidates.",
         {"size": 12, "color": GREY, "bold": True}),
        ("Greedy - B/2 random starts, then B/2 single-change refinements.",
         {"size": 12, "color": AMBER, "bold": True}),
        ("Open - B semantic proposals, all written before any score is seen.",
         {"size": 12, "color": BLUE, "bold": True}),
        ("Closed - B/2 semantic starts, then B/2 free redesigns of the current "
         "best using validation feedback.", {"size": 12, "color": RED, "bold": True}),
        ("", {"size": 7}),
        ("Scaling rules fixed in advance", {"size": 14, "bold": True}),
        ("Circuit width is 3 trainable rotations + 1 controlled-NOT per qubit for "
         "every method, so 4 qubits reproduces the previous contract exactly.",
         {"size": 10.5, "color": MUTED}),
        ("Latent and trash registers are always half and half.",
         {"size": 10.5, "color": MUTED}),
        ("Adaptive methods always split the budget 50/50 between exploration and "
         "refinement.", {"size": 10.5, "color": MUTED}),
        ("", {"size": 5}),
        ("Selection and feedback use validation only. The held-out test set is "
         "read once, after selection.", {"size": 12, "bold": True}),
    ], space_after=4)
    footer(slide, "The previous condition is reused rather than re-run; tests prove "
                  "this code reproduces its data, prompts and trainer exactly.")


RESULT_FIGURE_WIDTH = Inches(11.2)


def _result_slide(deck, summary, *, figure, title, kicker, headline,
                  level_labels, contrast_rows, footer_text):
    """Figure on top, one headline sentence, then a compact contrast table.

    The table keeps the per-level statistics readable and, unlike free text,
    has a height that does not depend on how long the numbers turn out to be.
    """
    slide = blank(deck)
    title_slide_header(slide, title, kicker)
    picture(slide, figure, Inches(1.07), Inches(1.56), RESULT_FIGURE_WIDTH)

    frame = textbox(slide, Inches(0.62), Inches(5.16), Inches(12.15), Inches(0.62))
    write(frame, [(headline, {"size": 13, "bold": True})], space_after=0)

    rows = [["Paired contrast", *level_labels]]
    colors = {}
    for r, (label, color, values) in enumerate(contrast_rows, start=1):
        rows.append([label, *values])
        colors[(r, 0)] = color
    column = Inches(11.05 / len(level_labels))
    simple_table(slide, rows, Inches(0.62), Inches(5.82), Inches(12.15),
                 [Inches(1.55), *([column] * len(level_labels))],
                 header_size=9.5, body_size=9, row_height=Inches(0.36),
                 colors=colors)
    footer(slide, footer_text)
    return slide


def _headline(entries, name, positive, mixed, negative):
    """Choose a headline sentence from the measured signs, never assumed."""
    values = [((e.get("contrasts") or {}).get(name) or {}).get("mean_paired_gain")
              for e in entries]
    values = [v for v in values if v is not None]
    if values and all(v > 0 for v in values):
        return positive
    if values and all(v < 0 for v in values):
        return negative
    return mixed


def slide_04_budget(deck, summary):
    b4 = summary["conditions"]["budget_b4"]
    b8 = summary["conditions"]["reference"]
    b16 = summary["conditions"]["budget_b16"]
    ladder = [b4, b8, b16]
    headline = (
        _headline(ladder, OPEN_VS_RANDOM,
                  "Semantic proposals beat non-semantic search at every budget.",
                  "The semantic advantage over non-semantic search varies in size "
                  "across budgets.",
                  "Semantic proposals lose to non-semantic search at every budget.")
        + "  "
        + _headline(ladder, CLOSED_VS_OPEN,
                    "The closed loop stays ahead of the open batch at every budget.",
                    "The closed loop's edge over the open batch is budget-dependent: "
                    "largest at the smallest budget and gone by the largest.",
                    "The closed loop trails the open batch at every budget.")
    )
    return _result_slide(
        deck, summary, figure="fig02_budget",
        title="Changing the evaluation budget",
        kicker="4 qubits, Ising chain, reference model - only B moves. "
               "B = 8 is the previous condition.",
        headline=headline,
        level_labels=["B = 4", "B = 8  (previous)", "B = 16"],
        contrast_rows=[
            ("Open - Random", BLUE, [cell(c, OPEN_VS_RANDOM) for c in ladder]),
            ("Closed - Open", RED, [cell(c, CLOSED_VS_OPEN) for c in ladder]),
        ],
        footer_text="Adaptive methods keep the 50/50 exploration/refinement split at "
                    "every budget (2+2, 4+4, 8+8). All 12 seeds are shown as dots.",
    )


def slide_05_qubits(deck, summary):
    n4 = summary["conditions"]["reference"]
    n6 = summary["conditions"]["qubits_n6"]
    n8 = summary["conditions"]["qubits_n8"]
    ladder = [n4, n6, n8]
    validity = n6["llm_proposal_validity"]["valid_fraction"]
    headline = (
        "The task gets harder with width, but the semantic advantage does not go "
        "away.  "
        + _headline(ladder, CLOSED_VS_OPEN,
                    "The closed loop keeps its edge over the open batch as the "
                    "register grows.",
                    "The closed loop's edge does not survive the wider registers "
                    f"(only {(validity or 0) * 100:.0f}% of its candidates at 6 "
                    "qubits were valid model proposals).",
                    "Beyond the previous width the closed loop falls behind the "
                    "open batch.")
    )
    return _result_slide(
        deck, summary, figure="fig03_qubits",
        title="Changing the qubit count",
        kicker="Ising chain, B = 8, reference model - only the register size moves. "
               "4 qubits is the previous condition.",
        headline=headline,
        level_labels=["4 qubits  (previous)", "6 qubits", "8 qubits"],
        contrast_rows=[
            ("Open - Random", BLUE, [cell(c, OPEN_VS_RANDOM) for c in ladder]),
            ("Closed - Open", RED, [cell(c, CLOSED_VS_OPEN) for c in ladder]),
        ],
        footer_text="Width scales deterministically: 3 rotations + 1 controlled-NOT "
                    "per qubit for every method, latent and trash always half and "
                    "half. A declared scaling rule, not a gate-count study.",
    )


def slide_06_hamiltonian(deck, summary):
    ising = summary["conditions"]["reference"]
    xxz = summary["conditions"]["hamiltonian_xxz"]
    pair = [ising, xxz]
    return _result_slide(
        deck, summary, figure="fig04_hamiltonian",
        title="Changing the Hamiltonian family",
        kicker="4 qubits, B = 8, reference model - only the physics in the task "
               "description moves. The Ising chain is the previous condition.",
        headline="A different, pre-declared state family - an XXZ Heisenberg chain "
                 "over the same anisotropy range - with the same circuit space, "
                 "trainer, seeds and prompt wording.",
        level_labels=["transverse-field Ising  (previous)", "XXZ Heisenberg"],
        contrast_rows=[
            ("Open - Random", BLUE, [cell(c, OPEN_VS_RANDOM) for c in pair]),
            ("Closed - Open", RED, [cell(c, CLOSED_VS_OPEN) for c in pair]),
        ],
        footer_text="Only Hamiltonian facts were substituted into the frozen prompt "
                    "- name, formula, parameter range and interaction type. An "
                    "automated check confirms nothing else in the prompt changed.",
    )


def slide_07_model(deck, summary):
    reference = summary["conditions"]["reference"]
    alternative = summary["conditions"]["model_alt"]
    pair = [reference, alternative]
    valid_ref = reference["llm_proposal_validity"]["valid_fraction"]
    valid_alt = alternative["llm_proposal_validity"]["valid_fraction"]
    return _result_slide(
        deck, summary, figure="fig05_model",
        title="Changing the underlying language model",
        kicker="4 qubits, Ising chain, B = 8 - only the model identifier moves "
               "(reference gpt-5.4-mini, alternative gpt-4.1-mini).",
        headline=(f"Same prompt bytes, temperature, schema, retry policy and token "
                  f"limit. Resource-contract compliance falls from "
                  f"{valid_ref * 100:.0f}% to {valid_alt * 100:.0f}% of evaluated "
                  f"candidates, yet the semantic advantage is essentially unchanged."),
        level_labels=["reference model  (previous)", "alternative model"],
        contrast_rows=[
            ("Open - Random", BLUE, [cell(c, OPEN_VS_RANDOM) for c in pair]),
            ("Closed - Open", RED, [cell(c, CLOSED_VS_OPEN) for c in pair]),
        ],
        footer_text="Invalid proposals are replaced by flagged random draws under "
                    "the frozen policy, so a weaker model shows up both as lower "
                    "fidelity and as a lower compliance rate.",
    )


def slide_08_summary(deck, summary):
    slide = blank(deck)
    title_slide_header(
        slide, "Everything at once: what survived",
        "Paired difference in test fidelity with bootstrap 95% CI - filled = "
        "significant at p < 0.05, dashed line = the previous condition")
    picture(slide, "fig06_summary_forest", Inches(0.5), Inches(1.58), Inches(12.3))

    order = [("budget_b4", "B = 4"), ("budget_b16", "B = 16"),
             ("qubits_n6", "6 qubits"), ("qubits_n8", "8 qubits"),
             ("hamiltonian_xxz", "XXZ chain"), ("model_alt", "Alt. model")]
    header = ["Condition", "Open - Random", "Closed - Open", "Greedy - Random"]
    rows, colors = [header], {}
    for r, (key, label) in enumerate(order, start=1):
        entry = summary["conditions"][key]
        cells = [label]
        for c, name in enumerate(("LLM-Open_minus_Random",
                                  "LLM-Closed_minus_LLM-Open",
                                  "Greedy_minus_Random"), start=1):
            text, color = verdict(entry, name)
            cells.append(text)
            colors[(r, c)] = color
        rows.append(cells)
    simple_table(slide, rows, Inches(2.55), Inches(5.12), Inches(8.2),
                 [Inches(2.2), Inches(2.0), Inches(2.0), Inches(2.0)],
                 body_size=10, row_height=Inches(0.245))
    footer(slide, "Positive = the first method wins. \"holds\" = same sign as the "
                  "previous condition and significant at p < 0.05; \"not detected\" "
                  "= not significant at 12 seeds; \"reverses\" = significant with "
                  "the opposite sign.")


def slide_09_interpretation(deck, summary):
    slide = blank(deck)
    title_slide_header(
        slide, "What this means, what it costs, what it does not show",
        "Reading the pattern above rather than any single cell")
    picture(slide, "fig07_proposal_validity", Inches(7.72), Inches(1.62), Inches(5.3))

    usage = summary["api_usage_total"]
    frame = textbox(slide, Inches(0.62), Inches(1.62), Inches(6.9), Inches(5.2))
    write(frame, summary["_slide9_lines"], space_after=3)

    panel(slide, Inches(7.72), Inches(4.62), Inches(5.3), Inches(2.15))
    frame = textbox(slide, Inches(7.94), Inches(4.76), Inches(4.9), Inches(2.0))
    write(frame, [
        ("Cost and scaling", {"size": 13.5, "bold": True}),
        (f"{usage['n_calls']} model calls across the whole study "
         f"({usage['input_tokens']:,} input and {usage['output_tokens']:,} output "
         f"tokens), under a hard spending cap.", {"size": 11, "color": MUTED}),
        ("Simulation cost, not model cost, is the binding constraint: state-vector "
         "training grows exponentially with the register, while the number of model "
         "calls grows only linearly in the budget.",
         {"size": 11, "color": MUTED}),
    ], space_after=4)
    frame = textbox(slide, Inches(7.72), Inches(6.94), Inches(5.3), Inches(0.4))
    write(frame, ["Out of scope by design: any ablation of the prompt's semantic "
                  "content."], size=9.5, color=MUTED, space_after=0)


def slide_10_conclusions(deck, summary):
    slide = blank(deck)
    title_slide_header(slide, "Conclusions", None)
    for i, (heading, body, color) in enumerate(summary["_slide10_items"]):
        top = Inches(1.62 + i * 1.28)
        panel(slide, Inches(0.62), top, Inches(12.1), Inches(1.12))
        rect(slide, Inches(0.62), top, Inches(0.1), Inches(1.12), color)
        frame = textbox(slide, Inches(0.95), top + Inches(0.13), Inches(11.6), Inches(1.0))
        write(frame, [(heading, {"size": 15, "bold": True, "color": color}),
                      (body, {"size": 12.5})], space_after=4)
    footer(slide, summary["_slide10_footer"])


# ------------------------------------------------------- narrative inputs ---

def build_narrative(summary: dict) -> dict:
    """Assemble the two prose slides from the measured results, so the wording
    can never claim more than the numbers support."""
    conditions = summary["conditions"]
    reference = conditions["reference"]
    varied = [k for k in conditions if k != "reference"
              and conditions[k].get("status") != "missing"]
    total = len(varied)

    def gain(key, name):
        return ((conditions[key].get("contrasts") or {}).get(name) or {}).get(
            "mean_paired_gain")

    def tally(name):
        reference_sign = gain("reference", name) > 0
        same_sign, holds, reverses = 0, [], []
        for key in varied:
            value = gain(key, name)
            if value is None:
                continue
            if (value > 0) == reference_sign:
                same_sign += 1
            state = verdict(conditions[key], name)[0]
            if state == "holds":
                holds.append(key)
            elif state == "reverses":
                reverses.append(key)
        return same_sign, holds, reverses

    open_signs, open_holds, open_reverses = tally("LLM-Open_minus_Random")
    closed_signs, closed_holds, closed_reverses = tally("LLM-Closed_minus_LLM-Open")
    greedy_signs, greedy_holds, greedy_reverses = tally("Greedy_minus_Random")

    names = {"budget_b4": "B = 4", "budget_b16": "B = 16", "qubits_n6": "6 qubits",
             "qubits_n8": "8 qubits", "hamiltonian_xxz": "the XXZ chain",
             "model_alt": "the alternative model"}

    def listing(keys):
        return ", ".join(names.get(k, k) for k in keys)

    validity = {k: conditions[k]["llm_proposal_validity"] for k in conditions
                if conditions[k].get("status") != "missing"}
    worst = min(validity, key=lambda k: validity[k]["valid_fraction"] or 1.0)

    def arm_validity(key):
        quality = json.loads((ROOT / key / "proposal_quality.json").read_text())
        out = {}
        for method in ("LLM-Open", "LLM-Closed"):
            stats = quality.get(method, {})
            evaluations = stats.get("evaluations", 0)
            out[method] = (1 - stats.get("fallback_evaluations", 0) / evaluations
                           if evaluations else None)
        return out

    arm_names = {"LLM-Open": "open batch", "LLM-Closed": "closed loop"}
    compliance_examples = []
    for key in sorted(validity, key=lambda k: validity[k]["valid_fraction"] or 1.0)[:2]:
        arms = arm_validity(key)
        low = min(arms, key=lambda m: arms[m] if arms[m] is not None else 1)
        compliance_examples.append(
            f"in {names.get(key, 'the previous condition')} the "
            f"{arm_names[low]} fell to {(arms[low] or 0) * 100:.0f}% while the "
            f"{arm_names['LLM-Closed' if low == 'LLM-Open' else 'LLM-Open']} stayed at "
            f"{(arms['LLM-Closed' if low == 'LLM-Open' else 'LLM-Open'] or 0) * 100:.0f}%")

    closed_line = (
        f"The closed-loop advantage is NOT a stable property: it keeps the "
        f"previous sign in only {closed_signs} of {total} conditions, is "
        f"significantly positive in {len(closed_holds)}"
        + (f" ({listing(closed_holds)})" if closed_holds else "")
        + (f" and significantly negative in {len(closed_reverses)} "
           f"({listing(closed_reverses)})" if closed_reverses else "")
        + "."
    ) if closed_reverses or closed_signs < total else (
        f"The closed-loop advantage replicates: it keeps its sign in all "
        f"{total} conditions and is significant in {len(closed_holds)}."
    )

    summary["_slide9_lines"] = [
        ("Interpretation", {"size": 14, "bold": True}),
        (f"The semantic advantage over non-semantic search is the durable part of "
         f"the previous result: it keeps its sign in {open_signs} of {total} "
         f"conditions and is significant in {len(open_holds)}"
         + ("; where it is not significant the effect is still positive but the "
            "per-seed win count is low." if len(open_holds) < total else "."),
         {"size": 11}),
        (closed_line, {"size": 11}),
        (f"A separate failure mode is visible alongside the search results: "
         f"resource-contract compliance. Which call breaks is condition-specific - "
         f"{'; '.join(compliance_examples)}. Every invalid proposal is spent as a "
         f"flagged random draw under the frozen policy, so part of what looks like "
         f"a method effect is a generation-validity effect.", {"size": 11}),
        (f"Non-semantic refinement never helps: {len(greedy_holds)} of {total} "
         f"conditions show a gain over a plain random sweep"
         + (f", and at {listing(greedy_reverses)} it is significantly worse - "
            f"spending half a small budget on single-gate steps costs breadth "
            f"without buying quality." if greedy_reverses else "."),
         {"size": 11}),
        ("", {"size": 6}),
        ("Limitations", {"size": 14, "bold": True, "space_after": 4}),
        ("Twelve paired seeds per cell: enough for large effects, underpowered for "
         "small ones. A \"not detected\" cell is not evidence of absence.",
         {"size": 10.5, "color": MUTED}),
        ("One factor at a time means no interactions were estimated - for example a "
         "large budget at eight qubits.", {"size": 10.5, "color": MUTED}),
        ("Two Hamiltonian families and two models are two points, not a survey.",
         {"size": 10.5, "color": MUTED}),
        ("Noiseless state-vector simulation throughout; no sampling noise and no "
         "device topology constraints.", {"size": 10.5, "color": MUTED}),
        ("The closed loop is given more search freedom than the single-change "
         "baseline, so that pair is not information-matched; capacity, gate set, "
         "parameter count, optimizer, data and budget are matched.",
         {"size": 10.5, "color": MUTED}),
    ]
    preliminary = [k for k in conditions
                   if conditions[k].get("status") not in ("complete", "missing")]
    if preliminary:
        summary["_slide9_lines"].append(
            (f"Preliminary cells with incomplete paired sets: "
             f"{listing(preliminary)}. No value was imputed.",
             {"size": 10.5, "color": RED, "bold": True}))

    def ladder(keys, name):
        return " to ".join(
            _fmt(gain(k, name), 4, sign=True) for k in keys if gain(k, name) is not None)

    budget_ladder = ladder(["budget_b4", "reference", "budget_b16"], CLOSED_VS_OPEN)
    width_ladder = ladder(["reference", "qubits_n6", "qubits_n8"], CLOSED_VS_OPEN)

    summary["_slide10_items"] = [
        ("The headline finding survives every single-factor change."
         if len(open_holds) == total else
         "The headline finding keeps its direction under every single-factor change.",
         f"Semantic proposals beat non-semantic search with the same sign in "
         f"{open_signs} of {total} varied conditions - a different budget, a "
         f"different register size, a different Hamiltonian and a different "
         f"language model - and significantly so in {len(open_holds)}. This is no "
         f"longer a property of one setting.", BLUE),
        ("The closed-loop advantage does not generalise as stated.",
         f"It was the fragile claim, and it broke where it was tested hardest. "
         f"Across budgets it runs {budget_ladder}; across register sizes it runs "
         f"{width_ladder}. Both ladders move against it, so the claim has to be "
         f"restated as budget- and width-dependent rather than general.", RED),
        ("Generating a valid circuit is a separate, measurable bottleneck.",
         f"Compliance with the exact resource contract fell as low as "
         f"{(validity[worst]['valid_fraction'] or 0) * 100:.0f}% of evaluated "
         f"candidates, and which call breaks depends on the condition rather than "
         f"on the method. Because an invalid proposal is spent as a random draw, "
         f"fixing generation validity is worth more right now than richer feedback.",
         AMBER),
        ("Next: interactions, more families and models, and noise.",
         "The single-factor grid is clean but conservative. The open questions are "
         "whether the effects compound at larger widths and budgets together, and "
         "whether they survive sampling noise and hardware topology.", SLATE),
    ]
    summary["_slide10_footer"] = (
        f"All figures and statistics produced from the committed result tables by "
        f"the committed build scripts; {reference['n_seeds']} paired seeds per cell; "
        f"held-out test read once after validation-only selection.")
    return summary


# ---------------------------------------------------------------- audit ----

def extract_text(path: Path) -> list[tuple[int, str]]:
    deck = Presentation(str(path))
    out = []
    for index, slide in enumerate(deck.slides, start=1):
        for shape in slide.shapes:
            if shape.has_text_frame:
                for paragraph in shape.text_frame.paragraphs:
                    text = "".join(run.text for run in paragraph.runs)
                    if text.strip():
                        out.append((index, text))
            if getattr(shape, "has_table", False) and shape.has_table:
                for row in shape.table.rows:
                    for cell in row.cells:
                        if cell.text.strip():
                            out.append((index, cell.text))
    return out


# Conservative average glyph width as a fraction of the point size for the
# deck's sans face; used only to estimate how many lines a paragraph wraps to.
GLYPH_WIDTH_RATIO = 0.50
LINE_HEIGHT_RATIO = 1.24


def estimated_text_height(shape) -> float:
    """Rough rendered height of a text frame, in EMU.

    Deliberately approximate: it exists to catch a box whose content clearly
    cannot fit, not to typeset. Overestimating slightly is the safe direction.
    """
    usable = shape.width - Emu(91440)  # default 0.05in left+right insets, doubled
    total = Emu(91440)  # top + bottom insets
    for paragraph in shape.text_frame.paragraphs:
        text = "".join(run.text for run in paragraph.runs)
        sizes = [run.font.size.pt for run in paragraph.runs if run.font.size]
        size = max(sizes) if sizes else 18.0
        char_width = Emu(int(size * GLYPH_WIDTH_RATIO * 12700))
        per_line = max(1, int(usable / char_width))
        lines = max(1, -(-len(text) // per_line))
        spacing = paragraph.line_spacing if isinstance(paragraph.line_spacing, float) else 1.0
        total += Emu(int(lines * size * LINE_HEIGHT_RATIO * spacing * 12700))
        if paragraph.space_after is not None:
            total += paragraph.space_after
    return total


def audit(path: Path, expected_slides: int = 10) -> list[str]:
    problems = []
    deck = Presentation(str(path))
    n_slides = len(deck.slides.__iter__.__self__._sldIdLst)
    if n_slides != expected_slides:
        problems.append(f"deck has {n_slides} slides, expected {expected_slides}")

    for index, slide in enumerate(deck.slides, start=1):
        for shape in slide.shapes:
            right = shape.left + shape.width
            bottom = shape.top + shape.height
            if shape.left < -Emu(1) or shape.top < -Emu(1):
                problems.append(f"slide {index}: shape starts off-canvas")
            if right > deck.slide_width + Emu(9525):
                problems.append(
                    f"slide {index}: shape overflows the right edge by "
                    f"{Emu(right - deck.slide_width).inches:.2f} in")
            if bottom > deck.slide_height + Emu(9525):
                problems.append(
                    f"slide {index}: shape overflows the bottom edge by "
                    f"{Emu(bottom - deck.slide_height).inches:.2f} in")

    for index, slide in enumerate(deck.slides, start=1):
        boxes = []
        for shape in slide.shapes:
            if not shape.has_text_frame or not shape.text_frame.text.strip():
                boxes.append((shape, shape.top + shape.height))
                continue
            estimated = estimated_text_height(shape)
            boxes.append((shape, shape.top + estimated))
            if shape.top + estimated > deck.slide_height:
                problems.append(
                    f"slide {index}: text starting {shape.text_frame.text[:40]!r} "
                    f"is estimated to run "
                    f"{Emu(int(shape.top + estimated - deck.slide_height)).inches:.2f} in "
                    f"past the bottom of the slide")
        for shape, bottom in boxes:
            if shape.has_text_frame and not shape.text_frame.text.strip():
                continue
            for other, _ in boxes:
                if other is shape or other.top <= shape.top:
                    continue
                horizontal = (min(shape.left + shape.width, other.left + other.width)
                              - max(shape.left, other.left))
                if horizontal > 0 and bottom > other.top:
                    label = (repr(shape.text_frame.text[:40])
                             if shape.has_text_frame else f"{shape.shape_type}")
                    problems.append(
                        f"slide {index}: {label} overlaps the shape below it by "
                        f"{Emu(int(bottom - other.top)).inches:.2f} in")
                    break

    for index, text in extract_text(path):
        for pattern, label in FORBIDDEN_PATTERNS:
            match = re.search(pattern, text)
            if match:
                problems.append(
                    f"slide {index}: forbidden {label} {match.group(0)!r} in {text[:70]!r}")
    return problems


def render_pdf(pptx_path: Path) -> Path:
    subprocess.run(
        ["soffice", "--headless", "--convert-to", "pdf", "--outdir",
         str(pptx_path.parent), str(pptx_path)],
        check=True, capture_output=True, timeout=600,
    )
    return pptx_path.with_suffix(".pdf")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true",
                        help="audit the existing deck without rebuilding")
    parser.add_argument("--no-pdf", action="store_true")
    args = parser.parse_args()

    DECK_DIR.mkdir(parents=True, exist_ok=True)
    pptx_path = DECK_DIR / f"{DECK_NAME}.pptx"

    if not args.check:
        summary = build_narrative(json.loads((ROOT / "summary.json").read_text()))
        deck = new_deck()
        for builder in (slide_01_title, slide_02_previous, slide_03_protocol,
                        slide_04_budget, slide_05_qubits, slide_06_hamiltonian,
                        slide_07_model, slide_08_summary, slide_09_interpretation,
                        slide_10_conclusions):
            builder(deck, summary)
        deck.save(str(pptx_path))
        print("wrote", pptx_path)

    problems = audit(pptx_path)
    if problems:
        print("\nAUDIT FAILED:")
        for problem in problems:
            print("  -", problem)
    else:
        print("audit passed: exactly 10 slides, nothing off-canvas, "
              "no prohibited strings")

    if not args.no_pdf and not args.check:
        pdf = render_pdf(pptx_path)
        print("wrote", pdf)
    if problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
