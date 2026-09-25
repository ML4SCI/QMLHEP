"""Build the 10-slide "tested one-factor slices" robustness deck (.pptx + .pdf).

Lean, large-type version: one visual idea per slide, body text at 12 pt or
larger, and a dedicated synthesis slide that states in three columns where
the closed loop beats the open batch, where no difference is detected, and
where the open batch wins. Content is the completed robustness study only;
every number is read from the committed result tables and no experiment is
run here.

Model tiers are named Low (gpt-4.1-mini) and High (gpt-5.4-mini); groups are
always ordered Random -> Greedy -> Low -> High; LLM-Open / LLM-Closed remain
the search-workflow names.

Usage:
  python scripts/qae/build_qae_robustness_slides.py          # build + PDF
  python scripts/qae/build_qae_robustness_slides.py --check  # audit only
"""
import argparse
import importlib.util
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
DECK_DIR = ROOT / "deck"
DECK_NAME = "20260904_GSoC"

_spec = importlib.util.spec_from_file_location(
    "qae_deck_common", Path("scripts/qae/build_qae_robustness_deck.py"))
_common = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_common)
geometry_audit = _common.audit
extract_text = _common.extract_text

FORBIDDEN_PATTERNS = _common.FORBIDDEN_PATTERNS + [
    (r"\bModel [AB]\b", "model letter label"),
    (r"\b(Open|Closed) [AB]\b", "model letter label"),
    (r"\b(reference|alternative) model\b", "model role label"),
    (r"(?i)across the full grid", "full-grid claim"),
    (r"(?i)for all qubit.budget combinations", "full-grid claim"),
    (r"(?i)\b(model|Hamiltonian)-independent\b", "over-general claim"),
]

SLIDE_W, SLIDE_H = Inches(13.333), Inches(7.5)

NAVY = RGBColor(0x1F, 0x2A, 0x44)
KICKER = RGBColor(0xC5, 0x22, 0x1F)
INK = RGBColor(0x20, 0x21, 0x24)
MUTED = RGBColor(0x5F, 0x63, 0x68)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
BLUE, BLUE_FILL = RGBColor(0x1A, 0x73, 0xE8), RGBColor(0xE8, 0xF0, 0xFE)
ORANGE, ORANGE_FILL = RGBColor(0xE8, 0x71, 0x0A), RGBColor(0xFE, 0xF3, 0xE0)
GREEN, GREEN_FILL = RGBColor(0x18, 0x80, 0x38), RGBColor(0xE6, 0xF4, 0xEA)
RED, RED_FILL = RGBColor(0xD9, 0x30, 0x25), RGBColor(0xFC, 0xE8, 0xE6)
AMBER, AMBER_FILL = RGBColor(0xF9, 0xAB, 0x00), RGBColor(0xFE, 0xF7, 0xE0)
GREY, GREY_FILL = RGBColor(0x9A, 0xA0, 0xA6), RGBColor(0xF1, 0xF3, 0xF4)
FONT = "Helvetica Neue"

LOW_MODEL, HIGH_MODEL = "gpt-4.1-mini", "gpt-5.4-mini"
OPEN_RANDOM = "LLM-Open_minus_Random"
CLOSED_OPEN = "LLM-Closed_minus_LLM-Open"
CLOSED_RANDOM = "LLM-Closed_minus_Random"
GREEDY_RANDOM = "Greedy_minus_Random"

BODY = 12        # minimum body size on content slides
SMALL = 10.5     # captions and footers
FOOTER_TOP = Inches(7.02)
TRANSITION_TOP = Inches(6.5)
TABLE_ROW_HEIGHT = Inches(0.4)
CHIP_HEIGHT = Inches(0.4)

# ------------------------------------------------------------ primitives ---


def _flat(shape):
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


def box(slide, left, top, width, height, fill, border, border_pt=1.25):
    shape = _flat(slide.shapes.add_shape(5, left, top, width, height))
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    shape.line.color.rgb = border
    shape.line.width = Pt(border_pt)
    return shape


def line(slide, x1, y1, x2, y2, color=GREY, width_pt=1.5):
    connector = slide.shapes.add_connector(1, x1, y1, x2, y2)
    connector.line.color.rgb = color
    connector.line.width = Pt(width_pt)
    return connector


def textbox(slide, left, top, width, height, align=PP_ALIGN.LEFT):
    frame = slide.shapes.add_textbox(left, top, width, height).text_frame
    frame.word_wrap = True
    frame.paragraphs[0].alignment = align
    return frame


def write(frame, lines, *, size=BODY, color=INK, bold=False, space_after=5,
          line_spacing=1.0, align=None):
    """lines: str | (str, opts) | [ (run_text, run_opts), ... ] per paragraph."""
    first = True
    for entry in lines:
        if isinstance(entry, list):
            runs, opts = entry, {}
        else:
            text, opts = (entry if isinstance(entry, tuple) else (entry, {}))
            runs = [(text, opts)]
        paragraph = frame.paragraphs[0] if first else frame.add_paragraph()
        first = False
        paragraph.space_after = Pt(opts.get("space_after", space_after))
        paragraph.line_spacing = opts.get("line_spacing", line_spacing)
        if align is not None or "align" in opts:
            paragraph.alignment = opts.get("align", align)
        for text, ropts in runs:
            run = paragraph.add_run()
            run.text = text
            font = run.font
            font.name = FONT
            font.size = Pt(ropts.get("size", opts.get("size", size)))
            font.bold = ropts.get("bold", opts.get("bold", bold))
            font.italic = ropts.get("italic", False)
            font.color.rgb = ropts.get("color", opts.get("color", color))
    return frame


def header(slide, kicker, title, subtitle=None):
    frame = textbox(slide, Inches(0.55), Inches(0.24), Inches(12.2), Inches(0.28))
    write(frame, [(kicker.upper(), {"size": 11.5, "bold": True, "color": KICKER})],
          space_after=0)
    frame = textbox(slide, Inches(0.55), Inches(0.56), Inches(12.2), Inches(0.52))
    write(frame, [(title, {"size": 26, "bold": True, "color": NAVY})], space_after=0)
    if subtitle:
        frame = textbox(slide, Inches(0.55), Inches(1.12), Inches(12.2), Inches(0.38))
        write(frame, [(subtitle, {"size": 12, "color": MUTED})], space_after=0)


def footer(slide, text, top=FOOTER_TOP):
    frame = textbox(slide, Inches(0.55), top, Inches(12.2), Inches(0.33))
    write(frame, [(text, {"size": 9.5, "color": MUTED})], space_after=0)


def transition(slide, text, top=TRANSITION_TOP):
    rect(slide, Inches(0.55), top, Inches(0.08), Inches(0.4), KICKER)
    frame = textbox(slide, Inches(0.75), top - Inches(0.01), Inches(12.0), Inches(0.44))
    write(frame, [[("Next:  ", {"bold": True, "color": KICKER}),
                   (text, {"color": INK})]], size=12.5, space_after=0)


def picture(slide, name, left, top, width=None, height=None):
    path = FIGURES / f"{name}.png"
    if not path.exists():
        raise SystemExit(f"missing figure {path}; run the slice figure builder first")
    if width is not None:
        return slide.shapes.add_picture(str(path), left, top, width=width)
    return slide.shapes.add_picture(str(path), left, top, height=height)


def titled_box(slide, left, top, width, height, fill, border, title, lines,
               title_color=None, title_size=13.5, body_size=BODY, space_after=4):
    box(slide, left, top, width, height, fill, border)
    frame = textbox(slide, left + Inches(0.16), top + Inches(0.1),
                    width - Inches(0.32), height - Inches(0.2))
    write(frame, [(title, {"size": title_size, "bold": True,
                           "color": title_color or border, "space_after": 4})]
          + [(entry if isinstance(entry, (tuple, list)) else (entry, {"size": body_size}))
             for entry in lines],
          size=body_size, space_after=space_after)
    return frame


def chip(slide, left, top, width, text, fill, border, color=INK, size=11.5, bold=False,
         height=CHIP_HEIGHT):
    shape = box(slide, left, top, width, height, fill, border, 1.0)
    frame = shape.text_frame
    frame.margin_left = frame.margin_right = Emu(0)
    frame.margin_top = frame.margin_bottom = Emu(0)
    frame.word_wrap = True
    paragraph = frame.paragraphs[0]
    paragraph.alignment = PP_ALIGN.CENTER
    run = paragraph.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.name = FONT
    run.font.color.rgb = color
    return shape


def table(slide, rows, left, top, col_widths, row_height=TABLE_ROW_HEIGHT,
          header_size=12, body_size=11.5, colors=None):
    width = sum(col_widths, Emu(0))
    shape = slide.shapes.add_table(len(rows), len(rows[0]), left, top, width,
                                   row_height * len(rows)).table
    for index, w in enumerate(col_widths):
        shape.columns[index].width = w
    for r, row in enumerate(rows):
        shape.rows[r].height = row_height
        for c, value in enumerate(row):
            cell = shape.cell(r, c)
            cell.text = str(value)
            cell.margin_left = cell.margin_right = Inches(0.08)
            cell.margin_top = cell.margin_bottom = Emu(19050)
            for paragraph in cell.text_frame.paragraphs:
                paragraph.alignment = PP_ALIGN.LEFT if c == 0 else PP_ALIGN.CENTER
                for run in paragraph.runs:
                    run.font.size = Pt(header_size if r == 0 else body_size)
                    run.font.bold = r == 0 or c == 0
                    run.font.name = FONT
                    run.font.color.rgb = INK
                    if colors and (r, c) in colors:
                        run.font.color.rgb = colors[(r, c)]
                        run.font.bold = True
            cell.fill.solid()
            cell.fill.fore_color.rgb = (RGBColor(0xE8, 0xEA, 0xED) if r == 0 else
                                        (WHITE if r % 2 else RGBColor(0xF8, 0xF9, 0xFA)))
    return shape


def bullet(text_runs, size=BODY):
    """A bullet paragraph made of (text, opts) runs."""
    return [("•  ", {"color": MUTED, "size": size})] + [
        (t, dict(o, size=o.get("size", size))) for t, o in text_runs]


# --------------------------------------------------------------- numbers ---


def _fmt(value, digits=3, sign=True):
    return f"{value:+.{digits}f}" if sign else f"{value:.{digits}f}"


def _p(value):
    return "p < 0.001" if value < 0.001 else f"p = {value:.3f}"


def stat(summary, key, contrast):
    return summary["conditions"][key]["contrasts"][contrast]


def mean_of(summary, key, method):
    return summary["conditions"][key]["per_method"][method]["mean"]


def validity(key, method):
    entry = json.loads((ROOT / key / "proposal_quality.json").read_text())[method]
    return 1 - entry["fallback_evaluations"] / entry["evaluations"]


def short(summary, key, contrast):
    entry = stat(summary, key, contrast)
    return (f"{_fmt(entry['mean_paired_gain'])} ({entry['wins_b']}/12, "
            f"{_p(entry['wilcoxon_exact_two_sided_p'])})")


def gain(summary, key, contrast):
    return stat(summary, key, contrast)["mean_paired_gain"]


def verdict(summary, key, contrast):
    """Evidence-calibrated word for one paired contrast in one condition."""
    entry = stat(summary, key, contrast)
    if entry["wilcoxon_exact_two_sided_p"] >= 0.05:
        return "not detected"
    return "robust" if entry["mean_paired_gain"] > 0 else "reversed"


# Every tested condition, with the label used on the synthesis slide.
CONDITIONS = [
    ("reference", "Baseline: B = 8"),
    ("budget_b4", "B = 4"),
    ("budget_b16", "B = 16"),
    ("qubits_n6", "6 qubits"),
    ("qubits_n8", "8 qubits"),
    ("hamiltonian_xxz", "XXZ anchor"),
    ("model_alt", "Low tier"),
]


# ---------------------------------------------------------------- slides ---


def slide_01_question(deck, summary):
    slide = deck.slides.add_slide(deck.slide_layouts[6])
    rect(slide, Emu(0), Emu(0), SLIDE_W, SLIDE_H, NAVY)
    frame = textbox(slide, Inches(0.9), Inches(0.95), Inches(11.6), Inches(1.9))
    write(frame, [("Which parts of the previous QAE result",
                   {"size": 38, "bold": True, "color": WHITE, "space_after": 0}),
                  ("survive one-factor changes?",
                   {"size": 38, "bold": True, "color": WHITE})], space_after=6)
    frame = textbox(slide, Inches(0.9), Inches(2.9), Inches(11.6), Inches(0.7))
    write(frame, [("A quantum autoencoder (QAE) robustness study — a follow-up to last "
                   "meeting's architecture-search result",
                   {"size": 17, "color": RGBColor(0xC8, 0xD4, 0xEC)})])
    box(slide, Inches(0.9), Inches(3.75), Inches(11.6), Inches(2.0),
        RGBColor(0x2A, 0x3A, 0x5C), RGBColor(0x5B, 0x8D, 0xEF))
    frame = textbox(slide, Inches(1.15), Inches(3.88), Inches(11.1), Inches(1.8))
    write(frame, [
        [("Baseline finding:  ", {"bold": True, "color": RGBColor(0xF9, 0xAB, 0x00)}),
         ("physics-informed language-model proposals beat random and greedy circuit "
          "search by a wide margin; a closed feedback loop added a small extra gain.",
          {"color": WHITE})],
        [("Four factors, one at a time:  ",
          {"bold": True, "color": RGBColor(0xF9, 0xAB, 0x00)}),
         ("evaluation budget · qubit count · Hamiltonian · language model.",
          {"color": WHITE})],
        [("This is a tested one-factor-slice study, not a complete factorial grid.",
          {"bold": True, "color": WHITE})],
    ], size=15, space_after=7)
    frame = textbox(slide, Inches(0.9), Inches(6.0), Inches(11.5), Inches(1.0))
    write(frame, [("Tomoya Hatanaka", {"size": 15, "bold": True, "color": WHITE}),
                  ("ML4SCI / Google Summer of Code 2026  ·  September 4, 2026",
                   {"size": 12, "color": RGBColor(0xC8, 0xD4, 0xEC)})], space_after=2)


def slide_02_baseline(deck, summary):
    slide = deck.slides.add_slide(deck.slide_layouts[6])
    header(slide, "1 · Previous baseline",
           "Where we left off: the previous baseline result",
           "Quantum autoencoder (QAE): compress ground states into fewer qubits  ·  "
           "4 qubits · Ising chain · B = 8 · High model tier · 12 paired seeds")
    flow = [
        ("Input |ψ(h)⟩", "ground state of the 4-qubit Ising chain\n"
         "H = −Σ Zᵢ Zᵢ₊₁ − h Σ Xᵢ, h ∈ [0.2, 2.0]", BLUE_FILL, BLUE),
        ("Encoder Uθ", "the circuit we search\n(16 gates, angles trained)", ORANGE_FILL, ORANGE),
        ("Latent qubits q0, q1", "the 2 qubits we keep", GREEN_FILL, GREEN),
        ("Trash qubits q2, q3", "should end in |00⟩ for every input", RED_FILL, RED),
    ]
    x = 0.55
    widths = [3.7, 2.55, 2.5, 2.7]
    for (title, body, fill, border), w in zip(flow, widths, strict=True):
        titled_box(slide, Inches(x), Inches(1.6), Inches(w), Inches(0.95), fill, border,
                   title, [(body, {"size": 10.5})], title_size=12, body_size=10.5,
                   space_after=0)
        if w != widths[-1]:
            frame = textbox(slide, Inches(x + w + 0.02), Inches(1.85), Inches(0.3), Inches(0.4))
            write(frame, [("→", {"size": 16, "color": MUTED})], space_after=0)
        x += w + 0.3
    box(slide, Inches(0.55), Inches(2.7), Inches(12.2), Inches(0.5), BLUE_FILL, BLUE)
    frame = textbox(slide, Inches(0.7), Inches(2.76), Inches(11.95), Inches(0.4))
    write(frame, [[("Trash fidelity  ", {"bold": True, "color": BLUE}),
                   ("F_trash = P(trash qubits q2 q3 = 00)  —  ", {}),
                   ("higher is better", {"bold": True, "color": GREEN}),
                   ("  ·  measured on a held-out test set of 64 unseen h values, "
                    "read once after selection", {})]], size=12, space_after=0)
    picture(slide, "baseline_groups", Inches(0.45), Inches(3.35), height=Inches(3.05))
    box(slide, Inches(6.6), Inches(3.35), Inches(6.15), Inches(1.75), GREY_FILL, GREY)
    frame = textbox(slide, Inches(6.75), Inches(3.4), Inches(5.9), Inches(1.68))
    write(frame, [
        [("Four search methods  ", {"bold": True}),
         ("(budget B = candidate circuits trained and evaluated per seed)",
          {"color": MUTED, "size": 10.5})],
        [("Random  ", {"bold": True, "color": GREY}), ("B uniformly sampled circuits", {})],
        [("Greedy  ", {"bold": True, "color": AMBER}),
         ("B/2 random starts + B/2 single-change refinements of the best", {})],
        [("LLM-Open  ", {"bold": True, "color": BLUE}),
         ("open-loop: all semantic proposals before any score is seen", {})],
        [("LLM-Closed  ", {"bold": True, "color": RED}),
         ("closed-loop: semantic starts + free redesigns of the current best, "
          "guided by validation scores", {})],
    ], size=11.5, space_after=2)
    box(slide, Inches(6.6), Inches(5.2), Inches(6.15), Inches(1.15), AMBER_FILL, AMBER)
    frame = textbox(slide, Inches(6.75), Inches(5.26), Inches(5.9), Inches(1.05))
    write(frame, [
        [("Two claims to stress-test", {"bold": True, "color": ORANGE})],
        [("1  Semantic proposals beat non-semantic search:  ", {"bold": True, "color": BLUE}),
         ("Open − Random " + short(summary, "reference", OPEN_RANDOM), {})],
        [("2  Closed-loop redesign adds a small extra gain:  ", {"bold": True, "color": RED}),
         ("Closed − Open " + short(summary, "reference", CLOSED_OPEN), {})],
    ], size=11.5, space_after=2)
    transition(slide, "do these two claims survive when exactly one condition changes?")
    footer(slide, "Dots = the 12 paired seeds (12 independent draws of training/validation "
                  "data shared by all methods); marker = mean; bar = bootstrap 95% "
                  "confidence interval (CI); p = exact paired Wilcoxon test. "
                  f"High tier = {HIGH_MODEL}.")


def slide_03_design(deck, summary):
    slide = deck.slides.add_slide(deck.slide_layouts[6])
    header(slide, "2 · Controlled robustness design",
           "One factor changes; everything else remains fixed",
           "Four controlled changes leave the baseline; each moves exactly one declared "
           "factor and holds the other three")
    box(slide, Inches(0.55), Inches(2.45), Inches(3.2), Inches(1.6), NAVY, NAVY)
    frame = textbox(slide, Inches(0.72), Inches(2.52), Inches(2.9), Inches(1.5))
    write(frame, [("BASELINE", {"size": 12, "bold": True, "color": AMBER}),
                  ("4 qubits · Ising · B = 8", {"size": 14, "bold": True, "color": WHITE}),
                  ("High model tier", {"size": 14, "bold": True, "color": WHITE}),
                  ("12 paired seeds", {"size": 12, "color": RGBColor(0xC8, 0xD4, 0xEC)})],
          space_after=2)
    branches = [
        ("Evaluation budget B", ["4", "8", "16"], 1, BLUE, BLUE_FILL),
        ("Qubit count", ["4", "6", "8"], 0, GREEN, GREEN_FILL),
        ("Hamiltonian", ["Ising", "XXZ"], 0, AMBER, AMBER_FILL),
        ("Language-model tier", ["Low", "High"], 1, RED, RED_FILL),
    ]
    for i, (name, levels, base_index, color, fill) in enumerate(branches):
        top = Inches(1.62 + i * 0.86)
        line(slide, Inches(3.75), Inches(3.25), Inches(4.45), top + Inches(0.36), color, 1.75)
        box(slide, Inches(4.45), top, Inches(3.7), Inches(0.76), fill, color)
        frame = textbox(slide, Inches(4.58), top + Inches(0.02), Inches(3.4), Inches(0.28))
        write(frame, [(name, {"size": 11.5, "bold": True, "color": color})], space_after=0)
        x = 4.6
        for j, level in enumerate(levels):
            is_base = j == base_index
            chip(slide, Inches(x), top + Inches(0.34), Inches(0.8), level,
                 NAVY if is_base else WHITE, NAVY if is_base else color,
                 WHITE if is_base else INK, size=11, bold=is_base, height=Inches(0.34))
            x += 0.9
    frame = textbox(slide, Inches(4.45), Inches(5.02), Inches(3.9), Inches(0.36))
    write(frame, [[("dark chip", {"bold": True, "color": NAVY}),
                   (" = baseline level · Low = " + LOW_MODEL + " · High = " + HIGH_MODEL,
                    {"color": MUTED})]], size=10, space_after=0)
    titled_box(slide, Inches(8.45), Inches(1.62), Inches(4.3), Inches(3.35), GREEN_FILL, GREEN,
               "FROZEN in every condition", [
        "•  prompt wording and output schema",
        "•  the four search methods",
        "•  circuit rule: 3 rotations + 1 CNOT (controlled-NOT) per qubit",
        "•  trainer, optimizer, data splits",
        "•  the same 12 paired seeds",
        "•  validation-only selection; held-out test read once",
        "•  statistical analysis",
    ], body_size=BODY, space_after=3)
    box(slide, Inches(0.55), Inches(5.48), Inches(12.2), Inches(0.92), AMBER_FILL, AMBER)
    frame = textbox(slide, Inches(0.72), Inches(5.54), Inches(11.9), Inches(0.82))
    write(frame, [
        [("No interaction condition was tested.  ", {"bold": True, "color": ORANGE}),
         ("Each change returns to the baseline before the next factor moves — e.g. a large "
          "budget at 8 qubits was never run.  XXZ = the second Hamiltonian family, an XXZ "
          "Heisenberg chain H = Σ (Xᵢ Xᵢ₊₁ + Yᵢ Yᵢ₊₁ + Δ Zᵢ Zᵢ₊₁), Δ ∈ [0.2, 2.0], "
          "declared before the run.", {})],
    ], size=BODY, space_after=0)
    transition(slide, "because only one factor moves at a time, read the results as four "
                      "tested slices, not as a full grid.")
    footer(slide, "An automated check refuses any condition whose frozen components differ "
                  "from the baseline's or that moves more than one factor; the baseline "
                  "cell is reused, not re-run.")


def slide_04_reading(deck, summary):
    slide = deck.slides.add_slide(deck.slide_layouts[6])
    header(slide, "3 · What was actually tested, and how to read it",
           "Four tested slices, one baseline, no empty panels",
           "Every result slide shows one of these four rows; untested combinations are "
           "omitted rather than displayed as empty panels")
    slices = [
        ("Budget slice", ["B = 4", "B = 8", "B = 16"], 1, "4 qubits · Ising · High fixed",
         BLUE, BLUE_FILL),
        ("Qubit slice", ["4 qubits", "6 qubits", "8 qubits"], 0,
         "B = 8 · Ising · High fixed", GREEN, GREEN_FILL),
        ("Hamiltonian slice", ["Ising", "XXZ"], 0, "4 qubits · B = 8 · High fixed",
         AMBER, AMBER_FILL),
        ("Model slice", ["Low", "High"], 1, "4 qubits · Ising · B = 8 fixed",
         RED, RED_FILL),
    ]
    for i, (name, levels, base_index, fixed, color, fill) in enumerate(slices):
        top = Inches(1.62 + i * 0.74)
        box(slide, Inches(0.55), top, Inches(7.55), Inches(0.64), fill, color, 1.0)
        frame = textbox(slide, Inches(0.7), top + Inches(0.14), Inches(1.85), Inches(0.4))
        write(frame, [(name, {"size": 12.5, "bold": True, "color": color})], space_after=0)
        x = 2.55
        for j, level in enumerate(levels):
            is_base = j == base_index
            chip(slide, Inches(x), top + Inches(0.12), Inches(1.05), level,
                 NAVY if is_base else WHITE, NAVY if is_base else color,
                 WHITE if is_base else INK, size=11.5, bold=is_base)
            x += 1.13
            if j < len(levels) - 1:
                frame = textbox(slide, Inches(x - 0.14), top + Inches(0.12), Inches(0.24),
                                Inches(0.36))
                write(frame, [("→", {"size": 12, "color": MUTED})], space_after=0,
                      align=PP_ALIGN.CENTER)
                x += 0.12
        frame = textbox(slide, Inches(6.2), top + Inches(0.12), Inches(1.9), Inches(0.4))
        write(frame, [(fixed, {"size": 10, "color": MUTED})], space_after=0)
    frame = textbox(slide, Inches(0.55), Inches(4.62), Inches(7.55), Inches(0.35))
    write(frame, [[("dark chip", {"bold": True, "color": NAVY}),
                   (" = the baseline level, present in every slice.", {"color": MUTED})]],
          size=11, space_after=0)
    titled_box(slide, Inches(8.35), Inches(1.62), Inches(4.4), Inches(1.35), RED_FILL, RED,
               "Model tiers (defined here once)", [
        [("Low", {"bold": True, "color": RED}), (f" = {LOW_MODEL}     ", {}),
         ("High", {"bold": True, "color": RED}), (f" = {HIGH_MODEL}", {})],
        ("Fixed capability-tier labels, not the observed score. Open / Closed name the "
         "search workflow; Low / High name the model.", {"size": 11, "color": MUTED}),
    ], body_size=BODY)
    titled_box(slide, Inches(8.35), Inches(3.1), Inches(4.4), Inches(2.5), GREY_FILL, GREY,
               "Reading every result figure", [
        "•  dots = 12 paired seeds · marker = mean · bar = 95% CI",
        "•  higher trash fidelity is better",
        "•  groups always: Random → Greedy → Low → High",
        "•  LLM-Open = blue triangle, LLM-Closed = red diamond",
        "•  High = filled marker, Low = hollow marker",
        "•  thick frame / shaded band = the baseline",
    ], title_color=NAVY, body_size=11.5, space_after=3)
    box(slide, Inches(0.55), Inches(5.05), Inches(7.55), Inches(1.3), AMBER_FILL, AMBER)
    frame = textbox(slide, Inches(0.72), Inches(5.12), Inches(7.25), Inches(1.2))
    write(frame, [
        [("Low was run only in the model slice.  ", {"bold": True, "color": ORANGE}),
         ("The other slices show Random, Greedy and High only; no Low value is invented.",
          {})],
        [("Untested combinations are omitted rather than displayed as empty panels.",
          {"bold": True})],
    ], size=BODY, space_after=4)
    transition(slide, "the first slice changes the amount of search — the evaluation "
                      "budget B.")
    footer(slide, "Random and Greedy do not depend on the language model, so each is one "
                  "result per condition; Low and High each hold an Open and a Closed result.")


def _result_slide(deck, kicker, title, subtitle, figure, figure_height, bullets,
                  transition_text, footer_text):
    slide = deck.slides.add_slide(deck.slide_layouts[6])
    header(slide, kicker, title, subtitle)
    pic = picture(slide, figure, Inches(0.5), Inches(1.55), height=figure_height)
    left = Inches(0.5) + pic.width + Inches(0.3)
    frame = textbox(slide, left, Inches(1.6), SLIDE_W - left - Inches(0.55), Inches(4.8))
    write(frame, bullets, size=BODY, space_after=9)
    transition(slide, transition_text)
    footer(slide, footer_text)
    return slide


def slide_05_budget(deck, summary):
    co = {k: gain(summary, k, CLOSED_OPEN) for k in ("budget_b4", "reference", "budget_b16")}
    return _result_slide(
        deck, "4 · Budget slice", "Budget changes the value of feedback",
        "B = 4 → 8 → 16 with 4 qubits, Ising chain and High tier fixed; B = 8 (thick "
        "frame) is the previous baseline",
        "slice_budget", Inches(4.45), [
            [("Semantic proposals stay above non-semantic search at every budget.",
              {"bold": True, "color": BLUE})],
            [("Open − Random is positive at all three budgets; at B = 4 it is not "
              f"detected ({short(summary, 'budget_b4', OPEN_RANDOM)}).", {})],
            [("The Closed − Open advantage shrinks as the budget grows:",
              {"bold": True, "color": RED})],
            [(f"{_fmt(co['budget_b4'])} at B = 4  →  {_fmt(co['reference'])} at B = 8  →  "
              f"{_fmt(co['budget_b16'])} at B = 16 (not detected).", {})],
            [("Free redesign pays most when the open batch is thinnest; a wide open batch "
              "buys the same breadth without feedback.", {"color": MUTED})],
        ],
        "budget changes the amount of search; next we ask whether register width "
        "changes the conclusion.",
        "Same trainer, prompts, seeds and selection rule at every budget; adaptive methods "
        "always split B as B/2 exploration + B/2 refinement (2+2, 4+4, 8+8).")


def slide_06_qubits(deck, summary):
    keys = ("reference", "qubits_n6", "qubits_n8")
    orr = [gain(summary, k, OPEN_RANDOM) for k in keys]
    co = [gain(summary, k, CLOSED_OPEN) for k in keys]
    v = [validity(k, "LLM-Closed") for k in keys]
    return _result_slide(
        deck, "5 · Qubit-count slice", "The semantic advantage grows as the register widens",
        "4 → 6 → 8 qubits with B = 8, Ising chain and High tier fixed; 4 qubits (thick "
        "frame) is the previous baseline",
        "slice_qubits", Inches(4.45), [
            [("1  Absolute fidelity falls as the task widens", {"bold": True, "color": NAVY})],
            [(f"Random {mean_of(summary, 'reference', 'Random'):.2f} → "
              f"{mean_of(summary, 'qubits_n8', 'Random'):.2f};  LLM-Open "
              f"{mean_of(summary, 'reference', 'LLM-Open'):.2f} → "
              f"{mean_of(summary, 'qubits_n8', 'LLM-Open'):.2f}.", {})],
            [("2  Open stays strong; Closed falls below Open", {"bold": True, "color": BLUE})],
            [(f"Open − Random  {_fmt(orr[0])} → {_fmt(orr[1])} → {_fmt(orr[2])}, "
              "significant at every width.", {})],
            [(f"Closed − Open  {_fmt(co[0])} → {_fmt(co[1])} → {_fmt(co[2])}, "
              "reversed at 6 and 8 qubits.", {})],
            [("Caveat: ", {"bold": True, "color": MUTED}),
             (f"closed-loop proposals meeting the gate-count contract (the required 3 "
              f"rotations + 1 CNOT per qubit) fell {v[0] * 100:.0f}% → {v[1] * 100:.0f}% → "
              f"{v[2] * 100:.0f}%; invalid proposals become random draws.",
              {"color": MUTED})],
            [("Tested for the Ising chain and the High tier only.",
              {"bold": True, "color": KICKER})],
        ],
        "width changes task difficulty; next we change the physical state family at the "
        "baseline width.",
        "Same B = 8, prompts (qubit count substituted), trainer and seeds at every width; "
        "latent and trash registers are always half and half.")


def slide_07_hamiltonian(deck, summary):
    x = "hamiltonian_xxz"
    return _result_slide(
        deck, "6 · Hamiltonian slice",
        "The broad ordering survives at the tested XXZ anchor",
        "Ising → XXZ with 4 qubits, B = 8 and High tier fixed; the Ising chain (thick "
        "frame) is the previous baseline",
        "slice_hamiltonian", Inches(4.45), [
            [("The same broad semantic advantage", {"bold": True, "color": BLUE})],
            [(f"Both LLM methods have higher means than Random on XXZ (Open "
              f"{mean_of(summary, x, 'LLM-Open'):.2f}, Closed "
              f"{mean_of(summary, x, 'LLM-Closed'):.2f} vs Random "
              f"{mean_of(summary, x, 'Random'):.2f}).", {})],
            [(f"Per seed: Closed − Random {short(summary, x, CLOSED_RANDOM)};  "
              f"Open − Random {short(summary, x, OPEN_RANDOM)} — Random's seeds spread "
              "widely, so that mean gain is not detected per seed.", {})],
            [("Closed exceeds Open at this anchor", {"bold": True, "color": RED})],
            [(f"Closed − Open {short(summary, x, CLOSED_OPEN)}, larger than at the Ising "
              "baseline.", {})],
            [("One XXZ point is not the XXZ landscape:", {"bold": True, "color": KICKER}),
             (" budget, width and model tier were not varied for XXZ.", {})],
        ],
        "the physics anchor gives one second-family result; the final controlled change "
        "is the model itself.",
        "Only the stated Hamiltonian facts change in the prompt (name, formula, parameter "
        "range, interaction type); circuit space, trainer and seeds are identical.")


def slide_08_model(deck, summary):
    r, low = "reference", "model_alt"
    return _result_slide(
        deck, "7 · Model slice",
        "Model tier changes reliability more than the Open advantage",
        "Low → High with 4 qubits, Ising chain and B = 8 fixed; High (shaded band) is the "
        "previous baseline; groups Random → Greedy → Low → High",
        "slice_model", Inches(4.45), [
            [("The Open advantage is similar for both tiers", {"bold": True, "color": BLUE})],
            [(f"Open − Random  {short(summary, low, OPEN_RANDOM)} for Low, "
              f"{short(summary, r, OPEN_RANDOM)} for High.", {})],
            [("The Closed − Open relationship differs", {"bold": True, "color": RED})],
            [(f"{short(summary, low, CLOSED_OPEN)} for Low vs "
              f"{short(summary, r, CLOSED_OPEN)} for High — the baseline's small closed-loop "
              "gain is not present for Low.", {})],
            [("Proposal validity is a separate failure mode", {"bold": True, "color": ORANGE})],
            [(f"Only {validity(low, 'LLM-Open') * 100:.0f}% of Low's open-batch proposals "
              f"met the gate-count contract (Low closed {validity(low, 'LLM-Closed') * 100:.0f}%; "
              f"High {validity(r, 'LLM-Open') * 100:.0f}% / "
              f"{validity(r, 'LLM-Closed') * 100:.0f}%). Invalid proposals become random "
              "draws, so validity and architecture quality are entangled in the score.", {})],
            [("Low and High are fixed tier labels, not a ranking by score.",
              {"bold": True, "color": KICKER})],
        ],
        "the four slices can now be combined to separate robust effects from fragile ones.",
        "Identical prompt bytes, temperature, retry policy and token limit for both tiers; "
        "only the model name changes. Random and Greedy are the same runs in both groups.")


def slide_09_synthesis(deck, summary):
    slide = deck.slides.add_slide(deck.slide_layouts[6])
    header(slide, "8 · Cross-slice synthesis and limitations",
           "What survived across the tested slices?",
           "Closed − Open sorted by what the paired test detected; the Open − Random "
           "advantage kept its sign in every slice")
    columns = {
        "robust": (RED_FILL, RED, "Closed beats Open"),
        "not detected": (GREY_FILL, GREY, "No difference detected"),
        "reversed": (BLUE_FILL, BLUE, "Open beats Closed"),
    }
    grouped = {k: [] for k in columns}
    for key, label in CONDITIONS:
        grouped[verdict(summary, key, CLOSED_OPEN)].append((key, label))
    for i, (word, (fill, border, title)) in enumerate(columns.items()):
        left = Inches(0.55 + i * 4.1)
        box(slide, left, Inches(1.6), Inches(3.95), Inches(2.58), fill, border, 1.5)
        frame = textbox(slide, left + Inches(0.16), Inches(1.68), Inches(3.65), Inches(0.4))
        write(frame, [(title, {"size": 14.5, "bold": True, "color": border})], space_after=0)
        y = 2.12
        for key, label in grouped[word]:
            entry = stat(summary, key, CLOSED_OPEN)
            chip(slide, left + Inches(0.16), Inches(y), Inches(3.65),
                 f"{label}   {_fmt(entry['mean_paired_gain'])}   "
                 f"{entry['wins_b']}/12 seeds", WHITE, border, INK, size=12,
                 bold=False, height=Inches(0.42))
            y += 0.5
        note = {
            "robust": "small budget, the XXZ anchor, and the baseline (slight)",
            "not detected": "a wide open batch, or the Low tier",
            "reversed": (f"wider registers; closed-loop validity "
                         f"{validity('reference', 'LLM-Closed') * 100:.0f}% → "
                         f"{validity('qubits_n6', 'LLM-Closed') * 100:.0f}%, "
                         f"{validity('qubits_n8', 'LLM-Closed') * 100:.0f}%"),
        }[word]
        frame = textbox(slide, left + Inches(0.16), Inches(y + 0.02), Inches(3.65),
                        Inches(0.42))
        write(frame, [(note, {"size": 10.5, "color": MUTED})], space_after=0)
    box(slide, Inches(0.55), Inches(4.3), Inches(12.2), Inches(0.62), GREEN_FILL, GREEN, 1.5)
    frame = textbox(slide, Inches(0.72), Inches(4.37), Inches(11.9), Inches(0.5))
    open_sig = sum(1 for k, _l in CONDITIONS if verdict(summary, k, OPEN_RANDOM) == "robust")
    write(frame, [[("Across the tested one-factor slices, ", {"bold": True, "color": GREEN}),
                   ("the Open − Random advantage is the robust finding: positive in all "
                    f"{len(CONDITIONS)} tested conditions, significant in {open_sig}. The "
                    "closed-loop advantage is budget-, width-, Hamiltonian- and model-"
                    "dependent — not a general result.", {})]], size=12.5, space_after=0)
    titled_box(slide, Inches(0.55), Inches(5.05), Inches(7.3), Inches(1.35), AMBER_FILL, AMBER,
               "Proposal validity can confound method differences", [
        (f"An invalid proposal is replaced by a random draw. Closed-loop validity fell to "
         f"{validity('qubits_n6', 'LLM-Closed') * 100:.0f}% at 6 qubits; Low's open batch "
         f"was only {validity('model_alt', 'LLM-Open') * 100:.0f}% valid. Part of what "
         "looks like a method effect is a generation-validity effect.", {"size": 11.5}),
    ], title_size=12.5, body_size=11.5)
    titled_box(slide, Inches(8.0), Inches(5.05), Inches(4.75), Inches(1.35), GREY_FILL, GREY,
               "Limitations", [
        ("no interactions · one XXZ anchor · Low only at the baseline condition · "
         "12 paired seeds · noiseless simulation · qubit count and circuit size scale "
         "together · not tested: everything else", {"size": 11.5}),
    ], title_color=NAVY, title_size=12.5, body_size=11.5)
    transition(slide, "ranking methods at a fixed budget is the wrong question for "
                      "practice — how much budget is actually required?")
    footer(slide, "\"beats\" = p < 0.05 in the exact paired Wilcoxon test over 12 seeds; "
                  "\"not detected\" = p ≥ 0.05, which is not evidence of no effect. "
                  "Values = mean paired difference in trash fidelity.")


def slide_10_next(deck, summary):
    slide = deck.slides.add_slide(deck.slide_layouts[6])
    header(slide, "9 · Conclusion and next action",
           "Next: how much budget is actually required?",
           "Three statements from the tested slices, then a targeted boundary-finding "
           "experiment rather than a factorial sweep")
    statements = [
        (GREEN_FILL, GREEN, "Robust",
         "Physics-informed Open proposals beat non-semantic search across the tested "
         "one-factor slices."),
        (RED_FILL, RED, "Fragile",
         "The closed-loop advantage is not general: it changes with budget, width, "
         "Hamiltonian and model tier."),
        (AMBER_FILL, AMBER, "Operational bottleneck",
         "Valid circuit generation must be separated from architecture quality."),
    ]
    for i, (fill, border, title, body) in enumerate(statements):
        titled_box(slide, Inches(0.55), Inches(1.6 + i * 1.28), Inches(4.7), Inches(1.16),
                   fill, border, title, [(body, {"size": BODY})], title_size=13.5)
    box(slide, Inches(5.5), Inches(1.6), Inches(7.25), Inches(4.75), BLUE_FILL, BLUE, 1.75)
    frame = textbox(slide, Inches(5.68), Inches(1.68), Inches(6.9), Inches(4.6))
    write(frame, [
        [("Next experiment: the minimum budget for a target fidelity",
          {"bold": True, "color": BLUE, "size": 13.5})],
        [("Before running, fix a target validation fidelity F_target (candidates 0.95 "
          "and 0.99). B_min = the smallest budget reaching F_target in at least 10 of the "
          "12 paired seeds — chosen on validation; the held-out test is read only after "
          "B_min is selected.", {})],
        [("API-minimising procedure", {"bold": True, "color": NAVY})],
        [("1  Reanalyse the existing logs first — no new calls.", {})],
        [("2  Build best-so-far validation curves versus candidate count from those logs.",
          {})],
        [("3  Check whether each target is already reached by B = 4, 8 or 16.", {})],
        [("4  Keep only the conditions whose B_min is unresolved.", {})],
        [("5  Run new LLM calls only at the nearest bracketing budget for those.", {})],
        [("6  Preserve the same 12 paired seeds for any confirmatory run.", {})],
        [("7  Report an interval for B_min where the exact minimum would cost too much.",
          {})],
        [("Not a blind full sweep — targeted boundary-finding, not factorial completion.",
          {"bold": True, "color": KICKER})],
    ], size=11.5, space_after=4)
    frame = textbox(slide, Inches(0.55), Inches(5.5), Inches(4.7), Inches(0.9))
    write(frame, [[("Why: ", {"bold": True, "color": NAVY}),
                   ("every comparison so far ranks methods at a fixed B; what decides "
                    "practical use is the budget needed to reach the required accuracy.",
                    {})]], size=11.5, space_after=0)
    footer(slide, "All figures and statistics come from the committed result tables via the "
                  "committed build scripts; no new experiment or model call was made for "
                  "this deck.")


# ------------------------------------------------------------------ QA ----

GLOSSARY = {
    "QAE": (r"\bQAE\b", r"[Qq]uantum autoencoder \(QAE\)"),
    "trash fidelity": (r"F_trash|trash fidelity", r"Trash fidelity\s+F_trash ="),
    "latent": (r"\blatent\b", r"Latent qubits q0, q1"),
    "trash": (r"\btrash\b", r"Trash qubits q2, q3"),
    "open-loop": (r"open-loop", r"open-loop: all semantic proposals"),
    "closed-loop": (r"closed-loop", r"closed-loop: semantic starts"),
    "budget B": (r"\bB\s*=\s*\d|budget B\b|\bB/2\b|\bB_min\b",
                 r"budget B = candidate circuits trained and evaluated per seed"),
    "Ising": (r"\bIsing\b", r"H = −Σ Zᵢ Zᵢ₊₁ − h Σ Xᵢ"),
    "XXZ": (r"\bXXZ\b", r"XXZ = the second Hamiltonian family"),
    "High tier": (r"\bHigh\b", r"High tier = gpt-5\.4-mini|High model tier"),
    "Low tier": (r"\bLow\b", r"Low = gpt-4\.1-mini"),
    "CI": (r"\bCI\b", r"confidence interval \(CI\)"),
    "seeds": (r"\bseeds?\b", r"12 paired seeds \(12 independent draws"),
    "Wilcoxon": (r"Wilcoxon", r"p = exact paired Wilcoxon test"),
    "CNOT": (r"\bCNOT\b", r"CNOT \(controlled-NOT\)"),
    "held-out": (r"held-out", r"held-out test set of 64 unseen h values"),
    "gate-count contract": (
        r"gate-count contract",
        r"gate-count contract \(the required 3 rotations \+ 1 CNOT per qubit\)"),
    "exploration/refinement": (
        r"\bexploration\b|\brefinement\b",
        r"B/2 exploration \+ B/2 refinement|single-change refinements"),
    "F_target": (r"F_target", r"target validation fidelity F_target"),
    "B_min": (r"B_min", r"B_min = the smallest budget"),
}

REQUIRED = {
    1: ["tested one-factor-slice study, not a complete factorial grid"],
    4: ["Untested combinations are omitted rather than displayed as empty panels",
        f"Low = {LOW_MODEL}", f"High = {HIGH_MODEL}", "Random → Greedy → Low → High"],
    9: ["Across the tested one-factor slices", "Closed beats Open", "No difference detected",
        "Open beats Closed", "no interactions", "12 paired seeds", "noiseless",
        "not tested"],
    10: ["F_target", "B_min", "0.95", "0.99", "10 of the", "validation",
         "held-out test is read only after", "Reanalyse the existing logs",
         "best-so-far", "B = 4, 8 or 16", "unresolved", "bracketing", "12 paired seeds",
         "interval"],
}
SECTIONS = ["survive one-factor changes", "previous baseline",
            "controlled robustness design", "what was actually tested",
            "budget slice", "qubit-count slice", "hamiltonian slice", "model slice",
            "cross-slice synthesis", "conclusion and next action"]
MIN_FONT_PT = 9.5


def slide_texts(path: Path) -> list[str]:
    deck = Presentation(str(path))
    texts = [""] * len(deck.slides._sldIdLst)
    for index, text in extract_text(path):
        texts[index - 1] += text + "\n"
    return texts


def glossary_audit(texts: list[str]) -> list[str]:
    problems = []
    for term, (use, definition) in GLOSSARY.items():
        first_use = next((i for i, t in enumerate(texts) if re.search(use, t)), None)
        defined = next((i for i, t in enumerate(texts) if re.search(definition, t)), None)
        if first_use is None:
            continue
        if defined is None:
            problems.append(f"'{term}' is used on slide {first_use + 1} but never defined")
        elif defined > first_use:
            problems.append(f"'{term}' is used on slide {first_use + 1} before its "
                            f"definition on slide {defined + 1}")
    return problems


def structure_audit(texts: list[str]) -> list[str]:
    problems = []
    if len(texts) != 10:
        problems.append(f"deck has {len(texts)} slides, expected 10")
    for index, (needle, text) in enumerate(zip(SECTIONS, texts, strict=False), start=1):
        if needle not in text.lower():
            problems.append(f"slide {index}: expected the section '{needle}'")
    for index, phrases in REQUIRED.items():
        text = texts[index - 1] if index <= len(texts) else ""
        for phrase in phrases:
            if phrase not in text:
                problems.append(f"slide {index}: missing required text {phrase!r}")
    manifest = json.loads((ROOT / "slices.json").read_text())
    if manifest["group_order"] != ["Random", "Greedy", "Low", "High"]:
        problems.append("slice figures do not use the Random -> Greedy -> Low -> High order")
    if manifest["figures"]["slice_model"]["groups"] != [
            "Random", "Greedy", "Low(Open, Closed)", "High(Open, Closed)"]:
        problems.append("model slice figure does not carry the four required groups")
    for name in ("slice_budget", "slice_qubits", "slice_hamiltonian"):
        if "Low" in " ".join(manifest["figures"][name]["groups"]):
            problems.append(f"{name} shows a Low group although Low was not run there")
    return problems


def prohibited_audit(texts: list[str]) -> list[str]:
    problems = []
    for index, text in enumerate(texts, start=1):
        for pattern, label in FORBIDDEN_PATTERNS:
            match = re.search(pattern, text)
            if match:
                problems.append(f"slide {index}: forbidden {label} {match.group(0)!r}")
    return problems


def font_audit(path: Path) -> list[str]:
    """No visible text run smaller than MIN_FONT_PT anywhere in the deck."""
    problems = []
    deck = Presentation(str(path))
    for index, slide in enumerate(deck.slides, start=1):
        for shape in slide.shapes:
            frames = []
            if shape.has_text_frame:
                frames.append(shape.text_frame)
            if getattr(shape, "has_table", False) and shape.has_table:
                frames += [cell.text_frame for row in shape.table.rows for cell in row.cells]
            for frame in frames:
                for paragraph in frame.paragraphs:
                    for run in paragraph.runs:
                        if run.text.strip() and run.font.size and run.font.size.pt < MIN_FONT_PT:
                            problems.append(
                                f"slide {index}: {run.font.size.pt:.1f} pt text "
                                f"{run.text[:30]!r} is below {MIN_FONT_PT} pt")
    return problems


def render_pdf(pptx_path: Path) -> Path:
    subprocess.run(["soffice", "--headless", "--convert-to", "pdf", "--outdir",
                    str(pptx_path.parent), str(pptx_path)],
                   check=True, capture_output=True, timeout=600)
    return pptx_path.with_suffix(".pdf")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--no-pdf", action="store_true")
    args = parser.parse_args()
    DECK_DIR.mkdir(parents=True, exist_ok=True)
    pptx_path = DECK_DIR / f"{DECK_NAME}.pptx"

    if not args.check:
        summary = json.loads((ROOT / "summary.json").read_text())
        deck = Presentation()
        deck.slide_width, deck.slide_height = SLIDE_W, SLIDE_H
        for builder in (slide_01_question, slide_02_baseline, slide_03_design,
                        slide_04_reading, slide_05_budget, slide_06_qubits,
                        slide_07_hamiltonian, slide_08_model, slide_09_synthesis,
                        slide_10_next):
            builder(deck, summary)
        deck.save(str(pptx_path))
        print("wrote", pptx_path)

    texts = slide_texts(pptx_path)
    geometry = [p for p in geometry_audit(pptx_path)
                if "forbidden" not in p and "LINE (9)" not in p]
    problems = (geometry + prohibited_audit(texts) + glossary_audit(texts)
                + structure_audit(texts) + font_audit(pptx_path))
    if problems:
        print("\nAUDIT FAILED:")
        for problem in problems:
            print("  -", problem)
    else:
        print("audit passed: 10 slides, required sections and phrases, tested-slice "
              "figures in Random -> Greedy -> Low -> High order, every term defined "
              "before use, no text below 9.5 pt, nothing off-canvas, no prohibited "
              "labels or claims")
    if not args.no_pdf and not args.check:
        print("wrote", render_pdf(pptx_path))
    if problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
