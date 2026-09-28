"""Guards on the deck build: the prohibited-string audit must actually bite,
and must not flag legitimate scientific vocabulary.
"""
import importlib.util
import re
from pathlib import Path

import pytest

SCRIPT = Path("scripts/qae/build_qae_robustness_deck.py")


def _load():
    spec = importlib.util.spec_from_file_location("qae_deck", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


deck_module = _load()


def _flags(text: str) -> list[str]:
    return [label for pattern, label in deck_module.FORBIDDEN_PATTERNS
            if re.search(pattern, text)]


@pytest.mark.parametrize("text", [
    "the v5 result",
    "QAE v3 vs v4",
    "outputs/qae_robustness/summary.json",
    "see candidate_results.csv",
    "pre-registered at commit 33740dc",
    "branch claude/qae-experiments-presentation",
    "neutral_v5.py",
    "qae_tfim_neutral run",
    "merged into main by a pull request",
    "QAE_PROTOCOL_V5",
])
def test_audit_rejects_coding_and_version_identifiers(text):
    assert _flags(text), f"{text!r} should have been flagged"


@pytest.mark.parametrize("text", [
    "Budget B = 4, 8, 16",
    "B/2 semantic warm starts + B/2 free redesigns",
    "train/val/test splits are unchanged",
    "50/50 exploration/refinement split",
    "reference model gpt-5.4-mini, alternative gpt-4.1-mini",
    "Closed - Open +0.0066 [+0.0023, +0.0112], 11/12 seeds, p = 0.012",
    "Held-out test trash fidelity (higher is better)",
    "XXZ Heisenberg chain over the same anisotropy range",
    "3 rotations + 1 controlled-NOT per qubit",
    "12 paired seeds; bootstrap 95% CI",
    "Research meeting - 4 September 2026",
])
def test_audit_accepts_legitimate_slide_vocabulary(text):
    assert not _flags(text), f"{text!r} was wrongly flagged: {_flags(text)}"


def test_the_deck_has_exactly_ten_slide_builders():
    source = SCRIPT.read_text()
    builders = re.findall(r"^def (slide_\d\d_\w+)\(", source, flags=re.M)
    assert len(builders) == 10
    assert builders == sorted(builders)
    ordered = source[source.index("for builder in ("):source.index("):\n            builder")]
    for name in builders:
        assert name in ordered, f"{name} is not in the build order"


def test_verdict_labels_follow_the_statistics():
    entry = {"contrasts": {
        "up": {"mean_paired_gain": 0.02, "wilcoxon_exact_two_sided_p": 0.01},
        "down": {"mean_paired_gain": -0.02, "wilcoxon_exact_two_sided_p": 0.01},
        "flat": {"mean_paired_gain": 0.02, "wilcoxon_exact_two_sided_p": 0.4},
    }}
    assert deck_module.verdict(entry, "up")[0] == "holds"
    assert deck_module.verdict(entry, "down")[0] == "reverses"
    assert deck_module.verdict(entry, "flat")[0] == "not detected"
    assert deck_module.verdict({}, "missing")[0] == "n/a"


def test_headline_sentence_tracks_the_measured_signs():
    def cell(value):
        return {"contrasts": {"x": {"mean_paired_gain": value,
                                    "wilcoxon_exact_two_sided_p": 0.01}}}
    args = ("all positive", "mixed", "all negative")
    assert deck_module._headline([cell(0.1), cell(0.2)], "x", *args) == "all positive"
    assert deck_module._headline([cell(0.1), cell(-0.2)], "x", *args) == "mixed"
    assert deck_module._headline([cell(-0.1), cell(-0.2)], "x", *args) == "all negative"


def test_contrast_sentence_reports_ci_wins_and_p():
    entry = {"contrasts": {"c": {
        "mean_paired_gain": 0.0066, "bootstrap_95ci": [0.0023, 0.0112],
        "wins_b": 11, "n": 12, "wilcoxon_exact_two_sided_p": 0.0122}}}
    text = deck_module.contrast_sentence(entry, "c", "Closed - Open")
    assert "+0.0066" in text and "+0.0023" in text and "+0.0112" in text
    assert "11/12" in text and "p = 0.012" in text
    assert deck_module.contrast_sentence({}, "c", "X") == "X: not estimable"
