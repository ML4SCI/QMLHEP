"""Guards on the tested-one-factor-slices deck: the audits must be able to
fail, and the figure contract (group order, no invented Low results) holds.
"""
import importlib.util
import json
import re
from pathlib import Path

SCRIPT = Path("scripts/qae/build_qae_robustness_slides.py")
FIGURES = Path("scripts/qae/build_qae_robustness_slice_figures.py")


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


slides = _load(SCRIPT, "qae_slides")
figures = _load(FIGURES, "qae_slice_figures")


def test_glossary_audit_flags_a_use_before_its_definition():
    texts = ["We measured F_trash on slide one.",
             "Trash fidelity  F_trash = P(trash qubits q2 q3 = 00)"]
    problems = slides.glossary_audit(texts)
    assert any("'trash fidelity'" in p and "before its definition" in p for p in problems)


def test_glossary_audit_flags_a_term_that_is_never_defined():
    problems = slides.glossary_audit(["closed-loop redesigns everywhere", ""])
    assert any("'closed-loop'" in p and "never defined" in p for p in problems)


def test_prohibited_audit_rejects_model_letter_labels_and_full_grid_claims():
    for text in ("LLM-Open Model A", "Open B", "the reference model", "the alternative model",
                 "robust across the full grid", "for all qubit–budget combinations",
                 "a model-independent effect", "Hamiltonian-independent"):
        assert slides.prohibited_audit([text]), text


def test_prohibited_audit_accepts_tier_and_workflow_vocabulary():
    for text in ("Random → Greedy → Low → High", "LLM-Open (High tier = filled)",
                 "Low = gpt-4.1-mini", "not a complete factorial grid",
                 "B = 8 · High model tier"):
        assert not slides.prohibited_audit([text]), text


def test_structure_audit_requires_the_next_action_ingredients():
    texts = [""] * 10
    problems = slides.structure_audit(texts)
    for phrase in ("F_target", "B_min", "bracketing", "Reanalyse the existing logs"):
        assert any(phrase in p for p in problems), phrase
    assert any("Across the tested one-factor slices" in p for p in problems)


def test_group_order_and_tier_mapping_are_fixed():
    assert figures.TIER == {"gpt-5.4-mini-2026-03-17": "High",
                            "gpt-4.1-mini-2025-04-14": "Low"}
    manifest = json.loads(Path("outputs/qae_robustness/slices.json").read_text())
    assert manifest["group_order"] == ["Random", "Greedy", "Low", "High"]
    assert manifest["figures"]["slice_model"]["groups"] == [
        "Random", "Greedy", "Low(Open, Closed)", "High(Open, Closed)"]
    for name in ("slice_budget", "slice_qubits", "slice_hamiltonian"):
        assert manifest["figures"][name]["groups"] == ["Random", "Greedy", "High(Open, Closed)"]


def test_slices_use_only_completed_conditions():
    manifest = json.loads(Path("outputs/qae_robustness/slices.json").read_text())
    run = {"reference", "budget_b4", "budget_b16", "qubits_n6", "qubits_n8",
           "hamiltonian_xxz", "model_alt"}
    for entry in manifest["figures"].values():
        assert set(entry["panels"]) <= run
    assert manifest["figures"]["slice_model"]["panels"] == ["reference", "model_alt"]


def test_the_deck_has_exactly_ten_slide_builders_in_the_required_order():
    source = SCRIPT.read_text()
    builders = re.findall(r"^def (slide_\d\d_\w+)\(", source, flags=re.M)
    assert len(builders) == 10
    assert builders == sorted(builders)
    ordered = source[source.index("for builder in ("):source.index("):\n            builder")]
    for name in builders:
        assert name in ordered


def test_verdict_words_are_evidence_calibrated():
    summary = {"conditions": {"x": {"contrasts": {
        "up": {"mean_paired_gain": 0.02, "wilcoxon_exact_two_sided_p": 0.01},
        "down": {"mean_paired_gain": -0.02, "wilcoxon_exact_two_sided_p": 0.01},
        "flat": {"mean_paired_gain": 0.02, "wilcoxon_exact_two_sided_p": 0.4}}}}}
    assert slides.verdict(summary, "x", "up") == "robust"
    assert slides.verdict(summary, "x", "down") == "reversed"
    assert slides.verdict(summary, "x", "flat") == "not detected"
