"""The one-factor-at-a-time guarantee, checked mechanically.

Each condition must (a) carry a frozen block byte-identical to the
reference's, (b) differ from the reference in exactly one declared factor,
(c) carry a derived block that is a pure function of its factors, and
(d) leave the prompt unchanged apart from the substituted Hamiltonian facts.
"""
import copy

import pytest

from llm_vqc.experiments.qae_robustness import conditions as C
from llm_vqc.experiments.qae_robustness import manifest as M
from llm_vqc.experiments.qae_robustness import prompts as P

NON_REFERENCE = [c for c in C.CONDITIONS if c.factor != "reference"]


def test_every_condition_verifies():
    manifests = M.all_manifests()
    assert manifests["violations"] == {c.key: [] for c in C.CONDITIONS}


@pytest.mark.parametrize("condition", NON_REFERENCE, ids=lambda c: c.key)
def test_frozen_block_is_identical_to_the_reference(condition):
    reference = M.condition_manifest(C.REFERENCE)["frozen"]
    assert C.canonical_json(M.condition_manifest(condition)["frozen"]) == \
        C.canonical_json(reference)


@pytest.mark.parametrize("condition", NON_REFERENCE, ids=lambda c: c.key)
def test_exactly_one_declared_factor_differs(condition):
    changed = C.changed_factors(condition)
    assert len(changed) == 1
    assert changed[0] == M._FACTOR_TO_FIELD[condition.factor]


def test_reference_condition_changes_nothing():
    assert C.changed_factors(C.REFERENCE) == []


@pytest.mark.parametrize("condition", C.CONDITIONS, ids=lambda c: c.key)
def test_derived_block_is_a_pure_function_of_the_factors(condition):
    manifest = M.condition_manifest(condition)
    assert manifest["derived"] == M.derive(manifest["factors"])


@pytest.mark.parametrize("condition", C.CONDITIONS, ids=lambda c: c.key)
def test_prompts_change_only_by_hamiltonian_fact_substitution(condition):
    check = M.hamiltonian_substitution_diff(condition)
    assert check["identical_after_undoing_fact_substitution"], \
        check["residual_difference"]
    if condition.family == C.REFERENCE_FAMILY:
        assert check["length_delta_bytes"] == 0


def test_a_tampered_manifest_is_detected():
    """The manifest must actually be able to fail."""
    sneaky = C.Condition(key="sneaky", factor="qubits", n_qubits=6, budget=16)
    problems = M.verify(sneaky)
    assert any("varies 2 factors" in p for p in problems)

    mislabelled = C.Condition(key="mislabelled", factor="budget", n_qubits=6)
    assert any("declares factor" in p for p in M.verify(mislabelled))


def test_derived_block_detects_a_hand_edited_value():
    manifest = M.condition_manifest(C.CONDITIONS_BY_KEY["qubits_n8"])
    tampered = copy.deepcopy(manifest)
    tampered["derived"]["n_cnots"] = 4
    assert tampered["derived"] != M.derive(tampered["factors"])


def test_model_condition_keeps_prompt_bytes_and_token_limits_identical():
    reference, alternative = C.REFERENCE, C.CONDITIONS_BY_KEY["model_alt"]
    assert alternative.model != reference.model
    for builder in (P.task_card, P.json_schema_card, P.open_loop_prompt,
                    P.warmstart_prompt):
        assert builder(alternative) == builder(reference)
    for requested in (1, alternative.n_warm, alternative.budget):
        assert (P.max_output_tokens(alternative, requested)
                == P.max_output_tokens(reference, requested)
                == P.REFERENCE_MAX_OUTPUT_TOKENS)


def test_budget_conditions_keep_the_fifty_fifty_split():
    for key, expected in (("budget_b4", (2, 2)), ("reference", (4, 4)),
                          ("budget_b16", (8, 8))):
        condition = C.CONDITIONS_BY_KEY[key]
        assert (condition.n_warm, condition.budget - condition.n_warm) == expected
