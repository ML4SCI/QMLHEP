"""Offline invariants for the QAE-TFIM v2 API benchmark. No network calls."""
import numpy as np

from llm_vqc.experiments.qae_tfim import api_v2
from llm_vqc.experiments.qae_tfim.pilot import architecture_key, verify_capacity


def test_reference_qae_capacity():
    verify_capacity(api_v2.REFERENCE_QAE)


def test_parse_candidate_valid():
    entry = {
        "name": "chain_funnel",
        "axes": [["Y"] * 4, ["Y"] * 4, ["Y"] * 4],
        "cnots_after_layer1": [[3, 2], [1, 0]],
        "cnots_after_layer2": [[2, 1], [3, 1]],
    }
    architecture, errors = api_v2.parse_candidate(entry)
    assert errors == []
    verify_capacity(architecture)


def test_parse_candidate_rejects_bad_axes_and_cnots():
    bad_axes = {
        "axes": [["Q"] * 4, ["Y"] * 4, ["Y"] * 4],
        "cnots_after_layer1": [[3, 2], [1, 0]],
        "cnots_after_layer2": [[2, 1], [3, 1]],
    }
    architecture, errors = api_v2.parse_candidate(bad_axes)
    assert architecture is None and errors

    self_loop = {
        "axes": [["Y"] * 4, ["Y"] * 4, ["Y"] * 4],
        "cnots_after_layer1": [[2, 2], [1, 0]],
        "cnots_after_layer2": [[2, 1], [3, 1]],
    }
    architecture, errors = api_v2.parse_candidate(self_loop)
    assert architecture is None and errors


def test_mutation_preserves_capacity_and_changes_arch():
    rng = np.random.default_rng(7)
    base = api_v2.REFERENCE_QAE
    changed = 0
    for _ in range(30):
        mutated = api_v2.mutate_arch(base, rng)
        verify_capacity(mutated)
        if architecture_key(mutated) != architecture_key(base):
            changed += 1
    assert changed == 30


def test_evolutionary_is_deterministic_per_seed():
    a = api_v2.evolutionary_candidates(3)
    b = api_v2.evolutionary_candidates(3)
    assert [architecture_key(x) for _, x in a] == [architecture_key(x) for _, x in b]
    for _, architecture in a:
        verify_capacity(architecture)


def test_closed_loop_prompt_never_mentions_test():
    history = [
        {"candidate": {"name": "c1"}, "val_fid": 0.91, "duplicate": False},
    ]
    prompt = api_v2.closed_loop_prompt(history)
    assert "validation trash fidelity: 0.9100" in prompt
    assert "test" not in prompt.lower() or "test-set information" in prompt.lower()
    # the only allowed mention is the prohibition itself
    assert prompt.lower().count("test") == prompt.lower().count("test-set information")


def test_open_loop_prompt_matches_frozen_skill_semantics():
    prompt = api_v2.OPEN_LOOP_PROMPT
    assert "8 distinct candidates" in prompt
    assert "three 4-qubit rotation layers" in prompt
    assert "protected" not in prompt.lower()
