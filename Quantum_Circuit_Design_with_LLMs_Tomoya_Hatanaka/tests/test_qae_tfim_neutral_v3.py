"""Offline invariants for the QAE-TFIM v3 neutral-space benchmark. No network."""
import numpy as np

from llm_vqc.experiments.qae_tfim import neutral_v3
from llm_vqc.experiments.qae_tfim.pilot import architecture_key


def test_random_sampler_capacity_and_free_ordering():
    rng = np.random.default_rng(1)
    saw_cnot_first = False
    for _ in range(50):
        arch = neutral_v3.sample_neutral_arch(rng)
        neutral_v3.verify_neutral_capacity(arch)
        if arch[0]["type"] == "CX":
            saw_cnot_first = True
    # ordering is genuinely free: a CNOT can appear before any rotation
    assert saw_cnot_first


def test_parse_candidate_valid_neutral():
    gates = (
        [{"g": "RY", "q": q} for q in range(4)]
        + [{"g": "CNOT", "c": 3, "t": 2}, {"g": "CNOT", "c": 1, "t": 0}]
        + [{"g": "RX", "q": q} for q in range(4)]
        + [{"g": "CNOT", "c": 2, "t": 1}, {"g": "CNOT", "c": 3, "t": 1}]
        + [{"g": "RZ", "q": q} for q in range(4)]
    )
    architecture, errors = neutral_v3.parse_candidate({"name": "x", "gates": gates})
    assert errors == []
    neutral_v3.verify_neutral_capacity(architecture)


def test_parse_candidate_rejects_wrong_counts_and_wires():
    gates = [{"g": "RY", "q": 0}] * 16  # 16 rotations, 0 CNOTs
    architecture, errors = neutral_v3.parse_candidate({"gates": gates})
    assert architecture is None and errors

    gates = (
        [{"g": "RY", "q": 0}] * 12
        + [{"g": "CNOT", "c": 2, "t": 2}]  # self-loop
        + [{"g": "CNOT", "c": 0, "t": 1}] * 3
    )
    architecture, errors = neutral_v3.parse_candidate({"gates": gates})
    assert architecture is None and errors


def test_mutation_changes_exactly_one_thing_and_keeps_capacity():
    rng = np.random.default_rng(5)
    base = neutral_v3.sample_neutral_arch(rng)
    for _ in range(50):
        mutated = neutral_v3.mutate_one_choice(base, rng)
        neutral_v3.verify_neutral_capacity(mutated)
        assert architecture_key(mutated) != architecture_key(base)


def test_greedy_is_deterministic_and_uses_full_budget():
    from llm_vqc.experiments.qae_tfim.pilot import state_sets

    train, val, test = state_sets(0)
    rows_a = neutral_v3.run_greedy_seed(0, train, val, test)
    rows_b = neutral_v3.run_greedy_seed(0, train, val, test)
    assert len(rows_a) == neutral_v3.BUDGET
    assert [r["candidate"] for r in rows_a] == [r["candidate"] for r in rows_b]
    assert [r["val_loss"] for r in rows_a] == [r["val_loss"] for r in rows_b]


def test_closed_loop_prompt_never_leaks_test():
    history = [{"candidate": {"name": "c1"}, "val_fid": 0.93, "duplicate": False}]
    prompt = neutral_v3.closed_loop_prompt(history)
    assert "validation trash fidelity: 0.9300" in prompt
    assert prompt.lower().count("test") == prompt.lower().count("test-set information")


def test_open_prompt_states_neutral_contract():
    prompt = neutral_v3.OPEN_LOOP_PROMPT
    assert "exactly 12 single-qubit rotations" in prompt
    assert "exactly 4 CNOTs" in prompt
    assert "ORDER freely" in prompt
    assert "protected" not in prompt.lower()
