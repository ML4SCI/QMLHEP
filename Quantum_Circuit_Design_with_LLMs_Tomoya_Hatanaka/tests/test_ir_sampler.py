"""Tests for the random IR sampler (also the future `random` search arm)."""

from __future__ import annotations

import numpy as np

from llm_vqc.ir.canonicalize import structural_hash
from llm_vqc.ir.sampler import sample_random_ir
from llm_vqc.ir.validators import validate_proposal


def test_sampler_produces_100_percent_valid_ir_over_many_draws():
    rng = np.random.default_rng(0)
    n = 500
    for i in range(n):
        ir = sample_random_ir(rng)
        result = validate_proposal(ir)
        assert result.valid, f"draw {i} invalid: {result.issues}"


def test_sampler_is_reproducible_given_the_same_seed():
    rng_a = np.random.default_rng(2026)
    rng_b = np.random.default_rng(2026)
    hashes_a = [structural_hash(sample_random_ir(rng_a)) for _ in range(20)]
    hashes_b = [structural_hash(sample_random_ir(rng_b)) for _ in range(20)]
    assert hashes_a == hashes_b


def test_sampler_produces_diverse_circuits():
    rng = np.random.default_rng(1)
    hashes = {structural_hash(sample_random_ir(rng)) for _ in range(200)}
    # Not a strict uniqueness requirement (collisions are possible and even
    # expected occasionally), but overwhelming duplication would indicate a
    # degenerate sampler.
    assert len(hashes) > 150


def test_sampler_covers_multiple_qubit_counts():
    rng = np.random.default_rng(3)
    qubit_counts = {sample_random_ir(rng).n_qubits for _ in range(200)}
    assert len(qubit_counts) >= 5


def test_sampler_covers_multiple_entangle_patterns():
    rng = np.random.default_rng(4)
    patterns_seen: set[str] = set()
    for _ in range(500):
        ir = sample_random_ir(rng)
        for layer in ir.layers:
            if layer.type == "entangle":
                patterns_seen.add(layer.pattern)
            elif layer.type == "repeat":
                for sub in layer.body:
                    if sub.type == "entangle":
                        patterns_seen.add(sub.pattern)
    assert len(patterns_seen) >= 4


def test_sampler_requires_explicit_generator_no_hidden_global_state():
    """`sample_random_ir` takes an explicit np.random.Generator; two
    independent default_rng(seed) instances must behave identically,
    proving there is no shared/global RNG state leaking between calls."""
    rng1 = np.random.default_rng(99)
    ir1 = sample_random_ir(rng1)
    rng2 = np.random.default_rng(99)
    ir2 = sample_random_ir(rng2)
    assert structural_hash(ir1) == structural_hash(ir2)
