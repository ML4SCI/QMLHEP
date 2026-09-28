"""Tests for deterministic seed derivation."""

from __future__ import annotations

from llm_vqc.evaluation.seeds import (
    TrainingSeeds,
    data_split_seed,
    derive_child_seed,
    preprocessing_seed,
    synthetic_fixture_seed,
    train_seed_for_circuit,
)


def test_derive_child_seed_is_deterministic():
    assert derive_child_seed(1, "a", "b") == derive_child_seed(1, "a", "b")


def test_derive_child_seed_is_sensitive_to_parent_and_labels():
    base = derive_child_seed(1, "a")
    assert base != derive_child_seed(2, "a")
    assert base != derive_child_seed(1, "b")
    assert base != derive_child_seed(1, "a", "extra")


def test_train_seed_for_circuit_depends_only_on_run_seed_and_hash():
    """Master plan Section 6.4: deterministic given (run_seed,
    structural_hash) alone -- the same circuit gets the same training
    seed regardless of which arm proposed it."""
    a = train_seed_for_circuit(42, "hash-abc")
    b = train_seed_for_circuit(42, "hash-abc")
    assert a == b
    assert a != train_seed_for_circuit(42, "hash-xyz")
    assert a != train_seed_for_circuit(7, "hash-abc")


def test_data_split_seed_and_preprocessing_seed_are_distinct_streams():
    a = data_split_seed(0, "T1")
    b = preprocessing_seed(0, "T1")
    assert a != b


def test_data_split_seed_distinguishes_tasks():
    assert data_split_seed(0, "T1") != data_split_seed(0, "T2")


def test_synthetic_fixture_seed_is_deterministic_and_distinct_per_name():
    a = synthetic_fixture_seed(0, "fixture_a")
    b = synthetic_fixture_seed(0, "fixture_b")
    assert a == synthetic_fixture_seed(0, "fixture_a")
    assert a != b


def test_training_seeds_produces_four_independent_deterministic_substreams():
    seeds1 = TrainingSeeds.from_train_seed(100)
    seeds2 = TrainingSeeds.from_train_seed(100)
    assert seeds1 == seeds2
    values = {seeds1.param_init, seeds1.minibatch, seeds1.training, seeds1.backend_sampling}
    assert len(values) == 4


def test_training_seeds_differ_for_different_train_seeds():
    seeds1 = TrainingSeeds.from_train_seed(1)
    seeds2 = TrainingSeeds.from_train_seed(2)
    assert seeds1 != seeds2
