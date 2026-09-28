"""Tests for T1 (Gaussian-peak regression)."""

from __future__ import annotations

import numpy as np

from llm_vqc.tasks.base import assert_disjoint_splits
from llm_vqc.tasks.t1_gaussian import N_TEST, N_TRAIN, N_VAL, T1GaussianPeakTask


def test_split_sizes_match_master_plan_proposal():
    task = T1GaussianPeakTask()
    data = task.build(seed=0)
    test = task.build_test(seed=0)
    assert len(data.train) == N_TRAIN == 150
    assert len(data.val) == N_VAL == 250
    assert len(test) == N_TEST == 2000


def test_features_and_targets_are_bounded_in_unit_interval():
    task = T1GaussianPeakTask()
    data = task.build(seed=0)
    assert (data.train.features >= 0).all() and (data.train.features <= 1).all()
    assert (data.train.targets >= 0).all() and (data.train.targets <= 1).all()


def test_train_val_test_are_mutually_disjoint():
    task = T1GaussianPeakTask()
    data = task.build(seed=0)
    test = task.build_test(seed=0)
    assert_disjoint_splits(data.train, data.val, test)  # must not raise


def test_build_is_deterministic_given_the_same_seed():
    task = T1GaussianPeakTask()
    a = task.build(seed=7)
    b = task.build(seed=7)
    assert np.array_equal(a.train.features, b.train.features)
    assert np.array_equal(a.train.targets, b.train.targets)
    assert np.array_equal(a.val.features, b.val.features)


def test_build_test_is_deterministic_given_the_same_seed():
    task = T1GaussianPeakTask()
    a = task.build_test(seed=3)
    b = task.build_test(seed=3)
    assert np.array_equal(a.features, b.features)
    assert np.array_equal(a.targets, b.targets)


def test_different_seeds_give_different_data():
    task = T1GaussianPeakTask()
    a = task.build(seed=1)
    b = task.build(seed=2)
    assert not np.array_equal(a.train.features, b.train.features)


def test_preprocessing_is_a_pure_per_sample_function_no_val_test_leakage():
    """T1's normalization is per-row (Knipfer et al. Eq. 2): applying it to
    a subset of rows must give bit-identical results to applying it to
    the full split, since no cross-row statistic is used. This is the
    "fit on train only" check for a task whose preprocessing has no
    fitted parameters at all."""
    task = T1GaussianPeakTask()
    data = task.build(seed=0)
    preprocessing = data.preprocessing
    # Re-transform a subset of the ALREADY-normalized val features and
    # confirm idempotence is irrelevant here; instead verify the
    # transform of raw features for a subset matches the transform of
    # the full batch, proving no cross-sample statistic leaks in.
    from llm_vqc.tasks.t1_gaussian import _generate_raw_split

    rng = np.random.default_rng(123)
    raw = _generate_raw_split(rng, 10, id_prefix="probe")
    full_transformed = preprocessing.transform(raw.features)
    subset_transformed = preprocessing.transform(raw.features[:3])
    assert np.allclose(full_transformed[:3], subset_transformed)
