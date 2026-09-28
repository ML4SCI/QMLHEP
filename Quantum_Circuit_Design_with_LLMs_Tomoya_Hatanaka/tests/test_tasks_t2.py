"""Tests for T2 (sklearn digits 3-vs-8 fallback; see DECISIONS.md Gate G1)."""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA

from llm_vqc.tasks.base import assert_disjoint_splits
from llm_vqc.tasks.t2_digits import (
    PCA_COMPONENTS,
    T2DigitsTask,
    _load_raw_binary_digits,
    _stratified_split_indices,
)


def test_total_sample_count_matches_known_class_sizes():
    task = T2DigitsTask()
    data = task.build(seed=0)
    test = task.build_test(seed=0)
    total = len(data.train) + len(data.val) + len(test)
    assert total == 357  # 183 (digit 3) + 174 (digit 8), verified in DECISIONS.md Gate G1


def test_features_are_scaled_into_0_to_pi():
    task = T2DigitsTask()
    data = task.build(seed=0)
    assert data.train.features.min() >= -1e-9
    assert data.train.features.max() <= np.pi + 1e-9
    assert data.train.features.shape[1] == PCA_COMPONENTS


def test_both_classes_present_in_every_split():
    task = T2DigitsTask()
    data = task.build(seed=0)
    test = task.build_test(seed=0)
    for split in (data.train, data.val, test):
        assert set(split.targets.tolist()) == {0.0, 1.0}


def test_train_val_test_are_mutually_disjoint():
    task = T2DigitsTask()
    data = task.build(seed=0)
    test = task.build_test(seed=0)
    assert_disjoint_splits(data.train, data.val, test)


def test_build_is_deterministic_given_the_same_seed():
    task = T2DigitsTask()
    a = task.build(seed=5)
    b = task.build(seed=5)
    assert np.array_equal(a.train.features, b.train.features)
    assert np.array_equal(a.train.targets, b.train.targets)


def test_different_seeds_give_different_splits():
    task = T2DigitsTask()
    a = task.build(seed=1)
    b = task.build(seed=2)
    assert a.train.sample_ids != b.train.sample_ids


def test_pca_is_fit_on_train_partition_only_not_val_or_test():
    """The core leakage check: PCA components fit on train-only data must
    differ from PCA components fit on train+val (or train+test) data —
    if they were identical, that would mean val/test rows silently
    influenced the fitted preprocessing.
    """
    pixels, labels, _ = _load_raw_binary_digits()
    rng = np.random.default_rng(0)
    train_idx, val_idx, test_idx = _stratified_split_indices(labels, rng)

    pca_train_only = PCA(n_components=PCA_COMPONENTS, random_state=0).fit(pixels[train_idx])
    pca_with_val_leak = PCA(n_components=PCA_COMPONENTS, random_state=0).fit(
        pixels[np.concatenate([train_idx, val_idx])]
    )
    pca_with_test_leak = PCA(n_components=PCA_COMPONENTS, random_state=0).fit(
        pixels[np.concatenate([train_idx, test_idx])]
    )
    assert not np.allclose(pca_train_only.components_, pca_with_val_leak.components_)
    assert not np.allclose(pca_train_only.components_, pca_with_test_leak.components_)


def test_task_build_reproduces_the_train_only_pca_fit():
    """The actual T2DigitsTask.build() output must match a PCA fit
    directly on the same train indices — confirming the task's internal
    preprocessing pipeline does not accidentally include val rows.

    Both the split RNG and the PCA random_state must use the exact same
    derived seeds the task uses internally (`data_split_seed` /
    `preprocessing_seed`), not an arbitrary raw seed — using a different
    seed for either would produce a different (but equally valid) split
    or a different (but equally valid, PCA components are sign-ambiguous)
    fit, which would fail this comparison for reasons unrelated to the
    thing actually being tested (train-only fitting).
    """
    from llm_vqc.evaluation.seeds import data_split_seed, preprocessing_seed

    pixels, labels, _ = _load_raw_binary_digits()
    split_seed = data_split_seed(0, T2DigitsTask.spec.name)
    rng = np.random.default_rng(split_seed)
    train_idx, _, _ = _stratified_split_indices(labels, rng)
    prep_seed = preprocessing_seed(0, T2DigitsTask.spec.name)
    reference_pca = PCA(n_components=PCA_COMPONENTS, random_state=prep_seed).fit(pixels[train_idx])

    task = T2DigitsTask()
    data = task.build(seed=0)
    task_pca = data.preprocessing.params["pca"]

    assert np.allclose(reference_pca.components_, task_pca.components_)
