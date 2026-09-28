# src/preprocessing/balance.py
# Shared class-balancing and disjoint-subset-partitioning helpers, used by both
# processor_top.py and processor_qg.py.
import numpy as np


def balance_classes(X, y):
    """
    Undersamples the majority class so each class has equal representation.
    Uses the current global numpy RNG state -- caller must set_seed(...) beforehand
    for reproducibility.
    """
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        return X, y
    min_count = counts.min()
    keep_indices = []
    for c in classes:
        class_idx = np.where(y == c)[0]
        if len(class_idx) > min_count:
            class_idx = np.random.choice(class_idx, size=min_count, replace=False)
        keep_indices.append(class_idx)
    keep_indices = np.concatenate(keep_indices)
    np.random.shuffle(keep_indices)
    return X[keep_indices], y[keep_indices]


def split_into_subsets(X, y, n_subsets):
    """
    Partitions an already-balanced (X, y) pool into n_subsets mutually disjoint,
    per-class-stratified chunks whose union covers every row exactly once
    (np.array_split absorbs uneven remainders into the earliest chunks -- no data
    loss, no overlap). Each subset preserves the parent pool's per-class ratio.
    Uses the current global numpy RNG state -- caller must set_seed(...) beforehand
    for reproducibility.

    Returns (X_subsets, y_subsets): two lists of length n_subsets.
    """
    classes = np.unique(y)
    per_class_chunks = {}
    for c in classes:
        class_idx = np.where(y == c)[0]
        permuted = np.random.permutation(class_idx)
        per_class_chunks[c] = np.array_split(permuted, n_subsets)

    X_subsets, y_subsets = [], []
    for k in range(n_subsets):
        idx_k = np.concatenate([per_class_chunks[c][k] for c in classes])
        idx_k = np.random.permutation(idx_k)
        X_subsets.append(X[idx_k])
        y_subsets.append(y[idx_k])
    return X_subsets, y_subsets


def resolve_sample_size(n, fraction=0.05, min_floor=500, fallback_size=5000):
    """
    Picks how many rows to draw for the symbolic-regression warm-up sample out of
    a pool of size n. Uses `fraction` of n normally, but if that would fall under
    `min_floor` (too small to be useful for symbolic fitting -- e.g. a 1/15 subset's
    own 5%), uses `fallback_size` instead. Always capped to the pool size itself.
    """
    target = int(fraction * n)
    if target < min_floor:
        target = fallback_size
    return min(target, n)
