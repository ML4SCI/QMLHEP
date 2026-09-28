# Covers the 15-way disjoint statistical-replicate subset splitting: balance,
# partition, and the symbolic warm-up sample-size floor logic. Synthetic tensors
# only -- no real data, no pykan/PennyLane dependency.
import numpy as np

from src.preprocessing.balance import balance_classes, resolve_sample_size, split_into_subsets


def test_balance_classes_undersamples_to_min_count():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 3))
    y = np.array([0] * 80 + [1] * 20)

    X_bal, y_bal = balance_classes(X, y)

    classes, counts = np.unique(y_bal, return_counts=True)
    assert set(classes.tolist()) == {0, 1}
    assert counts[0] == counts[1] == 20
    assert len(X_bal) == 40


def test_split_into_subsets_disjoint_and_covers_all_rows():
    rng = np.random.default_rng(1)
    n = 97  # deliberately not a multiple of n_subsets
    X = rng.normal(size=(n, 2))
    y = np.array([0, 1] * (n // 2) + [0])  # roughly balanced, odd total
    n_subsets = 15

    X_subsets, y_subsets = split_into_subsets(X, y, n_subsets)

    assert len(X_subsets) == n_subsets
    total_rows = sum(len(x) for x in X_subsets)
    assert total_rows == n  # no rows dropped despite uneven remainder

    # Reconstruct which original rows each subset holds by exact row content
    # (values are unique enough via float noise) to check disjointness.
    seen = set()
    for Xk in X_subsets:
        for row in Xk:
            key = tuple(row.tolist())
            assert key not in seen, "row appeared in more than one subset"
            seen.add(key)
    assert len(seen) == n


def test_split_into_subsets_preserves_class_balance_per_subset():
    rng = np.random.default_rng(2)
    n_per_class = 300
    X = rng.normal(size=(2 * n_per_class, 2))
    y = np.array([0] * n_per_class + [1] * n_per_class)
    n_subsets = 15

    _, y_subsets = split_into_subsets(X, y, n_subsets)

    for y_k in y_subsets:
        classes, counts = np.unique(y_k, return_counts=True)
        assert set(classes.tolist()) == {0, 1}
        # n_per_class / n_subsets = 20 exactly per class per subset here
        assert abs(int(counts[0]) - int(counts[1])) <= 1


def test_resolve_sample_size_uses_fraction_when_above_floor():
    # 5% of 100_000 = 5000, well above the 500 floor -> keep the 5% target.
    assert resolve_sample_size(100_000, fraction=0.05, min_floor=500, fallback_size=5000) == 5000


def test_resolve_sample_size_falls_back_when_below_floor():
    # 5% of 3000 = 150, under the 500 floor -> use fallback (5000), capped to pool size.
    assert resolve_sample_size(3000, fraction=0.05, min_floor=500, fallback_size=5000) == 3000


def test_resolve_sample_size_fallback_capped_to_larger_pool():
    # 5% of 20_000 = 1000 (>= floor) -> uses the 5% target, not the fallback.
    assert resolve_sample_size(20_000, fraction=0.05, min_floor=500, fallback_size=5000) == 1000


def test_resolve_sample_size_never_exceeds_pool():
    assert resolve_sample_size(200, fraction=0.05, min_floor=500, fallback_size=5000) == 200
