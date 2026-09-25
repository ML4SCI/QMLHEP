"""T2 fallback: scikit-learn `digits`, classes 3 vs 8 (binary classification).

**Why this task exists:** the master plan's primary T2 candidate — the
ML4SCI electron-vs-photon ECAL dataset — is not publicly accessible
without individually contacting ML4Sci maintainers (Gate G1, documented
in `DECISIONS.md`, evidenced from a prior official ML4SCI QMLHEP GSoC
repository's own README). This activates the master plan's own
pre-approved fallback (Section 13, risk 1).

**Scientific purpose:** a real, non-synthetic, non-trivial binary
classification benchmark with a HEP-style angle-encoded, PCA-reduced
feature pipeline, standing in for T2's original electron/photon
discrimination role. Digit pair 3-vs-8 was chosen (rather than an easy
pair like 0-vs-1) because it is a commonly used "hard" pair for this
dataset, closer in spirit to a genuinely non-trivial discrimination task
than an easy pair would be — see `DECISIONS.md` for the full Gate G1
write-up.

**Data source and provenance:** `sklearn.datasets.load_digits()` — the
public-domain UCI "Optical Recognition of Handwritten Digits" dataset
(E. Alpaydin, 1998), bundled with scikit-learn. No download, no
authentication, fully reproducible by any researcher with the pinned
`scikit-learn` version installed.

**Input dimensions:** 64 raw pixel features (8x8 image, integers 0-16)
per sample, reduced to `PCA_COMPONENTS = 10` components (within the
master plan's specified 8-16 range) fit on the training partition only.

**Target:** binary label, 0 = digit 3, 1 = digit 8.

**Preprocessing:** PCA (`PCA_COMPONENTS` components, fit on train only)
followed by per-component min-max scaling to `[0, pi]` (fit on train
only) — the PCA step reduces raw pixels to a compact feature vector, and
the scaling step maps that vector into a range suitable for angle
encoding, matching Knipfer et al.'s convention of scaling continuous
encoder inputs to `[0, pi]`. Both steps are fit exclusively on the
training partition and applied unchanged to validation and test.

**Quantum data encoding:** not fixed here, same rationale as T1 — a
trainable linear "embed" layer (`llm_vqc.evaluation.model`) adapts the
`PCA_COMPONENTS`-dimensional feature vector to whatever input width a
proposed circuit's IR encoding needs.

**Loss / metric:** binary cross-entropy loss; AUC is the search-visible
validation metric and the final test metric (`lower_is_better=False`).

**Split policy:** stratified 60% train / 20% val / 20% test (documented
deviation from the master plan's `~2k/1k/4k` sizing, driven by the
fallback dataset's much smaller total size of 357 samples for this class
pair — see `DECISIONS.md` Gate G1).

**Expected computational cost:** negligible — dataset loads instantly
from memory; PCA fit on ~214 training samples is near-instant.
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA

from llm_vqc.evaluation.seeds import data_split_seed, preprocessing_seed
from llm_vqc.tasks.base import (
    DataSplit,
    FittedPreprocessing,
    TaskSpec,
    TrainValData,
    assert_disjoint_splits,
)

CLASS_NEGATIVE = 3
CLASS_POSITIVE = 8
PCA_COMPONENTS = 10
TRAIN_FRACTION = 0.6
VAL_FRACTION = 0.2
# remaining ~0.2 goes to test
_EPS = 1e-12

SPEC = TaskSpec(
    name="T2_digits_3_vs_8",
    description=(
        "Binary classification, sklearn digits 3-vs-8 (T2 fallback; see DECISIONS.md Gate G1)"
    ),
    raw_feature_dim=PCA_COMPONENTS,
    classical_head_out_dim=1,
    metric_name="auc",
    lower_is_better=False,
    loss_name="bce",
)


def _load_raw_binary_digits() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load the full digits dataset, filtered to the two target classes.

    Returns (pixel_features [n,64], labels [n] in {0,1}, original_index [n]).
    """
    bunch = load_digits()
    mask = (bunch.target == CLASS_NEGATIVE) | (bunch.target == CLASS_POSITIVE)
    pixels = bunch.data[mask]
    labels = (bunch.target[mask] == CLASS_POSITIVE).astype(np.float64)
    original_index = np.nonzero(mask)[0]
    return pixels, labels, original_index


def _stratified_split_indices(
    labels: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stratified train/val/test index split, shuffled deterministically by `rng`."""
    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []
    for cls in (0.0, 1.0):
        cls_indices = np.nonzero(labels == cls)[0]
        shuffled = cls_indices[rng.permutation(len(cls_indices))]
        n = len(shuffled)
        n_train = int(round(n * TRAIN_FRACTION))
        n_val = int(round(n * VAL_FRACTION))
        train_idx.extend(shuffled[:n_train].tolist())
        val_idx.extend(shuffled[n_train : n_train + n_val].tolist())
        test_idx.extend(shuffled[n_train + n_val :].tolist())
    return np.array(train_idx), np.array(val_idx), np.array(test_idx)


def _fit_preprocessing(train_pixels: np.ndarray, seed: int) -> FittedPreprocessing:
    pca = PCA(n_components=PCA_COMPONENTS, random_state=seed)
    train_pca = pca.fit_transform(train_pixels)
    col_min = train_pca.min(axis=0, keepdims=True)
    col_max = train_pca.max(axis=0, keepdims=True)

    def _transform(raw_pixels: np.ndarray, params: dict) -> np.ndarray:
        pca_out = params["pca"].transform(raw_pixels)
        scaled = (pca_out - params["col_min"]) / (params["col_max"] - params["col_min"] + _EPS)
        return scaled * np.pi

    return FittedPreprocessing(
        kind="pca_then_minmax_to_pi",
        params={"pca": pca, "col_min": col_min, "col_max": col_max},
        _transform_fn=_transform,
    )


class T2DigitsTask:
    spec = SPEC

    def build(self, seed: int) -> TrainValData:
        pixels, labels, original_index = _load_raw_binary_digits()
        split_seed = data_split_seed(seed, self.spec.name)
        rng = np.random.default_rng(split_seed)
        train_idx, val_idx, test_idx = _stratified_split_indices(labels, rng)

        def _split(idx: np.ndarray, prefix: str) -> DataSplit:
            sample_ids = tuple(f"{prefix}-orig{original_index[i]}" for i in idx)
            return DataSplit(features=pixels[idx], targets=labels[idx], sample_ids=sample_ids)

        train_raw = _split(train_idx, "T2-train")
        val_raw = _split(val_idx, "T2-val")
        test_raw = _split(test_idx, "T2-test")
        assert_disjoint_splits(train_raw, val_raw, test_raw)

        prep_seed = preprocessing_seed(seed, self.spec.name)
        preprocessing = _fit_preprocessing(train_raw.features, prep_seed)

        train = DataSplit(
            features=preprocessing.transform(train_raw.features),
            targets=train_raw.targets,
            sample_ids=train_raw.sample_ids,
        )
        val = DataSplit(
            features=preprocessing.transform(val_raw.features),
            targets=val_raw.targets,
            sample_ids=val_raw.sample_ids,
        )

        return TrainValData(
            spec=self.spec, train=train, val=val, split_seed=split_seed, preprocessing=preprocessing
        )

    def build_test(self, seed: int) -> DataSplit:
        pixels, labels, original_index = _load_raw_binary_digits()
        split_seed = data_split_seed(seed, self.spec.name)
        rng = np.random.default_rng(split_seed)
        train_idx, val_idx, test_idx = _stratified_split_indices(labels, rng)

        test_sample_ids = tuple(f"T2-test-orig{original_index[i]}" for i in test_idx)

        prep_seed = preprocessing_seed(seed, self.spec.name)
        preprocessing = _fit_preprocessing(pixels[train_idx], prep_seed)

        return DataSplit(
            features=preprocessing.transform(pixels[test_idx]),
            targets=labels[test_idx],
            sample_ids=test_sample_ids,
        )
