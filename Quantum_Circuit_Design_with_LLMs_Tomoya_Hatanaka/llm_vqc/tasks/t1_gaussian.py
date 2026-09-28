"""T1: Gaussian-peak 1D regression (LLM-VQC_MASTER_PLAN.md Section 6.1).

**Scientific purpose:** direct comparability with Knipfer et al. 2026
(arXiv:2602.19387) — the ML4SCI mentors' own predecessor work — whose
"Simple QNN" architecture (linear embed -> VQC -> linear + sigmoid) and
data-generation formula this task replicates exactly, so that LAQS-Bench
results on this task can be read against their qualitative findings.

**Data source and provenance:** synthetic, generated on demand from a
fixed formula (not downloaded, no external dependency, no accessibility
risk). Reproduced from Knipfer et al. Section 3.3:

    y(x) = A / (sigma * sqrt(2*pi)) * exp(-(x - mu)^2 / (2*sigma^2)) + eps

with `mu ~ U[0, 1]` (peak position, the regression target), `A ~ U[0.5,
1.5]` (height), `sigma ~ U[0.01, 0.1]` (width), and `eps ~ N(0,
0.01^2)` i.i.d. per point. Each sample is evaluated at 21 fixed points
`x = linspace(0, 1, 21)`.

**Input dimensions:** 21 raw features per sample (fixed).
**Target:** `mu`, already in `[0, 1]` by construction — no further
target scaling needed.
**Preprocessing:** per-sample min-max normalization (Knipfer et al.
Eq. 2: `x_hat_i = (x_i - min_j(x_j)) / (max_j(x_j) - min_j(x_j))`,
the min/max taken *within* each sample's own 21 points, not across the
dataset). This has no fitted parameters — it is a stateless per-row
transform — so "fit on train only" is satisfied vacuously; it is
implemented as a `FittedPreprocessing` anyway for interface uniformity
with T2.
**Quantum data encoding:** not fixed here — a trainable linear "embed"
layer (added by `llm_vqc.evaluation.model`, not this module) maps the 21
raw features to whatever input width a proposed circuit's IR encoding
actually needs, then scales to `[0, pi]` for angle encoding, matching
Knipfer et al.'s tool convention exactly.
**Loss / metric:** MSE loss; RMSE is the search-visible validation
metric and the final test metric (both `lower_is_better=True`).
**Split policy:** master plan Section 6.1, using the document's own
`[PROPOSAL]` enlarged test size: 150 train / 250 val / 2000 test.
Because data is synthetic and unlimited, "overlap" is prevented by
construction (disjoint sample-id ranges) rather than merely checked.
**Expected computational cost:** negligible — pure NumPy array
generation, no I/O.
"""

from __future__ import annotations

import numpy as np

from llm_vqc.evaluation.seeds import data_split_seed
from llm_vqc.tasks.base import (
    DataSplit,
    FittedPreprocessing,
    TaskSpec,
    TrainValData,
    assert_disjoint_splits,
)

X_GRID = np.linspace(0.0, 1.0, 21)
N_TRAIN = 150
N_VAL = 250
N_TEST = 2000
_EPS = 1e-12  # numerical-stability floor for the per-sample min-max denominator

SPEC = TaskSpec(
    name="T1_gaussian_peak",
    description="1D Gaussian-peak regression (Knipfer et al. 2026 replication)",
    raw_feature_dim=21,
    classical_head_out_dim=1,
    metric_name="rmse",
    lower_is_better=True,
    loss_name="mse",
)


def _per_sample_minmax(raw_features: np.ndarray, params: dict) -> np.ndarray:
    row_min = raw_features.min(axis=1, keepdims=True)
    row_max = raw_features.max(axis=1, keepdims=True)
    return (raw_features - row_min) / (row_max - row_min + _EPS)


def _fit_preprocessing() -> FittedPreprocessing:
    # Stateless: no train-derived parameters, so "fitting" is a no-op.
    # Kept as a FittedPreprocessing (rather than skipped) for interface
    # parity with tasks that do have real fitted parameters (T2's PCA).
    return FittedPreprocessing(
        kind="per_sample_minmax", params={}, _transform_fn=_per_sample_minmax
    )


def _generate_raw_split(rng: np.random.Generator, n_samples: int, id_prefix: str) -> DataSplit:
    mu = rng.uniform(0.0, 1.0, size=n_samples)
    amplitude = rng.uniform(0.5, 1.5, size=n_samples)
    sigma = rng.uniform(0.01, 0.1, size=n_samples)
    noise = rng.normal(0.0, 0.01, size=(n_samples, len(X_GRID)))

    x = X_GRID[None, :]
    mu_col = mu[:, None]
    amplitude_col = amplitude[:, None]
    sigma_col = sigma[:, None]
    y = (
        amplitude_col
        / (sigma_col * np.sqrt(2 * np.pi))
        * np.exp(-((x - mu_col) ** 2) / (2 * sigma_col**2))
        + noise
    )

    sample_ids = tuple(f"{id_prefix}-{i}" for i in range(n_samples))
    return DataSplit(features=y, targets=mu, sample_ids=sample_ids)


class T1GaussianPeakTask:
    spec = SPEC

    def build(self, seed: int) -> TrainValData:
        split_seed = data_split_seed(seed, self.spec.name)
        rng = np.random.default_rng(split_seed)

        train = _generate_raw_split(rng, N_TRAIN, id_prefix="T1-train")
        val = _generate_raw_split(rng, N_VAL, id_prefix="T1-val")
        assert_disjoint_splits(train, val)

        preprocessing = _fit_preprocessing()
        train = DataSplit(
            features=preprocessing.transform(train.features),
            targets=train.targets,
            sample_ids=train.sample_ids,
        )
        val = DataSplit(
            features=preprocessing.transform(val.features),
            targets=val.targets,
            sample_ids=val.sample_ids,
        )

        return TrainValData(
            spec=self.spec, train=train, val=val, split_seed=split_seed, preprocessing=preprocessing
        )

    def build_test(self, seed: int) -> DataSplit:
        split_seed = data_split_seed(seed, self.spec.name)
        rng = np.random.default_rng(split_seed)
        # Advance the same rng stream past train+val (identical draw order
        # to `build`) so the test partition is disjoint from train/val by
        # construction, not merely by having a different id prefix.
        _generate_raw_split(rng, N_TRAIN, id_prefix="T1-train")
        _generate_raw_split(rng, N_VAL, id_prefix="T1-val")
        test = _generate_raw_split(rng, N_TEST, id_prefix="T1-test")

        preprocessing = _fit_preprocessing()  # stateless; safe to reconstruct independently
        return DataSplit(
            features=preprocessing.transform(test.features),
            targets=test.targets,
            sample_ids=test.sample_ids,
        )
