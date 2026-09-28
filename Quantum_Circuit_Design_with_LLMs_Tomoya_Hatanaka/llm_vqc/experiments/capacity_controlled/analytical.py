"""Strong analytical (non-learned) baselines for T1 peak-position estimation.

These estimate the Gaussian peak position mu directly from a sample's normalized
21-point curve — the same per-sample min-max-normalized features the learned
models receive. They are deterministic, use NO trainable parameters, and NEVER
inspect protected-test labels or the true generation parameters (A, sigma, mu).
Fitting for A4 uses only the input curve.

All estimators return mu_hat in [0, 1] (the X_GRID range).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit

from llm_vqc.tasks.t1_gaussian import X_GRID

_GRID = np.asarray(X_GRID, dtype=np.float64)
_DX = float(_GRID[1] - _GRID[0])


def a1_argmax(features: np.ndarray) -> np.ndarray:
    """A1: grid position of the maximum normalized input value."""
    f = np.asarray(features, dtype=np.float64)
    return _GRID[np.argmax(f, axis=1)]


def a2_quadratic_peak(features: np.ndarray) -> np.ndarray:
    """A2: sub-grid vertex of the parabola through the argmax and its two
    immediate neighbors. Edge argmax falls back to A1 deterministically."""
    f = np.asarray(features, dtype=np.float64)
    n, m = f.shape
    idx = np.argmax(f, axis=1)
    out = _GRID[idx].copy()
    for i in range(n):
        k = idx[i]
        if k == 0 or k == m - 1:
            continue  # edge -> A1 value already in out
        y0, y1, y2 = f[i, k - 1], f[i, k], f[i, k + 1]
        denom = (y0 - 2 * y1 + y2)
        if denom == 0:
            continue
        # vertex offset in grid units, clamped to the neighbor cell
        delta = 0.5 * (y0 - y2) / denom
        delta = float(np.clip(delta, -1.0, 1.0))
        out[i] = float(np.clip(_GRID[k] + delta * _DX, 0.0, 1.0))
    return out


def a3_weighted_centroid(features: np.ndarray) -> np.ndarray:
    """A3: weighted centroid with predeclared weights w = (normalized height)^2
    (nonnegative, not tuned)."""
    f = np.asarray(features, dtype=np.float64)
    w = np.clip(f, 0.0, None) ** 2
    denom = w.sum(axis=1)
    denom = np.where(denom == 0, 1.0, denom)
    mu = (w * _GRID[None, :]).sum(axis=1) / denom
    return np.clip(mu, 0.0, 1.0)


def _gaussian(x, amp, mu, sigma, offset):
    return amp * np.exp(-((x - mu) ** 2) / (2.0 * sigma**2)) + offset


def a4_gaussian_nlls(features: np.ndarray) -> tuple[np.ndarray, int]:
    """A4: per-sample Gaussian nonlinear least-squares fit; estimate mu while
    treating amplitude/width/offset as nuisances. Deterministic init (A1 for mu,
    grid span for sigma). Fit failures fall back to A1 and are counted.

    Returns (mu_hat array, n_failures). Never uses true A/sigma/mu.
    """
    f = np.asarray(features, dtype=np.float64)
    fallback = a1_argmax(f)
    out = fallback.copy()
    failures = 0
    lower = [0.0, 0.0, 0.005, -np.inf]
    upper = [np.inf, 1.0, 0.5, np.inf]
    for i in range(f.shape[0]):
        y = f[i]
        p0 = [float(y.max() - y.min()), float(fallback[i]), 0.1, float(y.min())]
        try:
            popt, _ = curve_fit(_gaussian, _GRID, y, p0=p0, bounds=(lower, upper), maxfev=2000)
            out[i] = float(np.clip(popt[1], 0.0, 1.0))
        except Exception:  # noqa: BLE001 - any fit failure -> A1 fallback, counted
            failures += 1
            out[i] = fallback[i]
    return out, failures


ANALYTICAL_ESTIMATORS = {
    "A1_argmax": lambda f: (a1_argmax(f), 0),
    "A2_quadratic_peak": lambda f: (a2_quadratic_peak(f), 0),
    "A3_weighted_centroid": lambda f: (a3_weighted_centroid(f), 0),
    "A4_gaussian_nlls": a4_gaussian_nlls,
}


def rmse(pred: np.ndarray, true: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    true = np.asarray(true, dtype=np.float64).reshape(-1)
    return float(np.sqrt(np.mean((pred - true) ** 2)))
