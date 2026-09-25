#!/usr/bin/env python
"""Predeclared simulation-based power analysis for the v2 TOST equivalence test.

Uses the v1 within-arm protected-test SD (~0.0025) as a proxy for the paired-diff
SD. Estimates, for the v2 design (n = 3 splits x 10 seeds = 30 paired cells), the
probability that paired TOST at delta = 0.002 concludes equivalence, and the
probability a paired t-test detects a difference, across true |diff| in
{0.000, 0.001, 0.002, 0.003}. Reports both a correlated-cell assumption
(sd_diff = 0.0025) and an independent-cell assumption (sd_diff ~ sqrt(2)*0.0025).

Run BEFORE reading v2 outcomes; the design is not changed on a favorable result.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "capacity_controlled_t1_v2"
N_CELLS = 30
DELTA = 0.002
ALPHA = 0.05
TRUE_DIFFS = (0.000, 0.001, 0.002, 0.003)
V1_WITHIN_ARM_SD = 0.0025
N_SIM = 20000


def tost_equivalent(d: np.ndarray, delta: float, alpha: float = ALPHA) -> bool:
    n = len(d)
    mean = float(np.mean(d))
    se = float(np.std(d, ddof=1) / np.sqrt(n))
    if se == 0:
        return abs(mean) < delta
    df = n - 1
    t_upper = (mean - delta) / se          # H0: mean >= delta
    p_upper = stats.t.cdf(t_upper, df)
    t_lower = (mean + delta) / se          # H0: mean <= -delta
    p_lower = stats.t.sf(t_lower, df)
    return max(p_upper, p_lower) < alpha


def detect_difference(d: np.ndarray, alpha: float = ALPHA) -> bool:
    _, p = stats.ttest_1samp(d, 0.0)
    return p < alpha


def simulate(sd_diff: float, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    res = {}
    for true in TRUE_DIFFS:
        eq, det = 0, 0
        for _ in range(N_SIM):
            d = rng.normal(true, sd_diff, size=N_CELLS)
            if tost_equivalent(d, DELTA):
                eq += 1
            if detect_difference(d):
                det += 1
        res[f"{true:.3f}"] = {"P_conclude_equivalence": eq / N_SIM,
                              "P_detect_difference": det / N_SIM}
    return res


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sd_corr = V1_WITHIN_ARM_SD
    sd_indep = float(np.sqrt(2) * V1_WITHIN_ARM_SD)
    result = {
        "note": "Predeclared power analysis; v1 variance is a small-sample estimate.",
        "design": {"n_paired_cells": N_CELLS, "splits": 3, "seeds": 10,
                   "equivalence_margin": DELTA, "alpha": ALPHA, "n_sim": N_SIM},
        "assumptions": {
            "v1_within_arm_test_sd": V1_WITHIN_ARM_SD,
            "correlated_cells_sd_diff": sd_corr,
            "independent_cells_sd_diff": round(sd_indep, 5)},
        "correlated_cells": simulate(sd_corr, seed=1),
        "independent_cells": simulate(sd_indep, seed=2),
        "interpretation": (
            "Under the correlated-cell assumption (sd_diff=0.0025), TOST at delta=0.002 "
            "has high power to conclude equivalence when the true arm difference is <=0.001, "
            "and low false-equivalence when the true difference is >=0.003. Power is weaker "
            "under the independent-cell assumption (sd_diff~0.0035). Actual v2 paired-diff SD "
            "is reported in equivalence_analysis.json and should be compared against these "
            "assumptions."),
        "limitation": "v1 SD comes from n=5 seeds on a single split; true multi-split "
                      "paired-diff variance may be larger.",
    }
    (OUT / "power_analysis.json").write_text(json.dumps(result, indent=2))
    print("Correlated-cell power (sd_diff=0.0025):")
    for k, v in result["correlated_cells"].items():
        print(f"  true_diff={k}: P(equiv)={v['P_conclude_equivalence']:.3f} "
              f"P(detect)={v['P_detect_difference']:.3f}")
    print(f"wrote {OUT}/power_analysis.json")


if __name__ == "__main__":
    main()
