# Covers src/utils/metrics.py::compute_efficiency_metrics: background
# efficiency/rejection at fixed signal-efficiency (TPR) working points, derived
# from the ROC curve. Uses a small hand-constructed y_true/y_probs pair whose
# ROC curve is known exactly, so expected values are computed by hand rather
# than re-deriving them from sklearn.
import math

from src.utils.metrics import compute_efficiency_metrics


def test_compute_efficiency_metrics_perfect_separation_has_zero_bkg_eff():
    # Perfectly separable: every positive scores higher than every negative.
    # The ROC curve only has points (0,0) -> (0,1) -> (1,1) (no intermediate
    # thresholds), so the smallest achieved TPR >= any target in (0, 1] is 1.0
    # -- still reached at FPR=0, i.e. zero background efficiency either way.
    y_true = [0, 0, 0, 0, 1, 1, 1, 1]
    y_probs = [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]

    metrics = compute_efficiency_metrics(y_true, y_probs, signal_efficiency_points=(0.5, 1.0))

    for target in (0.5, 1.0):
        assert metrics[f"Sig Eff {target:g} - Achieved Sig Eff"] == 1.0
        assert metrics[f"Sig Eff {target:g} - Bkg Eff"] == 0.0
        assert metrics[f"Sig Eff {target:g} - Bkg Rejection"] == math.inf


def test_compute_efficiency_metrics_known_roc_working_point():
    # 4 background (label 0) + 4 signal (label 1), interleaved so the ROC curve
    # has known, hand-computable points. Sorting by descending score:
    # [1, 0, 1, 0, 1, 0, 1, 0] -> TPR/FPR after each step (thresholds descending):
    #   after 1 pred-positive (the top score, a true 1): TPR=0.25, FPR=0.0
    #   after 2:  TPR=0.25, FPR=0.25
    #   after 3:  TPR=0.50, FPR=0.25
    #   after 4:  TPR=0.50, FPR=0.50
    #   after 5:  TPR=0.75, FPR=0.50
    #   after 6:  TPR=0.75, FPR=0.75
    #   after 7:  TPR=1.00, FPR=0.75
    #   after 8:  TPR=1.00, FPR=1.00
    y_true =  [1, 0, 1, 0, 1, 0, 1, 0]
    y_probs = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]

    metrics = compute_efficiency_metrics(y_true, y_probs, signal_efficiency_points=(0.5, 0.75))

    # Smallest TPR >= 0.5 is exactly 0.5, achieved at FPR=0.25.
    assert metrics["Sig Eff 0.5 - Achieved Sig Eff"] == 0.5
    assert metrics["Sig Eff 0.5 - Bkg Eff"] == 0.25
    assert metrics["Sig Eff 0.5 - Bkg Rejection"] == 4.0

    # Smallest TPR >= 0.75 is exactly 0.75, achieved at FPR=0.5.
    assert metrics["Sig Eff 0.75 - Achieved Sig Eff"] == 0.75
    assert metrics["Sig Eff 0.75 - Bkg Eff"] == 0.5
    assert metrics["Sig Eff 0.75 - Bkg Rejection"] == 2.0


def test_compute_efficiency_metrics_returns_flat_scalars_only():
    y_true = [0, 1, 0, 1]
    y_probs = [0.2, 0.8, 0.4, 0.6]

    metrics = compute_efficiency_metrics(y_true, y_probs, signal_efficiency_points=(0.5,))

    assert all(isinstance(v, float) for v in metrics.values())
    assert set(metrics.keys()) == {
        "Sig Eff 0.5 - Achieved Sig Eff",
        "Sig Eff 0.5 - Bkg Eff",
        "Sig Eff 0.5 - Bkg Rejection",
    }
