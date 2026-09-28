"""T2 capacity-controlled task wrapper + classification evaluation.

The controlled VQC circuit space is *identical* to T1-v2 (same genome grammar,
`validate_controlled`, structural limits, 4 qubits, angle-RY, Z-all, 12 quantum
params) — only the task/data differ. Because T2's raw feature dim is 10 (PCA),
the embed is `Linear(10→4)` and every controlled candidate has **61** total
trainable params (44 embed + 12 quantum + 5 head).

**Audited metric decision (see T2_TASK_AUDIT.md / PROTOCOL.md):** validation AUC
saturates at 1.0 (near-separable task), so it cannot rank candidates. The
selection metric is therefore **validation log-loss** (`lower_is_better=True`),
which stays discriminative. The primary protected-test metric for equivalence is
**balanced-accuracy error**. Both are documented, pre-registered choices.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch

from llm_vqc.evaluation.model import HybridQNNModel
from llm_vqc.experiments.capacity_controlled import space as S
from llm_vqc.tasks.base import DataSplit, TrainValData
from llm_vqc.tasks.t2_digits import PCA_COMPONENTS, T2DigitsTask

RAW_FEATURE_DIM = PCA_COMPONENTS            # 10
EMBED_PARAMS = RAW_FEATURE_DIM * S.N_QUBITS + S.N_QUBITS   # 44
HEAD_PARAMS = S.N_QUBITS * 1 + 1            # 5
QUANTUM_PARAMS = S.QUANTUM_PARAM_COUNT      # 12
TOTAL_TRAINABLE_PARAMS = EMBED_PARAMS + QUANTUM_PARAMS + HEAD_PARAMS   # 61

SELECTION_METRIC = "logloss"               # search selects lowest validation log-loss
PRIMARY_TEST_METRIC = "balanced_accuracy_error"
EQUIVALENCE_MARGIN = 0.03
TASK_NAME = "T2"


def _spec_for_selection():
    base = T2DigitsTask.spec
    # selection on validation log-loss (lower better); loss stays BCE
    return replace(base, metric_name=SELECTION_METRIC, lower_is_better=True)


def build_t2(split_seed: int) -> TrainValData:
    tv = T2DigitsTask().build(seed=split_seed)
    return replace(tv, spec=_spec_for_selection())


def build_t2_test(split_seed: int) -> DataSplit:
    return T2DigitsTask().build_test(seed=split_seed)


def total_trainable_params(ir) -> int:
    m = HybridQNNModel(ir, raw_feature_dim=RAW_FEATURE_DIM, head_out_dim=1)
    return int(sum(p.numel() for p in m.parameters()))


def assert_capacity(ir) -> int:
    total = total_trainable_params(ir)
    if total != TOTAL_TRAINABLE_PARAMS:
        from llm_vqc.ir.canonicalize import structural_hash
        raise RuntimeError(f"T2 CAPACITY VIOLATION: {total} != {TOTAL_TRAINABLE_PARAMS} "
                           f"(hash {structural_hash(ir)[:12]})")
    return total


def _predict(model: torch.nn.Module, features: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(features, dtype=torch.float64)).cpu().numpy().reshape(-1)


def full_classification_metrics(preds: np.ndarray, targets: np.ndarray,
                                with_curves: bool = False) -> dict:
    from llm_vqc.evaluation.metrics import (
        auc, balanced_accuracy_error, brier, classification_error, logloss,
    )
    p = np.asarray(preds, dtype=np.float64).reshape(-1)
    y = np.asarray(targets, dtype=np.float64).reshape(-1)
    yhat = (p >= 0.5).astype(np.float64)
    tp = int(((yhat == 1) & (y == 1)).sum()); tn = int(((yhat == 0) & (y == 0)).sum())
    fp = int(((yhat == 1) & (y == 0)).sum()); fn = int(((yhat == 0) & (y == 1)).sum())
    out = {
        "balanced_accuracy_error": balanced_accuracy_error(p, y),
        "classification_error": classification_error(p, y),
        "accuracy": float((yhat == y).mean()),
        "auc": auc(p, y) if len(set(y.tolist())) > 1 else float("nan"),
        "logloss": logloss(p, y), "brier": brier(p, y),
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }
    if with_curves:
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y, p)
        out["roc"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
        # 10-bin reliability
        bins = np.linspace(0, 1, 11)
        idx = np.clip(np.digitize(p, bins) - 1, 0, 9)
        cal = []
        for b in range(10):
            m = idx == b
            if m.any():
                cal.append({"bin": b, "conf": float(p[m].mean()), "acc": float(y[m].mean()), "n": int(m.sum())})
        out["calibration"] = cal
        out["pred"] = p.tolist(); out["true"] = y.tolist()
    return out


def evaluate_all_test_metrics(ir, classical_state: dict, test: DataSplit,
                              with_curves: bool = False) -> dict:
    """Reload an already-trained model (no retrain) and compute the full protected-
    test classification metric set. Preserves the test quarantine."""
    model = HybridQNNModel(ir, raw_feature_dim=RAW_FEATURE_DIM, head_out_dim=1)
    reloaded = {k: torch.tensor(v, dtype=torch.float64).reshape(model.state_dict()[k].shape)
                for k, v in classical_state.items()}
    model.load_state_dict(reloaded)
    preds = _predict(model, test.features)
    return full_classification_metrics(preds, test.targets, with_curves=with_curves)
