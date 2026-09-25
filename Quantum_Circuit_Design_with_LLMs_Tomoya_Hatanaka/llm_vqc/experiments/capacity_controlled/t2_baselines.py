"""Strong classical baselines for T2 (C0-C6).

Every learned baseline selects its config on VALIDATION (balanced-accuracy error,
the primary metric direction) and touches the protected test exactly once, after
selection — the same fairness contract as the VQC search. Exact trainable
parameter counts are recorded where meaningful.
"""

from __future__ import annotations

import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from llm_vqc.evaluation.metrics import balanced_accuracy_error
from llm_vqc.evaluation.seeds import derive_child_seed
from llm_vqc.experiments.capacity_controlled import t2_task as T2
from llm_vqc.experiments.capacity_controlled.init_policy import _init_linear
from llm_vqc.tasks.base import DataSplit, TrainValData

_ACT = {"relu": torch.relu, "tanh": torch.tanh, "sigmoid": torch.sigmoid}


def _val_metric(preds, y):
    return balanced_accuracy_error(preds, y)


def _fit_sklearn_grid(build, grid, tv: TrainValData, test: DataSplit):
    """Try each config, select best on validation balanced-acc-error, test once."""
    Xtr, ytr = tv.train.features, tv.train.targets
    Xva, yva = tv.val.features, tv.val.targets
    best = None
    cand = []
    for cfg in grid:
        clf = build(cfg).fit(Xtr, ytr)
        pv = clf.predict_proba(Xva)[:, 1]
        v = _val_metric(pv, yva)
        cand.append({"config": cfg, "val_bae": v})
        if best is None or v < best["val_bae"]:
            best = {"config": cfg, "val_bae": v, "clf": clf}
    pt = best["clf"].predict_proba(test.features)[:, 1]
    return best, cand, pt


def c0_trivial(tv: TrainValData, test: DataSplit) -> dict:
    prior = float(tv.train.targets.mean())
    preds = np.full(len(test.targets), prior)
    md = T2.full_classification_metrics(preds, test.targets)
    return {"baseline": "C0_trivial", "total_params": 0, "selected_config": {"prior": prior}, **md}


def c1_logreg(tv, test) -> dict:
    grid = [{"C": c} for c in (0.1, 1, 10)]
    best, cand, pt = _fit_sklearn_grid(lambda cfg: LogisticRegression(C=cfg["C"], max_iter=2000),
                                       grid, tv, test)
    md = T2.full_classification_metrics(pt, test.targets)
    return {"baseline": "C1_logreg", "total_params": tv.train.features.shape[1] + 1,
            "selected_config": best["config"], "val_bae": best["val_bae"], **md}


def c2_linear_svm(tv, test) -> dict:
    grid = [{"C": c} for c in (0.1, 1, 10)]
    best, cand, pt = _fit_sklearn_grid(
        lambda cfg: SVC(C=cfg["C"], kernel="linear", probability=True, random_state=0),
        grid, tv, test)
    md = T2.full_classification_metrics(pt, test.targets)
    return {"baseline": "C2_linear_svm", "total_params": tv.train.features.shape[1] + 1,
            "selected_config": best["config"], "val_bae": best["val_bae"], **md}


def c3_rbf_svm(tv, test) -> dict:
    grid = [{"C": c, "gamma": g} for c in (0.1, 1, 10) for g in ("scale", 0.1, 0.01)]
    best, cand, pt = _fit_sklearn_grid(
        lambda cfg: SVC(C=cfg["C"], gamma=cfg["gamma"], kernel="rbf", probability=True, random_state=0),
        grid, tv, test)
    md = T2.full_classification_metrics(pt, test.targets)
    return {"baseline": "C3_rbf_svm", "total_params": None,
            "selected_config": best["config"], "val_bae": best["val_bae"], **md}


def c4_trees(tv, test) -> dict:
    grid = ([{"kind": "rf", "n": n, "d": d} for n in (100, 300) for d in (3, 6, None)]
            + [{"kind": "gb", "n": 100, "d": d} for d in (2, 3)])

    def build(cfg):
        if cfg["kind"] == "rf":
            return RandomForestClassifier(n_estimators=cfg["n"], max_depth=cfg["d"], random_state=0)
        return GradientBoostingClassifier(n_estimators=cfg["n"], max_depth=cfg["d"], random_state=0)

    best, cand, pt = _fit_sklearn_grid(build, grid, tv, test)
    md = T2.full_classification_metrics(pt, test.targets)
    return {"baseline": "C4_trees", "total_params": None,
            "selected_config": best["config"], "val_bae": best["val_bae"], **md}


# --- torch classical MLP (C5 fixed, C6 NAS) --------------------------------

class MLP(torch.nn.Module):
    def __init__(self, hidden, activation):
        super().__init__()
        self.activation = activation
        dims = [T2.RAW_FEATURE_DIM, *hidden, 1]
        self.layers = torch.nn.ModuleList(
            [torch.nn.Linear(dims[i], dims[i + 1], dtype=torch.float64) for i in range(len(dims) - 1)])

    def forward(self, x):
        act = _ACT[self.activation]
        for i, layer in enumerate(self.layers):
            x = layer(x)
            x = torch.sigmoid(x) if i == len(self.layers) - 1 else act(x)
        return x

    def apply_init(self, seed):
        gen = torch.Generator(); gen.manual_seed(int(seed))
        for layer in self.layers:
            _init_linear(layer, gen)

    def n_params(self):
        return int(sum(p.numel() for p in self.parameters()))


def _train_mlp(model, tv, lr, wd, seed, epochs=20, batch=16):
    from llm_vqc.evaluation.seeds import TrainingSeeds
    s = TrainingSeeds.from_train_seed(seed)
    model.apply_init(s.param_init)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[7, 13, 17], gamma=0.5)
    loss_fn = torch.nn.BCELoss()
    Xtr = torch.tensor(tv.train.features, dtype=torch.float64)
    ytr = torch.tensor(tv.train.targets, dtype=torch.float64).reshape(-1, 1)
    rng = np.random.default_rng(s.minibatch)
    n = len(Xtr)
    for _ in range(epochs):
        model.train()
        perm = rng.permutation(n)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            loss = loss_fn(model(Xtr[idx]), ytr[idx])
            loss.backward(); opt.step()
        sched.step()
    model.eval()
    with torch.no_grad():
        pv = model(torch.tensor(tv.val.features, dtype=torch.float64)).numpy().reshape(-1)
    return pv


def c5_fixed_mlp(tv, test, seed) -> dict:
    model = MLP((4,), "tanh")
    ts = derive_child_seed(seed, "C5")
    _train_mlp(model, tv, 0.05, 1e-5, ts)
    with torch.no_grad():
        pt = model(torch.tensor(test.features, dtype=torch.float64)).numpy().reshape(-1)
    md = T2.full_classification_metrics(pt, test.targets)
    return {"baseline": "C5_fixed_mlp", "total_params": model.n_params(),
            "selected_config": {"hidden": [4], "activation": "tanh"}, **md}


C6_HIDDEN = ((4,), (6,), (8,), (4, 3), (6, 3), (8, 4))
C6_ACT = ("relu", "tanh", "sigmoid")
C6_LR = (0.01, 0.05, 0.1)
C6_WD = (0.0, 1e-5, 1e-4)


def c6_fair_nas(tv, test, seed, budget=25) -> dict:
    rng = np.random.default_rng(derive_child_seed(seed, "C6_nas"))
    best = None
    cand = []
    for i in range(budget):
        cfg = {"hidden": tuple(C6_HIDDEN[rng.integers(len(C6_HIDDEN))]),
               "activation": C6_ACT[rng.integers(len(C6_ACT))],
               "lr": float(C6_LR[rng.integers(len(C6_LR))]),
               "weight_decay": float(C6_WD[rng.integers(len(C6_WD))])}
        model = MLP(cfg["hidden"], cfg["activation"])
        ts = derive_child_seed(seed, "C6_cand", str(i))
        pv = _train_mlp(model, tv, cfg["lr"], cfg["weight_decay"], ts)
        v = _val_metric(pv, tv.val.targets)
        rec = {**cfg, "hidden": list(cfg["hidden"]), "total_params": model.n_params(), "val_bae": v}
        cand.append(rec)
        # keep the trained model of the best (re-eval on test at the end)
        if best is None or v < best["val_bae"]:
            with torch.no_grad():
                pt = model(torch.tensor(test.features, dtype=torch.float64)).numpy().reshape(-1)
            best = {**rec, "test_pred": pt}
    md = T2.full_classification_metrics(best["test_pred"], test.targets)
    return {"baseline": "C6_fair_nas", "total_params": best["total_params"],
            "selected_config": {k: best[k] for k in ("hidden", "activation", "lr", "weight_decay")},
            "val_bae": best["val_bae"], "n_candidates": len(cand),
            "candidates": [{k: c[k] for k in ("hidden", "activation", "lr", "weight_decay",
                                              "total_params", "val_bae")} for c in cand], **md}


def run_all_classical(tv, test, seed) -> list[dict]:
    return [c0_trivial(tv, test), c1_logreg(tv, test), c2_linear_svm(tv, test),
            c3_rbf_svm(tv, test), c4_trees(tv, test), c5_fixed_mlp(tv, test, seed),
            c6_fair_nas(tv, test, seed)]
