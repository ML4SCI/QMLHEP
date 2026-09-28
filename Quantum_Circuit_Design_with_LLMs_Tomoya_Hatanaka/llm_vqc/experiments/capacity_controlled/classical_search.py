"""C3 — fair classical MLP architecture+hyperparameter search.

Gives the classical model family the SAME selection budget the VQC arms get
(B=25 validation-evaluated candidates), over a predeclared grammar, with
protected test used exactly once after selection. This is the fair counterpart to
"the VQC side gets architecture search but the classical side is fixed."

Grammar (predeclared in PROTOCOL.md; not tuned on test):
  hidden ∈ {(4,),(5,),(4,3),(4,2),(3,3),(5,2),(6,2)}, activation ∈ {relu,tanh,sigmoid},
  lr ∈ {0.01,0.05,0.1}, weight_decay ∈ {0,1e-5,1e-4}, optimizer AdamW, 20 epochs.
Only configs with total trainable params in [90,120] are admissible (target 105);
the exact count is recorded per candidate. Selection is validation-only.
"""

from __future__ import annotations

import numpy as np
import torch

from llm_vqc.evaluation.training import TrainingConfig
from llm_vqc.evaluation.seeds import derive_child_seed
from llm_vqc.experiments.capacity_controlled import baselines as BL
from llm_vqc.experiments.capacity_controlled import space as S
from llm_vqc.experiments.capacity_controlled.init_policy import _init_linear
from llm_vqc.tasks.base import DataSplit, TrainValData

HIDDEN_CHOICES = ((4,), (5,), (4, 3), (4, 2), (3, 3), (5, 2), (6, 2))
ACTIVATIONS = ("relu", "tanh", "sigmoid")
LR_CHOICES = (0.01, 0.05, 0.1)
WD_CHOICES = (0.0, 1e-5, 1e-4)
PARAM_RANGE = (90, 120)
TARGET_PARAMS = 105
BUDGET = 25
_ACT = {"relu": torch.relu, "tanh": torch.tanh, "sigmoid": torch.sigmoid}


class ConfigurableMLP(torch.nn.Module):
    def __init__(self, hidden: tuple[int, ...], activation: str) -> None:
        super().__init__()
        self.activation = activation
        dims = [S.RAW_FEATURE_DIM, *hidden, 1]
        self.layers = torch.nn.ModuleList(
            [torch.nn.Linear(dims[i], dims[i + 1], dtype=torch.float64) for i in range(len(dims) - 1)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        act = _ACT[self.activation]
        for i, layer in enumerate(self.layers):
            x = layer(x)
            x = torch.sigmoid(x) if i == len(self.layers) - 1 else act(x)
        return x

    def apply_init(self, seed: int) -> None:
        gen = torch.Generator()
        gen.manual_seed(int(seed))
        for layer in self.layers:
            _init_linear(layer, gen)


def _sample_config(rng: np.random.Generator) -> dict:
    return {
        "hidden": tuple(HIDDEN_CHOICES[int(rng.integers(len(HIDDEN_CHOICES)))]),
        "activation": ACTIVATIONS[int(rng.integers(len(ACTIVATIONS)))],
        "lr": float(LR_CHOICES[int(rng.integers(len(LR_CHOICES)))]),
        "weight_decay": float(WD_CHOICES[int(rng.integers(len(WD_CHOICES)))]),
    }


def _params(cfg: dict) -> int:
    return BL.count_params(ConfigurableMLP(cfg["hidden"], cfg["activation"]))


def run_classical_search(train_val: TrainValData, test: DataSplit, seed: int,
                         budget: int = BUDGET) -> dict:
    """Random search over the classical grammar. Returns selected val/test + all
    candidate records. Deterministic given `seed`."""
    rng = np.random.default_rng(derive_child_seed(seed, "classical_search"))
    candidates = []
    best = None
    tries = 0
    while len(candidates) < budget and tries < budget * 50:
        tries += 1
        cfg = _sample_config(rng)
        n_params = _params(cfg)
        if not (PARAM_RANGE[0] <= n_params <= PARAM_RANGE[1]):
            continue
        model = ConfigurableMLP(cfg["hidden"], cfg["activation"])
        tcfg = TrainingConfig(learning_rate=cfg["lr"], weight_decay=cfg["weight_decay"])
        train_seed = derive_child_seed(seed, "classical_cand", str(len(candidates)))
        res = BL.train_classical_baseline(model, train_val, tcfg, train_seed)
        rec = {"index": len(candidates), **cfg, "hidden": list(cfg["hidden"]),
               "total_params": n_params, "val_rmse": res["final_val_metric"]}
        candidates.append(rec)
        if best is None or res["final_val_metric"] < best["val_rmse"]:
            best = {**rec, "state_dict": res["state_dict"]}

    # protected test once on the selected best
    sel_model = ConfigurableMLP(tuple(best["hidden"]), best["activation"])
    test_rmse = BL.evaluate_classical_on_test(sel_model, best["state_dict"], test,
                                              train_val.spec.metric_name)
    return {
        "seed": seed, "budget": budget, "n_candidates": len(candidates),
        "selected": {k: best[k] for k in ("index", "hidden", "activation", "lr",
                                          "weight_decay", "total_params", "val_rmse")},
        "selected_val_rmse": best["val_rmse"], "test_rmse": test_rmse,
        "candidates": candidates,
        "param_mismatch_vs_105": best["total_params"] - TARGET_PARAMS,
    }
