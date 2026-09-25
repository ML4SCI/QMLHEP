"""Classical and fixed-circuit baselines for the controlled experiment.

- **C1 — classical bottleneck**: `21 -> Linear(21->4) -> sigmoid -> Linear(4->1)
  -> sigmoid`. Same classical embed/head sizes as the controlled hybrid, but no
  VQC. Isolates "what does the classical shell alone achieve?"
- **C2 — parameter-matched MLP**: the classical MLP whose total trainable
  parameter count is closest to the controlled hybrid's 105. `21->4->3->1`
  (tanh hidden, sigmoid out) = 107 params (mismatch +2, recorded).
- **Q1 — fixed hand-designed VQC**: one predeclared hardware-efficient 4-qubit
  ansatz (RY layers interleaved with CNOT-ring entanglers), 12 quantum angles.
- **Q2 — fixed random VQC**: one architecture drawn with a predeclared seed,
  frozen, run under every repetition seed.

C1/C2 train through `train_classical_baseline` (this module) which mirrors the
shared protocol exactly (AdamW, 20 epochs, batch 16, same LR schedule, same seed
streams). Q1/Q2 are ordinary `CircuitIR`s trained by `train_model` + scored by
`evaluate_on_test`, using the controlled explicit init policy. None of these are
search arms; they are analyzed separately from candidate-budget curves.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from llm_vqc.evaluation.metrics import compute_metric
from llm_vqc.evaluation.seeds import TrainingSeeds
from llm_vqc.evaluation.training import TrainingConfig
from llm_vqc.experiments.capacity_controlled import space as S
from llm_vqc.experiments.capacity_controlled.init_policy import _init_linear
from llm_vqc.tasks.base import DataSplit, TrainValData

C2_HIDDEN = (4, 3)  # 21->4->3->1 = 107 params, closest to controlled hybrid's 105
CONTROLLED_TOTAL_PARAMS = S.TOTAL_TRAINABLE_PARAMS  # 105
Q2_FIXED_SEED = 20260719  # predeclared, before any results


class BottleneckModel(torch.nn.Module):
    """C1: 21 -> Linear(21->4) -> sigmoid -> Linear(4->1) -> sigmoid."""

    def __init__(self) -> None:
        super().__init__()
        self.embed = torch.nn.Linear(S.RAW_FEATURE_DIM, S.N_QUBITS, dtype=torch.float64)
        self.head = torch.nn.Linear(S.N_QUBITS, 1, dtype=torch.float64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.head(torch.sigmoid(self.embed(x))))

    def apply_init(self, seed: int) -> None:
        gen = torch.Generator()
        gen.manual_seed(int(seed))
        _init_linear(self.embed, gen)
        _init_linear(self.head, gen)


class MLPBaseline(torch.nn.Module):
    """C2: parameter-matched MLP, tanh hidden + sigmoid output."""

    def __init__(self, hidden: tuple[int, ...] = C2_HIDDEN) -> None:
        super().__init__()
        dims = [S.RAW_FEATURE_DIM, *hidden, 1]
        self.layers = torch.nn.ModuleList(
            [torch.nn.Linear(dims[i], dims[i + 1], dtype=torch.float64) for i in range(len(dims) - 1)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            x = torch.sigmoid(x) if i == len(self.layers) - 1 else torch.tanh(x)
        return x

    def apply_init(self, seed: int) -> None:
        gen = torch.Generator()
        gen.manual_seed(int(seed))
        for layer in self.layers:
            _init_linear(layer, gen)


def count_params(model: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def train_classical_baseline(
    model: torch.nn.Module,
    train_val: TrainValData,
    config: TrainingConfig,
    train_seed: int,
) -> dict:
    """Train a pure-classical baseline under the shared protocol. Returns a dict
    with train_loss_history, val_metric_history, final_val_metric, wall time, and
    the trained state_dict (for a separate protected-test pass)."""
    seeds = TrainingSeeds.from_train_seed(train_seed)
    start = time.perf_counter()
    if hasattr(model, "apply_init"):
        model.apply_init(seeds.param_init)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                  weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=list(config.lr_decay_epochs), gamma=config.lr_decay_factor)
    loss_fn = torch.nn.MSELoss()

    x_train = torch.tensor(train_val.train.features, dtype=torch.float64)
    y_train = torch.tensor(train_val.train.targets, dtype=torch.float64).reshape(-1, 1)
    x_val = torch.tensor(train_val.val.features, dtype=torch.float64)
    y_val = np.asarray(train_val.val.targets, dtype=np.float64)

    mb_rng = np.random.default_rng(seeds.minibatch)
    n = len(x_train)
    train_loss_history, val_metric_history = [], []
    for epoch in range(config.epochs):
        model.train()
        perm = mb_rng.permutation(n)
        losses = []
        for start_i in range(0, n, config.batch_size):
            idx = perm[start_i:start_i + config.batch_size]
            optimizer.zero_grad()
            pred = model(x_train[idx])
            loss = loss_fn(pred, y_train[idx])
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        scheduler.step()
        train_loss_history.append(float(np.mean(losses)))
        model.eval()
        with torch.no_grad():
            vp = model(x_val).cpu().numpy().reshape(-1)
        val_metric_history.append(compute_metric(train_val.spec.metric_name, vp, y_val))

    return {
        "train_loss_history": train_loss_history,
        "val_metric_history": val_metric_history,
        "final_val_metric": val_metric_history[-1],
        "wall_clock_seconds": time.perf_counter() - start,
        "state_dict": {k: v.detach().cpu().tolist() for k, v in model.state_dict().items()},
        "total_params": count_params(model),
    }


def evaluate_classical_on_test(model: torch.nn.Module, state_dict: dict,
                               test: DataSplit, metric_name: str) -> float:
    reloaded = {k: torch.tensor(v, dtype=torch.float64).reshape(model.state_dict()[k].shape)
                for k, v in state_dict.items()}
    model.load_state_dict(reloaded)
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(test.features, dtype=torch.float64)).cpu().numpy().reshape(-1)
    return compute_metric(metric_name, pred, test.targets)


# --- Fixed VQC baselines -----------------------------------------------------

def fixed_hea_genome() -> S.ControlledGenome:
    """Q1: hardware-efficient ansatz — 3 RY rotation layers (12 angles)
    interleaved with 2 CNOT-ring entanglers."""
    cnot_ring = next(i for i, t in enumerate(S.FREE_BLOCK_TEMPLATES) if t[1:3] == ("CNOT", "ring"))
    return S.ControlledGenome(blocks=[
        S.ParamBlock(kind="RY"),
        S.FreeBlock(template_index=cnot_ring),
        S.ParamBlock(kind="RY"),
        S.FreeBlock(template_index=cnot_ring),
        S.ParamBlock(kind="RY"),
    ])


def fixed_random_genome(seed: int = Q2_FIXED_SEED) -> S.ControlledGenome:
    """Q2: one architecture drawn with a predeclared seed, then frozen."""
    return S.random_genome(np.random.default_rng(seed))
