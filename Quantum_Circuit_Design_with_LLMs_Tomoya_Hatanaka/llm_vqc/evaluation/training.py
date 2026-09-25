"""The fixed training pipeline (LLM-VQC_MASTER_PLAN.md Section 6.3).

Every future search arm, on a given task, trains through this *one*
function with the *one* `TrainingConfig` for that experiment condition —
nothing here is chosen per-arm. A search method may only get a different
optimizer, epoch count, or initialization scheme if that difference is
itself an explicitly declared experimental variable (a different
`TrainingConfig`, recorded in the run's config, not a hidden default that
happens to differ).

Master-plan-specified defaults (Section 6.3): AdamW, learning rate 0.05,
20 epochs, batch size 16, decay at fixed epochs. The exact decay epochs
and weight_decay value are not pinned by the master plan text beyond
"AdamW... with decay at fixed epochs" — `DEFAULT_LR_DECAY_EPOCHS` and
`DEFAULT_WEIGHT_DECAY` are Phase 2 engineering defaults (see
`DECISIONS.md`), not master-plan-mandated numbers, exactly analogous to
how Phase 1 documented `MAX_TOP_LEVEL_LAYERS` as an engineering default.

No early stopping, no checkpointing (matches Knipfer et al.'s own
training protocol exactly: "No early stopping or model checkpointing is
used" — keeping this identical avoids introducing an extra uncontrolled
difference between this study and the one it is meant to be comparable
with).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import torch
from pydantic import BaseModel, Field

from llm_vqc.evaluation.metrics import compute_metric
from llm_vqc.evaluation.model import HybridQNNModel
from llm_vqc.evaluation.seeds import TrainingSeeds
from llm_vqc.ir.schema import CircuitIR
from llm_vqc.tasks.base import TrainValData

DEFAULT_LR_DECAY_EPOCHS = (7, 13, 17)
DEFAULT_WEIGHT_DECAY = 1e-5


class TrainingConfig(BaseModel):
    """Every training hyperparameter, explicit — none hidden as a code default
    a caller could silently miss."""

    optimizer: Literal["adamw"] = "adamw"
    learning_rate: float = Field(default=0.05, gt=0)
    epochs: int = Field(default=20, ge=1)
    batch_size: int = Field(default=16, ge=1)
    lr_decay_factor: float = 0.5
    lr_decay_epochs: tuple[int, ...] = DEFAULT_LR_DECAY_EPOCHS
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    device: Literal["cpu"] = "cpu"
    dtype: Literal["float64"] = "float64"


class TrainingDivergedError(Exception):
    """Raised internally when training produces a non-finite loss."""


@dataclass
class TrainingOutput:
    success: bool
    epochs_completed: int
    train_loss_history: list[float] = field(default_factory=list)
    val_metric_history: list[float] = field(default_factory=list)
    final_val_metric: float | None = None
    trained_circuit_weights: list[float] | None = None
    trained_classical_state: dict[str, list[float]] | None = None
    error_message: str | None = None
    wall_clock_seconds: float = 0.0


def _state_dict_to_lists(model: HybridQNNModel) -> dict[str, list[float]]:
    return {
        name: param.detach().cpu().flatten().tolist() for name, param in model.state_dict().items()
    }


def train_model(
    ir: CircuitIR,
    train_val: TrainValData,
    config: TrainingConfig,
    train_seed: int,
    init_policy: Callable[[HybridQNNModel, int], None] | None = None,
    freeze_quantum: bool = False,
) -> TrainingOutput:
    """Train a fresh model on `ir` against `train_val`, deterministic given
    `(ir, train_val, config, train_seed)`. Never raises — training failures
    (divergence, backend errors) are reported in the returned
    `TrainingOutput`, not propagated as exceptions, so the harness can
    record a failed candidate without aborting the whole evaluation loop.

    `init_policy` (optional, backward compatible): a callable
    `(model, param_init_seed) -> None` applied to the freshly-built model to
    set an *explicit, versioned* initialization. When `None` (the default, and
    the only behavior any legacy run used), the model keeps PyTorch/PennyLane
    dependency-default initialization under the seeded RNG — legacy runs are
    never silently re-initialized.

    `freeze_quantum` (optional, backward compatible, default False): when True,
    the quantum layer's angles are initialized (by `init_policy` if given) and
    then frozen — only the classical embed and head are trained. Used by the v2
    "frozen random quantum features" ablation; `False` reproduces all prior runs.
    """
    seeds = TrainingSeeds.from_train_seed(train_seed)
    start = time.perf_counter()

    torch.manual_seed(seeds.param_init)
    model = HybridQNNModel(
        ir,
        raw_feature_dim=train_val.spec.raw_feature_dim,
        head_out_dim=train_val.spec.classical_head_out_dim,
    )
    if init_policy is not None:
        init_policy(model, seeds.param_init)
    if freeze_quantum:
        model.q_layer.weights.requires_grad_(False)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params, lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=list(config.lr_decay_epochs), gamma=config.lr_decay_factor
    )
    loss_fn = torch.nn.MSELoss() if train_val.spec.loss_name == "mse" else torch.nn.BCELoss()

    x_train = torch.tensor(train_val.train.features, dtype=torch.float64)
    y_train = torch.tensor(train_val.train.targets, dtype=torch.float64).reshape(-1, 1)
    x_val = torch.tensor(train_val.val.features, dtype=torch.float64)
    y_val_np = np.asarray(train_val.val.targets, dtype=np.float64)

    minibatch_rng = np.random.default_rng(seeds.minibatch)
    n_train = len(x_train)

    train_loss_history: list[float] = []
    val_metric_history: list[float] = []

    try:
        for epoch in range(config.epochs):
            model.train()
            permutation = minibatch_rng.permutation(n_train)
            epoch_losses = []
            for batch_start in range(0, n_train, config.batch_size):
                batch_indices = permutation[batch_start : batch_start + config.batch_size]
                x_batch = x_train[batch_indices]
                y_batch = y_train[batch_indices]

                optimizer.zero_grad()
                prediction = model(x_batch)
                loss = loss_fn(prediction, y_batch)
                if not torch.isfinite(loss):
                    raise TrainingDivergedError(
                        f"non-finite loss at epoch {epoch}: {float(loss.item())!r}"
                    )
                loss.backward()
                optimizer.step()
                epoch_losses.append(float(loss.item()))
            scheduler.step()
            train_loss_history.append(float(np.mean(epoch_losses)))

            model.eval()
            with torch.no_grad():
                val_prediction = model(x_val).cpu().numpy().reshape(-1)
            val_metric = compute_metric(train_val.spec.metric_name, val_prediction, y_val_np)
            val_metric_history.append(val_metric)

        elapsed = time.perf_counter() - start
        return TrainingOutput(
            success=True,
            epochs_completed=config.epochs,
            train_loss_history=train_loss_history,
            val_metric_history=val_metric_history,
            final_val_metric=val_metric_history[-1],
            trained_circuit_weights=model.q_layer.weights.detach().cpu().tolist(),
            trained_classical_state=_state_dict_to_lists(model),
            error_message=None,
            wall_clock_seconds=elapsed,
        )
    except (TrainingDivergedError, RuntimeError, ValueError) as exc:
        elapsed = time.perf_counter() - start
        return TrainingOutput(
            success=False,
            epochs_completed=len(train_loss_history),
            train_loss_history=train_loss_history,
            val_metric_history=val_metric_history,
            final_val_metric=None,
            trained_circuit_weights=None,
            trained_classical_state=None,
            error_message=f"{type(exc).__name__}: {exc}",
            wall_clock_seconds=elapsed,
        )
