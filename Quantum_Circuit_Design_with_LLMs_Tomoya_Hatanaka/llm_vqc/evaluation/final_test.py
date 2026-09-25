"""The quarantined final-test evaluation path.

**This module is deliberately never imported by `llm_vqc.evaluation.
harness`** — that is the actual enforcement mechanism, not just a
comment. Nothing in the ordinary search-facing proposal-evaluation loop
can reach a test partition or a test metric, because the code that knows
how to compute one lives here, in a module the search path never
touches.

Per the master plan's protocol (Section 6.3): "final comparison on test
metric of each run's selected circuit (selected on validation)" — this
function does not retrain. It reloads the exact weights a circuit was
already trained to (persisted via `llm_vqc.evaluation.harness`'s
candidate cache / result store) and evaluates them once on the
quarantined test partition. If it retrained, the "selected on
validation, scored on test" protocol would collapse into scoring
whatever the *test-time* random initialization happened to favor —
which is not what "final test evaluation" is supposed to measure.
"""

from __future__ import annotations

import torch

from llm_vqc.evaluation.metrics import compute_metric
from llm_vqc.evaluation.model import HybridQNNModel
from llm_vqc.evaluation.results import FinalTestResult
from llm_vqc.ir.canonicalize import structural_hash
from llm_vqc.ir.schema import CircuitIR
from llm_vqc.tasks.base import DataSplit, TaskSpec


class FinalTestError(Exception):
    """Raised when a trained model cannot be reconstructed or evaluated."""


def evaluate_on_test(
    ir: CircuitIR,
    trained_classical_state: dict[str, list[float]],
    test_split: DataSplit,
    task_spec: TaskSpec,
    train_seed: int,
) -> FinalTestResult:
    """Reload an already-trained model's exact weights and score it once
    on `test_split`. Never retrains; never touches validation data.
    """
    model = HybridQNNModel(
        ir, raw_feature_dim=task_spec.raw_feature_dim, head_out_dim=task_spec.classical_head_out_dim
    )
    current_state = model.state_dict()
    if set(trained_classical_state.keys()) != set(current_state.keys()):
        expected = sorted(current_state.keys())
        got = sorted(trained_classical_state.keys())
        raise FinalTestError(
            "trained_classical_state keys do not match a freshly-built model for this IR "
            f"(expected {expected}, got {got}); the IR passed here must be identical to "
            "the one the weights were trained on."
        )
    reloaded_state = {
        key: torch.tensor(values, dtype=torch.float64).reshape(current_state[key].shape)
        for key, values in trained_classical_state.items()
    }
    model.load_state_dict(reloaded_state)
    model.eval()

    with torch.no_grad():
        x_test = torch.tensor(test_split.features, dtype=torch.float64)
        predictions = model(x_test).cpu().numpy().reshape(-1)

    metric_value = compute_metric(task_spec.metric_name, predictions, test_split.targets)

    return FinalTestResult(
        task_name=task_spec.name,
        structural_hash=structural_hash(ir),
        train_seed=train_seed,
        test_metric_name=task_spec.metric_name,
        test_metric_value=metric_value,
        n_test_samples=len(test_split),
    )
