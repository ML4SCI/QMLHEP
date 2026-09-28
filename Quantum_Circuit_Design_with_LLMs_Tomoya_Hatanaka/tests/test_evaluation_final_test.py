"""Tests for the quarantined final-test evaluation path."""

from __future__ import annotations

import pytest

from llm_vqc.evaluation.final_test import FinalTestError, evaluate_on_test
from llm_vqc.evaluation.seeds import train_seed_for_circuit
from llm_vqc.evaluation.training import TrainingConfig, train_model
from llm_vqc.ir.canonicalize import structural_hash
from llm_vqc.ir.schema import (
    CircuitIR,
    EncodingSpec,
    EntangleLayer,
    MeasurementSpec,
    RotationLayer,
)
from llm_vqc.tasks.t1_gaussian import T1GaussianPeakTask


def _small_ir() -> CircuitIR:
    return CircuitIR(
        n_qubits=3,
        encoding=EncodingSpec(type="angle", gate="RY", wires="all"),
        layers=[
            RotationLayer(gates=["RY", "RZ"], wires="all"),
            EntangleLayer(pattern="ring", gate="CNOT"),
        ],
        measurements=MeasurementSpec(observable="Z", wires="all"),
    )


@pytest.fixture(scope="module")
def trained_state():
    task = T1GaussianPeakTask()
    train_val = task.build(seed=0)
    ir = _small_ir()
    train_seed = train_seed_for_circuit(0, structural_hash(ir))
    output = train_model(ir, train_val, TrainingConfig(epochs=8), train_seed)
    assert output.success
    return ir, output, train_seed


def test_evaluate_on_test_does_not_retrain(trained_state):
    """Calling evaluate_on_test twice with the same weights must give the
    exact same test metric -- if it were retraining, results would vary
    with fresh random initialization."""
    ir, output, train_seed = trained_state
    task = T1GaussianPeakTask()
    test_split = task.build_test(seed=0)

    r1 = evaluate_on_test(ir, output.trained_classical_state, test_split, task.spec, train_seed)
    r2 = evaluate_on_test(ir, output.trained_classical_state, test_split, task.spec, train_seed)
    assert r1.test_metric_value == r2.test_metric_value


def test_evaluate_on_test_uses_the_full_quarantined_test_partition(trained_state):
    ir, output, train_seed = trained_state
    task = T1GaussianPeakTask()
    test_split = task.build_test(seed=0)

    result = evaluate_on_test(ir, output.trained_classical_state, test_split, task.spec, train_seed)
    assert result.n_test_samples == len(test_split) == 2000
    assert result.test_metric_name == "rmse"
    assert 0.0 < result.test_metric_value < 1.0


def test_evaluate_on_test_rejects_weights_from_a_different_circuit_shape(trained_state):
    """Loading a state dict whose keys don't match a freshly built model
    for the given IR must fail loudly, not silently reload a mismatched
    or partial set of weights."""
    ir, output, train_seed = trained_state
    task = T1GaussianPeakTask()
    test_split = task.build_test(seed=0)

    incompatible_state = {
        k: v for k, v in output.trained_classical_state.items() if "embed" not in k
    }
    with pytest.raises(FinalTestError):
        evaluate_on_test(ir, incompatible_state, test_split, task.spec, train_seed)


def test_final_test_result_matches_manual_forward_pass_computation(trained_state):
    """Cross-check against an independently constructed model + metric
    computation, not just internal self-consistency."""
    import torch

    from llm_vqc.evaluation.metrics import rmse
    from llm_vqc.evaluation.model import HybridQNNModel

    ir, output, train_seed = trained_state
    task = T1GaussianPeakTask()
    test_split = task.build_test(seed=0)

    model = HybridQNNModel(ir, raw_feature_dim=task.spec.raw_feature_dim, head_out_dim=1)
    state = {
        k: torch.tensor(v, dtype=torch.float64).reshape(model.state_dict()[k].shape)
        for k, v in output.trained_classical_state.items()
    }
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        x_test = torch.tensor(test_split.features, dtype=torch.float64)
        manual_pred = model(x_test).numpy().reshape(-1)
    manual_rmse = rmse(manual_pred, test_split.targets)

    result = evaluate_on_test(ir, output.trained_classical_state, test_split, task.spec, train_seed)
    assert result.test_metric_value == pytest.approx(manual_rmse)
