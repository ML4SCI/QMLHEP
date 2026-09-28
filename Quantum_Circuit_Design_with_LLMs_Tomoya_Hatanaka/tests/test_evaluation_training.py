"""Tests for the fixed training pipeline."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

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
            RotationLayer(gates=["RY"], wires="all"),
            EntangleLayer(pattern="ring", gate="CNOT"),
        ],
        measurements=MeasurementSpec(observable="Z", wires="all"),
    )


@pytest.fixture(scope="module")
def t1_data():
    task = T1GaussianPeakTask()
    return task.build(seed=0)


def test_training_config_rejects_zero_epochs():
    with pytest.raises(ValidationError):
        TrainingConfig(epochs=0)


def test_training_config_rejects_zero_batch_size():
    with pytest.raises(ValidationError):
        TrainingConfig(batch_size=0)


def test_training_config_rejects_nonpositive_learning_rate():
    with pytest.raises(ValidationError):
        TrainingConfig(learning_rate=0.0)
    with pytest.raises(ValidationError):
        TrainingConfig(learning_rate=-0.1)


def test_training_config_defaults_match_master_plan_section_6_3():
    config = TrainingConfig()
    assert config.optimizer == "adamw"
    assert config.learning_rate == 0.05
    assert config.epochs == 20
    assert config.batch_size == 16


def test_train_model_succeeds_and_loss_decreases(t1_data):
    ir = _small_ir()
    train_seed = train_seed_for_circuit(0, structural_hash(ir))
    result = train_model(ir, t1_data, TrainingConfig(epochs=8), train_seed)
    assert result.success
    assert result.epochs_completed == 8
    assert len(result.val_metric_history) == 8
    assert result.train_loss_history[-1] < result.train_loss_history[0]


def test_train_model_is_deterministic_given_the_same_inputs(t1_data):
    ir = _small_ir()
    train_seed = train_seed_for_circuit(0, structural_hash(ir))
    config = TrainingConfig(epochs=5)
    r1 = train_model(ir, t1_data, config, train_seed)
    r2 = train_model(ir, t1_data, config, train_seed)
    assert r1.train_loss_history == r2.train_loss_history
    assert r1.val_metric_history == r2.val_metric_history
    assert r1.trained_circuit_weights == r2.trained_circuit_weights


def test_train_model_different_train_seed_gives_different_result(t1_data):
    ir = _small_ir()
    config = TrainingConfig(epochs=5)
    r1 = train_model(ir, t1_data, config, train_seed_for_circuit(0, structural_hash(ir)))
    r2 = train_model(ir, t1_data, config, train_seed_for_circuit(1, structural_hash(ir)))
    assert r1.train_loss_history != r2.train_loss_history


def test_train_model_reports_divergence_as_failure_not_an_exception(t1_data):
    ir = _small_ir()
    train_seed = train_seed_for_circuit(0, structural_hash(ir))
    config = TrainingConfig(epochs=10, learning_rate=1e8)  # forces divergence
    result = train_model(ir, t1_data, config, train_seed)  # must not raise
    assert result.success is False
    assert result.error_message is not None
    assert "Diverged" in result.error_message
    assert result.trained_circuit_weights is None
    assert result.trained_classical_state is None


def test_failed_training_still_reports_partial_epoch_history(t1_data):
    ir = _small_ir()
    train_seed = train_seed_for_circuit(0, structural_hash(ir))
    result = train_model(ir, t1_data, TrainingConfig(epochs=10, learning_rate=1e8), train_seed)
    assert result.epochs_completed < 10
    assert result.epochs_completed == len(result.train_loss_history)


def test_trained_weights_have_expected_shapes(t1_data):
    ir = _small_ir()
    train_seed = train_seed_for_circuit(0, structural_hash(ir))
    result = train_model(ir, t1_data, TrainingConfig(epochs=3), train_seed)
    assert result.success
    assert len(result.trained_circuit_weights) == 3  # 1 RY per wire, n_qubits=3
    assert "embed.weight" in result.trained_classical_state
    assert "head.weight" in result.trained_classical_state
    assert "q_layer.weights" in result.trained_classical_state
