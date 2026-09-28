"""Tests for the hybrid classical-quantum torch model wrapper."""

from __future__ import annotations

import torch

from llm_vqc.evaluation.model import HybridQNNModel
from llm_vqc.ir.expand import build_program
from llm_vqc.ir.schema import (
    CircuitIR,
    EncodingSpec,
    EntangleLayer,
    MeasurementSpec,
    RotationLayer,
)


def _sample_ir(n_qubits=3) -> CircuitIR:
    return CircuitIR(
        n_qubits=n_qubits,
        encoding=EncodingSpec(type="angle", gate="RY", wires="all"),
        layers=[
            RotationLayer(gates=["RY", "RZ"], wires="all"),
            EntangleLayer(pattern="ring", gate="CNOT"),
        ],
        measurements=MeasurementSpec(observable="Z", wires="all"),
    )


def test_model_runs_on_cpu_only():
    ir = _sample_ir()
    model = HybridQNNModel(ir, raw_feature_dim=10, head_out_dim=1)
    for param in model.parameters():
        assert param.device.type == "cpu"
    out = model(torch.rand(4, 10, dtype=torch.float64))
    assert out.device.type == "cpu"


def test_output_shape_matches_head_out_dim():
    ir = _sample_ir()
    model = HybridQNNModel(ir, raw_feature_dim=7, head_out_dim=1)
    out = model(torch.rand(5, 7, dtype=torch.float64))
    assert out.shape == (5, 1)


def test_output_is_bounded_in_unit_interval_via_sigmoid():
    ir = _sample_ir()
    model = HybridQNNModel(ir, raw_feature_dim=7, head_out_dim=1)
    out = model(torch.rand(20, 7, dtype=torch.float64))
    assert (out >= 0).all() and (out <= 1).all()


def test_embed_layer_width_matches_circuit_num_inputs_not_n_qubits():
    """The embed layer must be sized from the circuit's own encoding
    width, not from n_qubits -- a circuit can encode into fewer wires
    than it has (e.g. data/compute qubit split)."""
    ir = CircuitIR(
        n_qubits=6,
        encoding=EncodingSpec(type="angle", gate="RY", wires=[0, 1, 2]),  # only 3 of 6 wires
        layers=[RotationLayer(gates=["H"], wires=[3, 4, 5])],
        measurements=MeasurementSpec(observable="Z", wires="all"),
    )
    program = build_program(ir)
    assert program.num_inputs == 3
    model = HybridQNNModel(ir, raw_feature_dim=21, head_out_dim=1)
    assert model.embed.out_features == 3


def test_gradients_flow_through_every_layer():
    ir = _sample_ir()
    model = HybridQNNModel(ir, raw_feature_dim=6, head_out_dim=1)
    out = model(torch.rand(4, 6, dtype=torch.float64))
    loss = ((out - 0.5) ** 2).mean()
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"{name} has no gradient"
        assert bool((param.grad != 0).any()), f"{name} has an all-zero gradient"


def test_batch_rows_produce_different_outputs():
    """Regression guard alongside the Phase 1 compiler-level test: at the
    model level, distinct rows of a batch must not collapse to the same
    prediction."""
    ir = _sample_ir()
    model = HybridQNNModel(ir, raw_feature_dim=5, head_out_dim=1)
    out = model(torch.rand(6, 5, dtype=torch.float64))
    distinct = len({round(float(v), 8) for v in out.detach().flatten()})
    assert distinct > 1


def test_zero_parameter_circuit_is_handled():
    """A circuit with only non-parameterized gates (H, CNOT) must still
    build and run -- q_layer.weights has shape (0,)."""
    ir = CircuitIR(
        n_qubits=2,
        encoding=EncodingSpec(type="angle", gate="RY", wires="all"),
        layers=[
            RotationLayer(gates=["H"], wires="all"),
            EntangleLayer(pattern="pairs", gate="CNOT", pairs=[(0, 1)]),
        ],
        measurements=MeasurementSpec(observable="Z", wires="all"),
    )
    model = HybridQNNModel(ir, raw_feature_dim=4, head_out_dim=1)
    assert model.q_layer.weights.shape == (0,)
    out = model(torch.rand(3, 4, dtype=torch.float64))
    assert out.shape == (3, 1)
