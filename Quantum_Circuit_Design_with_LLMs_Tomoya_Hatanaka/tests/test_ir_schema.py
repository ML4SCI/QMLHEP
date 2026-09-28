"""Schema-level (structural typing) tests for the circuit IR."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from llm_vqc.ir.schema import (
    MAX_QUBITS,
    MIN_QUBITS,
    CircuitIR,
    EncodingSpec,
    EntangleLayer,
    MeasurementSpec,
    RepeatBlock,
    RotationLayer,
)


def _minimal_ir(**overrides):
    defaults = dict(
        n_qubits=3,
        encoding=EncodingSpec(type="angle", gate="RY", wires="all"),
        layers=[RotationLayer(gates=["RX"], wires="all")],
        measurements=MeasurementSpec(observable="Z", wires="all"),
    )
    defaults.update(overrides)
    return CircuitIR(**defaults)


def test_minimal_valid_circuit_constructs():
    ir = _minimal_ir()
    assert ir.n_qubits == 3


@pytest.mark.parametrize("n_qubits", [MIN_QUBITS, MAX_QUBITS])
def test_qubit_count_bounds_are_inclusive(n_qubits):
    ir = _minimal_ir(n_qubits=n_qubits)
    assert ir.n_qubits == n_qubits


@pytest.mark.parametrize("n_qubits", [MIN_QUBITS - 1, MAX_QUBITS + 1, 0, -1])
def test_qubit_count_out_of_bounds_rejected(n_qubits):
    with pytest.raises(ValidationError):
        _minimal_ir(n_qubits=n_qubits)


def test_unknown_top_level_field_rejected():
    with pytest.raises(ValidationError):
        CircuitIR.model_validate(
            {
                "n_qubits": 3,
                "encoding": {"type": "angle", "gate": "RY"},
                "layers": [],
                "measurements": {"observable": "Z"},
                "totally_unexpected_field": 123,
            }
        )


def test_unknown_gate_name_rejected():
    with pytest.raises(ValidationError):
        RotationLayer(gates=["NOT_A_GATE"], wires="all")


def test_unknown_entangle_pattern_rejected():
    with pytest.raises(ValidationError):
        EntangleLayer(pattern="hypercube", gate="CNOT")


def test_unknown_layer_type_discriminator_rejected():
    with pytest.raises(ValidationError):
        CircuitIR.model_validate(
            {
                "n_qubits": 3,
                "encoding": {"type": "angle", "gate": "RY"},
                "layers": [{"type": "teleport"}],
                "measurements": {"observable": "Z"},
            }
        )


def test_repeat_block_requires_at_least_one_body_layer():
    with pytest.raises(ValidationError):
        RepeatBlock(times=2, body=[])


def test_repeat_block_requires_positive_times():
    with pytest.raises(ValidationError):
        RepeatBlock(times=0, body=[RotationLayer(gates=["RX"], wires="all")])


def test_repeat_block_rejects_nested_repeat():
    """Repeat-of-repeat is intentionally unsupported in Phase 1 (see README)."""
    with pytest.raises(ValidationError):
        RepeatBlock(
            times=2,
            body=[{"type": "repeat", "times": 2, "body": [{"type": "rot", "gates": ["RX"]}]}],
        )


def test_rotation_layer_requires_nonempty_gates():
    with pytest.raises(ValidationError):
        RotationLayer(gates=[], wires="all")


def test_wires_all_shorthand_and_explicit_list_both_accepted():
    layer_all = RotationLayer(gates=["RX"], wires="all")
    layer_explicit = RotationLayer(gates=["RX"], wires=[0, 2])
    assert layer_all.wires == "all"
    assert layer_explicit.wires == [0, 2]


def test_encoding_defaults_to_reupload_zero():
    enc = EncodingSpec(type="angle", gate="RY")
    assert enc.reupload == 0


def test_encoding_rejects_negative_reupload():
    with pytest.raises(ValidationError):
        EncodingSpec(type="angle", gate="RY", reupload=-1)


def test_measurement_defaults_to_z_and_all_wires():
    m = MeasurementSpec()
    assert m.observable == "Z"
    assert m.wires == "all"
