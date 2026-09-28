"""Semantic validation tests: qubit bounds, duplicate operands, pattern
requirements, size constraints, and the "collect every issue" contract.
"""

from __future__ import annotations

from llm_vqc.ir.validators import (
    MAX_EXPANDED_GATE_APPLICATIONS,
    MAX_TOP_LEVEL_LAYERS,
    validate_proposal,
)


def _valid_base(**overrides):
    base = {
        "n_qubits": 4,
        "encoding": {"type": "angle", "gate": "RY", "wires": "all"},
        "layers": [],
        "measurements": {"observable": "Z", "wires": "all"},
    }
    base.update(overrides)
    return base


def test_valid_minimal_circuit_passes():
    result = validate_proposal(_valid_base())
    assert result.valid
    assert result.issues == []
    assert result.ir is not None


def test_out_of_bounds_wire_in_rotation_layer_is_rejected():
    # n_qubits=4 -> valid indices are 0..3
    raw = _valid_base(layers=[{"type": "rot", "gates": ["RX"], "wires": [4]}])
    result = validate_proposal(raw)
    assert not result.valid
    assert any(i.code == "wires.out_of_bounds" for i in result.issues)


def test_duplicate_wire_in_rotation_layer_is_rejected():
    raw = _valid_base(layers=[{"type": "rot", "gates": ["RX"], "wires": [0, 0, 1]}])
    result = validate_proposal(raw)
    assert not result.valid
    assert any(i.code == "wires.duplicate" for i in result.issues)


def test_empty_explicit_wire_list_is_rejected():
    raw = _valid_base(layers=[{"type": "rot", "gates": ["RX"], "wires": []}])
    result = validate_proposal(raw)
    assert not result.valid
    assert any(i.code == "rot.empty_wires" for i in result.issues)


def test_star_pattern_without_center_is_rejected():
    raw = _valid_base(layers=[{"type": "entangle", "pattern": "star", "gate": "CNOT"}])
    result = validate_proposal(raw)
    assert not result.valid
    assert any(i.code == "entangle.missing_center" for i in result.issues)


def test_star_pattern_center_out_of_bounds_is_rejected():
    raw = _valid_base(
        layers=[{"type": "entangle", "pattern": "star", "gate": "CNOT", "center": 99}]
    )
    result = validate_proposal(raw)
    assert not result.valid
    assert any(i.code == "entangle.center_out_of_bounds" for i in result.issues)


def test_non_star_pattern_with_center_set_is_rejected():
    raw = _valid_base(
        layers=[{"type": "entangle", "pattern": "ring", "gate": "CNOT", "center": 0}]
    )
    result = validate_proposal(raw)
    assert not result.valid
    assert any(i.code == "entangle.unexpected_center" for i in result.issues)


def test_pairs_pattern_without_pairs_is_rejected():
    raw = _valid_base(layers=[{"type": "entangle", "pattern": "pairs", "gate": "CNOT"}])
    result = validate_proposal(raw)
    assert not result.valid
    assert any(i.code == "entangle.missing_pairs" for i in result.issues)


def test_pairs_self_loop_is_rejected():
    raw = _valid_base(
        layers=[
            {"type": "entangle", "pattern": "pairs", "gate": "CNOT", "pairs": [[2, 2]]}
        ]
    )
    result = validate_proposal(raw)
    assert not result.valid
    assert any(i.code == "entangle.self_loop" for i in result.issues)


def test_pairs_out_of_bounds_index_is_rejected():
    raw = _valid_base(
        layers=[
            {"type": "entangle", "pattern": "pairs", "gate": "CNOT", "pairs": [[0, 99]]}
        ]
    )
    result = validate_proposal(raw)
    assert not result.valid
    assert any(i.code == "entangle.pair_out_of_bounds" for i in result.issues)


def test_non_pairs_pattern_with_pairs_set_is_rejected():
    raw = _valid_base(
        layers=[
            {
                "type": "entangle",
                "pattern": "ring",
                "gate": "CNOT",
                "pairs": [[0, 1]],
            }
        ]
    )
    result = validate_proposal(raw)
    assert not result.valid
    assert any(i.code == "entangle.unexpected_pairs" for i in result.issues)


def test_ring_pattern_with_single_wire_is_rejected():
    raw = _valid_base(
        layers=[{"type": "entangle", "pattern": "ring", "gate": "CNOT", "wires": [0]}]
    )
    result = validate_proposal(raw)
    assert not result.valid
    assert any(i.code == "entangle.too_few_wires" for i in result.issues)


def test_pairs_pattern_with_customized_wires_is_rejected():
    """`wires` has no effect on the 'pairs' pattern; setting it non-default
    is proposer confusion, not silently ignored."""
    raw = _valid_base(
        layers=[
            {
                "type": "entangle",
                "pattern": "pairs",
                "gate": "CNOT",
                "pairs": [[0, 1]],
                "wires": [0, 1, 2],
            }
        ]
    )
    result = validate_proposal(raw)
    assert not result.valid
    assert any(i.code == "entangle.unexpected_wires" for i in result.issues)


def test_angle_encoding_without_gate_is_rejected():
    raw = _valid_base(encoding={"type": "angle", "wires": "all"})
    result = validate_proposal(raw)
    assert not result.valid
    assert any(i.code == "encoding.missing_gate" for i in result.issues)


def test_amplitude_encoding_with_gate_is_rejected():
    raw = _valid_base(encoding={"type": "amplitude", "gate": "RY", "wires": "all"})
    result = validate_proposal(raw)
    assert not result.valid
    assert any(i.code == "encoding.unexpected_gate" for i in result.issues)


def test_amplitude_encoding_with_reupload_is_rejected():
    raw = _valid_base(encoding={"type": "amplitude", "wires": "all", "reupload": 1})
    result = validate_proposal(raw)
    assert not result.valid
    assert any(i.code == "encoding.amplitude_reupload_unsupported" for i in result.issues)


def test_too_many_top_level_layers_is_rejected():
    layers = [{"type": "rot", "gates": ["RX"], "wires": "all"}] * (MAX_TOP_LEVEL_LAYERS + 1)
    raw = _valid_base(layers=layers)
    result = validate_proposal(raw)
    assert not result.valid
    assert any(i.code == "circuit.too_many_layers" for i in result.issues)


def test_excessive_expanded_gate_count_is_rejected():
    raw = _valid_base(
        n_qubits=10,
        layers=[
            {
                "type": "repeat",
                "times": 2,
                "body": [{"type": "rot", "gates": ["RX", "RY", "RZ"], "wires": "all"}] * 5,
            }
        ]
        * 5,
    )
    result = validate_proposal(raw)
    assert not result.valid
    assert any(i.code == "circuit.too_many_gate_applications" for i in result.issues)
    assert MAX_EXPANDED_GATE_APPLICATIONS > 0  # sanity: constant is a positive bound


def test_repeat_block_body_layers_are_validated_too():
    """A validation problem nested inside a repeat block must still surface."""
    raw = _valid_base(
        layers=[
            {
                "type": "repeat",
                "times": 2,
                "body": [{"type": "rot", "gates": ["RX"], "wires": [99]}],
            }
        ]
    )
    result = validate_proposal(raw)
    assert not result.valid
    assert any(i.code == "wires.out_of_bounds" and "body[0]" in i.path for i in result.issues)


def test_multiple_independent_issues_are_all_collected_in_one_pass():
    """The core design contract: validate_proposal does not fail-fast."""
    raw = _valid_base(
        encoding={"type": "angle", "wires": [0, 0, 50]},  # missing gate + duplicate + oob
        layers=[
            {"type": "entangle", "pattern": "star", "gate": "CNOT"},  # missing center
            {
                "type": "entangle",
                "pattern": "pairs",
                "gate": "CNOT",
                "pairs": [[1, 1]],
            },  # self-loop
        ],
    )
    result = validate_proposal(raw)
    assert not result.valid
    codes = {i.code for i in result.issues}
    assert {
        "encoding.missing_gate",
        "wires.duplicate",
        "wires.out_of_bounds",
        "entangle.missing_center",
        "entangle.self_loop",
    } <= codes


def test_schema_level_failure_short_circuits_before_semantic_checks():
    """A raw dict that fails pydantic typing never reaches semantic checks
    (they assume a well-typed model and would error on a bad type)."""
    result = validate_proposal({"n_qubits": "not-an-int"})
    assert not result.valid
    assert all(i.code.startswith("schema.") for i in result.issues)
    assert result.ir is None


def test_already_parsed_circuitir_instance_is_accepted_directly():
    from llm_vqc.ir.schema import CircuitIR

    ir = CircuitIR.model_validate(_valid_base())
    result = validate_proposal(ir)
    assert result.valid
    assert result.ir is ir
