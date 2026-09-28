"""Canonical serialization, structural hashing, and round-trip tests."""

from __future__ import annotations

import json

from llm_vqc.ir.canonicalize import canonical_json, structural_hash
from llm_vqc.ir.schema import (
    CircuitIR,
    EncodingSpec,
    EntangleLayer,
    MeasurementSpec,
    RotationLayer,
)


def _sample_ir(**overrides) -> CircuitIR:
    defaults = dict(
        n_qubits=3,
        encoding=EncodingSpec(type="angle", gate="RY", wires="all"),
        layers=[
            RotationLayer(gates=["RY", "RZ"], wires="all"),
            EntangleLayer(pattern="ring", gate="CNOT"),
        ],
        measurements=MeasurementSpec(observable="Z", wires="all"),
    )
    defaults.update(overrides)
    return CircuitIR(**defaults)


def test_canonical_json_round_trips_to_an_equal_model():
    ir = _sample_ir()
    text = canonical_json(ir)
    restored = CircuitIR.model_validate(json.loads(text))
    assert restored == ir


def test_structural_hash_is_deterministic_across_repeated_calls():
    ir = _sample_ir()
    assert structural_hash(ir) == structural_hash(ir)


def test_structural_hash_is_stable_across_independently_constructed_equal_ir():
    ir_a = _sample_ir()
    ir_b = _sample_ir()
    assert ir_a is not ir_b
    assert structural_hash(ir_a) == structural_hash(ir_b)


def test_structural_hash_is_sha256_hex():
    h = structural_hash(_sample_ir())
    assert len(h) == 64
    int(h, 16)  # raises ValueError if not valid hex


def test_layer_order_changes_the_hash():
    ir_a = _sample_ir(
        layers=[
            RotationLayer(gates=["RX"], wires="all"),
            EntangleLayer(pattern="ring", gate="CNOT"),
        ]
    )
    ir_b = _sample_ir(
        layers=[
            EntangleLayer(pattern="ring", gate="CNOT"),
            RotationLayer(gates=["RX"], wires="all"),
        ]
    )
    assert structural_hash(ir_a) != structural_hash(ir_b)


def test_gate_order_within_a_layer_changes_the_hash():
    ir_a = _sample_ir(layers=[RotationLayer(gates=["RX", "RY"], wires="all")])
    ir_b = _sample_ir(layers=[RotationLayer(gates=["RY", "RX"], wires="all")])
    assert structural_hash(ir_a) != structural_hash(ir_b)


def test_metadata_differences_change_the_hash():
    """Metadata is part of the syntactic content and is not stripped."""
    ir_a = _sample_ir()
    ir_b = _sample_ir(metadata={"note": "variant"})
    assert structural_hash(ir_a) != structural_hash(ir_b)


def test_syntactically_different_but_physically_similar_circuits_hash_differently():
    """Canonicalization is syntactic identity, not physical equivalence —
    see canonicalize.py's module docstring. A star pattern and the
    equivalent explicit pairs list are NOT collapsed to the same hash."""
    ir_star = _sample_ir(
        n_qubits=3, layers=[EntangleLayer(pattern="star", gate="CNOT", center=0)]
    )
    ir_pairs = _sample_ir(
        n_qubits=3,
        layers=[EntangleLayer(pattern="pairs", gate="CNOT", pairs=[(0, 1), (0, 2)])],
    )
    assert structural_hash(ir_star) != structural_hash(ir_pairs)


def test_wires_all_shorthand_and_equivalent_explicit_list_hash_differently():
    """"all" and an explicit [0,1,2] list are syntactically distinct even
    though they resolve to the same wires for a given n_qubits — this is
    intentional (canonicalization does not resolve WireSpec shorthand)."""
    ir_all = _sample_ir(n_qubits=3, layers=[RotationLayer(gates=["RX"], wires="all")])
    ir_explicit = _sample_ir(n_qubits=3, layers=[RotationLayer(gates=["RX"], wires=[0, 1, 2])])
    assert structural_hash(ir_all) != structural_hash(ir_explicit)


def test_dict_key_order_does_not_affect_hash():
    """model_dump + sort_keys means hash is independent of Python dict
    insertion order, which pydantic could in principle vary across
    versions/paths."""
    ir = _sample_ir(metadata={"b": "2", "a": "1"})
    ir_reordered_metadata = _sample_ir(metadata={"a": "1", "b": "2"})
    assert structural_hash(ir) == structural_hash(ir_reordered_metadata)
