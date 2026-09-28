"""Behavior under malformed or adversarially structured inputs.

None of these should ever raise an uncaught exception from
`validate_proposal` — a search algorithm or LLM agent can send arbitrary
garbage, and the contract is "return a structured ValidationResult",
never "crash the caller".
"""

from __future__ import annotations

import pytest

from llm_vqc.ir.validators import validate_proposal


@pytest.mark.parametrize(
    "raw",
    [
        None,
        {},
        [],
        "just a string",
        42,
        {"n_qubits": None},
        {"n_qubits": 3.5},  # float where int expected
        {"n_qubits": "3"},  # numeric string (pydantic v2 strict-ish for models w/o coercion config)
        {"n_qubits": 3, "encoding": None},
        {"n_qubits": 3, "encoding": "angle"},  # string instead of object
        {"n_qubits": 3, "encoding": {}, "layers": "not-a-list", "measurements": {}},
        {
            "n_qubits": 3,
            "encoding": {"type": "angle", "gate": "RY"},
            "layers": None,
            "measurements": {},
        },
        {
            "n_qubits": 3,
            "encoding": {"type": "angle", "gate": "RY"},
            "layers": [None],
            "measurements": {"observable": "Z"},
        },
        {
            "n_qubits": 3,
            "encoding": {"type": "angle", "gate": "RY"},
            "layers": [{"type": "rot", "gates": "RX"}],  # gates should be a list, not a bare string
            "measurements": {"observable": "Z"},
        },
        {
            "n_qubits": 3,
            "encoding": {"type": "angle", "gate": "RY"},
            "layers": [
                {
                    "type": "entangle",
                    "pattern": "pairs",
                    "gate": "CNOT",
                    "pairs": "not-a-list",
                }
            ],
            "measurements": {"observable": "Z"},
        },
        {
            "n_qubits": 3,
            "encoding": {"type": "angle", "gate": "RY"},
            "layers": [
                {
                    "type": "entangle",
                    "pattern": "pairs",
                    "gate": "CNOT",
                    "pairs": [[0, 1, 2]],  # triple instead of pair
                }
            ],
            "measurements": {"observable": "Z"},
        },
        {
            "n_qubits": 3,
            "encoding": {"type": "angle", "gate": "RY", "wires": [10**9]},  # huge index
            "layers": [],
            "measurements": {"observable": "Z"},
        },
        {
            "n_qubits": 3,
            "encoding": {"type": "angle", "gate": "RY", "wires": [-1]},  # negative index
            "layers": [],
            "measurements": {"observable": "Z"},
        },
        {
            "n_qubits": 3,
            "encoding": {"type": "angle", "gate": "RY"},
            "layers": [
                {"type": "rot", "gates": ["RX; DROP TABLE circuits;--"], "wires": "all"}
            ],
            "measurements": {"observable": "Z"},
        },
        {
            "n_qubits": 3,
            "encoding": {"type": "angle", "gate": "RY"},
            "layers": [{"type": "rot", "gates": ["RX"], "wires": "all", "unexpected_extra": True}],
            "measurements": {"observable": "Z"},
        },
    ],
)
def test_malformed_or_adversarial_input_never_raises_and_is_reported_invalid(raw):
    result = validate_proposal(raw)
    assert result.valid is False
    assert len(result.issues) >= 1
    assert result.ir is None


def test_deeply_nested_repeat_body_with_invalid_inner_layer_is_still_reported():
    raw = {
        "n_qubits": 3,
        "encoding": {"type": "angle", "gate": "RY"},
        "layers": [
            {
                "type": "repeat",
                "times": 3,
                "body": [
                    {"type": "rot", "gates": ["RX"], "wires": [999]},
                    {"type": "entangle", "pattern": "pairs", "gate": "CNOT", "pairs": [[0, 0]]},
                ],
            }
        ],
        "measurements": {"observable": "Z"},
    }
    result = validate_proposal(raw)
    assert not result.valid
    codes = {i.code for i in result.issues}
    assert "wires.out_of_bounds" in codes
    assert "entangle.self_loop" in codes


def test_very_large_but_schema_valid_layer_count_is_rejected_by_size_limit_not_a_crash():
    raw = {
        "n_qubits": 10,
        "encoding": {"type": "angle", "gate": "RY"},
        "layers": [{"type": "rot", "gates": ["RX"], "wires": "all"}] * 10_000,
        "measurements": {"observable": "Z"},
    }
    result = validate_proposal(raw)
    assert not result.valid
    assert any(i.code == "circuit.too_many_layers" for i in result.issues)
