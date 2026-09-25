"""Fixture tests: re-encoded published circuits from Knipfer et al. 2026.

See tests/fixtures/knipfer_circuits.py for exactly which circuit is
re-encoded and why only one of the paper's circuits is claimed as an
exact reproduction.
"""

from __future__ import annotations

import numpy as np

from llm_vqc.ir.compiler_pennylane import to_qnode
from llm_vqc.ir.compiler_qiskit import to_qiskit_bound, to_qiskit_symbolic
from llm_vqc.ir.expand import build_program, count_parameters
from llm_vqc.ir.validators import validate_proposal
from tests.fixtures.knipfer_circuits import (
    CLAUDE_SIMPLE_QNN_EXPECTED_N_QUBITS,
    CLAUDE_SIMPLE_QNN_EXPECTED_PARAMETER_COUNT,
    claude_simple_qnn_best_circuit,
)


def test_claude_simple_qnn_fixture_is_valid():
    ir = claude_simple_qnn_best_circuit()
    result = validate_proposal(ir)
    assert result.valid, result.issues


def test_claude_simple_qnn_fixture_matches_paper_qubit_count():
    ir = claude_simple_qnn_best_circuit()
    assert ir.n_qubits == CLAUDE_SIMPLE_QNN_EXPECTED_N_QUBITS


def test_claude_simple_qnn_fixture_matches_paper_parameter_count():
    """The paper (Section 4.1.1) states: 'The final model also has 45
    trainable parameters'. This is the fixture's core correctness check:
    an independently-derived count from re-encoding the published literal
    code must match the independently-published number."""
    ir = claude_simple_qnn_best_circuit()
    assert count_parameters(ir) == CLAUDE_SIMPLE_QNN_EXPECTED_PARAMETER_COUNT


def test_claude_simple_qnn_fixture_measures_only_data_qubits():
    ir = claude_simple_qnn_best_circuit()
    program = build_program(ir)
    measured_wires = {m.wire for m in program.measurements}
    assert measured_wires == {0, 1, 2, 3, 4}


def test_claude_simple_qnn_fixture_compiles_on_pennylane():
    ir = claude_simple_qnn_best_circuit()
    program = build_program(ir)
    qnode = to_qnode(ir)
    rng = np.random.default_rng(0)
    inputs = rng.uniform(0, np.pi, size=program.num_inputs)
    weights = rng.uniform(-np.pi, np.pi, size=program.num_parameters)
    out = qnode(inputs, weights)
    assert len(out) == 5


def test_claude_simple_qnn_fixture_compiles_on_qiskit():
    ir = claude_simple_qnn_best_circuit()
    program = build_program(ir)
    rng = np.random.default_rng(0)
    inputs = rng.uniform(0, np.pi, size=program.num_inputs)
    weights = rng.uniform(-np.pi, np.pi, size=program.num_parameters)
    circuit = to_qiskit_bound(ir, inputs=list(inputs), weights=list(weights))
    assert circuit.num_qubits == 9
    assert circuit.num_parameters == 0  # fully bound

    symbolic = to_qiskit_symbolic(ir)
    assert symbolic.num_parameters == program.num_inputs + program.num_parameters
