"""Structural cost metrics for a valid CircuitIR.

Deliberately limited to *structural* circuit properties (qubit count,
depth, gate counts, parameter count) — no evaluation metrics (loss,
accuracy, RMSE) and no task-specific logic live here. That separation
matters: these numbers must be computable, and comparable across search
arms, before any task/data exists (Phase 2+).

`depth` reuses Qiskit's own DAG-based depth calculation (`QuantumCircuit.
depth()`) rather than reimplementing critical-path analysis — gates on
disjoint wires can execute in parallel, and getting that right from
scratch would just re-derive what Qiskit already does correctly. All
other counts are computed directly from the linearized `CircuitProgram`
and do not depend on Qiskit at all.
"""

from __future__ import annotations

from pydantic import BaseModel

from llm_vqc.ir.compiler_qiskit import to_qiskit_bound
from llm_vqc.ir.expand import CircuitProgram, EntangleInstruction, build_program
from llm_vqc.ir.schema import CircuitIR


class CircuitCostSummary(BaseModel):
    n_qubits: int
    depth: int
    gate_count: int
    two_qubit_gate_count: int
    parameter_count: int
    input_count: int


def _dummy_bind_values(program: CircuitProgram) -> tuple[list[float], list[float]]:
    """Concrete placeholder values used *only* to materialize a circuit for
    structural counting (depth). Never used for evaluation — the specific
    values are arbitrary and must not be interpreted as meaningful.
    """
    weights = [0.0] * program.num_parameters
    if program.encoding_type == "amplitude":
        inputs = [1.0] + [0.0] * (program.num_inputs - 1)
    else:
        inputs = [0.0] * program.num_inputs
    return inputs, weights


def circuit_cost_summary(
    ir: CircuitIR, program: CircuitProgram | None = None
) -> CircuitCostSummary:
    program = program if program is not None else build_program(ir)

    inputs, weights = _dummy_bind_values(program)
    compiled = to_qiskit_bound(ir, inputs=inputs, weights=weights, program=program)

    two_qubit_gate_count = sum(
        1 for instr in program.body if isinstance(instr, EntangleInstruction)
    )
    gate_count = len(program.encoding_instructions) + len(program.body)
    if program.encoding_type == "amplitude":
        gate_count += 1  # the single state-preparation operation

    return CircuitCostSummary(
        n_qubits=program.n_qubits,
        depth=compiled.depth(),
        gate_count=gate_count,
        two_qubit_gate_count=two_qubit_gate_count,
        parameter_count=program.num_parameters,
        input_count=program.num_inputs,
    )
