"""IR re-encodings of published circuits from Knipfer, Roman, Matchev,
Matcheva, Gleyzer, "AI Agents for Variational Quantum Circuit Design"
(arXiv:2602.19387, Feb 2026) — the ML4SCI mentors' direct predecessor work
(see LLM-VQC_MASTER_PLAN.md Section 2.1).

Only ONE circuit from that paper is re-encoded here with a claim of exact
fidelity: the "Simple QNN" best circuit found by Claude 3.7 Sonnet
(paper Section 4.1.1, Figure 5), because the paper prints its literal
PennyLane source code plus its exact reported parameter count (45,
Section 4.1.1: "The final model also has 45 trainable parameters").

The Llama 3.3 70B best circuits (Simple QNN, §4.2; QuanvNN, §4.4) are
**not** re-encoded here, deliberately: the paper describes them only in
prose ("input encoding with an RX gate, followed by parameterized RY and
RZ gates...") without a literal code listing or an independently
derivable parameter count from the text alone (the QuanvNN 81-parameter
figure and Simple-QNN 48-parameter figure could not be reproduced exactly
from the qubit-count and gate-list description given — e.g. 5*n_qubits
does not divide evenly into 48 for any integer n_qubits). Constructing a
fixture that merely resembles the prose description, while silently
labeling it as reproducing the paper's circuit, would overstate what was
actually verified — the task instructions this fixture was built under
are explicit that placeholder or fabricated "reproductions" are not
acceptable. If a literal code listing for these circuits becomes
available, they can be added as exact fixtures the same way.
"""

from __future__ import annotations

from llm_vqc.ir.schema import (
    CircuitIR,
    EncodingSpec,
    EntangleLayer,
    MeasurementSpec,
    RotationLayer,
)

# Data qubits: 0-4 (5 qubits). Computation qubits: 5-8 (4 qubits). 9 total.
_DATA_QUBITS = [0, 1, 2, 3, 4]
_COMP_QUBITS = [5, 6, 7, 8]

# Paper's cited ground truth (Knipfer et al., Section 4.1.1 / 4.1.2):
# "9 qubits divided into 5 data qubits and 4 computation qubits"
# "The final model also has 45 trainable parameters ... and a test RMSE of 0.0326."
CLAUDE_SIMPLE_QNN_EXPECTED_N_QUBITS = 9
CLAUDE_SIMPLE_QNN_EXPECTED_PARAMETER_COUNT = 45


def claude_simple_qnn_best_circuit() -> CircuitIR:
    """Re-encoding of Knipfer et al. Figure 5 / Section 4.1.1's literal code:

    ```
    n_qubits = 9
    data_qubits = range(5); comp_qubits = range(5, 9)
    for i in comp_qubits: qml.Hadamard(wires=i)
    for i in data_qubits: qml.RY(inputs[i], wires=i)
    for i in range(n_qubits):
        qml.RX(weights[i, 0], wires=i); qml.RY(weights[i, 1], wires=i)
    central_qubit = 0
    for i in range(1, n_qubits): qml.CNOT(wires=[central_qubit, i])
    for i in data_qubits:
        qml.RZ(weights[i, 2], wires=i); qml.RY(weights[i, 3], wires=i)
    for i in comp_qubits:
        qml.RX(weights[i, 2], wires=i); qml.RZ(weights[i, 3], wires=i)
    for i, data_q in enumerate(data_qubits):
        if i < len(comp_qubits): qml.CNOT(wires=[data_q, comp_qubits[i]])
    for i in range(len(comp_qubits)):
        qml.CNOT(wires=[comp_qubits[i], comp_qubits[(i + 1) % len(comp_qubits)]])
    for i in range(n_qubits): qml.RX(weights[i, 4], wires=i)
    return [qml.expval(qml.PauliZ(i)) for i in range(5)]
    ```

    The one grammar extension this fixture required — adding `H` (a fixed,
    non-parameterized gate) to the rotation-layer gate set — is recorded
    as a Phase 1 design decision in DECISIONS.md; it is additive and does
    not change the validity of any previously-valid IR.
    """
    return CircuitIR(
        n_qubits=9,
        encoding=EncodingSpec(type="angle", gate="RY", wires=_DATA_QUBITS, reupload=0),
        layers=[
            # for i in comp_qubits: qml.Hadamard(wires=i)
            RotationLayer(gates=["H"], wires=_COMP_QUBITS),
            # for i in range(n_qubits): qml.RX(...); qml.RY(...)
            RotationLayer(gates=["RX", "RY"], wires="all"),
            # central_qubit = 0; for i in range(1, n_qubits): qml.CNOT([0, i])
            EntangleLayer(pattern="star", gate="CNOT", center=0, wires="all"),
            # for i in data_qubits: qml.RZ(...); qml.RY(...)
            RotationLayer(gates=["RZ", "RY"], wires=_DATA_QUBITS),
            # for i in comp_qubits: qml.RX(...); qml.RZ(...)
            RotationLayer(gates=["RX", "RZ"], wires=_COMP_QUBITS),
            # one-to-one data_qubits[i] -> comp_qubits[i] CNOTs
            EntangleLayer(
                pattern="pairs",
                gate="CNOT",
                pairs=list(zip(_DATA_QUBITS, _COMP_QUBITS, strict=False)),
            ),
            # ring CNOT among comp_qubits only
            EntangleLayer(pattern="ring", gate="CNOT", wires=_COMP_QUBITS),
            # final rotation layer: qml.RX per qubit
            RotationLayer(gates=["RX"], wires="all"),
        ],
        measurements=MeasurementSpec(observable="Z", wires=_DATA_QUBITS),
        metadata={
            "source": "Knipfer et al. 2026 (arXiv:2602.19387), Figure 5 / Section 4.1.1",
            "description": "Simple QNN best circuit found by Claude 3.7 Sonnet",
        },
    )
