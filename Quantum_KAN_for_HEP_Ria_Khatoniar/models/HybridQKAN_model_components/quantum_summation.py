import pennylane as qml
import torch
import torch.nn as nn
import math

class QuantumSumBlock(nn.Module):
    def __init__(self, num_polynomials):
        """
        Quantum summation over N polynomial outputs using Hadamard test.
        Supports arbitrary N (must be <= 2^num_index_qubits).
        
        Args:
            phi_vals (tensor): shape (N,), tensor of real-valued polynomial outputs.
        
        Returns:
            torch.Tensor: scalar value (quantum sum approximation via Hadamard test)
        """
        super().__init__()
        self.num_polynomials = num_polynomials
        self.weights = nn.Parameter(torch.randn(num_polynomials))  # trainable weights

        self.num_index_qubits = math.ceil(math.log2(num_polynomials))
        self.total_wires = self.num_index_qubits + 2  # index + hadamard + target
        self.wires = list(range(self.total_wires))

        self.dev = qml.device("default.qubit", wires=self.total_wires)

        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def circuit(phi_vals, wts):
            index_wires = self.wires[:self.num_index_qubits]
            hadamard_wire = self.wires[self.num_index_qubits]
            target_wire = self.wires[-1]

            for i in index_wires:
                qml.Hadamard(wires=i)
            qml.Hadamard(wires=hadamard_wire)

            for p in range(self.num_polynomials):
                ctrl_bin = [int(b) for b in f"{p:0{self.num_index_qubits}b}"]
                ctrl_values = [1] + ctrl_bin
                qml.ctrl(qml.RZ, control=[hadamard_wire] + index_wires, control_values=ctrl_values)(
                    2 * wts[p] * phi_vals[p], wires=target_wire
                )

            qml.Hadamard(wires=hadamard_wire)
            return qml.expval(qml.PauliZ(hadamard_wire))

        self.circuit = circuit

    def forward(self, phi_vals):
        return self.circuit(phi_vals, self.weights)
