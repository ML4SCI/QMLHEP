import torch
import torch.nn as nn
import pennylane as qml

class QCBMState(nn.Module):
    """
    Quantum Circuit Born Machine (QCBM) state preparation.
    Produces probability distribution over 2^(L+P) outcomes.
    """
    def __init__(self, n_label_qubits: int, n_pos_qubits: int, depth: int = 3, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.L = n_label_qubits
        self.P = n_pos_qubits
        self.n_qubits = self.L + self.P
        self.depth = depth

        init = 0.01 * torch.randn(depth, self.n_qubits, 3, dtype=torch.float32)
        self.theta = nn.Parameter(init)

        self.dev = qml.device("default.qubit", wires=self.n_qubits)

        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def qnode(weights):
            qml.templates.StronglyEntanglingLayers(weights, wires=list(range(self.n_qubits)))
            return qml.probs(wires=list(range(self.n_qubits)))

        self._qprobs = qnode

    def forward(self):
        return self._qprobs(self.theta).to(torch.float32)

    @torch.no_grad()
    def freeze(self):
        self.theta.requires_grad_(False)
