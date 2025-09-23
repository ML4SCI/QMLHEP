import torch
import torch.nn as nn
import pennylane as qml
from QCBM import QCBMState

class LabelMixer(nn.Module):
    """
    Applies an extra entangling block on label qubits,
    after the QCBM has been prepared.
    """
    def __init__(self, qcbm: QCBMState, depth: int = 2, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.qcbm = qcbm
        self.L = qcbm.L
        self.P = qcbm.P
        self.n_qubits = qcbm.n_qubits
        self.depth = depth

        init = 0.01 * torch.randn(depth, self.L, 3, dtype=torch.float32)
        self.phi = nn.Parameter(init)

        self.dev = qml.device("default.qubit", wires=self.n_qubits)

        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def qnode(weights_qcbm, weights_label):
            qml.templates.StronglyEntanglingLayers(weights_qcbm, wires=list(range(self.n_qubits)))
            if self.L > 0:
                qml.templates.StronglyEntanglingLayers(weights_label, wires=list(range(self.L)))
            return qml.probs(wires=list(range(self.n_qubits)))

        self._qprobs = qnode

    def forward(self):
        return self._qprobs(self.qcbm.theta, self.phi).to(torch.float32)
