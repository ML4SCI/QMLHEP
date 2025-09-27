# models/pqc.py
import numpy as np
import torch
import torch.nn as nn
import pennylane as qml

class PQCModel(nn.Module):
    """
    PennyLane QNode wrapper with angle encoding and StronglyEntanglingLayers.
    Uses lightning.qubit + adjoint on CPU when available (fast), otherwise falls
    back to default.qubit + parameter-shift. Supports QMAML via weights_override.
    Multi-qubit readout: returns expvals for all wires, then a linear head -> 2 logits.
    """
    def __init__(
        self,
        num_qubits: int,
        depth: int,
        init_type: str = "qmaml",
        use_lightning: bool = True,
        bound_angles: bool = False,   # if True, tanh-bound inputs to (-pi, pi) for stability
        verbose: bool = False,
    ):
        super().__init__()
        self.num_qubits = num_qubits
        self.depth = depth
        self.bound_angles = bound_angles

        # Device selection + differentiation method

        _diff_method = "parameter-shift"
        if use_lightning:
            try:
                self.dev = qml.device("lightning.qubit", wires=num_qubits)
                _diff_method = "adjoint"
                if verbose: 
                    print("[PQC] Using lightning.qubit + adjoint")
            except Exception as e:
                if verbose: 
                    print(f"[PQC] Lightning unavailable ({e}); falling back to default.qubit + parameter-shift")
                self.dev = qml.device("default.qubit", wires=num_qubits, shots=None)
        else:
            self.dev = qml.device("default.qubit", wires=num_qubits, shots=None)
            if verbose: 
                print("[PQC] Using default.qubit + parameter-shift")

        # Quantum circuit

        @qml.qnode(self.dev, interface="torch", diff_method=_diff_method)
        def circuit(sample: torch.Tensor, weights: torch.Tensor):
            # sample: (num_qubits,), weights: (depth, num_qubits, 3)
            s = torch.tanh(sample) * np.pi if self.bound_angles else sample
            for i in range(num_qubits):
                qml.RY(s[i], wires=i)
            try:
                qml.StronglyEntanglingLayers(weights, wires=range(num_qubits))
            except AttributeError:
                qml.templates.StronglyEntanglingLayers(weights, wires=range(num_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]

        self.circuit = circuit

        # Initialization of weights

        if init_type == "zero":
            w = torch.zeros(depth, num_qubits, 3, dtype=torch.float64)
        elif init_type == "pi":
            w = torch.full((depth, num_qubits, 3), np.pi, dtype=torch.float64)
        elif init_type == "uniform":
            w = torch.rand(depth, num_qubits, 3, dtype=torch.float64) * (0.05 * np.pi)
        elif init_type == "gaussian":
            gamma = 1.0 / (4 * num_qubits * (depth + 2))
            w = torch.normal(0.0, gamma, size=(depth, num_qubits, 3), dtype=torch.float64)
        else:  # "qmaml" or fallback
            w = torch.randn(depth, num_qubits, 3, dtype=torch.float64)
        self.weights = nn.Parameter(w)

        # Linear classifier head
        self.fc = nn.Linear(self.num_qubits, 2)

    # Forward pass

    def forward(self, x: torch.Tensor, weights_override: torch.Tensor = None) -> torch.Tensor:
        """
        x: (B, num_qubits) or (num_qubits,)
        weights_override: optional (depth, num_qubits, 3) for QMAML inner-loop evaluation.
        """
        w = weights_override if weights_override is not None else self.weights

        x64 = x.to(torch.float64)
        w64 = w.to(torch.float64)
        if (weights_override is not None) and weights_override.requires_grad and not w64.requires_grad:
            w64.requires_grad_(True)

        # Evaluate circuit batch-wise
        if x64.dim() == 2:
            outs = [self.circuit(x64[i], w64) for i in range(x64.shape[0])]
            vals = torch.stack([torch.stack(o) for o in outs], dim=0)  # (B, num_qubits)
        elif x64.dim() == 1:
            vals = torch.stack(self.circuit(x64, w64)).unsqueeze(0)    # (1, num_qubits)
        else:
            raise ValueError(f"Expected x with dim 1 or 2, got {tuple(x.shape)}")

        logits = self.fc(vals.to(torch.float32))  # (B, 2)
        return logits

__all__ = ["PQCModel"]
