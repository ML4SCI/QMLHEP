import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml

class QuantumFourierBlock(nn.Module):
    """
    Quantum Fourier Residual block.
    Maps input scalars in [0,1] into Fourier-like features
    using quantum rotations and entangling layers.
    """
    def __init__(self, k_frequencies: int = 4, entangle_depth: int = 1, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.K = k_frequencies
        self.depth = entangle_depth

        self.log_omega = nn.Parameter(torch.randn(self.K, dtype=torch.float32) * 0.05)
        self.phase = nn.Parameter(torch.zeros(self.K, dtype=torch.float32))
        self.w_cos = nn.Parameter(torch.randn(self.K, dtype=torch.float32) * 0.1)
        self.w_sin = nn.Parameter(torch.randn(self.K, dtype=torch.float32) * 0.1)

        self.dev = qml.device("default.qubit", wires=self.K)

        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def qnode(alpha_vec):
            for k in range(self.K):
                qml.RY(alpha_vec[k], wires=k)
            for _ in range(self.depth):
                for k in range(self.K):
                    qml.CNOT(wires=[k, (k + 1) % self.K])
            z = [qml.expval(qml.PauliZ(k)) for k in range(self.K)]
            x = [qml.expval(qml.PauliX(k)) for k in range(self.K)]
            return z + x

        self._qnode = qnode

    def forward_scalar(self, x01_scalar: torch.Tensor) -> torch.Tensor:
        x01 = torch.clamp(x01_scalar.reshape(()), 0.0, 1.0)
        omega = F.softplus(self.log_omega) + 1e-4
        alpha = omega * (2.0 * math.pi * x01) + self.phase
        outs = self._qnode(alpha.to(torch.float32))
        outs = torch.stack([torch.as_tensor(o, dtype=torch.float32) for o in outs], dim=0)
        z = outs[: self.K]
        x = outs[self.K:]
        return (self.w_cos * z).sum() + (self.w_sin * x).sum()

    def forward_batch(self, x01_vec: torch.Tensor) -> torch.Tensor:
        x01_vec = torch.clamp(x01_vec.to(torch.float32), 0.0, 1.0)
        vals = [self.forward_scalar(x01_vec[i]) for i in range(x01_vec.shape[0])]
        return torch.stack(vals, dim=0).to(torch.float32)
