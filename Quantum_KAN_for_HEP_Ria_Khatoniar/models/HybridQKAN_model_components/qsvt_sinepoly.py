import torch
import torch.nn as nn
import numpy as np
import pennylane as qml

class QSVT(nn.Module):
    def __init__(self, wires=4, degree=5):
        super().__init__()
        self.wires = wires
        self.dev = qml.device("default.qubit", wires=wires)

        xs = np.linspace(-1, 1, 200)
        fx = sum(np.sin(k * np.pi * xs) for k in [1, 3, 5])
        poly = np.polyfit(xs, fx, deg=degree)
        poly /= np.max(np.abs(np.polyval(poly, xs))) + 1e-6
        odd_poly = [c if i % 2 == 1 else 0 for i, c in enumerate(poly[::-1])]
        self.qsvt_phis = nn.Parameter(torch.tensor(odd_poly, dtype=torch.float32), requires_grad=True)
        self.theta = nn.Parameter(torch.randn(wires))

        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def circuit(x, phis, theta):
            for i in range(wires):
                qml.RY(x[i], wires=i)
            A = torch.diag(x.detach())
            ph_np = phis.detach().cpu().numpy()
            ph_np /= np.max(np.abs(ph_np)) + 1e-6
            qml.qsvt(A, ph_np, encoding_wires=list(range(wires)), block_encoding="embedding")
            for i in range(wires):
                qml.RZ(theta[i], wires=i)
            return [qml.expval(qml.PauliZ(i)) for i in range(wires)]

        self.qnode = circuit

    def forward(self, x):
        return self.qnode(x, self.qsvt_phis, self.theta)