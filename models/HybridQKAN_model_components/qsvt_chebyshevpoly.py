from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_iris
import pennylane as qml
import numpy as np
import torch
import torch.nn as nn

target_polys = [
    [0, 1],
    [-1, 0, 2],
    [0, -3, 0, 4],
    [1, 0, -8, 0, 8]
]
class QSVT(nn.Module):
    def __init__(self, degree=5, wires=4):
        super().__init__()
        self.degree = degree
        self.wires = wires
        self.target_polys = target_polys
        self.dev = qml.device("default.qubit", wires=wires)

        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def circuit(x, poly):
            for i in range(wires):
                qml.RY(x[i], wires=i)
            A = torch.diag(x)
            qml.qsvt(A, poly, encoding_wires=list(range(wires)), block_encoding="embedding")
            return [qml.expval(qml.PauliZ(i)) for i in range(wires)]

        self.qnode = circuit

    def forward(self, x):
        outputs = []
        for poly in self.target_polys:
            outputs.append(torch.tensor(self.qnode(x, poly)))
        return torch.cat(outputs)