import torch
import torch.nn as nn
from qsvt_sinepoly import QSVT
from LCU import quantum_lcu_block
from quantum_summation import QuantumSumBlock
from SplineKANlayer import KANLayer

class QuantumKANRegressor(nn.Module):
    def __init__(self, num_features, degree=3):
        super().__init__()
        self.num_features = num_features
        self.degree = degree

        self.qsvt = QSVT(wires=1, degree=degree, depth=2)
        self.lcu_weights = nn.Parameter(torch.rand(num_features, degree))  # (F, P)
        self.sum_blocks = nn.ModuleList([QuantumSumBlock(degree) for _ in range(num_features)])
        self.kan = KANLayer(in_features=num_features, out_features=1)

    def forward(self, X):
        """
        Input: X of shape (B, F)
        Output: y_hat of shape (B, 1)
        """
        B = X.size(0)
        all_features = []

        for i in range(B):
            xi = X[i]  # shape: (F,)
            qsvt_vecs = [self.qsvt(xi[f]) for f in range(self.num_features)]  # each: (P,)
            lcu_vals = [quantum_lcu_block(qsvt_vecs[f], self.lcu_weights[f]) for f in range(self.num_features)]
            summed = [self.sum_blocks[f](lcu_vals[f]) for f in range(self.num_features)]
            all_features.append(torch.stack(summed))  # shape: (F,)

        features = torch.stack(all_features)  # shape: (B, F)
        return self.kan(features)  # shape: (B, 1)
