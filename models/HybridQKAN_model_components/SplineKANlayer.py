import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.interpolate import BSpline

class KANLayer(nn.Module):
    def __init__(self, in_features, out_features, num_basis=30, order=3):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.knots = nn.Parameter(torch.tensor([np.linspace(-2, 2, num_basis+order+1) for _ in range(in_features)], dtype=torch.float32))
        self.weights = nn.Parameter(torch.randn(out_features, in_features, num_basis))
        self.bias = nn.Parameter(torch.randn(out_features))

    def forward(self, x):
        x = x.detach()
        B = []
        for i in range(self.in_features):
            xi = x[:, i].detach().cpu().numpy()
            knots_i = np.sort(self.knots[i].detach().cpu().numpy())
            bi = []
            for j in range(self.weights.shape[2]):
                coeff = np.zeros(self.weights.shape[2])
                coeff[j] = 1
                spline = BSpline(knots_i, coeff, 3, extrapolate=False)
                val_tensor = torch.from_numpy(np.nan_to_num(spline(xi))).float().unsqueeze(-1)
                bi.append(val_tensor)
            B.append(torch.cat(bi, dim=1).unsqueeze(1))
        basis_tensor = torch.cat(B, dim=1).to(x.device)
        out = torch.einsum("bij,kij->bk", basis_tensor, self.weights)
        return out + self.bias