import math
import numpy as np
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float32)


def bspline_basis_matrix(num_splines: int, degree: int, grid: np.ndarray) -> np.ndarray:
    assert num_splines >= degree + 1
    n = num_splines - 1
    p = degree
    if n - p > 0:
        interior = np.linspace(0.0, 1.0, (n - p) + 2, dtype=float)[1:-1]
    else:
        interior = np.array([], dtype=float)
    knots = np.concatenate([np.zeros(p + 1), interior, np.ones(p + 1)])

    def N(i, r, t):
        if r == 0:
            left, right = knots[i], knots[i + 1]
            return np.where(((t >= left) & (t < right)) | ((right == 1.0) & (t == 1.0)), 1.0, 0.0)
        left_den = knots[i + r] - knots[i]
        right_den = knots[i + r + 1] - knots[i + 1]
        left_term = ((t - knots[i]) / left_den) * N(i, r - 1, t) if left_den > 0 else 0
        right_term = ((knots[i + r + 1] - t) / right_den) * N(i + 1, r - 1, t) if right_den > 0 else 0
        return left_term + right_term

    tgrid = np.asarray(grid, dtype=float)
    return np.vstack([N(i, p, tgrid) for i in range(num_splines)])


class QCBMState(nn.Module):
    def __init__(self, n_label_qubits: int, n_pos_qubits: int, depth: int = 3, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.L, self.P = n_label_qubits, n_pos_qubits
        self.n_qubits = self.L + self.P
        self.theta = nn.Parameter(0.01 * torch.randn(depth, self.n_qubits, 3).float())

        self.dev = qml.device("default.qubit", wires=self.n_qubits)

        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def qnode(weights):
            qml.templates.StronglyEntanglingLayers(weights, wires=range(self.n_qubits))
            return qml.probs(wires=range(self.n_qubits))

        self._qprobs = qnode

    def forward(self):
        return self._qprobs(self.theta.float()).to(torch.float32)


class LabelMixer(nn.Module):
    def __init__(self, qcbm: QCBMState, depth=1, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.qcbm = qcbm
        self.L, self.P = qcbm.L, qcbm.P
        self.phi = nn.Parameter(0.01 * torch.randn(depth, self.L, 3).float())

        self.dev = qml.device("default.qubit", wires=self.L + self.P)

        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def qnode(weights_qcbm, weights_label):
            qml.templates.StronglyEntanglingLayers(weights_qcbm, wires=range(self.L + self.P))
            if self.L > 0:
                qml.templates.StronglyEntanglingLayers(weights_label, wires=range(self.L))
            return qml.probs(wires=range(self.L + self.P))

        self._qprobs = qnode

    def forward(self):
        return self._qprobs(self.qcbm.theta.float(), self.phi.float()).to(torch.float32)


class QuantumBlock(nn.Module):
    def __init__(self, k_frequencies=3, depth=1, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.K = k_frequencies
        self.log_omega = nn.Parameter(torch.randn(self.K).float() * 0.05)
        self.phase = nn.Parameter(torch.zeros(self.K).float())
        self.w_cos = nn.Parameter(torch.randn(self.K).float() * 0.1)
        self.w_sin = nn.Parameter(torch.randn(self.K).float() * 0.1)

        self.dev = qml.device("default.qubit", wires=self.K)

        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def qnode(alpha_vec):
            for k in range(self.K):
                qml.RY(alpha_vec[k], wires=k)
            for k in range(self.K - 1):
                qml.CNOT(wires=[k, k + 1])
            z = [qml.expval(qml.PauliZ(k)) for k in range(self.K)]
            x = [qml.expval(qml.PauliX(k)) for k in range(self.K)]
            return z + x

        self._qnode = qnode

    def forward_scalar(self, x01_scalar: torch.Tensor) -> torch.Tensor:
        x01 = torch.clamp(x01_scalar.reshape(()).float(), 0.0, 1.0)
        omega = F.softplus(self.log_omega.float()) + 1e-4
        alpha = omega * (2 * math.pi * x01) + self.phase.float()
        outs = self._qnode(alpha.float())
        outs = torch.stack([torch.as_tensor(o, dtype=torch.float32) for o in outs], 0)
        z, x = outs[:self.K], outs[self.K:]
        return (self.w_cos.float() * z).sum() + (self.w_sin.float() * x).sum()

    def forward_batch(self, x01_vec: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.forward_scalar(val) for val in x01_vec.float()], 0)


class QuKANResidualEdge(nn.Module):
    def __init__(self, mixer: LabelMixer, n_label_qubits, n_pos_qubits, k=3):
        super().__init__()
        self.mixer = mixer
        self.Nlabel, self.Npos = 2 ** n_label_qubits, 2 ** n_pos_qubits
        self.wf = nn.Parameter(torch.tensor(0.5).float())
        self.wq = nn.Parameter(torch.tensor(0.5).float())
        self.qfour = QuantumBlock(k)

    def batch_forward(self, x_pos01, probs_flat):
        lp = probs_flat.view(self.Nlabel, self.Npos)
        idx = torch.round(torch.clamp(x_pos01.float(), 0, 1) * (self.Npos - 1)).long()
        idx = torch.clamp(idx, 0, self.Npos - 1)
        p_vals = lp[:, idx].sum(0).float()
        qfr_vals = self.qfour.forward_batch(x_pos01.float())
        return self.wf * p_vals + self.wq * qfr_vals


class QuKANRegressor(nn.Module):
    def __init__(self, input_dim=1, hidden_nodes=6, seed=0):
        super().__init__()
        self.qcbm = QCBMState(2, 5, depth=3, seed=seed)
        self.mixers, self.edges = nn.ModuleList(), nn.ModuleList()
        for m in range(hidden_nodes):
            for j in range(input_dim):
                mixer = LabelMixer(self.qcbm, depth=1, seed=seed + 97 * m + j)
                edge = QuKANResidualEdge(mixer, 2, 5, k=3)
                self.mixers.append(mixer)
                self.edges.append(edge)

        print(f"[QuKANRegressor] built edges: {hidden_nodes} nodes × {input_dim} inputs = {len(self.edges)} edges")

        self.readout = QuantumBlock(k_frequencies=3, seed=seed + 123)

    def pretrain_qcbm(self, degree=2, epochs=50, lr=5e-2):
        num_spl, Npos = 2 ** self.qcbm.L, 2 ** self.qcbm.P
        grid = np.linspace(0, 1, Npos)
        B = np.maximum(bspline_basis_matrix(num_spl, degree, grid), 0.0)
        B = (B + 1e-8) / B.sum(1, keepdims=True)
        target = torch.tensor((B / num_spl).reshape(-1), dtype=torch.float32)

        opt = torch.optim.Adam(self.qcbm.parameters(), lr=lr)
        for ep in range(epochs):
            opt.zero_grad()
            probs = self.qcbm()
            loss = F.mse_loss(probs, target)
            loss.backward()
            opt.step()
            if ep % 10 == 0 or ep == epochs - 1:
                tv = 0.5 * torch.sum(torch.abs(probs - target)).item()
                print(f"[QCBM pretrain] {ep:03d} | MSE={loss.item():.6f} | TV={tv:.6f}")

        self.qcbm.theta.requires_grad_(False)
        print("QCBM frozen.")

    def forward(self, X):
        X01 = torch.sigmoid(X.float())
        edge_probs = [mix().float() for mix in self.mixers]
        nodes, eidx = [], 0
        for m in range(len(self.mixers) // X.shape[1]):
            acc = torch.zeros(X.shape[0]).float()
            for j in range(X.shape[1]):
                out = self.edges[eidx].batch_forward(X01[:, j], edge_probs[eidx])
                acc = acc + out
                eidx += 1
            nodes.append(acc)
        H = torch.stack(nodes, 1).float()
        return self.readout.forward_batch(H.mean(1))





def f_func(x): return torch.tanh(10*x + 0.5 + F.relu(x**2) * 10)
def g_func(x): return torch.sin(x) + torch.cos(5*x) * torch.exp(-x**2) + F.relu(x - 0.5)
def h_func(x): return torch.sigmoid(3*x) + F.relu(torch.sin(2*x) + x**3)
def k_func(x): return torch.tanh(5*x - 2) + 3 * F.relu(torch.cos(x**2))
def m_func(x): return F.softplus(x**2 - 1) + torch.tanh(4*x + 0.1)
def n_func(x): return torch.exp(-x**2 + 0.3*x) + F.relu(torch.tanh(2*x - 1))

FUNCTION_MAP = {
    "f_func": f_func,
    "g_func": g_func,
    "h_func": h_func,
    "k_func": k_func,
    "m_func": m_func,
    "n_func": n_func,
}


def train_one_function(name, func, epochs=100, batch=64, seed=0):
    x = torch.linspace(-1, 1, 500).unsqueeze(1).float()
    y = func(x).float()
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    model = QuKANRegressor(input_dim=1, hidden_nodes=6, seed=seed)
    model.pretrain_qcbm()

    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=8e-4)
    mse = nn.MSELoss()

    train_losses, test_losses = [], []

    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        pred = model(X_train)
        loss = mse(pred, y_train.squeeze())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        with torch.no_grad():
            model.eval()
            test_loss = mse(model(X_test), y_test.squeeze())

        train_losses.append(loss.item())
        test_losses.append(test_loss.item())

        if (ep + 1) % 5 == 0 or ep == epochs - 1:
            print(f"[{name}] Epoch {ep+1} | Train Loss={loss.item():.5f} | Test Loss={test_loss.item():.5f}")

    
    model.eval()
    with torch.no_grad():
        preds = model(X_test).cpu().numpy()
        true = y_test.cpu().numpy()
        x_plot = X_test.cpu().numpy().squeeze()
        sort_idx = x_plot.argsort()

    plt.figure(figsize=(12, 5))

    
    plt.subplot(1, 2, 1)
    plt.plot(x_plot[sort_idx], true[sort_idx], label='Ground Truth', color='blue')
    plt.plot(x_plot[sort_idx], preds[sort_idx], '--', label='Prediction', color='red')
    plt.title(f"{name} – Prediction vs Ground Truth")
    plt.xlabel("Input x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid(True)

    
    
    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.title(f"{name} – Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"{name}_results.png")  
    plt.close()


def main():
    for name, fn in FUNCTION_MAP.items():
        print(f"\nTraining QuKAN Regressor on {name}")
        train_one_function(name, fn, epochs=500, batch=64, seed=0)



if __name__ == "__main__":
    main()
