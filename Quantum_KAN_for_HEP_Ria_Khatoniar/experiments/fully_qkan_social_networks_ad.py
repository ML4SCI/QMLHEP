import os
import math
import numpy as np
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


CSV_PATH = "C:\\Users\\riakh\\Downloads\\Social_Network_Ads.csv"


torch.set_default_dtype(torch.float32)


def bspline_basis_matrix(num_splines: int, degree: int, grid: np.ndarray) -> np.ndarray:
    """
    Open-uniform B-spline basis on [0,1].
    num_splines = n+1, degree = p. Knot vector length must be n+p+2 with p+1 repeats at each end,
    and exactly (n-p) interior knots.
    """
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
            left = knots[i]
            right = knots[i + 1]
            
            return np.where(((t >= left) & (t < right)) | ((right == 1.0) & (t == 1.0)), 1.0, 0.0)
        left_den = knots[i + r] - knots[i]
        right_den = knots[i + r + 1] - knots[i + 1]
        left_term = 0.0
        right_term = 0.0
        if left_den > 0:
            left_term = ((t - knots[i]) / left_den) * N(i, r - 1, t)
        if right_den > 0:
            right_term = ((knots[i + r + 1] - t) / right_den) * N(i + 1, r - 1, t)
        return left_term + right_term

    tgrid = np.asarray(grid, dtype=float)
    B = np.vstack([N(i, p, tgrid) for i in range(num_splines)])
    return np.maximum(B, 0.0)




class QCBMState(nn.Module):
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




class LabelMixer(nn.Module):
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




class QuantumBlock(nn.Module):
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
        outs = torch.stack([o if isinstance(o, torch.Tensor) else torch.as_tensor(o, dtype=torch.float32)
                            for o in outs], dim=0).to(torch.float32)
        z = outs[: self.K]
        x = outs[self.K:]
        return (self.w_cos * z).sum() + (self.w_sin * x).sum()

    def forward_batch(self, x01_vec: torch.Tensor) -> torch.Tensor:
        x01_vec = torch.clamp(x01_vec.to(torch.float32), 0.0, 1.0)
        vals = [self.forward_scalar(x01_vec[i]) for i in range(x01_vec.shape[0])]
        return torch.stack(vals, dim=0).to(torch.float32)




class QuKANResidualEdge(nn.Module):
    def __init__(self, mixer: LabelMixer, n_label_qubits: int, n_pos_qubits: int,
                 fourier_k: int = 4, fourier_depth: int = 1, seed: int = 0, w_init=0.5):
        super().__init__()
        self.mixer = mixer
        self.L = n_label_qubits
        self.P = n_pos_qubits
        self.Nlabel = 2 ** self.L
        self.Npos = 2 ** self.P

        self.wf = nn.Parameter(torch.tensor(float(w_init), dtype=torch.float32))  
        self.wq = nn.Parameter(torch.tensor(float(w_init), dtype=torch.float32))  

        self.qfour = QuantumBlock(k_frequencies=fourier_k, entangle_depth=fourier_depth, seed=seed)

    def batch_forward(self, x_pos01: torch.Tensor, probs_flat: torch.Tensor) -> torch.Tensor:
        """
        x_pos01: (B,) clamped to [0,1] (for position index AND Fourier phase)
        probs_flat: (Nlabel*Npos,) from the edge's LabelMixer()
        returns: (B,)
        """
        x_pos01 = x_pos01.to(torch.float32)
        probs_flat = probs_flat.to(torch.float32)

        B = x_pos01.shape[0]
        lp = probs_flat.view(self.Nlabel, self.Npos)  

        idx = torch.round(torch.clamp(x_pos01, 0.0, 1.0) * (self.Npos - 1)).long()  
        idx = torch.clamp(idx, 0, self.Npos - 1)

        p_vals = lp[:, idx].sum(dim=0).to(torch.float32)  
        qfr_vals = self.qfour.forward_batch(x_pos01)      

        out = (self.wf * p_vals + self.wq * qfr_vals).to(torch.float32)  
        return out




@dataclass
class QuKANLayerCfg:
    n_nodes: int = 6
    n_label_qubits: int = 2   
    n_pos_qubits: int = 5     
    qcbm_depth: int = 3
    label_mixer_depth: int = 2
    fourier_k: int = 4
    fourier_depth: int = 1

class QuKANLayer(nn.Module):
    """
    KAN-style: node_m = sum_j f_edge_{m,j}(x_j)
    Quantum part is independent of x (probabilities over pos bins);
    x affects which position bin is read out and the Fourier phase.
    """
    def __init__(self, cfg: QuKANLayerCfg, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.cfg = cfg
        self.L = cfg.n_label_qubits
        self.P = cfg.n_pos_qubits

        self.qcbm = QCBMState(self.L, self.P, depth=cfg.qcbm_depth, seed=seed)
        self.mixers = nn.ModuleList()
        self.edges = nn.ModuleList()
        self._built = False

    def build(self, input_dim: int, seed: int = 0):
        self.input_dim = input_dim
        torch.manual_seed(seed)
        for m in range(self.cfg.n_nodes):
            for j in range(input_dim):
                mixer = LabelMixer(self.qcbm, depth=self.cfg.label_mixer_depth, seed=seed + 97 * m + j)
                edge = QuKANResidualEdge(
                    mixer,
                    self.L, self.P,
                    fourier_k=self.cfg.fourier_k,
                    fourier_depth=self.cfg.fourier_depth,
                    seed=seed + 991 * m + 13 * j,
                    w_init=0.5
                )
                self.mixers.append(mixer)
                self.edges.append(edge)
        self._built = True
        print(f"built edges: nodes={self.cfg.n_nodes}, in_dim={input_dim}, total_edges={len(self.edges)}")

    def pretrain_qcbm_on_splines(self, degree=2, epochs=200, lr=5e-2, verbose=True):
        num_splines = 2 ** self.L
        Npos = 2 ** self.P
        grid = np.linspace(0.0, 1.0, Npos, dtype=float)
        B = bspline_basis_matrix(num_splines, degree, grid)  

        B = B + 1e-8
        B = B / B.sum(axis=1, keepdims=True)
        target = torch.tensor((B / num_splines).reshape(-1), dtype=torch.float32)  

        opt = torch.optim.Adam(self.qcbm.parameters(), lr=lr)
        for ep in range(1, epochs + 1):
            opt.zero_grad()
            probs = self.qcbm().to(torch.float32)            
            loss = F.mse_loss(probs, target)
            loss.backward()
            opt.step()
            if verbose and (ep % 50 == 0 or ep == 1):
                with torch.no_grad():
                    tv = 0.5 * torch.sum(torch.abs(probs - target)).item()
                print(f"[QCBM pretrain] epoch {ep:03d} | MSE={loss.item():.6f} | TV={tv:.6f}")
        self.qcbm.freeze()

    def forward(self, X_in: torch.Tensor, input_is_01: bool) -> torch.Tensor:
        """
        X_in: (B, D) inputs to the layer
        input_is_01: True if X_in is already in [0,1] (layer 1); False for layer 2 (we'll squash for position/phase)
        returns: (B, M) node outputs
        """
        assert self._built, "Call build(input_dim) first."
        X_in = X_in.to(torch.float32)
        B, D = X_in.shape
        M = self.cfg.n_nodes

        edge_probs = [mix().to(torch.float32) for mix in self.mixers]  
        X01_pos = (X_in if input_is_01 else torch.sigmoid(X_in)).to(torch.float32)

        nodes = []
        eidx = 0
        for m in range(M):
            acc = torch.zeros(B, dtype=torch.float32, device=X_in.device)
            for j in range(D):
                probs_flat = edge_probs[eidx]
                edge = self.edges[eidx]
                x_pos = X01_pos[:, j].to(torch.float32)    
                out_j = edge.batch_forward(x_pos, probs_flat).to(torch.float32)  
                acc = acc + out_j
                eidx += 1
            nodes.append(acc)
        nodes = torch.stack(nodes, dim=1).to(torch.float32)  
        return nodes


@dataclass
class KANReadoutCfg:
    n_classes: int
    in_dim: int
    fourier_k: int = 3
    fourier_depth: int = 1

class KANReadout(nn.Module):
    def __init__(self, cfg: KANReadoutCfg, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.cfg = cfg
        C, M = cfg.n_classes, cfg.in_dim

        self.qfr = nn.ModuleList([
            QuantumBlock(k_frequencies=cfg.fourier_k,
                                entangle_depth=cfg.fourier_depth,
                                seed=seed + 131 * c + m)
            for c in range(C) for m in range(M)
        ])
        self.b = nn.Parameter(torch.zeros(C, dtype=torch.float32))

    def _edge_idx(self, c: int, m: int) -> int:
        return c * self.cfg.in_dim + m

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        H = H.to(torch.float32)
        B, M = H.shape
        C = self.cfg.n_classes
        H01 = torch.sigmoid(H)  

        logits = []
        for c in range(C):
            acc_c = torch.zeros(B, dtype=torch.float32, device=H.device)
            for m in range(M):
                qfr = self.qfr[self._edge_idx(c, m)]
                acc_c = acc_c + qfr.forward_batch(H01[:, m])
            logits.append(acc_c + self.b[c])
        return torch.stack(logits, dim=1).to(torch.float32)  




@dataclass
class QuKANNetCfg:
    layer1: QuKANLayerCfg = field(default_factory=lambda: QuKANLayerCfg(n_nodes=6, n_label_qubits=2, n_pos_qubits=5, fourier_k=4, fourier_depth=1))
    layer2: QuKANLayerCfg = field(default_factory=lambda: QuKANLayerCfg(n_nodes=6, n_label_qubits=2, n_pos_qubits=5, fourier_k=4, fourier_depth=1))
    n_classes: int = 2  

class QuKANNet(nn.Module):
    def __init__(self, cfg: QuKANNetCfg, input_dim: int, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.cfg = cfg
        
        self.l1 = QuKANLayer(cfg.layer1, seed=seed)
        self.l1.build(input_dim=input_dim, seed=seed)
        
        self.l2 = QuKANLayer(cfg.layer2, seed=seed + 1)
        self.l2.build(input_dim=cfg.layer1.n_nodes, seed=seed + 1)

        
        self.readout = KANReadout(
            KANReadoutCfg(
                n_classes=cfg.n_classes,
                in_dim=cfg.layer2.n_nodes,
                fourier_k=3,        
                fourier_depth=1
            ),
            seed=seed + 1234
        )

    def pretrain_qcbms(self, degree=2, epochs=200, lr=5e-2, verbose=True):
        print("[Pretrain] Layer 1 QCBM on degree-2 B-splines")
        self.l1.pretrain_qcbm_on_splines(degree=degree, epochs=epochs, lr=lr, verbose=verbose)
        print("[Pretrain] Layer 2 QCBM on degree-2 B-splines")
        self.l2.pretrain_qcbm_on_splines(degree=degree, epochs=epochs, lr=lr, verbose=verbose)

    def forward(self, X01: torch.Tensor) -> torch.Tensor:
        X01 = X01.to(torch.float32)
        h1 = self.l1(X01, input_is_01=True).to(torch.float32)     
        h2 = self.l2(h1,  input_is_01=False).to(torch.float32)    
        return self.readout(h2)                                    




def run_social(seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)

    assert os.path.exists(CSV_PATH), f"CSV not found: {CSV_PATH}"
    df = pd.read_csv(CSV_PATH)

    
    cols = [c.lower() for c in df.columns]
    col_map = {c.lower(): c for c in df.columns}
    needed = ["age", "estimatedsalary", "purchased"]
    for k in needed:
        assert k in cols, f"Column '{k}' not found in CSV. Found columns: {df.columns.tolist()}"

    X_np = df[[col_map["age"], col_map["estimatedsalary"]]].values.astype(np.float32)
    y_np = df[col_map["purchased"]].values.astype(np.int64)

    
    scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    X01 = scaler.fit_transform(X_np).astype(np.float32)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X01, y_np, test_size=0.3, random_state=seed, stratify=y_np
    )

    X_tr = torch.tensor(X_tr, dtype=torch.float32)
    X_te = torch.tensor(X_te, dtype=torch.float32)
    y_tr = torch.tensor(y_tr, dtype=torch.long)
    y_te = torch.tensor(y_te, dtype=torch.long)

    cfg = QuKANNetCfg(
        layer1=QuKANLayerCfg(n_nodes=6, n_label_qubits=2, n_pos_qubits=5,
                             qcbm_depth=3, label_mixer_depth=2, fourier_k=4, fourier_depth=1),
        layer2=QuKANLayerCfg(n_nodes=6, n_label_qubits=2, n_pos_qubits=5,
                             qcbm_depth=3, label_mixer_depth=2, fourier_k=4, fourier_depth=1),
        n_classes=2,  
    )
    model = QuKANNet(cfg, input_dim=2, seed=seed)

    
    model.pretrain_qcbms(degree=2, epochs=200, lr=5e-2, verbose=True)

    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=8e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=60)
    ce = nn.CrossEntropyLoss(label_smoothing=0.03)

    print("\nTraining QuKAN on Social_Network_Ads")
    epochs = 60
    B = 32
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(X_tr.shape[0])
        Xb_all, yb_all = X_tr[perm], y_tr[perm]

        tot, corr, loss_sum = 0, 0, 0.0
        for i in range(0, Xb_all.shape[0], B):
            xb = Xb_all[i:i+B]
            yb = yb_all[i:i+B]
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = ce(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            loss_sum += float(loss.item()) * xb.size(0)
            tot += xb.size(0)
            corr += (logits.argmax(1) == yb).sum().item()

        sched.step()
        train_acc = 100.0 * corr / tot
        train_loss = loss_sum / tot

        model.eval()
        with torch.no_grad():
            logits_te = model(X_te)
            val_acc = 100.0 * (logits_te.argmax(1) == y_te).float().mean().item()

        if ep % 2 == 1 or ep >= epochs - 10:
            print(f"Epoch {ep:03d} | Loss={train_loss:.4f} | Train Acc={train_acc:.2f}% | Val Acc={val_acc:.2f}%")

    print("Done.")

    
    with torch.no_grad():
        pred = model(X_te).argmax(1).cpu().numpy()
        acc = (pred == y_te.cpu().numpy()).mean() * 100
    print(f"\nFinal Test Accuracy: {acc:.2f}%")

if __name__ == "__main__":
    run_social(seed=0)
