import math, numpy as np
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

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
    def __init__(self, n_label_qubits, n_pos_qubits, depth=3, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.L, self.P = n_label_qubits, n_pos_qubits
        self.n_qubits = self.L + self.P
        self.theta = nn.Parameter(0.01 * torch.randn(depth, self.n_qubits, 3, dtype=torch.float32))
        self.dev = qml.device("default.qubit", wires=self.n_qubits)

        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def qnode(weights):
            qml.templates.StronglyEntanglingLayers(weights, wires=range(self.n_qubits))
            return qml.probs(wires=range(self.n_qubits))
        self._qprobs = qnode

    def forward(self):
        return self._qprobs(self.theta).to(torch.float32)

    def freeze(self):
        self.theta.requires_grad_(False)

class LabelMixer(nn.Module):
    def __init__(self, qcbm: QCBMState, depth=1, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.qcbm = qcbm
        self.L, self.P = qcbm.L, qcbm.P
        self.phi = nn.Parameter(0.01 * torch.randn(depth, self.L, 3, dtype=torch.float32))
        self.dev = qml.device("default.qubit", wires=self.L + self.P)

        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def qnode(weights_qcbm, weights_label):
            qml.templates.StronglyEntanglingLayers(weights_qcbm, wires=range(self.L + self.P))
            if self.L > 0:
                qml.templates.StronglyEntanglingLayers(weights_label, wires=range(self.L))
            return qml.probs(wires=range(self.L + self.P))
        self._qprobs = qnode

    def forward(self):
        return self._qprobs(self.qcbm.theta, self.phi).to(torch.float32)


class QuantumBlock(nn.Module):
    def __init__(self, k_frequencies=3, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.K = k_frequencies
        self.log_omega = nn.Parameter(torch.randn(self.K) * 0.05)
        self.phase = nn.Parameter(torch.zeros(self.K))
        self.w_cos = nn.Parameter(torch.randn(self.K) * 0.1)
        self.w_sin = nn.Parameter(torch.randn(self.K) * 0.1)

    def forward_batch(self, x01_vec):
        x01_vec = torch.clamp(x01_vec, 0, 1)
        omega = F.softplus(self.log_omega) + 1e-4  
        vals = []
        for val in x01_vec:
            alpha = omega * (2*math.pi*val) + self.phase  
            z = torch.cos(alpha)  
            x = torch.sin(alpha)  
            vals.append((self.w_cos*z).sum() + (self.w_sin*x).sum())
        return torch.stack(vals)

class QuKANResidualEdge(nn.Module):
    def __init__(self, mixer, n_label_qubits, n_pos_qubits, fourier_k=3, seed=0, w_init=0.5):
        super().__init__()
        self.mixer = mixer
        self.L, self.P = n_label_qubits, n_pos_qubits
        self.Nlabel, self.Npos = 2**self.L, 2**self.P
        self.wf = nn.Parameter(torch.tensor(float(w_init)))
        self.wq = nn.Parameter(torch.tensor(float(w_init)))
        self.qfour = QuantumBlock(fourier_k, seed=seed)

    def batch_forward(self, x_pos01, probs_flat):
        lp = probs_flat.view(self.Nlabel, self.Npos)
        idx = torch.round(torch.clamp(x_pos01,0,1)*(self.Npos-1)).long()
        idx = torch.clamp(idx, 0, self.Npos-1)
        p_vals = lp[:,idx].sum(0)
        qfr_vals = self.qfour.forward_batch(x_pos01)
        return self.wf*p_vals + self.wq*qfr_vals


@dataclass
class QuKANLayerCfg:
    n_nodes: int = 4
    n_label_qubits: int = 2
    n_pos_qubits: int = 6   
    qcbm_depth: int = 3
    label_mixer_depth: int = 1
    fourier_k: int = 3
    mixers_trainable: bool = False  

class QuKANLayer(nn.Module):
    def __init__(self, cfg: QuKANLayerCfg, seed=0):
        super().__init__()
        self.cfg = cfg
        self.qcbm = QCBMState(cfg.n_label_qubits, cfg.n_pos_qubits, cfg.qcbm_depth, seed)
        self.mixers, self.edges = nn.ModuleList(), nn.ModuleList()
        self._built=False
        self._train_mixers = cfg.mixers_trainable

    def build(self, input_dim, seed=0):
        for m in range(self.cfg.n_nodes):
            for j in range(input_dim):
                mixer = LabelMixer(self.qcbm, self.cfg.label_mixer_depth, seed+97*m+j)
                edge = QuKANResidualEdge(mixer, self.cfg.n_label_qubits, self.cfg.n_pos_qubits,
                                         self.cfg.fourier_k, seed=seed+991*m+13*j)
                self.mixers.append(mixer); self.edges.append(edge)
        self._built=True
        print(f"built edges: {self.cfg.n_nodes} nodes × {input_dim} inputs = {len(self.edges)} edges")

    def pretrain_qcbm_on_splines(self, degree=2, epochs=80, lr=5e-2, verbose=True):
        num_spl, Npos = 2**self.cfg.n_label_qubits, 2**self.cfg.n_pos_qubits
        grid = np.linspace(0,1,Npos)
        B = bspline_basis_matrix(num_spl, degree, grid)
        B = (B+1e-8)/B.sum(1,keepdims=True)
        target = torch.tensor((B/num_spl).reshape(-1), dtype=torch.float32)
        opt=torch.optim.Adam(self.qcbm.parameters(), lr=lr)
        for ep in range(epochs):
            opt.zero_grad(); probs=self.qcbm()
            loss=F.mse_loss(probs, target); loss.backward(); opt.step()
            if verbose and (ep%20==0 or ep==epochs-1):
                tv=0.5*torch.sum(torch.abs(probs-target)).item()
                print(f"[QCBM pretrain] {ep:03d} | MSE={loss.item():.6f} | TV={tv:.6f}")
        self.qcbm.freeze()
        print("QCBM frozen.")

    def forward(self,X, input_is_01=True):
        X01 = (X if input_is_01 else torch.sigmoid(X))
        if self._train_mixers:
            edge_probs=[mix() for mix in self.mixers]
        else:
            with torch.no_grad():
                edge_probs=[mix() for mix in self.mixers]
        nodes=[]; eidx=0
        for m in range(self.cfg.n_nodes):
            acc=torch.zeros(X.shape[0], dtype=torch.float32)
            for j in range(X.shape[1]):
                out=self.edges[eidx].batch_forward(X01[:,j], edge_probs[eidx])
                acc=acc+out; eidx+=1
            nodes.append(acc)
        return torch.stack(nodes,1)

@dataclass
class KANReadoutCfg:
    n_classes:int; in_dim:int; fourier_k:int=3

class KANReadout(nn.Module):
    def __init__(self,cfg:KANReadoutCfg,seed=0):
        super().__init__()
        self.cfg=cfg; C,M=cfg.n_classes,cfg.in_dim
        self.qfr=nn.ModuleList([QuantumBlock(cfg.fourier_k,seed+131*c+m)
                                for c in range(C) for m in range(M)])
        self.b=nn.Parameter(torch.zeros(C))
    def _idx(self,c,m): return c*self.cfg.in_dim+m
    def forward(self,H):
        H01=torch.sigmoid(H); logits=[]
        for c in range(self.cfg.n_classes):
            acc=torch.zeros(H.shape[0], dtype=torch.float32)
            for m in range(H.shape[1]):
                acc=acc+self.qfr[self._idx(c,m)].forward_batch(H01[:,m])
            logits.append(acc+self.b[c])
        return torch.stack(logits,1)


@dataclass
class QuKANNetCfg:
    layer1:QuKANLayerCfg=field(default_factory=QuKANLayerCfg)
    layer2:QuKANLayerCfg=field(default_factory=lambda: QuKANLayerCfg(n_pos_qubits=6))
    n_classes:int=10

class QuKANNet(nn.Module):
    def __init__(self,cfg,input_dim,seed=0):
        super().__init__()
        self.l1=QuKANLayer(cfg.layer1,seed); self.l1.build(input_dim,seed)
        self.l2=QuKANLayer(cfg.layer2,seed+1); self.l2.build(cfg.layer1.n_nodes,seed+1)
        self.readout=KANReadout(KANReadoutCfg(cfg.n_classes,cfg.layer2.n_nodes),seed+123)
    def pretrain_qcbms(self,degree=2,epochs=80,lr=5e-2):
        print("\n[Pretrain] Layer 1 QCBM"); self.l1.pretrain_qcbm_on_splines(degree,epochs,lr)
        print("\n[Pretrain] Layer 2 QCBM"); self.l2.pretrain_qcbm_on_splines(degree,epochs,lr)
    def forward(self,X):
        h1=self.l1(X,True); h2=self.l2(h1,False); return self.readout(h2)

def run_digits(seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    digits = load_digits()
    X, y = digits.data.astype(np.float32), digits.target.astype(np.int64)
    X, y = X[:1000], y[:1000]

    X = MinMaxScaler((0,1)).fit_transform(X).astype(np.float32)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
    X_tr, X_te = torch.tensor(X_tr), torch.tensor(X_te)
    y_tr, y_te = torch.tensor(y_tr), torch.tensor(y_te)

    model = QuKANNet(QuKANNetCfg(), input_dim=64, seed=seed)
    model.pretrain_qcbms()

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=8e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=40)
    ce = nn.CrossEntropyLoss(label_smoothing=0.05)

    print("\nTraining QuKAN on Digits Dataset (1000 samples)")
    for ep in range(1, 41):
        model.train()
        perm = torch.randperm(X_tr.shape[0])
        Xb_all, yb_all = X_tr[perm], y_tr[perm]
        loss_sum, tot, corr = 0.0, 0, 0
        for i in range(0, Xb_all.shape[0], 64):
            xb, yb = Xb_all[i:i+64], yb_all[i:i+64]
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = ce(logits, yb)
            loss.backward()
            opt.step()
            loss_sum += float(loss.item()) * xb.size(0)
            tot += xb.size(0)
            corr += (logits.argmax(1) == yb).sum().item()
        sched.step()
        train_acc = 100.0 * corr / tot
        val_acc = (model(X_te).argmax(1) == y_te).float().mean().item() * 100.0
        print(f"Epoch {ep:03d} | Train Acc={train_acc:.2f}% | Val Acc={val_acc:.2f}%")

if __name__ == "__main__":
    run_digits(0)
