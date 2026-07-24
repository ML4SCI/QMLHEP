import math, numpy as np
from dataclasses import dataclass, field
import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

torch.set_default_dtype(torch.float32)

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

class LabelMixer(nn.Module):
    def __init__(self, qcbm: QCBMState, depth=2, seed=0):
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
    def __init__(self, k_frequencies=4, entangle_depth=1, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.K = k_frequencies
        self.log_omega = nn.Parameter(torch.randn(self.K) * 0.05)
        self.phase = nn.Parameter(torch.zeros(self.K))
        self.w_cos = nn.Parameter(torch.randn(self.K) * 0.1)
        self.w_sin = nn.Parameter(torch.randn(self.K) * 0.1)
        self.dev = qml.device("default.qubit", wires=self.K)

        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def qnode(alpha_vec):
            
            for k in range(self.K):
                qml.RY(alpha_vec[k], wires=k)
            
            for _ in range(entangle_depth):
                for k in range(self.K):
                    qml.CNOT([k, (k + 1) % self.K])
            z = [qml.expval(qml.PauliZ(k)) for k in range(self.K)]
            x = [qml.expval(qml.PauliX(k)) for k in range(self.K)]
            return z + x
        self._qnode = qnode

    def forward_batch(self, x01_vec: torch.Tensor):
        
        x01_vec = torch.clamp(x01_vec, 0, 1)
        omega = F.softplus(self.log_omega) + 1e-4
        vals = []
        for val in x01_vec:
            
            alpha = omega * (2 * math.pi * val) + self.phase
            outs = self._qnode(alpha.to(torch.float32))
            outs = torch.stack([torch.as_tensor(o, dtype=torch.float32) for o in outs])
            
            vals.append((self.w_cos * outs[:self.K]).sum() + (self.w_sin * outs[self.K:]).sum())
        return torch.stack(vals)

class QuKANResidualEdge(nn.Module):
    """
    Edge output = w_f * (QCBM label⊗position probability at discretized x)
                + w_q * (QuantumFourierBlock(x))
    """
    def __init__(self, mixer, n_label_qubits, n_pos_qubits, fourier_k=4, fourier_depth=1, seed=0, w_init=0.5):
        super().__init__()
        self.mixer = mixer
        self.L, self.P = n_label_qubits, n_pos_qubits
        self.Nlabel, self.Npos = 2 ** self.L, 2 ** self.P
        self.wf = nn.Parameter(torch.tensor(float(w_init)))
        self.wq = nn.Parameter(torch.tensor(float(w_init)))
        self.qfour = QuantumBlock(fourier_k, fourier_depth, seed=seed)

    def batch_forward(self, x_pos01: torch.Tensor, probs_flat: torch.Tensor):
        """
        x_pos01: (B,) in [0,1]
        probs_flat: (2^(L+P),) from LabelMixer (shared across batch; edge-wise)
        """
        
        lp = probs_flat.view(self.Nlabel, self.Npos)
        
        idx = torch.round(torch.clamp(x_pos01, 0, 1) * (self.Npos - 1)).long()
        idx = torch.clamp(idx, 0, self.Npos - 1)
        
        p_vals = lp[:, idx].sum(0)
        
        qfr_vals = self.qfour.forward_batch(x_pos01)
        return self.wf * p_vals + self.wq * qfr_vals

@dataclass
class QuKANLayerCfg:
    n_nodes: int = 5          
    n_label_qubits: int = 2     
    n_pos_qubits: int = 6       
    qcbm_depth: int = 3
    label_mixer_depth: int = 2
    fourier_k: int = 4
    fourier_depth: int = 1

class QuKANLayer(nn.Module):
    def __init__(self, cfg: QuKANLayerCfg, seed=0):
        super().__init__()
        self.cfg = cfg
        self.qcbm = QCBMState(cfg.n_label_qubits, cfg.n_pos_qubits, cfg.qcbm_depth, seed)
        self.mixers, self.edges = nn.ModuleList(), nn.ModuleList()

    def build(self, input_dim, seed=0):
        print(f"[QuKANLayer] Building with {input_dim} inputs...")
        for m in range(self.cfg.n_nodes):
            for j in range(input_dim):
                mixer = LabelMixer(self.qcbm, self.cfg.label_mixer_depth, seed + 97 * m + j)
                edge = QuKANResidualEdge(
                    mixer, self.cfg.n_label_qubits, self.cfg.n_pos_qubits,
                    self.cfg.fourier_k, self.cfg.fourier_depth, seed + 991 * m + 13 * j
                )
                self.mixers.append(mixer)
                self.edges.append(edge)
        print(f"[QuKANLayer] Built edges: {self.cfg.n_nodes} nodes × {input_dim} inputs = {len(self.edges)} edges")

    def forward(self, X):
        
        X01 = torch.sigmoid(X)
        
        edge_probs = [mix() for mix in self.mixers]
        nodes = []
        eidx = 0
        for m in range(self.cfg.n_nodes):
            acc = torch.zeros(X.shape[0], dtype=torch.float32, device=X.device)
            for j in range(X.shape[1]):
                out = self.edges[eidx].batch_forward(X01[:, j], edge_probs[eidx])
                acc = acc + out
                eidx += 1
            nodes.append(acc)
        return torch.stack(nodes, 1)

@dataclass
class KANReadoutCfg:
    n_classes: int
    in_dim: int
    fourier_k: int = 3
    fourier_depth: int = 1

class KANReadout(nn.Module):
    """
    KAN-style readout: for each output class, sum QuantumFourier transforms of each hidden unit (no Linear).
    """
    def __init__(self, cfg: KANReadoutCfg, seed=0):
        super().__init__()
        self.cfg = cfg
        C, M = cfg.n_classes, cfg.in_dim
        self.qfr = nn.ModuleList([
            QuantumBlock(cfg.fourier_k, cfg.fourier_depth, seed + 131 * c + m)
            for c in range(C) for m in range(M)
        ])
        self.b = nn.Parameter(torch.zeros(C))

    def _idx(self, c, m):
        return c * self.cfg.in_dim + m

    def forward(self, H):
        H01 = torch.sigmoid(H)
        logits = []
        for c in range(self.cfg.n_classes):
            acc = torch.zeros(H.shape[0], dtype=torch.float32, device=H.device)
            for m in range(H.shape[1]):
                acc = acc + self.qfr[self._idx(c, m)].forward_batch(H01[:, m])
            logits.append(acc + self.b[c])
        return torch.stack(logits, 1)

@dataclass
class QuKANNetCfg:
    layer1: QuKANLayerCfg = field(default_factory=QuKANLayerCfg)
    layer2: QuKANLayerCfg = field(default_factory=QuKANLayerCfg)
    n_classes: int = 2 

class QuKANNet(nn.Module):
    def __init__(self, cfg, input_dim, seed=0):
        super().__init__()
        print("[QuKANNet] Initializing network...")
        self.l1 = QuKANLayer(cfg.layer1, seed);   self.l1.build(input_dim, seed)
        self.l2 = QuKANLayer(cfg.layer2, seed+1); self.l2.build(cfg.layer1.n_nodes, seed+1)
        self.readout = KANReadout(KANReadoutCfg(cfg.n_classes, cfg.layer2.n_nodes), seed+123)
        print("[QuKANNet] Build complete.")

    def forward(self, X):
        h1 = self.l1(X)
        h2 = self.l2(h1)
        return self.readout(h2)

def load_higgs_csv_first_n(csv_path: str, n_samples: int):
    data = np.loadtxt(csv_path, delimiter=",", max_rows=n_samples)
    y = data[:, 0].astype(np.int64)        
    X = data[:, 1:29].astype(np.float32)   
    scaler = MinMaxScaler((0, 1))
    X = scaler.fit_transform(X).astype(np.float32)
    return X, y

def run_higgs(csv_path: str,
              n_samples: int = 20000,
              epochs: int = 20,
              batch_size: int = 128,
              seed: int = 0):
    torch.manual_seed(seed); np.random.seed(seed)

    print(f"Loading first {n_samples} rows from: {csv_path}")
    X, y = load_higgs_csv_first_n(csv_path, n_samples)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    X_tr = torch.tensor(X_tr, dtype=torch.float32)
    X_te = torch.tensor(X_te, dtype=torch.float32)
    y_tr = torch.tensor(y_tr, dtype=torch.long)
    y_te = torch.tensor(y_te, dtype=torch.long)

    
    input_dim = X_tr.shape[1] 
    model = QuKANNet(QuKANNetCfg(), input_dim=input_dim, seed=seed)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss(label_smoothing=0.05)

    print("\nTraining QuKAN on HIGGS (28 features)")
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(X_tr.shape[0])
        xb_all, yb_all = X_tr[perm], y_tr[perm]

        tot, corr = 0, 0
        epoch_loss = 0.0

        for i in range(0, xb_all.shape[0], batch_size):
            xb = xb_all[i:i+batch_size]
            yb = yb_all[i:i+batch_size]

            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = ce(logits, yb)
            loss.backward()
            opt.step()

            epoch_loss += loss.item() * xb.size(0)
            tot += xb.size(0)
            corr += (logits.argmax(1) == yb).sum().item()

        train_acc = 100.0 * corr / tot
        with torch.no_grad():
            val_logits = model(X_te)
            val_acc = (val_logits.argmax(1) == y_te).float().mean().item() * 100.0

        avg_loss = epoch_loss / tot
        print(f"Epoch {ep:03d} | Loss={avg_loss:.4f} | Train Acc={train_acc:.2f}% | Val Acc={val_acc:.2f}%")

if __name__ == "__main__":
    CSV_PATH = r"C:\Users\riakh\Downloads\archive (26)\HIGGS.csv"
    run_higgs(CSV_PATH, n_samples=2000, epochs=20, batch_size=128, seed=0)
