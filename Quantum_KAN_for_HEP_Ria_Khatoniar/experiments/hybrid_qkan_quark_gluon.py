import torch
import torch.nn as nn
import torch.optim as optim
import pennylane as qml
import math
import h5py
import numpy as np
from sklearn.model_selection import train_test_split


def load_and_preprocess(file_path, Nsamples=1000, crop_size=8):
    with h5py.File(file_path, "r") as f:
        X_jets = f["X_jets"][:Nsamples]
        y = f["y"][:Nsamples]

    def crop_center(img, cx, cy):
        x, y = img.shape[1:3]
        sx, sy = x // 2 - cx // 2, y // 2 - cy // 2
        return img[:, sx:sx + cx, sy:sy + cy, :]

    cropped = crop_center(X_jets, crop_size, crop_size)
    ch0 = cropped[..., 0]
    ch1 = cropped[..., 1]

    def normalize(ch):
        div = np.max(ch, axis=(1, 2), keepdims=True)
        div[div == 0] = 1
        return ch / (div + 1e-5)

    X = np.concatenate([normalize(ch0), normalize(ch1)], axis=-1)
    return train_test_split(
        torch.tensor(X.reshape(Nsamples, -1), dtype=torch.float32),
        torch.tensor(y, dtype=torch.long),
        test_size=0.2, stratify=y, random_state=42
    )


class QSVTLayer(nn.Module):
    def __init__(self, chunk_size, coeffs):
        super().__init__()
        self.coeffs = coeffs
        self.theta = nn.Parameter(torch.randn(1))
        self.dev = qml.device("default.qubit", wires=1)

        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def circuit(x_scalar, theta):
            x_scalar = torch.clamp(x_scalar, -1.0, 1.0)
            qml.RY(x_scalar, wires=0)
            A = x_scalar.reshape(1, 1)
            qml.qsvt(A, self.coeffs, encoding_wires=[0], block_encoding="embedding")
            qml.RZ(theta[0], wires=0)
            return qml.expval(qml.PauliZ(0))

        self.circuit = circuit

    def forward(self, x_chunk):  # shape: [B, chunk_size]
        outputs = []
        for i in range(x_chunk.shape[1]):
            col = [self.circuit(x_chunk[b, i], self.theta) for b in range(x_chunk.shape[0])]
            outputs.append(torch.stack(col))
        return torch.stack(outputs, dim=1)  # shape: [B, chunk_size]


class QuantumSummation(nn.Module):
    def __init__(self, chunk_size):
        super().__init__()
        self.weights = nn.Parameter(torch.rand(chunk_size))
        self.num_index_qubits = math.ceil(math.log2(chunk_size))
        self.total_wires = self.num_index_qubits + 2
        self.dev = qml.device("default.qubit", wires=self.total_wires)

        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def circuit(phi_vals, wts):
            index_wires = list(range(self.num_index_qubits))
            hadamard_wire = self.num_index_qubits
            target_wire = self.total_wires - 1
            for i in index_wires:
                qml.Hadamard(wires=i)
            qml.Hadamard(wires=hadamard_wire)
            for p in range(len(phi_vals)):
                ctrl_bin = [int(b) for b in f"{p:0{self.num_index_qubits}b}"]
                qml.ctrl(qml.RZ, control=[hadamard_wire] + index_wires, control_values=[1] + ctrl_bin)(
                    2 * wts[p] * phi_vals[p], wires=target_wire)
            qml.Hadamard(wires=hadamard_wire)
            return qml.expval(qml.PauliZ(hadamard_wire))

        self.circuit = circuit

    def forward(self, phi_vals):  # shape: (chunk_size,)
        return self.circuit(phi_vals, self.weights)


def make_lcu_layer(chunk_size):
    dev = qml.device("default.qubit", wires=3)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def lcu_circuit(vals, weights):
        qml.RY(vals[0], wires=0)
        qml.RY(weights[0], wires=1)
        qml.CNOT(wires=[0, 2])
        angle = torch.clamp(vals[0] * weights[0], -math.pi, math.pi)
        qml.ctrl(qml.RZ, control=0)(angle, wires=2)
        return qml.expval(qml.PauliZ(2))

    def forward_fn(inputs, weights):
        return torch.stack([lcu_circuit(x, weights) for x in inputs])

    return forward_fn

def forward_step(i_n, grid_size, A, K, C):
    ratio = A * grid_size**(-K) + C
    return ratio * i_n

class SineKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, device='cpu', grid_size=5, is_first=False, add_bias=True, norm_freq=True):
        super(SineKANLayer, self).__init__()
        self.grid_size = grid_size
        self.device = device
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.is_first = is_first
        self.add_bias = add_bias

        # Hyperparameters for phase evolution (used in forward_step)
        self.A, self.K, self.C = 0.9724, 0.9884, 0.9994
        self.grid_norm_factor = (torch.arange(grid_size, dtype=torch.float32) + 1).reshape(1, 1, grid_size)

        # Amplitudes
        if is_first:
            amp = torch.empty(output_dim, input_dim, 1).normal_(0, .4) / output_dim / self.grid_norm_factor
        else:
            amp = torch.empty(output_dim, input_dim, 1).uniform_(-1, 1) / output_dim / self.grid_norm_factor
        self.amplitudes = nn.Parameter(amp.float())

        # Frequency
        grid_phase = torch.arange(1, grid_size + 1, dtype=torch.float32).reshape(1, 1, 1, grid_size) / (grid_size + 1)
        self.input_phase = torch.linspace(0, math.pi, input_dim, dtype=torch.float32).reshape(1, 1, input_dim, 1).to(device)
        phase = grid_phase.to(device) + self.input_phase  # [1, 1, input_dim, grid_size]

        if norm_freq:
            self.freq = nn.Parameter((torch.arange(1, grid_size + 1).float().reshape(1, 1, 1, grid_size) / (grid_size + 1)**(1 - is_first)).float())
        else:
            self.freq = nn.Parameter(torch.arange(1, grid_size + 1).float().reshape(1, 1, 1, grid_size))

        # Modified phase update using A, K, C
        for i in range(1, self.grid_size):
            phase = self.forward_step(phase, i)

        self.register_buffer('phase', phase.float())  # buffer, not a parameter

        if self.add_bias:
            self.bias = nn.Parameter(torch.ones(1, output_dim).float() / output_dim)

    def forward_step(self, i_n, i):
        ratio = self.A * (self.grid_size ** (-self.K)) + self.C
        return ratio * i_n

    def forward(self, x):
        x = x.to(self.amplitudes.dtype) 
        x = x.view(x.shape[0], -1)       # [B, in_dim]
        x_reshaped = x.view(x.shape[0], 1, x.shape[1], 1)  # [B, 1, in_dim, 1]
        s = torch.sin(x_reshaped * self.freq + self.phase)  # [B, 1, in_dim, grid_size]
        y = torch.einsum('bijd,ojd->bo', s, self.amplitudes)  # [B, out_dim]
        if self.add_bias:
            y = y + self.bias
        return y

class SineKAN(nn.Module):
    def __init__(self, input_dim, hidden=64, output_dim=2):
        super().__init__()
        self.l1 = SineKANLayer(input_dim, hidden, is_first=True)
        self.l2 = SineKANLayer(hidden, output_dim)

    def forward(self, x):
        return self.l2(self.l1(x))


class HybridQKAN(nn.Module):
    def __init__(self, input_dim, cheb_coeffs, chunk_size=8):
        super().__init__()
        self.chunk_size = chunk_size
        self.num_chunks = input_dim // chunk_size
        self.qsvt_layers = nn.ModuleList([
            QSVTLayer(chunk_size, coeffs=cheb_coeffs[i % len(cheb_coeffs)])
            for i in range(self.num_chunks)
        ])
        self.lcu_weights = nn.Parameter(torch.randn(self.num_chunks, chunk_size))
        self.lcu_layer = make_lcu_layer(chunk_size)
        self.sum_blocks = nn.ModuleList([QuantumSummation(chunk_size) for _ in range(self.num_chunks)])
        self.kan = SineKAN(self.num_chunks)

    def forward(self, x):  # x: [B, D]
        B, D = x.shape
        x = x.view(B, self.num_chunks, self.chunk_size)
        features = []
        for i in range(self.num_chunks):
            chunk = x[:, i, :]  # shape: [B, chunk_size]
            qsvt_out = self.qsvt_layers[i](chunk)  # [B, chunk_size]
            lcu_out = self.lcu_layer(qsvt_out, self.lcu_weights[i])  # [B]
            summed = torch.stack([self.sum_blocks[i](lcu_out[b]) for b in range(B)])
            features.append(summed)
        return self.kan(torch.stack(features, dim=1).squeeze(-1))  # [B, num_chunks] -> KAN


def train_qkan_model(filepath):
    X_train, X_test, y_train, y_test = load_and_preprocess(filepath, Nsamples=1000, crop_size=8)
    input_dim = X_train.shape[1]
    cheb_coeffs = [[0, 1], [-1, 0, 2], [0, -3, 0, 4]]

    model = HybridQKAN(input_dim=input_dim, cheb_coeffs=cheb_coeffs, chunk_size=8)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(50):
        model.train()
        optimizer.zero_grad()
        output = model(X_train.to(device))
        loss = criterion(output, y_train.to(device))
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            preds = model(X_test.to(device)).argmax(dim=1)
            acc = (preds == y_test.to(device)).float().mean().item()
        print(f"[HybridQKAN] Epoch {epoch+1:02d} | Loss: {loss.item():.4f} | Test Acc: {acc:.4f}")


# Call this with data path
# train_qkan_model("C:\\Users\\riakh\\Downloads\\quark-gluon_train-set_n793900-001.hdf5")
