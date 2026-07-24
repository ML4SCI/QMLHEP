import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import math
from typing import *

SEED = 10
torch.manual_seed(SEED)
np.random.seed(SEED)


crop_size = 8
Nsamples = 1000
file_path = "C:\\Users\\riakh\\Downloads\\quark-gluon_train-set_n793900-001.hdf5"

def load_and_preprocess(file_path, Nsamples=1000, crop_size=8):
    with h5py.File(file_path, "r") as f:
        X_jets = f["X_jets"][:Nsamples]
        y = f["y"][:Nsamples]

    def crop_center(img, cropx, cropy):
        x, y = img.shape[1:3]
        startx = x // 2 - (cropx // 2)
        starty = y // 2 - (cropy // 2)
        return img[:, startx:startx + cropx, starty:starty + cropy, :]

    cropped = crop_center(X_jets, crop_size, crop_size)
    
    ch0 = cropped[:, :, :, 1]
    ch1 = cropped[:, :, :, 2]

    def normalize_channel(ch):
        div = np.max(ch, axis=(1, 2), keepdims=True)
        div[div == 0] = 1
        return ch / (div + 1e-5)

    ch0 = normalize_channel(ch0)
    ch1 = normalize_channel(ch1)

    combined = np.concatenate([ch0, ch1], axis=-1)
    X_tensor = torch.tensor(combined.reshape(Nsamples, -1), dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    return train_test_split(X_tensor, y_tensor, test_size=0.2, stratify=y, random_state=42)

X_train, X_test, y_train, y_test = load_and_preprocess(file_path, Nsamples, crop_size)

def forward_step(i_n, grid_size, A, K, C):
    ratio = A * grid_size**(-K) + C
    return ratio * i_n

class SineKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, device='cpu', grid_size=5, is_first=False, add_bias=True, norm_freq=True):
        super().__init__()
        self.grid_size = grid_size
        self.device = device
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.is_first = is_first
        self.add_bias = add_bias

        self.A, self.K, self.C = 0.9724, 0.9884, 0.9994
        self.grid_norm_factor = (torch.arange(grid_size) + 1).reshape(1, 1, grid_size)

        if is_first:
            self.amplitudes = nn.Parameter(
                torch.empty(output_dim, input_dim, 1).normal_(0, .4) / output_dim / self.grid_norm_factor
            )
        else:
            self.amplitudes = nn.Parameter(
                torch.empty(output_dim, input_dim, 1).uniform_(-1, 1) / output_dim / self.grid_norm_factor
            )

        grid_phase = torch.arange(1, grid_size + 1).reshape(1, 1, 1, grid_size) / (grid_size + 1)
        self.input_phase = torch.linspace(0, math.pi, input_dim).reshape(1, 1, input_dim, 1).to(device)
        phase = grid_phase.to(device) + self.input_phase

        if norm_freq:
            self.freq = nn.Parameter(torch.arange(1, grid_size + 1).float().reshape(1, 1, 1, grid_size) / (grid_size + 1)**(1 - is_first))
        else:
            self.freq = nn.Parameter(torch.arange(1, grid_size + 1).float().reshape(1, 1, 1, grid_size))

        for i in range(1, self.grid_size):
            phase = forward_step(phase, i, self.A, self.K, self.C)

        self.register_buffer('phase', phase)

        if self.add_bias:
            self.bias = nn.Parameter(torch.ones(1, output_dim) / output_dim)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x_reshaped = x.view(x.shape[0], 1, x.shape[1], 1)
        s = torch.sin(x_reshaped * self.freq + self.phase)
        y = torch.einsum('ijkl,jkl->ij', s, self.amplitudes)
        if self.add_bias:
            y += self.bias
        return y

class SineKAN(nn.Module):
    def __init__(self, layers_hidden: List[int], grid_size=8, device='cpu'):
        super().__init__()
        self.layers = nn.ModuleList([
            SineKANLayer(in_dim, out_dim, device, grid_size=grid_size, is_first=(i == 0))
            for i, (in_dim, out_dim) in enumerate(zip(layers_hidden[:-1], layers_hidden[1:]))
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

def train_sinekan_model(X_train, X_test, y_train, y_test, input_dim=128, output_dim=2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = SineKAN([input_dim, 64, output_dim], grid_size=8, device=device).to(device)
    model = SineKAN([input_dim, 64, 32, output_dim], grid_size=8, device=device).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-2)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    best_state = None

    for epoch in range(300):
        model.train()
        optimizer.zero_grad()
        out = model(X_train.to(device))
        loss = criterion(out, y_train.to(device))
        loss.backward()
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            model.eval()
            test_preds = model(X_test.to(device)).argmax(dim=1)
            acc = (test_preds == y_test.to(device)).float().mean().item()

        print(f"Epoch {epoch+1:03d} | Loss: {loss.item():.4f} | Test Acc: {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            best_state = model.state_dict()
            torch.save(best_state, "best_sinekan.pt")

    print(f"\nBest Test Accuracy: {best_acc:.4f} | Weights saved to 'best_sinekan.pt'")


train_sinekan_model(X_train, X_test, y_train, y_test, input_dim=8*8*2)
