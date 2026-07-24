import h5py
import numpy as np
import torch
import torch.nn as nn
import random
from sklearn.decomposition import PCA

def load_data(filepath):
    with h5py.File(filepath, 'r') as jet_mass_data:
        X_jet = jet_mass_data['image']
        print("Original shape:", X_jet.shape)
        X_jet = np.array(X_jet)
        X_jet = torch.tensor(X_jet, dtype=torch.float32)
    return X_jet

def normalize_and_resize(X, size=(16, 16)):
    X = (X - X.min()) / (X.max() - X.min())
    X = X.unsqueeze(1)
    X_resized = nn.functional.interpolate(X, size=size, mode='bilinear', align_corners=False)
    print("Resized shape:", X_resized.shape)
    return X_resized

def sample_data(X, num_samples=10000):
    indices = random.sample(range(X.shape[0]), num_samples)
    X_sampled = X[indices]
    print("Sampled shape:", X_sampled.shape)
    return X_sampled

def apply_pca(X, n_components=8):
    X_flat = X.view(-1, 256).numpy()
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_flat)
    print(f"PCA applied with {n_components} components.")
    return pca, X_pca

class JetDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = torch.tensor(data, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
