# src/data_preprocessing.py

import h5py
import torch
import torch.nn as nn
import numpy as np
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

def apply_pca(X, n_components=2):
    X_flat = X.view(X.size(0), -1).numpy()
    pca = PCA(n_components=n_components)
    pca_data = pca.fit_transform(X_flat)
    print(f"PCA applied with {n_components} components.")
    return pca, pca_data

def normalize_pca(pca_data, n_components=2):
    pca_descaler = [[] for _ in range(n_components)]
    for i in range(n_components):
        if pca_data[:, i].min() < 0:
            pca_descaler[i].append(pca_data[:, i].min())
            pca_data[:, i] += np.abs(pca_data[:, i].min())
        else:
            pca_descaler[i].append(pca_data[:, i].min())
            pca_data[:, i] -= pca_data[:, i].min()
        pca_descaler[i].append(pca_data[:, i].max())
        pca_data[:, i] /= pca_data[:, i].max()
    pca_data_rot = np.arcsin(np.sqrt(pca_data))
    return pca_data_rot, pca_descaler
