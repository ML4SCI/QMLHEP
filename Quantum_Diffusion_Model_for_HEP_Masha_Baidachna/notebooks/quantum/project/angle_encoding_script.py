# %%
from scipy.stats import unitary_group
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pennylane as qml
from pennylane.templates import RandomLayers

import torch
import torch.nn as nn
import torch.optim as optim

# %%
# filename = "C:/Users/realc/OneDrive/Documents/GSOC/data/QG1_normalized_16_xl"
# data_X = np.array(h5py.File(filename, "r")['X'])

# num_samples = 100
seed = 42

num_qubits = 64 # data.shape[1]**2
dev = qml.device("default.qubit", wires=num_qubits)

@qml.qnode(dev)
def angle_circuit(phi):
    for j in range(4):
        qml.RX(np.pi * phi[j], wires=j)
    return [qml.expval(qml.PauliZ(j)) for j in range(4)]

def angle_encoding(data, sample=0):
    new_dim = data.shape[1] // 2

    out = np.zeros((new_dim, new_dim, 4))
    for i in range(0, data.shape[1], 2):
        for j in range(0, data.shape[1], 2):
            q_results = angle_circuit(
                [
                    data[sample, i, j],
                    data[sample, i, j+1],
                    data[sample, i+1, j],
                    data[sample, i+1, j+1]
                ]
            )
            for c in range(4):
                out[i//2, j//2, c] = q_results[c]
    return out

# %%
# encoded_data = [angle_encoding(data_X, sample) for sample in range(num_samples)]
# encoded_data = np.array(encoded_data)
# encoded_data = torch.tensor(encoded_data, dtype=torch.float32)


