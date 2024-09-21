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
filename = "C:/Users/realc/OneDrive/Documents/GSOC/data/QG1_normalized_16_xl"
data_X = np.array(h5py.File(filename, "r")['X'])

num_samples = 100
seed = 42

num_qubits = 64
dev = qml.device("default.qubit", wires=num_qubits)

@qml.qnode(dev)
def angle_circuit(phi):
    for j in range(4):
        qml.RX(np.pi * phi[j], wires=j)
    return [qml.expval(qml.PauliZ(j)) for j in range(4)]

def angle_encoding(data, sample=0):
    out = np.zeros((8, 8, 4))
    for i in range(0, 16, 2):
        for j in range(0, 16, 2):
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

def haar_random_unitary(seed=None):
    if seed is not None:
        np.random.seed(seed)
    return unitary_group.rvs(4)

def scramble_state_vectors(encoded_data, seed):
    scrambled_vectors = []

    for sample in range(num_samples):
        scrambled_vector = []
        for i in range(8):
            channels = []
            for j in range(8):
                U = haar_random_unitary(seed)
                scrambled_state = np.dot(U, encoded_data[sample, i, j, :])
                scrambled_state /= np.linalg.norm(scrambled_state)
                channels.append(scrambled_state)

                if seed is not None:
                    seed += 1
            scrambled_vector.append(channels)
        scrambled_vectors.append(scrambled_vector)

    return np.array(scrambled_vectors)

# %%
encoded_data = [angle_encoding(data_X, sample) for sample in range(num_samples)]
encoded_data = np.array(encoded_data)
encoded_data = torch.tensor(encoded_data, dtype=torch.float32)


