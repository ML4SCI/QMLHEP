# Previously known as haar nosing, but is really just normal random unitaries
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
def normal_random_unitary(seed=None):
    if seed is not None:
        np.random.seed(seed)
    return unitary_group.rvs(4)

def apply_haar_scrambling(encoded_data, num_samples, seed):
    scrambled_vectors = []

    for sample in range(num_samples):
        scrambled_vector = []
        for i in range(8):
            channels = []
            for j in range(8):
                U = normal_random_unitary(seed)
                scrambled_state = np.dot(U, encoded_data[sample, i, j, :])
                scrambled_state /= np.linalg.norm(scrambled_state)
                channels.append(scrambled_state)

                if seed is not None:
                    seed += 1
            scrambled_vector.append(channels)
        scrambled_vectors.append(scrambled_vector)

    return np.array(scrambled_vectors)


