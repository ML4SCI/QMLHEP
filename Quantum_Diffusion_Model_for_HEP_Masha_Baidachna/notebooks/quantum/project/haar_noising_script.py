from scipy.stats import rv_continuous
import numpy as np
import pennylane as qml

dev = qml.device("default.qubit", wires=2) 

class sin_prob_dist(rv_continuous):
    def _pdf(self, theta):
        return 0.5 * np.sin(theta)

sin_sampler = sin_prob_dist(a=0, b=np.pi)

@qml.qnode(dev)
def haar_random_unitary():
    phi1, omega1 = 2 * np.pi * np.random.uniform(size=2)
    theta1 = sin_sampler.rvs(size=1)
    
    phi2, omega2 = 2 * np.pi * np.random.uniform(size=2)
    theta2 = sin_sampler.rvs(size=1)
    
    qml.Rot(phi1, theta1, omega1, wires=0)
    qml.Rot(phi2, theta2, omega2, wires=1)
    
    return qml.state()

def apply_haar_scrambling(encoded_data, num_samples, seed=None):
    scrambled_vectors = []
    new_dim = encoded_data.shape[1]

    for sample in range(num_samples):
        scrambled_vector = []
        for _ in range(new_dim):
            channels = []
            for _ in range(new_dim):
                if seed is not None:
                    np.random.seed(seed)

                # Haar random unitary for 4D vector with 2 qubits
                scrambled_state = haar_random_unitary()

                scrambled_state = np.reshape(scrambled_state, (4,))
                scrambled_state /= np.linalg.norm(scrambled_state)

                channels.append(scrambled_state)

                if seed is not None:
                    seed += 1
            scrambled_vector.append(channels)
        scrambled_vectors.append(scrambled_vector)

    return np.array(scrambled_vectors)
