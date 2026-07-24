from __future__ import annotations

import torch.nn as nn
import pennylane as qml
import torch
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from torch.optim import Optimizer
from torch.nn import Module
from IPython.display import clear_output



def plot_training_progress(epoch, iterations, metric_1, metric_2, generator, real_data, dist_shape: tuple[int, int]):
    # we don't plot if we don't have enough data
    if len(metric_1) < 2:
        return

    clear_output(wait=True)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(25, 6))

    # Metric 1
    ax1.set_title("metric 1", fontsize=15)
    ax1.plot(iterations, metric_1, color="royalblue", linewidth=3)
    ax1.set_xlabel("Epoch")
    ax1.set_yscale("log")
    ax1.grid()

    # Metric 2
    ax2.set_title("metric 2", fontsize=15)
    ax2.plot(iterations, metric_2, color="cornflowerblue", linewidth=3)
    ax2.set_xlabel("Epoch")
    ax2.set_yscale("log")
    ax2.grid()

    # Generated distribution
    if len(dist_shape) == 1:
        gen = generator().detach().numpy()
        x = list(range(dist_shape[0]))
        ax3.bar(x, gen[0], color="cornflowerblue", label="generated distribution")
        ax3.bar(x, real_data.numpy().reshape(*dist_shape), color="plum", alpha=0.7, label="real distribution")
        ax3.legend()
        ax3.set_title('Probability Distribution', fontsize=15)
        ax3.set_xlabel('number of heads')
    else:
        gen = generator().detach().numpy().reshape(*dist_shape)
        im = ax3.imshow(gen, extent=[-4, 4, -4, 4], origin='lower', cmap='inferno', aspect='auto', vmin=0, vmax=0.1)
        ax3.set_title('Probability Distribution', fontsize=15)
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')

        fig.colorbar(im, ax=ax3)

    plt.suptitle(f"Epoch {epoch + 1}", fontsize=25)
    plt.show()


def model_training(discriminator: Discriminator, generator: QuantumGenerator, probability_distribution: np.array, dist_shape: tuple[int, int],
                   device: torch.device, criterion: Module, disc_optimizer: Optimizer, gen_optimizer: Optimizer, metrics: list, epochs: int,
                   batch_size: int) -> None:

    gen_loss = []
    disc_loss = []
    metric_1 = []
    metric_2 = []
    iterations = []

    real_labels = torch.full((1,), 1.0, dtype=torch.float, device=device)
    fake_labels = torch.full((1,), 0.0, dtype=torch.float, device=device)

    dataset = Dataset(probability_distribution)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    for epoch in range(epochs):
        for i, data in enumerate(dataloader):

            # Data for training the discriminator
            real_data = data.to(device)
            fake_data = generator()

            # Calculate Frenchet Distance
            wd = metrics[0](real_data.numpy().reshape(*dist_shape),fake_data.detach().numpy().reshape(*dist_shape))
            fd = metrics[1](real_data.numpy().reshape(*dist_shape),fake_data.detach().numpy().reshape(*dist_shape))

            # Training the discriminator
            discriminator.zero_grad()
            outD_real = discriminator(real_data).view(-1)
            outD_fake = discriminator(fake_data.detach()).view(-1)

            errD_real = criterion(outD_real, real_labels)
            errD_fake = criterion(outD_fake, fake_labels)
            # Propagate gradients
            errD_real.backward()
            errD_fake.backward()

            errD = errD_real + errD_fake
            

            disc_optimizer.step()
            

            # Training the generator
            generator.zero_grad()
            outD_fake = discriminator(fake_data).view(-1)
            errG = criterion(outD_fake, real_labels)
            

            errG.backward()
            gen_optimizer.step()

        # Show loss values
        if (epoch + 1)% 10 == 0:
            #print(f'Iteration: {counter}, Discriminator Loss: {errD:0.3f}, Generator Loss: {errG:0.3f}, Frenchet Distance: {fd:0.6f}')
            gen_loss.append(errG.detach())
            disc_loss.append(errD.detach())
            metric_1.append(wd)
            metric_2.append(fd)
            iterations.append(epoch)
            plot_training_progress(epoch, iterations, metric_1, metric_2, generator, real_data, dist_shape)



class Dataset(Dataset):
    def __init__(self, probs):
        self.data = torch.tensor(probs, dtype=torch.float32) 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    

class Discriminator(nn.Module):
    """Fully connected classical discriminator"""

    def __init__(self, input_shape, layers):
        super().__init__()

        self.input_shape = input_shape
        self.layers = layers
        self.model = self.set_model()
    
    def set_model(self):
        input = self.input_shape
        modules = []
        for layer in self.layers:
            modules.append(nn.Linear(input, layer))
            modules.append(nn.LeakyReLU())
            input = layer
        
        modules.append(nn.Linear(input, 1))
        modules.append(nn.Sigmoid())
        
        return nn.Sequential(*modules)

    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        
        return self.model(x)
    
    
    
class QuantumAnsatz:
    def __init__(self, n_qubits, q_depth) -> None:
        
        self.n_qubits = n_qubits
        self.q_depth = q_depth
        self.dev = qml.device("default.qubit", wires=n_qubits)


    def circuit(self):

        @qml.qnode(self.dev, diff_method="backprop")
        def quantum_circuit(weights):

            weights = weights.reshape(self.q_depth, self.n_qubits)

            # Initialise latent vectors
            for i in range(self.n_qubits):
                qml.Hadamard(wires=i)

            # Repeated layer
            for i in range(self.q_depth):
                # Parameterised layer
                for y in range(self.n_qubits):
                    qml.RY(weights[i][y], wires=y)

                # Control Z gates
                for y in range(self.n_qubits - 1):
                    qml.CZ(wires=[y, y + 1])

                qml.Barrier(wires=list(range(self.n_qubits)), only_visual=True)

            return qml.probs(wires=list(range(self.n_qubits)))

        return quantum_circuit
    
    def plot_circuit(self):
        weights = torch.rand(1, self.n_qubits*self.q_depth) 
        qml.draw_mpl(self.circuit())(weights=weights)
        plt.show()
    

    
    
class QuantumGenerator(nn.Module):
    """Quantum generator class for the patch method"""

    def __init__(self, total_qubits, auxiliar_qubits, circuit_depth, quantum_circuit, partial_measure=False, states_range=(0, )):
        """
        Args:
            n_generators (int): Number of sub-generators to be used in the patch method.
            q_delta (float, optional): Spread of the random distribution for parameter initialisation.
        """

        super().__init__()

        self.n_qubits = total_qubits
        self.q_depth = circuit_depth
        self.a_qubits = auxiliar_qubits
        self.q_params = nn.ParameterList(
            [
                nn.Parameter(torch.rand(circuit_depth * total_qubits), requires_grad=True)
            ]
        )

        self.ansatz = quantum_circuit
        self.partial_measure = partial_measure

        if len(states_range) == 1:
            self.states_range = (0, 2 ** (self.n_qubits - self.a_qubits))
        else:
            self.states_range = states_range


    def part_measure(self):
        # Non-linear Transform
        probs = self.ansatz(self.q_params[0]).float().unsqueeze(0)[0]
        probsgiven0 = probs[self.states_range[0]: self.states_range[1]]
        probsgiven0 /= torch.sum(probsgiven0)

        # Post-Processing
        # probsgiven = probsgiven0 / torch.max(probsgiven0)
        return probsgiven0.reshape(1, len(probsgiven0))

    def forward(self):
        
        if self.partial_measure:
            qc_out = self.part_measure()
        else:
            qc_out = self.ansatz(self.q_params[0]).float().unsqueeze(0)
        
        return qc_out
    
    
    def trained_circuit(self, shots: int):
        dev = qml.device("default.qubit", wires=self.n_qubits, shots=shots)

        @qml.qnode(dev)
        def quantum_circuit():

            weights = self.q_params[0].reshape(self.q_depth, self.n_qubits).detach()

            # Initialise latent vectors
            for i in range(self.n_qubits):
                qml.Hadamard(wires=i)

            # Repeated layer
            for i in range(self.q_depth):
                # Parameterised layer
                for y in range(self.n_qubits):
                    qml.RY(weights[i][y], wires=y)

                # Control Z gates
                for y in range(self.n_qubits - 1):
                    qml.CZ(wires=[y, y + 1])

                #qml.CZ(wires=[5, 0])

                qml.Barrier(wires=list(range(self.n_qubits)), only_visual=True)

            return qml.sample(wires=list(range(self.n_qubits)))
        
        bin_samples = quantum_circuit()
        samples = [self.binary_to_decimal(row) for row in bin_samples]
    
        return np.array(samples)
    

    def filtered_distribution(self, shots: int, excluded_states: list[int] = []):
        
        if self.partial_measure:
            states = 2 ** self.n_qubits
            excluded_states = list(set(range(states)) - set(range(self.states_range[0], self.states_range[1])))

        generated_samples = self.trained_circuit(shots=2*shots)

        # Filter out the unwanted states
        samples = generated_samples[~np.isin(generated_samples, excluded_states)]
        #print(filtered_samples, filtered_samples.shape)
        s = 0
        factor = 2

        while s < shots:
            generated_samples = self.trained_circuit(shots=factor*shots)
            filtered_samples = generated_samples[~np.isin(generated_samples, excluded_states)]

            # Filter out the unwanted states
            samples = np.concatenate((samples, filtered_samples), axis=None)
            s += np.size(filtered_samples)

            factor += 1

        
        if len(samples) >= shots:
            sampled_array = np.random.choice(samples, size=shots, replace=False)
            return sampled_array
        else:
            print("There is not enough samples after filtering")

    
    @staticmethod
    def binary_to_decimal(binary: np.array) -> int:
        return int("".join(str(x) for x in binary), 2)


