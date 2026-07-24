"""
Quantum autoencoder implementations using pennylane
"""

import os
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import pennylane as qml
from pennylane import numpy as np
from qutip import Bloch
from sklearn.metrics import roc_auc_score, roc_curve

class ConvSQAE(pl.LightningModule):
    """SQAE with a convolutional-like path encoding of the data."""

    def __init__(self, data_qbits, latent_qbits, device, img_dim, kernel_size, stride, DRCs, diff_method="best", learning_rate=0.01):
        """Create basic convolutional-like SQAE

        Args:
            data_qbits (int): Number of qubits to upload data and use as encoder
            latent_qbits (int): Number of latent qubits
            device (pennylane.Device): Pennylane device to use for circuit evaluation
            img_dim (int): Dimension of the images (width)
            kernel_size (int): Size of the kernel to use when uploading the data
            stride (int): Stride to use when uploading the data
            DRCs (int): Number of times to repeat the encoding upload circuit in the encoder
            diff_method (str): Method to differentiate quantum circuit, usually "adjoint" is best
            learning_rate (float): Learning rate for optimizer
        """
        super(ConvSQAE, self).__init__()
        self.dev = device
        self.data_qbits = data_qbits
        self.latent_qbits = latent_qbits
        self.trash_qbits = self.data_qbits - self.latent_qbits
        self.total_qbits = data_qbits + self.trash_qbits + 1
        self.circuit_node = qml.QNode(self.circuit, device, diff_method=diff_method)
        self.latent_node = qml.QNode(self.visualize_latent_circuit, device, diff_method=diff_method)

        self.auc_hist = []
        self.animation_filenames = []
        self.bloch_animation_filenames = []

        self.kernel_size = kernel_size
        self.stride = stride
        self.DRCs = DRCs

        self.data_shape = (img_dim, img_dim)
        self.number_of_kernel_uploads = len(list(range(0, img_dim - kernel_size + 1, stride)))**2
        self.parameters_shape = (DRCs * 2 * self.number_of_kernel_uploads * kernel_size ** 2,)
        self.params_per_layer = self.parameters_shape[0] // DRCs

        self.params = np.random.uniform(size=self.parameters_shape, requires_grad=True)
        self.rnd_init = np.random.uniform(0, np.pi, size=self.trash_qbits, requires_grad=False)

        self.learning_rate = learning_rate

    def single_upload(self, params, data, wire):
        """Upload data on a single qubit

        Args:
            params (list): Parameters to use for upload, must be twice as long as data
            data (list): Data to upload
            wire (int): On which wire to upload
        """
        for i, d in enumerate(data.flatten()):
            if i % 3 == 0:
                qml.RZ(params[i * 2] + params[i * 2 + 1] * d, wires=wire)
            if i % 3 == 1:
                qml.RY(params[i * 2] + params[i * 2 + 1] * d, wires=wire)
            if i % 3 == 2:
                qml.RZ(params[i * 2] + params[i * 2 + 1] * d, wires=wire)

    def conv_upload(self, params, img, kernel_size, stride, wires):
        """Upload image using the convolution-like method

        Args:
            params (list): Parameters to use
            img (2D list): The actual image
            kernel size (int): Kernel size for upload
            stride (int): Stride for upload
            wires (list): List of integers to use as qubit index for upload
        """
        number_of_kernel_uploads = len(list(range(0, img.shape[1]-kernel_size+1, stride))) * len(list(range(0, img.shape[0]-kernel_size+1, stride)))
        params_per_upload = len(params) // number_of_kernel_uploads
        upload_counter = 0
        wire = 0
        for y in range(0, img.shape[1]-kernel_size+1, stride):
            for x in range(0, img.shape[0]-kernel_size+1, stride):
                self.single_upload(params[upload_counter * params_per_upload: (upload_counter + 1) * params_per_upload],
                                   img[y:y+kernel_size, x:x+kernel_size], wires[wire])
                upload_counter = upload_counter + 1
                wire = wire + 1

    def circular_entanglement(self, wires):
        """Entangles wires in a circular shape

        Args:
            wires (list): List of qubits to entangle
        """
        qml.CNOT(wires=[wires[-1], 0])
        for i in range(len(wires)-1):
            qml.CNOT(wires=[i, i+1])

    def encoder(self, params, data):
        """The encoder circuit for the SQAE

        Args:
            params (list): Parameters to use
            data (list): Data to upload
        """
        for i in range(self.DRCs):
            self.conv_upload(params[i * self.params_per_layer:(i + 1) * self.params_per_layer], data, self.kernel_size, self.stride, list(range(self.number_of_kernel_uploads)))
            self.circular_entanglement(list(range(self.number_of_kernel_uploads)))

    def circuit(self, params, data):
        """Full circuit to be used as SQAE, includes encoder and SWAP test

        Args:
            params (list): Parameters to be used for PQC
            data (list): Data for the circuit

        Returns:
            Expectation value of readout bit
        """
        self.encoder(params, data)

        qml.Hadamard(wires=self.total_qbits-1)
        for i in range(self.trash_qbits):
            qml.CSWAP(wires=[self.total_qbits - 1, self.latent_qbits + i, self.data_qbits + i])
        qml.Hadamard(wires=self.total_qbits-1)

        return qml.expval(qml.PauliZ(self.total_qbits-1))

    def visualize_latent_circuit(self, params, data, bit, measure_gate):
        """Encoder circuit with measurement on chosen qubit

        Args:
            params (list): Parameters to be used for PQC
            data (list): Data for the circuit
            bit (int): Qubit index to be measured
            measure_gate (function): Function of gate to be measured (qml.PauliX, qml.PauliY, qml.PauliZ)

        Returns:
            Expectation value of chosen bit
        """
        self.encoder(params, data)
        return qml.expval(measure_gate(bit))

    def plot_latent_space(self, latent_bit, x_test_bg, x_test_signal, save_fig=None):
        """Plots the Bloch sphere of a chosen qubit for given signal and background data

        Args:
            latent_bit (int): Bit to visualize
            x_test_bg (list): Background data
            x_test_signal (list): Signal data
            save_fig (str): If string is given, plot will be saved with given name
        """
        b = Bloch()
        points_bg_x = [self.latent_node(self.params, i, latent_bit, qml.PauliX) for i in x_test_bg]
        points_bg_y = [self.latent_node(self.params, i, latent_bit, qml.PauliY) for i in x_test_bg]
        points_bg_z = [self.latent_node(self.params, i, latent_bit, qml.PauliZ) for i in x_test_bg]
        x = [points_bg_x[i] for i in range(len(points_bg_x))]
        y = [points_bg_y[i] for i in range(len(points_bg_y))]
        z = [points_bg_z[i] for i in range(len(points_bg_z))]
        bloch_points_bg = [x, y, z]
        b.add_points(bloch_points_bg)

        points_sig_x = [self.latent_node(self.params, i, latent_bit, qml.PauliX) for i in x_test_signal]
        points_sig_y = [self.latent_node(self.params, i, latent_bit, qml.PauliY) for i in x_test_signal]
        points_sig_z = [self.latent_node(self.params, i, latent_bit, qml.PauliZ) for i in x_test_signal]
        xs = [points_sig_x[i] for i in range(len(points_sig_x))]
        ys = [points_sig_y[i] for i in range(len(points_sig_y))]
        zs = [points_sig_z[i] for i in range(len(points_sig_z))]
        bloch_points_sig = [xs, ys, zs]
        b.add_points(bloch_points_sig)

        if save_fig:
            b.save(name=save_fig)
        else:
            b.show()
        b.clear()

    def plot_circuit(self):
        """Plots the circuit with dummy data."""
        data = np.random.uniform(size=self.data_shape)
        fig, _ = qml.draw_mpl(self.circuit_node)(self.params, data)
        fig.show()

    def training_step(self, batch, batch_idx):
        """Training step executed by PyTorch Lightning

        Args:
            batch: Training batch
            batch_idx: Batch index

        Returns:
            Loss value
        """
        x_train = batch[0]
        loss = self.cost_batch(self.params, x_train)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step executed by PyTorch Lightning

        Args:
            batch: Validation batch
            batch_idx: Batch index

        Returns:
            Loss value
        """
        x_val = batch[0]
        val_loss = self.cost_batch(self.params, x_val)
        self.log('val_loss', val_loss)
        return val_loss

    def configure_optimizers(self):
        """Configure optimizers for PyTorch Lightning

        Returns:
            Optimizer
        """
        return torch.optim.Adam([self.params], lr=self.learning_rate)

    def iterate_minibatches(self, data, batch_size):
        """Generator of data as batches

        Args:
            data (list): List of data
            batch_size (int): Size of batches to return
        """
        for start_idx in range(0, data.shape[0] - batch_size + 1, batch_size):
            idxs = slice(start_idx, start_idx + batch_size)
            yield data[idxs]

    def cost_batch(self, params, batch):
        """Evaluate cost function on batch - MSE of fidelities

        Args:
            params: Parameters to use for evaluation
            batch: Batch of data

        Returns:
            Average loss (1 - fidelity) of batch
        """
        loss = 0.0
        for i in batch:
            sample_loss = self.circuit_node(params, i)
            loss = loss + (1 - sample_loss) ** 2
        return loss / len(batch)

    def evaluate(self, x_test_bg, x_test_signal, save_fig=None):
        """Evaluate tagging performance on test data giving AUC and ROC curve

        Args:
            x_test_bg (list): Background data (same class the model was trained on)
            x_test_signal (list): Signal data (model has never seen data from this class)
            save_fig (str): If a string is given the resulting plots will be saved to the given path
        """
        pred_bg = np.array([self.circuit_node(self.params, i) for i in x_test_bg])
        if not save_fig: print("Median fidelities bg: ", np.median(pred_bg))

        pred_signal = np.array([self.circuit_node(self.params, i) for i in x_test_signal])
        if not save_fig: print("Median fidelities signal: ", np.median(pred_signal))

        bce_background = 1 - pred_bg
        bce_signal = 1 - pred_signal

        fig, axs = plt.subplots(1, 3, figsize=(11, 4))

        if not save_fig: print(f'Median background: {np.median(bce_background):.3}')
        if not save_fig: print(f'Median signal: {np.median(bce_signal):.3}')
        bins = np.histogram(np.hstack((bce_background, bce_signal)), bins=25)[1]
        axs[0].hist(bce_background, histtype='step', label="background", bins=bins)
        axs[0].hist(bce_signal, histtype='step', label="signal", bins=bins)
        axs[0].set_xlabel("loss")
        axs[0].legend()

        thresholds = np.linspace(0, max(np.max(bce_background), np.max(bce_signal)), 1000)

        accs = []
        for i in thresholds:
            num_background_right = np.sum(bce_background < i)
            num_signal_right = np.sum(bce_signal > i)
            acc = (num_background_right + num_signal_right) / (len(pred_bg) + len(pred_signal))
            accs.append(acc)

        if not save_fig: print(f'Maximum accuracy: {np.max(accs):.3}')
        axs[1].plot(thresholds, accs)
        axs[1].set_xlabel("anomaly threshold")
        axs[1].set_ylabel("tagging accuracy")

        y_true = np.append(np.zeros(len(bce_background)), np.ones(len(bce_signal)))
        y_pred = np.append(bce_background, bce_signal)
        auc = roc_auc_score(y_true, y_pred)
        if not save_fig: print(f'AUC: {auc:.4}')
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        tnr = 1 - fpr
        x = np.linspace(0, 1, 200)
        y_rnd = 1 - x
        axs[2].plot(tnr, tpr, label=f"anomaly tagger, auc:{auc:.3}")
        axs[2].plot(x, y_rnd, label="random tagger", color='grey')
        axs[2].set_xlabel("fpr")
        axs[2].set_ylabel("tpr")
        axs[2].legend(loc="lower left")

        fig.tight_layout()
        if save_fig:
            plt.savefig(save_fig)
            plt.close()

    def plot_train_hist(self, logscale=False):
        """Plot the history of the training and validation loss

        Args:
            logscale (bool): If true y scale will be plotted logarithmically
        """
        plt.plot(self.train_hist['loss'], label="train")
        plt.plot(self.train_hist['val_loss'], label="val")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend()
        if logscale:
            plt.yscale('log')
        plt.show()
