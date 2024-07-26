import torch
import torch.nn as nn
import pytorch_lightning as pl

import torchmetrics
import pennylane as qml
import matplotlib.pyplot as plt

from pytorch_metric_learning import losses

from ..utils import get_preprocessing
from .classical import ConvEncoder


class QuantumHead(nn.Module):
    def __init__(self, in_features, n_qubits, out_features, n_qlayers):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_qubits = n_qubits
        self.n_qlayers = n_qlayers
        
        self.proj = nn.Linear(in_features, n_qubits)
        
        self.device = qml.device('default.qubit', wires=self.n_qubits)
        @qml.qnode(self.device, interface='torch')
        def quantum_circuit(inputs, weights):
            # print(inputs.shape)
            qml.templates.AngleEmbedding(inputs, wires=range(self.n_qubits))
            
            # Apply layers of rotation gates and CNOTs for entanglement
            for layer in range(self.n_qlayers):
                for qubit in range(self.n_qubits):
                    qml.RX(weights[layer, qubit, 0], wires=qubit)
                    qml.RY(weights[layer, qubit, 1], wires=qubit)
                    qml.RZ(weights[layer, qubit, 2], wires=qubit)
                for qubit in range(self.n_qubits - 1):
                    qml.CNOT(wires=[qubit, qubit + 1])
                qml.CNOT(wires=[self.n_qubits - 1, 0])
            return [qml.expval(qml.PauliZ(i)) for i in range(self.out_features)]
        
        self.quantum_circuit = quantum_circuit
        dummy_inputs = torch.randn(self.n_qubits)
        dummy_weights = torch.randn(self.n_qlayers, self.n_qubits, 3)
        print(qml.draw(self.quantum_circuit)(dummy_inputs, dummy_weights))
        
        qml.draw_mpl(self.quantum_circuit)(dummy_inputs, dummy_weights)
        plt.show()
        
        self.quantum_layer = qml.qnn.TorchLayer(self.quantum_circuit, {"weights": (n_qlayers, self.n_qubits, 3)})
        
        # batch_dim = 8
        # x = torch.zeros((batch_dim, self.in_features))
        # print(self.quantum_layer(x).shape)
        
    def forward(self, inputs):
        q_inputs = self.proj(inputs)
        q_outputs = self.quantum_layer(q_inputs)
        return q_outputs

class Hybrid_Contrastive(pl.LightningModule):
    """
    A PyTorch Lightning module for supervised contrastive learning on the MNIST dataset.
    """
    def __init__(self, activ_type, pool_type, head_output, n_qubits, n_qlayers, lr, pos_margin=0.25, neg_margin=1.5, preprocess=None):
        super().__init__()
        self.save_hyperparameters()
        self.preprocessing = get_preprocessing(preprocess)
        self.encoder = ConvEncoder(activ_type, pool_type)
        self.head = QuantumHead(ConvEncoder.backbone_output_size, n_qubits, head_output, n_qlayers)
        self.loss = losses.ContrastiveLoss(pos_margin=pos_margin, neg_margin=neg_margin)
        self.train_loss = torchmetrics.MeanMetric()
        self.valid_loss = torchmetrics.MeanMetric()

    def forward(self, x):
        x = self.encoder(x)
        x = self.head(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        if self.preprocessing:
            x = self.preprocessing(x)
        embeddings = self.forward(x)
        loss = self.loss(embeddings, y)
        self.train_loss.update(loss)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        if self.preprocessing:
            x = self.preprocessing(x)
        embeddings = self.forward(x)
        loss = self.loss(embeddings, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
