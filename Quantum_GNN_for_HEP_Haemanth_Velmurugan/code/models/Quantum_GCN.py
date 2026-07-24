import torch
from torch.nn import Module, ModuleList, Linear, LeakyReLU
from torch_geometric.nn import global_mean_pool
from QNN_Node_Embedding import quantum_net
from GCNConv_Layers import GCNConv
from Quantum_Classifiers import MPS, TTN


class QGCN(Module):

    def __init__(self, input_dims, q_depths, output_dims, activ_fn=LeakyReLU(0.2), classifier=None, readout=False):

        super().__init__()
        layers = []
        self.n_qubits = input_dims

        for q_depth in q_depths:
            nodeNN = quantum_net(self.n_qubits, q_depth)
            QGCNConv = GCNConv(self.n_qubits, nodeNN)
            layers.append(QGCNConv)

        self.layers = ModuleList(layers)
        self.activ_fn = activ_fn

        if readout:
            self.readout = Linear(1, 1)
        else:
            self.readout = None

        if classifier == "MPS":
            meas_qubits = [i for i in range(
                self.n_qubits-1, self.n_qubits-1-output_dims, -1)]
            self.classifier = MPS(input_dims, meas_qubits)

        elif classifier == "TTN":
            meas_qubits = [i for i in range(
                self.n_qubits-1, self.n_qubits-1-output_dims, -1)]
            self.classifier = TTN(input_dims, meas_qubits)

        else:
            self.classifier = Linear(input_dims, output_dims)

    def forward(self, x, edge_index, batch):
        """
        Defining how tensors are supposed to move through the *dressed* quantum
        net.
        """

        h = x
        for i in range(len(self.layers)):
            h = self.layers[i](h, edge_index)
            h = self.activ_fn(h)

        # readout layer to get the embedding for each graph in batch
        h = global_mean_pool(h, batch)
        h = self.classifier(h)

        if self.readout is not None:
            h = self.readout(h)

        # return the prediction from the postprocessing layer
        return h
