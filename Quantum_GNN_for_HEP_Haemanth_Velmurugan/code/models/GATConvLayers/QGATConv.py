import torch
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, remove_self_loops
from QNN_Node_Embedding import quantum_net
from Quantum_Classifiers import MPS, TTN


class QGATConv(MessagePassing):
    def __init__(self, in_channels, depth, attn_model):
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).
        self.bias = nn.Parameter(torch.empty(in_channels))
        self.reset_parameters()
        self.n_qubits = in_channels
        self.qc, _ = quantum_net(self.n_qubits, depth)

        if attn_model == "MPS":
            self.attn, _ = MPS(in_channels*2)
        else:
            self.attn, _ = TTN(in_channels*2)

        self.readout = Linear(1, 1)
        self.attn = Linear(in_channels*2, 1)

    def reset_parameters(self):
        self.bias.data.zero_()

    def forward(self, x, edge_index, edge_attr):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Add self-loops to the adjacency matrix.
        edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
        edge_index, edge_attr = add_self_loops(
            edge_index, edge_attr, num_nodes=x.size(0))

        # Linearly transform node feature matrix
        q_out = self.qc(x).float()

        # Start propagating messages.
        out = self.propagate(edge_index, x=q_out)

        # Apply a final bias vector.
        out = out + self.bias

        return out

    def message(self, x_i, x_j):

        x_edge = torch.cat((x_i, x_j), dim=1)
        x_edge = self.attn(x_edge)
        x_edge = self.readout(x_edge)
        return x_edge.view(-1, 1) * x_j
