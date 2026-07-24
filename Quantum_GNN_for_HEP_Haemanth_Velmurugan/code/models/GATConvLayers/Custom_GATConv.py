import torch
from torch.nn import Linear, Parameter, Sequential, ReLU, LeakyReLU
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, remove_self_loops


class GATConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).
#         self.heads = heads
#         seld.concat = concat
        self.lin = Linear(in_channels, out_channels, bias=False)
        self.attn = Sequential(Linear(out_channels*2, 8),
                               ReLU(),
                               Linear(8, 1),
                               LeakyReLU(0.2)
                               )
        total_out_channels = out_channels
        self.bias = Parameter(torch.empty(total_out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin.reset_parameters()

        for layer in self.attn:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

        self.bias.data.zero_()

    def forward(self, x, edge_index, edge_attr):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # H, C = self.heads, self.out_channels

        x = self.lin(x)  # .view(-1, H, C)

        # Add self-loops to the adjacency matrix.
        edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
        edge_index, edge_attr = add_self_loops(
            edge_index, edge_attr, num_nodes=x.size(0))

        # alpha = self.edge_update(x_i, x_j, edge_attr=edge_attr)

        # propagate_type: (x: OptPairTensor, alpha: Tensor)
        out = self.propagate(edge_index, x=x)

        # out = out.mean(dim=1)

        # Apply a final bias vector.
        out = out + self.bias

        return out

    def message(self, x_i, x_j):
        x_edge = torch.cat((x_i, x_j), dim=1)
        x_edge = self.attn(x_edge)
        return x_edge.view(-1, 1) * x_j
