

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch.nn import Linear, ReLU, Sigmoid, ModuleList, LeakyReLU, Linear, BatchNorm1d, BatchNorm
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.data import Data, Dataset
from torch.utils.data import Dataset
from torch_geometric.utils import to_networkx
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from torch_geometric.utils import subgraph
import torch.nn.functional as F

# Hybrid Quantum-Classical GNN model
class HybridQuantumGNN(torch.nn.Module):
    def __init__(self, input_dims, hidden_dims, output_dims, activ_fn, QuantumLayer, n_qubits):
        super().__init__()
        layers = []
        self.norm_layers = []
        self.residual_projections = []  # To project residuals if needed
        
        # Initial GAT Layer
        layers.append(GATConv(input_dims, hidden_dims[0]))
        self.norm_layers.append(BatchNorm1d(hidden_dims[0]))
        if input_dims != hidden_dims[0]:
            self.residual_projections.append(Linear(input_dims, hidden_dims[0]))
        else:
            self.residual_projections.append(None)
        
        # Additional Graph Layers with Residuals and Normalization
        for i in range(len(hidden_dims) - 1):
            layers.append(GATConv(hidden_dims[i], hidden_dims[i+1]))
            self.norm_layers.append(BatchNorm1d(hidden_dims[i+1]))
            if hidden_dims[i] != hidden_dims[i+1]:
                self.residual_projections.append(Linear(hidden_dims[i], hidden_dims[i+1]))
            else:
                self.residual_projections.append(None)
        
        self.layers = ModuleList(layers)
        self.norm_layers = ModuleList(self.norm_layers)
        self.residual_projections = ModuleList(self.residual_projections)
        self.activ_fn = activ_fn
        
        # Quantum Layer
        self.quantum_layer = QuantumLayer
        self.n_qubits = n_qubits

        # Classical Readout Layer
        self.readout_layer = Linear(hidden_dims[-1] * 2 + self.n_qubits, output_dims)  # Combining GNN and quantum outputs
        self.dropout = torch.nn.Dropout(p=0.5)  

    def forward(self, x, edge_index, batch):
        h = x
        for i in range(len(self.layers)):
            residual = h  # Residual connection
            h = self.layers[i](h, edge_index)
            h = self.norm_layers[i](h)
            h = self.activ_fn(h)
            h = self.dropout(h)
            
            # Project residual if needed
            if self.residual_projections[i] is not None:
                residual = self.residual_projections[i](residual)
            
            h = h + residual  # Adding residual connection
        
        # Multi-head readout: mean pooling + max pooling
        h_mean = global_mean_pool(h, batch)
        h_max = global_max_pool(h, batch)
        h_gnn = torch.cat([h_mean, h_max], dim=1)

        # Quantum layer output
        h_quantum = self.quantum_layer(h_gnn)

        # Combine GNN and quantum layer outputs
        h_combined = torch.cat([h_gnn, h_quantum], dim=1)

        # Final readout layer for graph-level embedding
        return self.readout_layer(h_combined)


class GNN(torch.nn.Module):
    def __init__(self, input_dims, hidden_dims, output_dims, activ_fn):
        super().__init__()
        layers = []
        self.norm_layers = []
        self.residual_projections = []  # To project residuals if needed
        
        # Initial GAT Layer
        layers.append(GATConv(input_dims, hidden_dims[0]))
        self.norm_layers.append(BatchNorm(hidden_dims[0]))
        if input_dims != hidden_dims[0]:
            self.residual_projections.append(Linear(input_dims, hidden_dims[0]))
        else:
            self.residual_projections.append(None)
        
        # Additional Graph Layers with Residuals and Normalization
        for i in range(len(hidden_dims) - 1):
            layers.append(GATConv(hidden_dims[i], hidden_dims[i+1]))
            self.norm_layers.append(BatchNorm(hidden_dims[i+1]))
            if hidden_dims[i] != hidden_dims[i+1]:
                self.residual_projections.append(Linear(hidden_dims[i], hidden_dims[i+1]))
            else:
                self.residual_projections.append(None)
        
        self.layers = ModuleList(layers)
        self.norm_layers = ModuleList(self.norm_layers)
        self.residual_projections = ModuleList(self.residual_projections)
        self.activ_fn = activ_fn
        self.readout_layer = Linear(hidden_dims[-1] * 2, output_dims)  # Expanded output for multiple readouts
        self.dropout = torch.nn.Dropout(p=0.5)  

    def forward(self, x, edge_index, batch):
        h = x
        for i in range(len(self.layers)):
            residual = h  # Residual connection
            h = self.layers[i](h, edge_index)
            h = self.norm_layers[i](h)
            h = self.activ_fn(h)
            h = self.dropout(h)
            
            # Project residual if needed
            if self.residual_projections[i] is not None:
                residual = self.residual_projections[i](residual)
            
            h = h + residual  # Adding residual connection
        
        # Multi-head readout: mean pooling + max pooling
        h_mean = global_mean_pool(h, batch)
        h_max = global_max_pool(h, batch)
        h = torch.cat([h_mean, h_max], dim=1)
        
        # Final readout layer for graph-level embedding
        return self.readout_layer(h)
