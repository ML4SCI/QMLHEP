from .graph_syn_qg import *


class TopKMomentum(T.BaseTransform):
    def __init__(self, k: int):
        self.k = k
    
    def __call__(self, data):
        if data.num_nodes > self.k:
            _, top_k_indices = torch.topk(data.h[:, 0], self.k)
            data.h = data.h[top_k_indices]
            data.particleid = data.particleid[top_k_indices]
            data.num_nodes = self.k
        return data


class ToTopMomentum(T.BaseTransform):
    def __call__(data):
        top_index = torch.argemax(data.h[:, 0])

        edges = []
        for i in range(data.num_nodes):
            if i != top_index:
                edges.append([i, top_index])
                

        return data

# Transform: Edge Creation
class KNNGroup(T.BaseTransform):
    def __init__(self, k: int, attr_name: str):
        self.k = k
        self.attr_name = attr_name

    def __call__(self, data, self_loop=True):
        if hasattr(data, self.attr_name):
            attr = getattr(data, self.attr_name)
            edge_index = pyg_nn.knn_graph(attr, k=self.k)
            data.edge_index = edge_index
            
            if self_loop:
                # Create self-loops
                num_nodes = attr.size(0)
                self_loops = torch.arange(0, num_nodes, dtype=torch.long).unsqueeze(0).repeat(2, 1)
                
                # Concatenate self-loops to edge_index
                edge_index = torch.cat([edge_index, self_loops], dim=1)
                data.edge_index = edge_index
        else:
            raise ValueError(f"Attribute '{self.attr_name}' not found in data.")
        return data
