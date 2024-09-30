import os
import os.path as osp
import shutil

import torch
from torch_geometric.data import InMemoryDataset, download_url, extract_zip, Data
from torch_geometric.io import read_tu_data

from itertools import repeat, product
import numpy as np

from copy import deepcopy
import pdb

from typing import Callable, Optional

from scipy.io import loadmat

class QM7b_aug(InMemoryDataset):
    url = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm7.mat'

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        aug=None, 
        rho=0.9,
        force_reload: bool = True,
    ) -> None:
        
        
        super().__init__(root, transform, pre_transform, pre_filter, force_reload=force_reload)

        self.data, self.slices = torch.load(self.processed_paths[0])
        # print(self._data)
        self.node_score = torch.zeros(self._data['x'].shape[0], dtype=torch.half)
        self.aug = aug
        self.rho = rho
    @property
    def raw_file_names(self) -> str:
        return 'qm7.mat'

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self) -> None:
        download_url(self.url, self.raw_dir)

    def process(self) -> None:
        data = loadmat(self.raw_paths[0])
        print(data.keys(), data["X"].shape)
        coulomb_matrix = torch.tensor(data['X'])
        target = torch.tensor(data["T"][0])

        z = data["Z"]
        z = torch.tensor(z, dtype=torch.float)

        pos = torch.tensor(data["R"], dtype=torch.float)

        data_list = []
        for i in range(target.shape[0]):
            edge_index = coulomb_matrix[i].nonzero(as_tuple=False).t().contiguous()
            edge_attr = coulomb_matrix[i, edge_index[0], edge_index[1]]
            graph_data = Data(edge_index=edge_index, 
                              edge_attr=edge_attr, 
                              y=target[i].unsqueeze(0),
                              x=z[i].unsqueeze(-1),
                              z=z[i],
                            #   pos=pos[i]
                            )
            graph_data.num_nodes = torch.tensor([edge_index.max().item() + 1])
            mask = graph_data.z != 0
            graph_data.z = graph_data.z[mask]
            graph_data.x = graph_data.x[mask]
            # graph_data.pos = graph_data.pos[mask]

            data_list.append(graph_data)

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        torch.save(self.collate(data_list), self.processed_paths[0])

    def get(self, idx):
        data = self._data.__class__()

        if hasattr(self._data, '__num_nodes__'):
            data.num_nodes = self._data.__num_nodes__[idx]
        # print(self._data.keys(), self._data)
        # print("dsdf",  self.slices)
        for key in self._data.keys():
            if key == "num_nodes":
                continue
            item, slices = self._data[key], self.slices[key]
            if torch.is_tensor(item):
                s = list(repeat(slice(None), item.dim()))
                s[self._data.__cat_dim__(key, item)] = slice(slices[idx], slices[idx + 1])
            else:
                s = slice(slices[idx], slices[idx + 1])
            data[key] = item[s]

        """
        edge_index = data.edge_index
        node_num = data.x.size()[0]
        edge_num = data.edge_index.size()[1]
        data.edge_index = torch.tensor([[edge_index[0, n], edge_index[1, n]] for n in range(edge_num) if edge_index[0, n] < node_num and edge_index[1, n] < node_num] + [[n, n] for n in range(node_num)], dtype=torch.int64).t()
        """

        node_num = data.edge_index.max()
        # sl = torch.tensor([[n, n] for n in range(node_num)]).t()
        # data.edge_index = torch.cat((data.edge_index, sl), dim=1)

        if self.aug == 'drop_ra':

            nodes_score = self.node_score[self.slices['x'][idx]:self.slices['x'][idx + 1]]
            data_aug = self.drop_nodes_prob(deepcopy(data), nodes_score, self.rho)
            data_cp = self.drop_nodes_cp(deepcopy(data), nodes_score, self.rho)

        elif self.aug == 'none':
            data_aug = deepcopy(data)
            data_aug.x = torch.ones((data.edge_index.max() + 1, 1))
            data_cp = deepcopy(data)
            data_cp.x = torch.ones((data.edge_index.max() + 1, 1))
        else:
            print('augmentation error')
            assert False

        return data, data_aug, data_cp

    def drop_nodes_prob(self, data, node_score, rho):
        # print(data.x)
        node_num = data.x.size(0)
        # print(node_num)
        drop_num = int(node_num * (1.0 - rho))

        # Convert node_score to a probability distribution
        node_prob = node_score.float() + 0.001
        node_prob = np.array(node_prob)
        node_prob /= node_prob.sum()

        idx_nondrop = np.random.choice(node_num, node_num - drop_num, replace=False, p=node_prob)
        idx_drop = np.setdiff1d(np.arange(node_num), idx_nondrop)
        idx_nondrop.sort()

        # Adjust edge index to reflect node drops
        edge_index = data.edge_index.numpy()
        adj = torch.zeros((node_num, node_num))
        adj[edge_index[0], edge_index[1]] = 1
        adj[idx_drop, :] = 0
        adj[:, idx_drop] = 0
        edge_index = adj.nonzero().t()

        # Filter nodes and adjust edges
        edge_idx = edge_index.numpy()
        idx_not_missing = [n for n in range(node_num) if (n in edge_idx[0] or n in edge_idx[1])]
        node_num_aug = len(idx_not_missing)

        data.x = data.x[idx_not_missing]
        idx_dict = {idx_not_missing[n]: n for n in range(node_num_aug)}
        edge_idx = [[idx_dict[edge_idx[0, n]], idx_dict[edge_idx[1, n]]] for n in range(edge_idx.shape[1])
                    if not edge_idx[0, n] == edge_idx[1, n]]
        data.edge_index = torch.tensor(edge_idx).transpose_(0, 1)

        return data

    def drop_nodes_cp(self, data, node_score, rho):
        node_num = data.x.size(0)
        drop_num = int(node_num * (1.0 - rho))

        # Invert node scores (complement probability)
        node_prob = max(node_score.float()) - node_score.float() + 0.001
        node_prob = np.array(node_prob)
        node_prob /= node_prob.sum()

        idx_nondrop = np.random.choice(node_num, node_num - drop_num, replace=False, p=node_prob)
        idx_drop = np.setdiff1d(np.arange(node_num), idx_nondrop)
        idx_nondrop.sort()

        # Adjust edge index to reflect node drops
        edge_index = data.edge_index.numpy()
        adj = torch.zeros((node_num, node_num))
        adj[edge_index[0], edge_index[1]] = 1
        adj[idx_drop, :] = 0
        adj[:, idx_drop] = 0
        edge_index = adj.nonzero().t()

        # Filter nodes and adjust edges
        edge_idx = edge_index.numpy()
        idx_not_missing = [n for n in range(node_num) if (n in edge_idx[0] or n in edge_idx[1])]
        node_num_aug = len(idx_not_missing)

        data.x = data.x[idx_not_missing]
        idx_dict = {idx_not_missing[n]: n for n in range(node_num_aug)}
        edge_idx = [[idx_dict[edge_idx[0, n]], idx_dict[edge_idx[1, n]]] for n in range(edge_idx.shape[1])
                    if not edge_idx[0, n] == edge_idx[1, n]]
        data.edge_index = torch.tensor(edge_idx).transpose_(0, 1)

        return data
