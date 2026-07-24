import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torch_geometric as pyg
import torch_geometric.nn as pyg_nn
import torch_geometric.data as pyg_data
import torch_geometric.loader as pyg_loader
import torch_geometric.transforms as T

import networkx as nx
import numpy as np
import glob, os, shutil

class QG_Jets(pyg_data.InMemoryDataset):
    def __init__(self, root,
                 transform=None, pre_transform=None, pre_filter=None, force_reload=True):
        super().__init__(root, transform, pre_transform)
        self.root = root

        self.load(self.processed_paths[0])
        
    @property
    def raw_file_names(self):
        files = glob.glob(self.root + '/raw/*.npz') 
        return [file.split('/')[-1] for file in files]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self) -> None:
        pass
    def process(self):
        data_list = []
        print(self.raw_paths)
        for raw_path in self.raw_paths:
            
            data = np.load(raw_path)

            jets = data['X']
            ys = data['y']  
            
            # PDGid to float dictionary
            PID2FLOAT_MAP = {22: 0,
                 211: .1, -211: .2,
                 321: .3, -321: .4,
                 130: .5,
                 2112: .6, -2112: .7,
                 2212: .8, -2212: .9,
                 11: 1.0, -11: 1.1,
                 13: 1.2, -13: 1.3,
                 0: 1.5,  # Representing undefined PID
                 }
            
            pids = np.unique(jets[:, :, 3]).flatten()
            print("PIDS:", pids)
            for index, jet in enumerate(jets):
                # Remove zero-padding
                jet = jet[~np.all(jet == 0, axis=1)]

                pt_total = np.sum(jet[:, 0])  # Total scalar sum of p_T
                
                # Center y and phi
                weighted_y = np.sum(jet[:, 0] * jet[:, 1]) / pt_total
                weighted_phi = np.sum(jet[:, 0] * jet[:, 2]) / pt_total
                jet[:, 1] = jet[:, 1] - weighted_y
                jet[:, 2] = jet[:, 2] - weighted_phi
                
                # Normalize p_T
                jet[:, 0] = jet[:, 0] / pt_total
            
                for pid in pids:
                    np.place(jet[:, 3], jet[:, 3] == pid, PID2FLOAT_MAP[pid])
        
                data = pyg_data.Data(particleid = torch.tensor(jet[:, 3:], 
                                            dtype=torch.float), 
                                        h = torch.tensor(jet[:, :3], 
                                            dtype=torch.float),
                                        y=int(ys[index]),
                                        num_nodes=jet.shape[0])

                data_list.append(data)

        self.save(data_list, self.processed_paths[0])

        return data_list
    
    
class QG_Jets_old(pyg_data.InMemoryDataset):
    def __init__(self, root,
                 transform=None, pre_transform=None, pre_filter=None, force_reload=True):
        super().__init__(root, transform, pre_transform)
        self.root = root

        self.load(self.processed_paths[0])
        
    @property
    def raw_file_names(self):
        files = glob.glob(self.root + '/raw/*.npz') 
        return [file.split('/')[-1] for file in files]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self) -> None:
        pass
    def process(self):
        data_list = []
        print(self.raw_paths)
        for raw_path in self.raw_paths:
            
            data = np.load(raw_path)

            jets = data['X']
            ys = data['y']  
            
            # PDGid to float dictionary
            PID2FLOAT_MAP = {22: 0,
                 211: .1, -211: .2,
                 321: .3, -321: .4,
                 130: .5,
                 2112: .6, -2112: .7,
                 2212: .8, -2212: .9,
                 11: 1.0, -11: 1.1,
                 13: 1.2, -13: 1.3,
                 0: 1.5,  # Representing undefined PID
                 }
            
            pids = np.unique(jets[:, :, 3]).flatten()
            print("PIDS:", pids)
            for index, jet in enumerate(jets):
                # Remove zero-padding
                jet = jet[~np.all(jet == 0, axis=1)]

                pt_total = np.sum(jet[:, 0])  # Total scalar sum of p_T
                
                # Center y and phi
                weighted_y = np.sum(jet[:, 0] * jet[:, 1]) / pt_total
                weighted_phi = np.sum(jet[:, 0] * jet[:, 2]) / pt_total
                jet[:, 1] = jet[:, 1] - weighted_y
                jet[:, 2] = jet[:, 2] - weighted_phi
                
                # Normalize p_T
                jet[:, 0] = jet[:, 0] / pt_total
            
                for pid in pids:
                    np.place(jet[:, 3], jet[:, 3] == pid, PID2FLOAT_MAP[pid])
        
                data = pyg_data.Data(particleid = torch.tensor(jet[:, 3:], 
                                            dtype=torch.float), 
                                        h = torch.tensor(jet[:, :3], 
                                            dtype=torch.float),
                                        y=int(ys[index]),
                                        num_nodes=jet.shape[0])

                data_list.append(data)

        self.save(data_list, self.processed_paths[0])

        return data_list