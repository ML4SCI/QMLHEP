import energyflow
from particle import Particle
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import os
from training.utils import read_configurations


class QGJets():

    def __init__(self, num_data=None):
        self.config = read_configurations("./config.json")

        if num_data is None:
            self.N = self.config["NUM JETS"]
        else:
            self.N = num_data

        self.data = energyflow.qg_jets.load(num_data=self.N, pad=True, ncol=4, generator='pythia',
                                            with_bc=False, cache_dir='~/data')
        self.path = self.config["PATH"]

        dataset = np.load(os.path.expanduser(
            '~/data/datasets/QG_jets.npz'))   # make it self.dataset and use another loader function to work with raw data
        self.X = dataset['X']
        self.y = dataset['y']

    def filter(self):
        threshold = self.config["PT THRESHOLD"]
        min_particles = self.config["MIN PARTICLES"]
        max_particles = self.config["MAX PARTICLES"]

        indices = np.logical_and(np.sum(self.X[:, :, 0] > threshold, axis=1) >= min_particles,
                                 np.sum(self.X[:, :, 0] > threshold, axis=1) <= max_particles)

        self.X = self.X[indices, :, :]
        self.y = self.y[indices, :, :]

    def select_highest_pt(self):

        self.n_particles = self.config["NODES"]
        X_sorted = np.zeros(
            (self.X.shape[0], self.n_particles, self.X.shape[2]))
        for i in range(self.X.shape[0]):
            sort_nodes = np.argsort(self.X[i, :, 0])[::-1]
            X_sorted[i, :, :] = self.X[i, sort_nodes[:self.n_particles], :]

        self.X = X_sorted

    def feature_engineer_dataset(self):

        # unpack all features
        pt = self.X[:, :, 0]
        rapidity = self.X[:, :, 1]
        phi = self.X[:, :, 2]
        pdgids = self.X[:, :, 3]

        # Calculate new feature values
        nonzero_pdgids = np.where(pdgids > 0)
        mass = np.zeros(pdgids.shape)
        mass_nonzeros = np.vectorize(
            Particle.from_pdgid)(pdgids[nonzero_pdgids])
        mass[nonzero_pdgids] = np.fromiter(
            (i.mass for i in mass_nonzeros), float)

        mt = np.sqrt(np.square(mass) + np.square(pt))
        energy = mass*np.cosh(rapidity)
        px = pt*np.cos(phi)
        py = pt*np.sin(phi)
        pz = mt*np.sinh(rapidity)

        self.X = np.stack((pt, rapidity, phi, mt, energy, px, py, pz), axis=2)

    def create_graph(self, x):
        M = x.shape[0]  # number of nodes in graph (particles)
        # number of neighbors of each node to be considered for edge connections (1 extra for self loops)
        N = self.connections
        # (√N is considered optimal number of neighbors for k-NN on N points)
        edge_index = torch.zeros((2, M*N))
        edge_attr = torch.zeros((M*N, 1))

        for i, features in enumerate(x):
            # find N nearest neighbors in (𝜙,𝑦) space
            distances = torch.sqrt(
                (features[2] - x[:, 2])**2 + (features[1] - x[:, 1])**2)
            N_nearest = np.argsort(distances)[:N]

            edge_index[0][i*N:(i+1)*N] = torch.tensor([i for _ in range(N)])
            edge_index[1][i*N:(i+1)*N] = N_nearest
            edge_attr[i*N:(i+1)*N] = distances[N_nearest].reshape(N, 1)

        edge_index = edge_index.to(torch.int)
        return edge_index, edge_attr

    def create_graph_dataset(self):
        dataset = []
        self.connections = self.config["CONNECTIONS"]

        for Xi, yi in zip(self.X, self.y):

            edge_index, edge_attr = self.create_graph(Xi)
            data = Data(x=torch.tensor(Xi, dtype=torch.float64), edge_index=edge_index, edge_attr=edge_attr,
                        y=torch.tensor(yi, dtype=torch.long))
            dataset.append(data)

        self.graph_dataset = dataset

    def generate_loaders_graph(self):
        train_ratio = self.config["TRAIN"]
        val_ratio = self.config["VAL"]
        test_ratio = self.config["TEST"]
        batch_size = self.config["BATCH SIZE"]

        data_train, data_val, data_test = torch.utils.data.random_split(
            self.graph_dataset, [train_ratio, val_ratio, test_ratio])

        train_dataloader = DataLoader(data_train,
                                      batch_size=batch_size,
                                      shuffle=True)
        val_dataloader = DataLoader(data_val,
                                    batch_size=batch_size,
                                    shuffle=True)
        test_dataloader = DataLoader(data_test,
                                     batch_size=batch_size,
                                     shuffle=True)

        return train_dataloader, val_dataloader, test_dataloader
