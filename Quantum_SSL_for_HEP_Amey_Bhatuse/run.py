import torch
import torchdata
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
from torchquantum.layer.entanglement.op2_layer import Op2QAllLayer
from torchquantum.layer.layers.layers import Op1QAllLayer, Op2QAllLayer
from torchquantum.measurement import measure

import numpy as np
import cupy as cp
import os
from tqdm import tqdm
import tempfile

import scipy
import warnings
import dgl
from dgl.data import DGLDataset
from dgl.dataloading import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler


from sklearn import metrics
from sklearn.preprocessing import normalize

import scipy.sparse as sp
import csv
import time
import pandas as pd
from collections import OrderedDict
from functools import partial
import pickle
import multiprocessing
import joblib

import torch_geometric
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import add_self_loops, degree, softmax
import torch.optim as optim

from copy import deepcopy
import gc

from particle import Particle
import pennylane as qml
import matplotlib.pyplot as plt

import torch.multiprocessing as mp
import torch.distributed as dist
import torch.multiprocessing as mp
import os

### Reference - https://github.com/ML4SCI/QMLHEP/blob/main/Quantum_GNN_for_HEP_Roy_Forestano/utils/preprocess.py

def preprocess_fixed_nodes(x_data,y_data,nodes_per_graph=10): #,masses):
    print('--- Finding All Unique Particles ---')
    unique_particles = np.unique(x_data[:,:,3])
    x_data = torch.tensor(x_data)
    y_data = torch.tensor(y_data)
    print()
    print('--- Inserting Masses ---')
    masses = torch.zeros((x_data.shape[0],x_data.shape[1]))
    for i,particle in tqdm(enumerate(unique_particles)):
        if particle!=0:
            mass = Particle.from_pdgid(particle).mass/1000
            inds = torch.where(particle==x_data[:,:,3])
            masses[inds]=mass # GeV
    print()
    print('--- Calculating Momenta and Energies ---')
    #theta = torch.arctan(torch.exp(-X[:,:,1]))*2 # polar angle
    pt        = x_data[:,:,0]     # transverse momentum
    rapidity  = x_data[:,:,1]     # rapidity
    phi       = x_data[:,:,2]     # azimuthal angle

    mt        = (pt**2+masses**2).sqrt() # Transverse mass
    energy    = mt*torch.cosh(rapidity) # Energy per multiplicity bin
    e_per_jet = energy.sum(axis=1)  # total energy per jet summed across multiplicity bins

    px = pt*torch.cos(phi)  # momentum in x
    py = pt*torch.sin(phi)  # momentum in y
    pz = mt*torch.sinh(rapidity)  # momentum in z

    # three momentum
    p  = torch.cat(( px[:,:,None],
                     py[:,:,None],
                     pz[:,:,None]), dim=2 )

    p_per_jet        = (p).sum(axis=1)  # total componet momentum per jet
    pt_per_Mbin      = (p_per_jet[:,:2]**2).sum(axis=1).sqrt()  # transverse momentum per jet
    mass_per_jet     = (e_per_jet**2-(p_per_jet**2).sum(axis=1)).sqrt() # mass per jet
    rapidity_per_jet = torch.log( (e_per_jet+p_per_jet[:,2])/(e_per_jet-p_per_jet[:,2]) )/2  # rapidity per jet from analytical formula
    end_multiplicity_indx_per_jet = (pt!=0).sum(axis=1).int() # see where the jet (graph) ends

    x_data = torch.cat( ( x_data[:,:,:3],
                          x_data[:,:,4:],
                          masses[:,:,None],
                          energy[:,:,None],
                          p), dim=2)

    x_data_max = (x_data.max(dim=1).values).max(dim=0).values
    x_data = x_data/x_data_max

    print()
    print('--- Calculating Edge Tensors ---')
    N = x_data[:,0,3].shape[0]  # number of jets (graphs)
    M = nodes_per_graph #x_data[0,:,3].shape[0]  # number of max multiplicty
    connections = nodes_per_graph
    edge_tensor = torch.zeros((N,M,M))
    edge_indx_tensor = torch.zeros((N,2,connections*(connections-1) )) # M*(connections-1) is the max number of edges we allow per jet
    edge_attr_matrix = torch.zeros((N,connections*(connections-1),1))
#     fixed_edges_list = torch.tensor([ [i,j] for i in range(connections) for j in range(connections) if i!=j]).reshape(2,90)

    for jet in tqdm(range(N)):
        stop_indx = end_multiplicity_indx_per_jet[jet] #connections # stop finding edges once we hit zeros -> when we hit 10
        if end_multiplicity_indx_per_jet[jet]>=connections:
            for m in range(connections):
#                 inds_edge = np.argsort((energy[jet,m]+energy[jet,:stop_indx])**2-torch.sum((p[jet,m,:stop_indx]+p[jet,:stop_indx,:])**2,axis=1))[:connections]
#                 edge_tensor[jet,m,:] = (energy[jet,m]+energy[jet,:connections])**2-torch.sum((p[jet,m,:]+p[jet,:connections,:])**2,axis=1)
#                 edge_tensor[jet,m,m] = 0.
#                 edge_tensor[jet,m,m]=((energy[jet,m]+energy[jet,m])**2-torch.sum((p[jet,m,:]+p[jet,m,:])**2,axis=0))
                # inds_edge = torch.sqrt( (phi[jet,m]-phi[jet,:])**2 + (rapidity[jet,m]-rapidity[jet,:])**2 ).argsort()[:connections]
                # edge_tensor[jet,m,:] = torch.sqrt( (phi[jet,m]-phi[jet,inds_edge])**2 + (rapidity[jet,m]-rapidity[jet,inds_edge])**2 )
                edge_tensor[jet,m,:] = torch.sqrt( (phi[jet,m]-phi[jet,:connections])**2 + (rapidity[jet,m]-rapidity[jet,:connections])**2 )
#                 inds_edge = np.argsort( (energy[jet,m]+energy[jet,:stop_indx])**2-torch.sum((p[jet,m,:stop_indx]+p[jet,:stop_indx,:])**2,axis=1) )[:connections]
#                 edge_tensor[jet,m,inds_edge] = (energy[jet,m]+energy[jet,inds_edge])**2-torch.sum((p[jet,m,:]+p[jet,inds_edge,:])**2,axis=1)
            edges_exist_at = torch.where(edge_tensor[jet,:,:].abs()>0)

#             edge_indx_tensor[jet,:,:(edge_tensor[jet,:,:].abs()>0).sum()] = fixed_edges_list
            edge_indx_tensor[jet,:,:(edge_tensor[jet,:,:].abs()>0).sum()] = torch.cat((edges_exist_at[0][None,:],edges_exist_at[1][None,:]),dim=0).reshape((2,edges_exist_at[0].shape[0]))
            edge_attr_matrix[jet,:(edge_tensor[jet,:,:].abs()>0).sum(),0]  =  edge_tensor[jet,edges_exist_at[0],edges_exist_at[1]].flatten()

    end_edges_indx_per_jet = (edge_attr_matrix!=0).sum(axis=1).int()
    keep_inds =  torch.where(end_edges_indx_per_jet>=connections)[0]

    edge_tensor = edge_tensor/edge_tensor.max()
    edge_attr_matrix = edge_attr_matrix/edge_attr_matrix.max()

    graph_help = torch.cat( ( (energy.max(axis=1).values/e_per_jet).reshape(x_data[:,0,3].shape[0],1),
                              (mass_per_jet).reshape(x_data[:,0,3].shape[0],1),
                              (end_multiplicity_indx_per_jet).reshape(x_data[:,0,3].shape[0],1).int(),
                              (end_edges_indx_per_jet).reshape(x_data[:,0,3].shape[0],1).int() ), dim=1)

    return x_data[keep_inds,:nodes_per_graph], y_data[keep_inds].long(), edge_tensor[keep_inds], edge_indx_tensor[keep_inds].long(), edge_attr_matrix[keep_inds], graph_help[keep_inds], masses

### Reference - https://github.com/bmdillon/JetCLR/blob/main/scripts/modules/jet_augs.py

def distort_jets( batch, strength=0.1, pT_clip_min=0.1 ):
    '''
    Input: batch of jets, shape (batchsize, 3, n_constit)
    dim 1 ordering: (pT, eta, phi)
    Output: batch of jets with each constituents position shifted independently, shifts drawn from normal with mean 0, std strength/pT, same shape as input
    '''
    pT = batch[:,0]   # (batchsize, n_constit)
    shift_eta = np.nan_to_num( strength * np.random.randn(batch.shape[0], batch.shape[2]) / pT.clip(min=pT_clip_min), posinf = 0.0, neginf = 0.0 )# * mask
    shift_phi = np.nan_to_num( strength * np.random.randn(batch.shape[0], batch.shape[2]) / pT.clip(min=pT_clip_min), posinf = 0.0, neginf = 0.0 )# * mask
    shift = np.stack( [ np.zeros( (batch.shape[0], batch.shape[2]) ), shift_eta, shift_phi ], 1)
    return batch + shift

def collinear_fill_jets( batch ):
    '''
    Input: batch of jets, shape (batchsize, 3, n_constit)
    dim 1 ordering: (pT, eta, phi)
    Output: batch of jets with collinear splittings, the function attempts to fill as many of the zero-padded args.nconstit
    entries with collinear splittings of the constituents by splitting each constituent at most once, same shape as input
    '''
    batchb = batch.copy()
    nc = batch.shape[2]
    nzs = np.array( [ np.where( batch[:,0,:][i]>0.0)[0].shape[0] for i in range(len(batch)) ] )

    for k in range(len(batch)):
        nzs1 = np.max( [ nzs[k], int(nc/2) ] )
        zs1 = int(nc-nzs1)
        els = np.random.choice( np.linspace(0,nzs1-1,nzs1), size=zs1, replace=False )
        rs = np.random.uniform( size=zs1 )
        for j in range(zs1):
            batchb[k,0,int(els[j])] = rs[j]*batch[k,0,int(els[j])]
            batchb[k,0,int(nzs[k]+j)] = (1-rs[j])*batch[k,0,int(els[j])]
            batchb[k,1,int(nzs[k]+j)] = batch[k,1,int(els[j])]
            batchb[k,2,int(nzs[k]+j)] = batch[k,2,int(els[j])]

    return batchb

class QuarkGluonGraphDataset(dgl.data.dgl_dataset.DGLDataset):

  def __init__(self, dataset_name, raw_dir, save_dir, data_folder_name, datafile_name, labelsfile_name, datatype='particles', dataset_size=12500,
               nodes_per_graph = 5, spectral_augmentation=False, irc_safety_aug=False, url=None, hash_key=..., force_reload=False, verbose=False, transform=None,
              device='cpu'):
    self.data_folder = data_folder_name
    self.datafile_name = datafile_name
    self.labelsfile_name = labelsfile_name
    self.datatype = datatype
    self.nodes_per_graph = nodes_per_graph
    self.spectral_augmentation = spectral_augmentation
    self.drop_ra_nodes = False
    self.drop_cp_nodes = False
    self.aug_ratio = None
    self.irc_safety_aug = irc_safety_aug
    self.device = device
    self.dataset_size = dataset_size
    self.augment = False
    self.nodes_per_aug_graph = None
    super().__init__(dataset_name, url, raw_dir, save_dir, hash_key, force_reload, verbose, transform)

  @property
  def data_folder_name(self):
    return self.data_folder

  @property
  def raw_path(self):
    return os.path.join(self.raw_dir, self.data_folder_name)

  @property
  def save_path(self):
    return os.path.join(self.save_dir, self.data_folder_name)

  @property
  def graph_path(self):
    return os.path.join(self.save_path, 'graphs_and_labels')

  @property
  def info_path(self):
    return os.path.join(self.save_path, 'graphs_and_labels')

  def load(self):
    graphs, label_dict = dgl.load_graphs(str(self.graph_path))
    info_dict = dgl.data.utils.load_info(str(self.info_path))

    self.graph_lists = graphs
    self.graph_labels = label_dict["labels"]
    self.max_num_node = info_dict["max_num_node"]
    self.num_labels = info_dict["num_labels"]

  # def save(self,):
  #   label_dict = {"labels": self.graph_labels}
  #   info_dict = {
  #           "max_num_node": self.max_num_node,
  #           "num_labels": self.num_labels,
  #       }
  #   dgl.save_graphs(str(self.graph_path), self.graph_lists, label_dict)
  #   dgl.data.utils.save_info(str(self.info_path), info_dict)

  def process(self,):
    data = np.load(os.path.join(self.raw_path, self.datafile_name))
    X = data['X']
    y = data['y']
    X_l, y_l = [], []
    i = 0

    while len(X_l)!=self.dataset_size:
        if np.unique(X[i].sum(axis=1).nonzero()).shape[0] >= self.nodes_per_graph:
            sorted_inds = np.argsort(X[i,:,0])[::-1]
            x = X[i][sorted_inds]
            X_l.append(x[:self.nodes_per_graph, :])
            y_l.append(y[i])
        i += 1
    X = np.array(X_l)
    y = np.array(y_l)


    if self.datatype == 'particles':
      self.graph_lists = []
      self.rationale_augmented_graph_lists_1 = []
      self.rationale_augmented_graph_lists_2 = []
      self.complement_augmented_graph_lists = []
      x_data_proc, y_data_proc, edge_tensor, edge_indx_tensor, edge_attr_matrix, graph_help, masses = preprocess_fixed_nodes(X,y,nodes_per_graph = self.nodes_per_graph) #,masses[:N])
      self.max_num_node = x_data_proc.shape[1]
      self.graph_labels = y_data_proc
      self.num_labels = y_data_proc.shape[0]

      print('--- Creating graphs ---')
      for i in tqdm(range(x_data_proc.shape[0])):
        g = dgl.graph((edge_indx_tensor[i][0], edge_indx_tensor[i][1]))
        g.ndata['node_attr'] = x_data_proc[i]
        g.ndata['node_indices'] = torch.arange(x_data_proc[i].shape[0]).reshape(-1,1)
        g.ndata['node_mass'] = masses[i][:self.nodes_per_graph]
        g.edata['edge_attr'] = edge_attr_matrix[i].view(-1,)
        g.to(self.device)
        self.graph_lists.append(g)
        self.rationale_augmented_graph_lists_1.append(g)
        self.rationale_augmented_graph_lists_2.append(g)
        self.complement_augmented_graph_lists.append(g)

      if self.spectral_augmentation:
        self.spectral_graph_lists = []
        print('--- Creating spectral graphs ---')
        for i in tqdm(range(x_data_proc.shape[0])):
          g = SpectralGraph((edge_indx_tensor[i][0], edge_indx_tensor[i][1]), theta=0.1, delta_origin=0.05, edge_weights_matrix=edge_tensor[i])
          g.ndata['node_attr'] = x_data_proc[i]
          g.edata['edge_attr'] = edge_attr_matrix[i].view(-1,)
          self.spectral_graph_lists.append(g)
        # print(self.graph_lists)

      if self.irc_safety_aug:
        for idx in range(len(self.graph_lists)):
          g = self.graph_lists[idx]
          g.ndata['node_attr_irc'] = g.ndata['node_attr'].clone()
          if self.device=='cuda':
            g.ndata['node_attr_irc'][:,:3] = torch.Tensor(distort_jets(collinear_fill_jets(g.ndata['node_attr'][:,:3].T.unsqueeze(0).cpu().numpy()))).squeeze(0).T.cuda()
          else:
            g.ndata['node_attr_irc'][:,:3] = torch.Tensor(distort_jets(collinear_fill_jets(g.ndata['node_attr'][:,:3].T.unsqueeze(0).numpy()))).squeeze(0).T
          pt, rapidity, phi = g.ndata['node_attr_irc'][:, 0], g.ndata['node_attr_irc'][:, 1], g.ndata['node_attr_irc'][:, 2]
          mt = (pt**2+g.ndata['node_mass']**2).sqrt()
          energy = mt*torch.cosh(rapidity)
          px, py, pz = pt*torch.cos(phi), pt*torch.sin(phi), mt*torch.sinh(rapidity)
          g.ndata['node_attr_irc'][:,3] =  mt
          g.ndata['node_attr_irc'][:,4] = energy
          g.ndata['node_attr_irc'][:,5] = px
          g.ndata['node_attr_irc'][:,6] = py
          g.ndata['node_attr_irc'][:,7] = pz

  def has_cache(self):
    if os.path.exists(self.graph_path) and os.path.exists(self.info_path):
      return True
    return False

  def __len__(self,):
    return len(self.graph_lists)

  def augment_dataset(self, type, batched_graph, batch_size):
    self.augment = True

    if type == 'rationale':
      return drop_nodes_prob_batch(batched_graph, batch_size), drop_nodes_prob_batch(batched_graph, batch_size)

    if type == 'complement':
      return drop_nodes_cp_batch(batched_graph, batch_size)

  def __getitem__(self, idx):
    if self.spectral_augmentation:
      g1 = self.graph_lists[idx]
      g2 = self.spectral_graph_lists[idx]
      if self._transform is not None:
        g1 = self._transform(g1)
        g2 = self._transform(g2)
      return g1, g2, self.graph_labels[idx]

    else:
      g = self.graph_lists[idx]
      if self._transform is not None:
        g = self._transform(g)
      return g, self.graph_labels[idx]

  @property
  def num_classes(self):
    return int(self.num_labels)

class GNN_imp_estimator(torch.nn.Module):
    """

    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        JK (str): last, concat, max or sum.
        max_pool_layer (int): the layer from which we use max pool rather than add pool for neighbor aggregation
        drop_ratio (float): dropout rate
        gnn_type: gin, gcn, graphsage, gat

    Output:
        node representations

    """

    def __init__(self, num_layer, emb_dim, in_dim, JK="last", drop_ratio=0):
        super(GNN_imp_estimator, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")


        ###List of MLPs
        self.gnns = torch.nn.ModuleList()
        # self.gnns.append(torch_geometric.nn.conv.GCNConv(in_dim, 32))
        # self.gnns.append(torch_geometric.nn.conv.GCNConv(32, 16))
        # self.gnns.append(torch_geometric.nn.conv.GCNConv(16, 8))
        
        self.gnns.append(dgl.nn.GraphConv(in_dim, 32, weight=True, bias=True))
        self.gnns.append(dgl.nn.GraphConv(32, 16, weight=True, bias=True))
        self.gnns.append(dgl.nn.GraphConv(16, 8, weight=True, bias=True))

        ###List of batchnorms
        self.batch_norms = torch.nn.ModuleList()
        self.batch_norms.append(torch.nn.BatchNorm1d(32))
        self.batch_norms.append(torch.nn.BatchNorm1d(16))
        self.batch_norms.append(torch.nn.BatchNorm1d(8))

        self.linear = torch.nn.Linear(8, 1)

    def forward(self, *argv):
        if len(argv) == 4:
            x, edge_index, edge_attr, batch = argv[0], argv[1], argv[2], argv[3]
        elif len(argv) == 2:
            graph, batch = argv[0], argv[1]
            x, edge_index, edge_attr = graph.ndata['node_attr'], graph.edges(), graph.edata['edge_attr']
        else:
            raise ValueError("unmatched number of arguments.")

        # x = self.x_embedding1(x[:, 0]) + self.x_embedding2(x[:, 1])

        # h_list = [x]
        # for layer in range(len(self.gnns)):
        #     print('Layer : ',layer)
        #     h = self.gnns[layer](graph, h_list[layer].float(), edge_weight=edge_attr)   #
        #     h = self.batch_norms[layer](h) 
        #     if layer == len(self.gnns) - 1:
        #         # remove relu for the last layer
        #         h = F.dropout(h, self.drop_ratio, training=self.training)
        #     else:
        #         h = F.dropout(nn.ReLU()(h), self.drop_ratio, training=self.training)
        #     h_list.append(h)
        
        h0 = self.gnns[0](graph, x.float(), edge_weight=edge_attr)
        h0 = self.batch_norms[0](h0)
        h0 = F.dropout(nn.ReLU()(h0), self.drop_ratio, training=self.training)
        h1 = self.gnns[1](graph, h0, edge_weight=edge_attr)
        h1 = self.batch_norms[1](h1)
        h1 = F.dropout(nn.ReLU()(h1), self.drop_ratio, training=self.training)
        h2 = self.gnns[2](graph, h1, edge_weight=edge_attr)
        h2 = self.batch_norms[2](h2) 
        h2 = F.dropout(h2, self.drop_ratio, training=self.training)
        
        node_representation = h2  #h_list[-1]
        node_representation = self.linear(node_representation)
        node_representation = softmax(node_representation, batch)
        
        return node_representation


class GNN(torch.nn.Module):
    """


    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        JK (str): last, concat, max or sum.
        max_pool_layer (int): the layer from which we use max pool rather than add pool for neighbor aggregation
        drop_ratio (float): dropout rate
        gnn_type: gin, gcn, graphsage, gat

    Output:
        node representations

    """
    def __init__(self, num_layer, in_dim, emb_dim, inter_dim, JK = "last", drop_ratio=0, gnn_type = "gin"):
        super(GNN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        ###List of MLPs
        self.gnns = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        # self.gnns.append(torch_geometric.nn.conv.GCNConv(in_dim, emb_dim))
        if gnn_type == "gin":
            self.gnns.append(GINConv(in_dim, aggr="add"))
            self.batch_norms.append(torch.nn.BatchNorm1d(inter_dim))
        elif gnn_type == "gcn":
            self.gnns.append(torch_geometric.nn.conv.GCNConv(in_dim, inter_dim))
            self.batch_norms.append(torch.nn.BatchNorm1d(inter_dim))
        elif gnn_type == "gat":
            self.gnns.append(torch_geometric.nn.conv.GATConv(in_dim, inter_dim, heads=3, concat=False))
            self.batch_norms.append(torch.nn.BatchNorm1d(inter_dim))
        elif gnn_type == "graphsage":
            self.gnns.append(GraphSAGEConv(in_dim))
            self.batch_norms.append(torch.nn.BatchNorm1d(inter_dim))
            
        for layer in range(num_layer):
            if gnn_type == "gin":
                self.gnns.append(GINConv(inter_dim, aggr="add"))
                self.batch_norms.append(torch.nn.BatchNorm1d(inter_dim))
            elif gnn_type == "gcn":
                self.gnns.append(torch_geometric.nn.conv.GCNConv(inter_dim, inter_dim))
                self.batch_norms.append(torch.nn.BatchNorm1d(inter_dim))
            elif gnn_type == "gat":
                self.gnns.append(torch_geometric.nn.conv.GATConv(inter_dim, inter_dim, heads=3, concat=False))
                self.batch_norms.append(torch.nn.BatchNorm1d(inter_dim))
            elif gnn_type == "graphsage":
                self.gnns.append(GraphSAGEConv(inter_dim))
                self.batch_norms.append(torch.nn.BatchNorm1d(inter_dim))
                
        if gnn_type == "gin":
            self.gnns.append(GINConv(emb_dim, aggr="add"))
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))
        elif gnn_type == "gcn":
            self.gnns.append(torch_geometric.nn.conv.GCNConv(inter_dim, emb_dim))
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))
        elif gnn_type == "gat":
            self.gnns.append(torch_geometric.nn.conv.GATConv(inter_dim, emb_dim, heads=3, concat=False))
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))
        elif gnn_type == "graphsage":
            self.gnns.append(GraphSAGEConv(emb_dim))
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))
        

    #def forward(self, x, edge_index, edge_attr):
    def forward(self, *argv):
        if len(argv) == 3:
            x, edge_index, edge_attr = argv[0], argv[1], argv[2]
        elif len(argv) == 1:
            # data = argv[0]
            # x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
            graph = argv[0]
            x, edge_index, edge_attr = graph.ndata['node_attr'], graph.edges(), graph.edata['edge_attr']
        else:
            raise ValueError("unmatched number of arguments.")

        # print(x)
        # h_list = [x]
        h = x
        for layer in range(self.num_layer+2):
            h = self.gnns[layer](h, edge_index, edge_attr)   #h_list[layer]
            h = self.batch_norms[layer](h)
            #h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            if layer == self.num_layer+1:
                #remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            # h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim = 1)
        elif self.JK == "last":
            node_representation = h   #h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim = 0), dim = 0)[0]
        elif self.JK == "sum":
            h_list = [h.unsqueeze(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim = 0), dim = 0)[0]

        return node_representation

    def forward_gradc(self, *argv):
        if len(argv) == 3:
            x, edge_index, edge_attr = argv[0], argv[1], argv[2]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        else:
            raise ValueError("unmatched number of arguments.")

        h_list = [x]
        for layer in range(self.num_layer+2):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            if layer == self.num_layer - 1:
                #remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim = 1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim = 0), dim = 0)[0]
        elif self.JK == "sum":
            h_list = [h.unsqueeze(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim = 0), dim = 0)[0]

        return node_representation
    
class BetterBetterTorchLayer(torch.nn.Module):
  def __init__(self, nodes_per_graph, num_layers, input_dim, device):
    super(BetterBetterTorchLayer, self).__init__()
    self.device =  device
    self.num_qubits = nodes_per_graph
    inputs = []
    self.node_attr_count = 0
    for q in range(self.num_qubits):
      for d in range(input_dim):
        inputs.append({'input_idx':self.node_attr_count, 'func':'ry', 'wires':[q]})
        self.node_attr_count += 1
    self.edge_attr_count = self.node_attr_count + 1
    for q in range(self.num_qubits):
      for e in range(q+1, self.num_qubits):
        inputs.append({'input_idx':self.edge_attr_count, 'func':'crz', 'wires':[q,e]})
        self.edge_attr_count += 1
    self.edge_attr_count -= self.node_attr_count
    self.encoder = tq.GeneralEncoder(inputs)
    self.q_layers = tq.QuantumModuleList()
    for layer in range(num_layers):
      self.q_layers.append(
                Op1QAllLayer(
                    op=tq.RX, n_wires=self.num_qubits, has_params=True, trainable=True
                )
            )
      self.q_layers.append(
                Op1QAllLayer(
                    op=tq.RY, n_wires=self.num_qubits, has_params=True, trainable=True
                )
            )
      self.q_layers.append(
                Op1QAllLayer(
                    op=tq.RZ, n_wires=self.num_qubits, has_params=True, trainable=True
                )
            )
      self.q_layers.append(
                Op2QAllLayer(op=tq.CNOT, n_wires=self.num_qubits, jump = (layer+1)%(self.num_qubits-1), circular=True)
            )
    # self.measure = tq.MeasureAll(tq.PauliZ)


  def forward(self, node_inputs, node_indices, edge_inputs):
    qdev = tq.QuantumDevice(n_wires=self.num_qubits, bsz=node_inputs.shape[0], device=self.device, record_op=True)
    self.encoder(qdev, torch.cat((node_inputs.reshape(node_inputs.shape[0],-1), edge_inputs.reshape(edge_inputs.shape[0],-1)), dim=1))
    for l in range(len(self.q_layers)):
      self.q_layers[l](qdev)
    return torch.abs(qdev.get_states_1d())**2

class QGNN_node_estimator(torch.nn.Module):
  def __init__(self, nodes_per_graph, num_layers, input_dim=None, device='cpu'):
    super(QGNN_node_estimator, self).__init__()
    self.device = device
    self.nodes_per_graph = nodes_per_graph
    self.num_layers = num_layers
    self.input_dim = input_dim
    # self.quantum_nn = BetterTorchLayer(self.nodes_per_graph, num_layers)
    self.quantum_nn = BetterBetterTorchLayer(self.nodes_per_graph, self.num_layers, self.input_dim, self.device)

  def edge_attr_relevant(self, edge_attr, edge_attr_r):
    count=0
    for n in range(edge_attr.shape[0]):
        for e in range(n+1, edge_attr.shape[1]):
          edge_attr_r[count] = edge_attr[n, e]
          count += 1
    return edge_attr_r

  def forward(self, x, node_indices, edge_attr=None):
    if edge_attr is not None:
      edge_attr_r = torch.zeros((edge_attr.shape[0], int(edge_attr.shape[1]*edge_attr.shape[2]/2 + 1)), device=self.device)
      edge_attr = torch.vmap(self.edge_attr_relevant)(edge_attr, edge_attr_r)
    output = self.quantum_nn(x, node_indices, edge_attr)
    output = output[:, [2**i for i in range(self.nodes_per_graph)]]
    return torch.vmap(lambda out, ind: out[ind])(output, node_indices)


### Reference - https://github.com/colizz/weaver-benchmark/blob/main/top_tagging/networks/particlenet_pf.py

'''Based on https://github.com/WangYueFt/dgcnn/blob/master/pytorch/model.py.'''


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k + 1, dim=-1)[1][:, :, 1:]  # (batch_size, num_points, k)
    return idx


# v1 is faster on GPU
def get_graph_feature_v1(x, k, idx):
    batch_size, num_dims, num_points = x.size()

    idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    fts = x.transpose(2, 1).reshape(-1, num_dims)  # -> (batch_size, num_points, num_dims) -> (batch_size*num_points, num_dims)
    fts = fts[idx, :].view(batch_size, num_points, k, num_dims)  # neighbors: -> (batch_size*num_points*k, num_dims) -> ...
    fts = fts.permute(0, 3, 1, 2).contiguous()  # (batch_size, num_dims, num_points, k)
    x = x.view(batch_size, num_dims, num_points, 1).repeat(1, 1, 1, k)
    fts = torch.cat((x, fts - x), dim=1)  # ->(batch_size, 2*num_dims, num_points, k)
    return fts


# v2 is faster on CPU
def get_graph_feature_v2(x, k, idx):
    batch_size, num_dims, num_points = x.size()

    idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    fts = x.transpose(0, 1).reshape(num_dims, -1)  # -> (num_dims, batch_size, num_points) -> (num_dims, batch_size*num_points)
    fts = fts[:, idx].view(num_dims, batch_size, num_points, k)  # neighbors: -> (num_dims, batch_size*num_points*k) -> ...
    fts = fts.transpose(1, 0).contiguous()  # (batch_size, num_dims, num_points, k)

    x = x.view(batch_size, num_dims, num_points, 1).repeat(1, 1, 1, k)
    fts = torch.cat((x, fts - x), dim=1)  # ->(batch_size, 2*num_dims, num_points, k)

    return fts


class EdgeConvBlock(nn.Module):
    r"""EdgeConv layer.
    Introduced in "`Dynamic Graph CNN for Learning on Point Clouds
    <https://arxiv.org/pdf/1801.07829>`__".  Can be described as follows:
    .. math::
       x_i^{(l+1)} = \max_{j \in \mathcal{N}(i)} \mathrm{ReLU}(
       \Theta \cdot (x_j^{(l)} - x_i^{(l)}) + \Phi \cdot x_i^{(l)})
    where :math:`\mathcal{N}(i)` is the neighbor of :math:`i`.
    Parameters
    ----------
    in_feat : int
        Input feature size.
    out_feat : int
        Output feature size.
    batch_norm : bool
        Whether to include batch normalization on messages.
    """

    def __init__(self, k, in_feat, out_feats, batch_norm=True, activation=True, cpu_mode=False):
        super(EdgeConvBlock, self).__init__()
        self.k = k
        self.batch_norm = batch_norm
        self.activation = activation
        self.num_layers = len(out_feats)
        self.get_graph_feature = get_graph_feature_v2 if cpu_mode else get_graph_feature_v1

        self.convs = nn.ModuleList()
        for i in range(self.num_layers):
            self.convs.append(nn.Conv2d(2 * in_feat if i == 0 else out_feats[i - 1], out_feats[i], kernel_size=1, bias=False if self.batch_norm else True))

        if batch_norm:
            self.bns = nn.ModuleList()
            for i in range(self.num_layers):
                self.bns.append(nn.BatchNorm2d(out_feats[i]))

        if activation:
            self.acts = nn.ModuleList()
            for i in range(self.num_layers):
                self.acts.append(nn.ReLU())

        if in_feat == out_feats[-1]:
            self.sc = None
        else:
            self.sc = nn.Conv1d(in_feat, out_feats[-1], kernel_size=1, bias=False)
            self.sc_bn = nn.BatchNorm1d(out_feats[-1])

        if activation:
            self.sc_act = nn.ReLU()

    def forward(self, points, features):

        topk_indices = knn(points, self.k)
        x = self.get_graph_feature(features, self.k, topk_indices)

        for conv, bn, act in zip(self.convs, self.bns, self.acts):
            x = conv(x)  # (N, C', P, K)
            if bn:
                x = bn(x)
            if act:
                x = act(x)

        fts = x.mean(dim=-1)  # (N, C, P)

        # shortcut
        if self.sc:
            sc = self.sc(features)  # (N, C_out, P)
            sc = self.sc_bn(sc)
        else:
            sc = features

        return self.sc_act(sc + fts)  # (N, C_out, P)


class ParticleNet(nn.Module):

    def __init__(self,
                 input_dims,
                 num_classes,
                 conv_params=[(7, (32, 32, 32)), (7, (64, 64, 64))],
                 fc_params=[(128, 0.1)],
                 use_fusion=True,
                 use_fts_bn=True,
                 use_counts=True,
                 for_inference=False,
                 for_segmentation=True,
                 **kwargs):
        super(ParticleNet, self).__init__(**kwargs)

        self.use_fts_bn = use_fts_bn
        if self.use_fts_bn:
            self.bn_fts = nn.BatchNorm1d(input_dims)

        self.use_counts = use_counts

        self.edge_convs = nn.ModuleList()
        for idx, layer_param in enumerate(conv_params):
            k, channels = layer_param
            in_feat = input_dims if idx == 0 else conv_params[idx - 1][1][-1]
            self.edge_convs.append(EdgeConvBlock(k=k, in_feat=in_feat, out_feats=channels, cpu_mode=for_inference))

        self.use_fusion = use_fusion
        if self.use_fusion:
            in_chn = sum(x[-1] for _, x in conv_params)
            out_chn = np.clip((in_chn // 128) * 128, 128, 1024)
            self.fusion_block = nn.Sequential(nn.Conv1d(in_chn, out_chn, kernel_size=1, bias=False), nn.BatchNorm1d(out_chn), nn.ReLU())

        self.for_segmentation = for_segmentation

        fcs = []
        for idx, layer_param in enumerate(fc_params):
            channels, drop_rate = layer_param
            if idx == 0:
                in_chn = out_chn if self.use_fusion else conv_params[-1][1][-1]
            else:
                in_chn = fc_params[idx - 1][0]
            if self.for_segmentation:
                fcs.append(nn.Sequential(nn.Conv1d(in_chn, channels, kernel_size=1, bias=False),
                                         nn.BatchNorm1d(channels), nn.ReLU(), nn.Dropout(drop_rate)))
            else:
                fcs.append(nn.Sequential(nn.Linear(in_chn, channels), nn.ReLU(), nn.Dropout(drop_rate)))
        # if self.for_segmentation:
        #     fcs.append(nn.Conv1d(fc_params[-1][0], num_classes, kernel_size=1))
        # else:
        #     fcs.append(nn.Linear(fc_params[-1][0], num_classes))
        self.fc = nn.Sequential(*fcs)

        self.for_inference = for_inference

    def forward(self, points, features, mask=None):
#         print('points:\n', points)
#         print('features:\n', features)
        if mask is None:
            mask = (features.abs().sum(dim=1, keepdim=True) != 0)  # (N, 1, P)
            # print(mask)
            # print(mask.shape)
        points *= mask
        features *= mask
        coord_shift = (mask == 0) * 1e9
        if self.use_counts:
            counts = mask.float().sum(dim=-1)
            counts = torch.max(counts, torch.ones_like(counts))  # >=1

        if self.use_fts_bn:
            fts = self.bn_fts(features) * mask
        else:
            fts = features
        outputs = []
        for idx, conv in enumerate(self.edge_convs):
            pts = (points if idx == 0 else fts) + coord_shift
            fts = conv(pts, fts) * mask
            if self.use_fusion:
                outputs.append(fts)
        if self.use_fusion:
            fts = self.fusion_block(torch.cat(outputs, dim=1)) * mask

#         assert(((fts.abs().sum(dim=1, keepdim=True) != 0).float() - mask.float()).abs().sum().item() == 0)
        
        if self.for_segmentation:
            x = fts
        else:
            if self.use_counts:
                x = fts.sum(dim=-1) / counts  # divide by the real counts
            else:
                x = fts.mean(dim=-1)

        output =  self.fc(x)
        
        if self.for_inference:
            output = torch.softmax(output, dim=1)
        # print('output:\n', output)
        return output


class FeatureConv(nn.Module):

    def __init__(self, in_chn, out_chn, **kwargs):
        super(FeatureConv, self).__init__(**kwargs)
        self.conv = nn.Sequential(
            nn.BatchNorm1d(in_chn),
            nn.Conv1d(in_chn, out_chn, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_chn),
            nn.ReLU()
            )

    def forward(self, x):
        return self.conv(x)


class ParticleNetTagger1Path(nn.Module):

    def __init__(self,
                 pf_features_dims,
                 num_classes,
                 conv_params= [(6, (32, 32, 32)), (6, (64, 64, 64)), (6, (128, 128, 128))],   #[(7, (32, 32, 32)), (7, (64, 64, 64))],  #    
                 fc_params=[(128, 0.1)],
                 use_fusion=False,
                 use_fts_bn=False,
                 use_counts=True,
                 pf_input_dropout=None,
                 for_inference=False,
                 **kwargs):
        super(ParticleNetTagger1Path, self).__init__(**kwargs)
        self.pf_input_dropout = nn.Dropout(pf_input_dropout) if pf_input_dropout else None
        # self.pf_conv = FeatureConv(pf_features_dims, 32)
        self.gnn = ParticleNet(input_dims=pf_features_dims,
                              num_classes=num_classes,
                              conv_params=conv_params,
                              fc_params=fc_params,
                              use_fusion=use_fusion,
                              use_fts_bn=use_fts_bn,
                              use_counts=use_counts,
                              for_inference=for_inference)

    def forward(self, pf_points, pf_features, pf_mask):
        if self.pf_input_dropout:
            pf_mask = (self.pf_input_dropout(pf_mask) != 0).float()
            pf_points *= pf_mask
            pf_features *= pf_mask
        # mod_feats = self.pf_conv(pf_features)   #* pf_mask , * pf_mask
        return self.gnn(pf_points, pf_features, pf_mask)  

def get_model(data_config, **kwargs):
    # conv_params = [
    #     (16, (64, 64, 64)),
    #     (16, (128, 128, 128)),
    #     (16, (256, 256, 256)),
    #     ]
    ec_k = kwargs.get('ec_k', 16)
    ec_c1 = kwargs.get('ec_c1', 64)
    ec_c2 = kwargs.get('ec_c2', 128)
    ec_c3 = kwargs.get('ec_c3', 256)
    fc_c, fc_p = kwargs.get('fc_c', 256), kwargs.get('fc_p', 0.1)
    conv_params = [
        (ec_k, (ec_c1, ec_c1, ec_c1)),
        (ec_k, (ec_c2, ec_c2, ec_c2)),
        (ec_k, (ec_c3, ec_c3, ec_c3)),
        ]
    fc_params = [(fc_c, fc_p)]
    use_fusion = True

    pf_features_dims = len(data_config.input_dicts['pf_features'])
    num_classes = len(data_config.label_value)
    model = ParticleNetTagger1Path(pf_features_dims, num_classes,
                                   conv_params, fc_params,
                                   use_fusion=use_fusion,
                                   use_fts_bn=kwargs.get('use_fts_bn', False),
                                   use_counts=kwargs.get('use_counts', True),
                                   pf_input_dropout=kwargs.get('pf_input_dropout', None),
                                   for_inference=kwargs.get('for_inference', False)
                                   )
    model_info = {
        'input_names':list(data_config.input_names),
        'input_shapes':{k:((1,) + s[1:]) for k, s in data_config.input_shapes.items()},
        'output_names':['softmax'],
        'dynamic_axes':{**{k:{0:'N', 2:'n_' + k.split('_')[0]} for k in data_config.input_names}, **{'softmax':{0:'N'}}},
        }

    return model, model_info


def drop_nodes_prob_batch(graph, batch_size, aug_ratio, device):

    node_num = graph.num_nodes()
    edge_num = graph.num_edges()
    node_num_sin = int(graph.num_nodes()/batch_size)
    edge_num_sin = int(graph.num_edges()/batch_size)
    drop_num_sin = int(node_num_sin * aug_ratio)
    drop_num = batch_size*drop_num_sin
    node_score = graph.ndata['node_score'].reshape(batch_size, -1)
    node_prob = node_score.float()
    node_prob += 0.001
    node_prob = cp.array(node_prob)
    node_prob /= node_prob.sum(axis=1)[:, None]
    node_prob = node_prob.get()
    
    idx_nondrop = np.zeros((batch_size, node_num_sin-drop_num_sin), dtype=np.int64)
    idx_drop_set = []
    idx_nondrop_e = np.zeros((batch_size, node_num_sin-drop_num_sin), dtype=np.int64)
    for b in range(batch_size):
      idx_nondrop[b] = np.random.choice(node_num_sin, node_num_sin-drop_num_sin, replace=False, p=node_prob[b])
      idx_drop_set.append(set(np.setdiff1d(np.arange(node_num_sin), idx_nondrop[b]).tolist()))
      idx_nondrop[b].sort()
      idx_nondrop_e[b] += idx_nondrop[b] + b*node_num_sin

    idx_nondrop_e = idx_nondrop_e.reshape(-1,)

    idx_drop_set_e = set(np.setdiff1d(np.arange(node_num), idx_nondrop_e).tolist())

    idx_dict = np.zeros((idx_nondrop_e[-1] + 1,), dtype=np.int64)
    idx_dict[idx_nondrop_e] = np.arange(len(idx_nondrop_e), dtype=np.int64)

    # edge_index = data.edge_index.numpy()
    edge_index = np.array([graph.edges()[0].cpu().numpy(), graph.edges()[1].cpu().numpy()])

    edge_mask = []
    for n in range(edge_num):
        if edge_index[0, n] not in idx_drop_set_e and edge_index[1, n] not in idx_drop_set_e:
          edge_mask.append(n)
    edge_mask = np.asarray(edge_mask, dtype=np.int64)
    edge_index = idx_dict[edge_index[:, edge_mask]]

    ng = dgl.graph([])
    src = graph.edges()[0]
    dst = graph.edges()[1]
    nsrc = edge_index[0]
    ndst = edge_index[1]
    ng.add_edges(nsrc, ndst)
    ng.to(device)
    ng.ndata['node_attr'] = graph.ndata['node_attr'].clone().reshape(batch_size,node_num_sin,-1)[torch.arange(batch_size, device=device).unsqueeze(-1), idx_nondrop].reshape(node_num-drop_num,-1)
    ng.ndata['node_attr_irc'] = graph.ndata['node_attr_irc'].clone().reshape(batch_size,node_num_sin,-1)[torch.arange(batch_size, device=device).unsqueeze(-1), idx_nondrop].reshape(node_num-drop_num,-1)
    ng.ndata['node_indices'] = torch.Tensor(idx_nondrop, device=device).int().reshape(-1,1)
    ng.edata['edge_attr'] = graph.edata['edge_attr'].clone()[edge_mask]
    ng.edata['edge_indices'] = torch.Tensor(edge_mask, device=device).int()

    return ng

def drop_nodes_cp_batch(graph, batch_size, aug_ratio, device):

    node_num = graph.num_nodes()
    edge_num = graph.num_edges()
    node_num_sin = int(graph.num_nodes()/batch_size)
    edge_num_sin = int(graph.num_edges()/batch_size)
    drop_num_sin = int(node_num_sin * aug_ratio)
    drop_num = batch_size*drop_num_sin
    node_score = graph.ndata['node_score'].reshape(batch_size, -1)
    node_prob = torch.sub(torch.max(node_score, dim=1).values.reshape(-1,1), node_score)
    node_prob += 0.001
    node_prob = cp.array(node_prob)
    node_prob /= node_prob.sum(axis=1)[:, None]
    node_prob = node_prob.get()
    
    idx_nondrop = np.zeros((batch_size, node_num_sin-drop_num_sin), dtype=np.int64)
    idx_drop_set = []
    idx_nondrop_e = np.zeros((batch_size, node_num_sin-drop_num_sin), dtype=np.int64)
    for b in range(batch_size):
      idx_nondrop[b] = np.random.choice(node_num_sin, node_num_sin-drop_num_sin, replace=False, p=node_prob[b])
      idx_drop_set.append(set(np.setdiff1d(np.arange(node_num_sin), idx_nondrop[b]).tolist()))
      idx_nondrop[b].sort()
      idx_nondrop_e[b] += idx_nondrop[b] + b*node_num_sin

    idx_nondrop_e = idx_nondrop_e.reshape(-1,)

    idx_drop_set_e = set(np.setdiff1d(np.arange(node_num), idx_nondrop_e).tolist())

    idx_dict = np.zeros((idx_nondrop_e[-1] + 1,), dtype=np.int64)
    idx_dict[idx_nondrop_e] = np.arange(len(idx_nondrop_e), dtype=np.int64)

    # edge_index = data.edge_index.numpy()
    edge_index = np.array([graph.edges()[0].cpu().numpy(), graph.edges()[1].cpu().numpy()])

    edge_mask = []
    for n in range(edge_num):
        if edge_index[0, n] not in idx_drop_set_e and edge_index[1, n] not in idx_drop_set_e:
          edge_mask.append(n)
    edge_mask = np.asarray(edge_mask, dtype=np.int64)
    edge_index = idx_dict[edge_index[:, edge_mask]]

    ng = dgl.graph([])
    src = graph.edges()[0]
    dst = graph.edges()[1]
    nsrc = edge_index[0]
    ndst = edge_index[1]
    ng.add_edges(nsrc, ndst)
    ng.to(device)
    ng.ndata['node_attr'] = graph.ndata['node_attr'].clone().reshape(batch_size,node_num_sin,-1)[torch.arange(batch_size,  device=device).unsqueeze(-1), idx_nondrop].reshape(node_num-drop_num,-1)
    ng.ndata['node_attr_irc'] = graph.ndata['node_attr_irc'].clone().reshape(batch_size,node_num_sin,-1)[torch.arange(batch_size, device=device).unsqueeze(-1), idx_nondrop].reshape(node_num-drop_num,-1)
    ng.ndata['node_indices'] = torch.Tensor(idx_nondrop, device=device).int().reshape(-1,1)
    ng.edata['edge_attr'] = graph.edata['edge_attr'].clone()[edge_mask]
    ng.edata['edge_indices'] = torch.Tensor(edge_mask, device=device).int()

    return ng

class graphcl(nn.Module):
    def __init__(self, gnn, node_imp_estimator, emb_dim, out_dim):
        super(graphcl, self).__init__()
        self.gnn = gnn
        self.node_imp_estimator = node_imp_estimator
        self.pool = global_mean_pool
        self.projection_head = nn.Sequential(nn.Linear(emb_dim, out_dim), nn.ReLU(inplace=True), nn.Linear(out_dim, out_dim))

    def prepare_variables_rg(self, gr_n, g_n, n_i):  
      g_n[n_i] = gr_n
      return g_n

    def forward_cl(self, x, edge_index, edge_attr, batch, nodes_per_graph, g_batch=None, q_edge_attr=False, device='cpu', node_est='classical'):
      if node_est == 'classical':
          node_imp = self.node_imp_estimator(x, edge_index, edge_attr, batch)
      if node_est == 'quantum':
          # ############ Quantum Node Est.

          g_n = g_batch.ndata['node_attr'].clone().reshape(3*batch_size, nodes_per_graph, -1)
          n_i = g_batch.ndata['node_indices'].clone().reshape(3*batch_size, nodes_per_graph)
          g_n_0 = torch.zeros((3*batch_size, nodes_per_graph_original, g_n.shape[2]), device=device)
          g_n = torch.vmap(self.prepare_variables_rg)(g_n.float(), g_n_0, n_i.long())

          e_n = g_batch.edata['edge_attr'].clone()
          e_i = g_batch.edata['edge_indices'].clone()
          e_n_0 = torch.zeros((3*batch_size*nodes_per_graph_original*(nodes_per_graph_original-1)), device=device)
          e_n_0[e_i.long()]  = e_n
          e_n = e_n_0.reshape(3*batch_size, nodes_per_graph_original, -1)

          if q_edge_attr:
            node_imp = model.node_imp_estimator(g_n, n_i, e_n)
          else:
            node_imp = model.node_imp_estimator(g_n, n_i)
          node_imp = node_imp.reshape(-1,1)
          
          ############

      out = torch.max(node_imp.reshape(-1, nodes_per_graph), 1)[0]
      out = out.reshape(-1, 1)
      out = out[batch]
      node_imp /= (out * 10)
      node_imp += 0.9
        
      pf_feats = x.reshape(3*batch_size, nodes_per_graph, -1)
      points = pf_feats[:,:,1:3]
    
      x = self.gnn(points.reshape(points.shape[0], points.shape[2], points.shape[1])
                     , pf_feats.reshape(pf_feats.shape[0], pf_feats.shape[2], pf_feats.shape[1]), None)
      x = x.reshape(x.shape[0], x.shape[2], x.shape[1])
      x = x.reshape(x.shape[0]*x.shape[1], x.shape[2])
        
      node_imp = node_imp.expand(-1, x.shape[1])
      x = torch.mul(x, node_imp)
      x = self.pool(x, batch)
      x = self.projection_head(x.float())

      return x

    def loss_cl(self, x1, x2, temp):
        T = temp
        batch_size, _ = x1.size()
        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss).mean()
        return loss

    def loss_infonce(self, x1, x2, temp):
        T = temp
        batch_size, _ = x1.size()
        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / sim_matrix.sum(dim=1)
        loss = - torch.log(loss).mean()
        return loss

    def loss_ra(self, x1, x2, x3, temp, lamda):
        batch_size, _ = x1.size()
        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)
        x3_abs = x3.norm(dim=1)

        cp_sim_matrix = torch.einsum('ik,jk->ij', x1, x3) / torch.einsum('i,j->ij', x1_abs, x3_abs)
        cp_sim_matrix = torch.exp(cp_sim_matrix / temp)

        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
        sim_matrix = torch.exp(sim_matrix / temp)

        pos_sim = sim_matrix[range(batch_size), range(batch_size)]

        ra_loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        ra_loss = - torch.log(ra_loss).mean()

        cp_loss = pos_sim / (cp_sim_matrix.sum(dim=1) + pos_sim)
        cp_loss = - torch.log(cp_loss).mean()

        uni_loss_1 = self.lunif(torch.nn.functional.normalize(x1, dim=1))
        uni_loss_2 = self.lunif(torch.nn.functional.normalize(x2, dim=1))
        uni_loss = (uni_loss_1 + uni_loss_2) / 2
        al_loss = self.lalign(torch.nn.functional.normalize(x1, dim=1), torch.nn.functional.normalize(x2, dim=1))

        loss = ra_loss + lamda * cp_loss
        # loss = 0.5*uni_loss + al_loss + lamda * cp_loss

        return ra_loss, cp_loss, loss, uni_loss, al_loss

    def lalign(self, x, y, alpha=2):
      return (x - y).norm(dim=1).pow(alpha).mean()

    def lunif(self, x, t=2):
      sq_pdist = torch.pdist(x, p=2).pow(2)
      return sq_pdist.mul(-t).exp().mean().log()

def train(epoch, model, device, dataset, optimizer, batch_size, nodes_per_graph, aug_ratio, loss_temp, lamda, irc_safety, q_edge_attr=False, loader=None, node_est='classical'):
    
    torch.autograd.set_detect_anomaly(True)
    dataset.aug = "none"
    imp_batch_size = batch_size
    loader = GraphDataLoader(dataset, batch_size=imp_batch_size, shuffle=False, drop_last=False)
    model.eval()
    torch.set_grad_enabled(False)
    node_imp_l = []
    # loader.set_epoch(epoch)
    for step, (g_batch, _) in enumerate(loader):
        
        batch = torch.arange(0, g_batch.batch_size, device=device).reshape(-1,1).expand(g_batch.batch_size, nodes_per_graph).reshape(-1,)
        if node_est == 'classical':
          node_imp = model.node_imp_estimator(g_batch.ndata['node_attr'], torch.stack(g_batch.edges()), g_batch.edata['edge_attr'], batch).detach()
        else:
          ############ Quantum Node Est.
          g_batch.to(device)
          g_n = g_batch.ndata['node_attr'].clone().reshape(batch_size, nodes_per_graph, -1)
          e_n = g_batch.edata['edge_attr'].clone().reshape(batch_size, nodes_per_graph, -1)
          n_i = g_batch.ndata['node_indices'].clone().reshape(batch_size, nodes_per_graph)

          if q_edge_attr:
            node_imp = model.node_imp_estimator(g_n, n_i, e_n)
          else:
            node_imp = model.node_imp_estimator(g_n, n_i)
          node_imp = node_imp.reshape(-1,1)
       
          ############

        node_imp_l.append(node_imp.squeeze())

    for i, b in enumerate(node_imp_l):
      n = b.reshape(-1,nodes_per_graph)
      for g in range(len(n)):
        dataset[i*len(n)+g][0].ndata['node_score'] = torch.Tensor(n[g])

    dataset.nodes_per_aug_graph = dataset.nodes_per_graph-int(aug_ratio*dataset.nodes_per_graph)

    torch.set_grad_enabled(True)
    model.train()

    train_loss_accum = 0
    ra_loss_accum = 0
    cp_loss_accum = 0
    uni_loss_accum = 0
    alignment_loss_accum = 0

    for step, (g_batch, _) in enumerate(loader):
        batch1, batch2 = dataset.augment_dataset('rationale', g_batch, batch_size)
        batch3 = dataset.augment_dataset('complement', g_batch, batch_size)
  
        batch1 = batch1.to(device)
        batch2 = batch2.to(device)
        batch3 = batch3.to(device)

        optimizer.zero_grad()
        batch_1 = torch.arange(0, batch_size, device=device).reshape(-1,1).expand(batch_size, dataset.nodes_per_aug_graph).reshape(-1,)
        batch_2 = torch.arange(0, batch_size, device=device).reshape(-1,1).expand(batch_size, dataset.nodes_per_aug_graph).reshape(-1,)
        batch_3 = torch.arange(0, batch_size, device=device).reshape(-1,1).expand(batch_size, dataset.nodes_per_aug_graph).reshape(-1,)
        
        node_attr_123 = torch.cat((batch1.ndata['node_attr'].float(), batch2.ndata['node_attr'].float(), batch3.ndata['node_attr'].float()), dim=0)
        edges_123 = torch.stack(dgl.batch([batch1, batch2, batch3]).edges())
        edge_attr_123 = torch.cat((batch1.edata['edge_attr'], batch2.edata['edge_attr'], batch3.edata['edge_attr']), dim=0)
        overall_batch = torch.arange(0, batch_size*3).reshape(-1,1).expand(batch_size*3, dataset.nodes_per_aug_graph).reshape(-1,)
        output = model.forward_cl(node_attr_123, edges_123, edge_attr_123, overall_batch, dataset.nodes_per_aug_graph, dgl.batch([batch1, batch2, batch3]), q_edge_attr, device=device, node_est=node_est)
        x1, x2, x3 = torch.split(output, [batch_size, batch_size, batch_size], dim=0)

        ra_loss, cp_loss, loss, uni_loss, al_loss = model.loss_ra(x1, x2, x3, loss_temp, lamda)

        loss.backward()
        optimizer.step()

        train_loss_accum += float(loss.detach().cpu().item())
        ra_loss_accum += float(ra_loss.detach().cpu().item())
        cp_loss_accum += float(cp_loss.detach().cpu().item())
        uni_loss_accum += float(uni_loss.detach().cpu().item())
        alignment_loss_accum += float(al_loss.detach().cpu().item())

    gc.collect()
    return train_loss_accum/(step+1), ra_loss_accum/(step+1), cp_loss_accum/(step+1), uni_loss_accum/(step+1), alignment_loss_accum/(step+1)

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()
    
class MyDistributedDataParallel(nn.parallel.DistributedDataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)
        
def run(rank, world_size, epochs, model_type, npg_original):
    setup(rank, world_size)
    
    main_dir = ''
    epochs=epochs
    nodes_per_graph_original = npg_original
    num_layer_gnn = 2
    num_layer_gnn_est = 2
    qnn_layers = 3
    emb_dim = 128
    in_dim = 8
    inter_dim = 256
    out_dim = 128
    JK = 'last'
    dropout_ratio = 0.1
    gnn_type = 'gat'    #'gcn'
    lr = 0.001
    decay = 0
    aug_ratio = 0.2
    batch_size = 2000
    loss_temp = 0.1
    lamda = 0.1
    node_est = 'quantum'

    torch.set_default_tensor_type('torch.cuda.FloatTensor')    
    
    # if rank == '0':
    qg_dataset = QuarkGluonGraphDataset(dataset_name='Quark Gluon', raw_dir='/global/homes/a/ameyb', save_dir='/content',
                                    data_folder_name=['sample_data', 'energyflow/datasets'][1], datafile_name=['x10_sorted_12500.npy', 'QG_jets.npz'][1], labelsfile_name=['y10_sorted_12500.npy', 'QG_jets.npz'][1],
                                    datatype='particles', dataset_size=10000, nodes_per_graph=nodes_per_graph_original, spectral_augmentation=False, irc_safety_aug=True, 
                                    device=rank)
    
    # gnn = GNN(num_layer=num_layer_gnn, in_dim=in_dim, emb_dim=emb_dim, inter_dim=inter_dim, JK=JK, drop_ratio=dropout_ratio, gnn_type=gnn_type)
    gnn = ParticleNetTagger1Path(in_dim, 2)
    if node_est == 'classical':
       node_imp_estimator = GNN_imp_estimator(num_layer=num_layer_gnn_est, emb_dim=emb_dim, in_dim=in_dim, JK=JK, drop_ratio=dropout_ratio)
    if node_est == 'quantum':
       node_imp_estimator = QGNN_node_estimator(nodes_per_graph_original, qnn_layers, in_dim, device=rank)
    
    model = graphcl(gnn, node_imp_estimator, emb_dim, out_dim)
    model.to(rank)
    model = MyDistributedDataParallel(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=decay)
    # loader = GraphDataLoader(qg_dataset, batch_size=batch_size, shuffle=False, drop_last=False, use_ddp=True, ddp_seed=0)
    
    # CHECKPOINT_PATH = tempfile.gettempdir() + "/model.checkpoint"
    if model_type == 'ParticleNet':
        CHECKPOINT_PATH = main_dir + "/particle_net_model.checkpoint"
    elif model_type == 'LorentzNet':
        CHECKPOINT_PATH = main_dir + "/lorentz_net_model.checkpoint"
    elif model_type == 'MPNN':
        CHECKPOINT_PATH = main_dir + "/mpnn_model.checkpoint"
    
    if rank == 0:
        # All processes should see same parameters as they all start from same
        # random parameters and gradients are synchronized in backward passes.
        # Therefore, saving it in one process is sufficient.
        torch.save(model.state_dict(), CHECKPOINT_PATH)

    # Use a barrier() to make sure that process 1 loads the model after process
    # 0 saves it.
    dist.barrier()
    # configure map_location properly
    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
    model.load_state_dict(
        torch.load(CHECKPOINT_PATH, map_location=map_location, weights_only=True))
        
    for epoch in range(1, epochs + 1):
        # loader.set_epoch(epoch)
        print("====epoch " + str(epoch))
        qg_dataset.augment = False
        train_loss, ra_loss, cp_loss, uni_loss, al_loss = train(epoch, model=model, device=rank, dataset=qg_dataset, optimizer=optimizer, batch_size=batch_size, nodes_per_graph=qg_dataset.nodes_per_graph, aug_ratio=aug_ratio, loss_temp=loss_temp, lamda=lamda,
                                      irc_safety=True, q_edge_attr=True, node_est=node_est)
        
        if rank == 0:
            torch.save(model.state_dict(), CHECKPOINT_PATH)
          
    
    cleanup()

# if __name__ == '__main__':
#     world_size = 2
#     mp.spawn(
#         main,
#         args=(world_size),
#          nprocs=world_size, 
#          join=False
#          )
