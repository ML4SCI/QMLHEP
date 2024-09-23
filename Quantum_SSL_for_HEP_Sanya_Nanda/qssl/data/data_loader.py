import numpy as np
import torch
from particle import Particle
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import os
import matplotlib.pyplot as plt
from collections import defaultdict
from .data_preprocessing_augmentation import graph_augment


class DataLoader:
    '''Loads the training and test pairs'''
    def __init__(self, file_path):
        data = np.load(file_path, allow_pickle=True)
        self.pairs_train = np.expand_dims(data["pairs_train"], -1)
        self.labels_train = data["labels_train"]
        self.pairs_test = np.expand_dims(data["pairs_test"], -1)
        self.labels_test = data["labels_test"]
    
    def get_train_data(self):
        return self.pairs_train, self.labels_train
    
    def get_test_data(self):
        return self.pairs_test, self.labels_test

class LabeledContrastiveDatasetQG():
    """
    Dataset class to load images from .npz files, convert them to PyTorch tensors, and return x1 and x2.
    """

    def __init__(self, file, transforms=None):
        #self.files = [os.path.join(folder, file) for file in os.listdir(folder) if file.endswith('.npz')]
        self.file=file
        self.transform = transforms
        
    def __len__(self):
        return len(self.file)
    
    def __getitem__(self, idx):
        """
        Load the npz file, convert x1 and x2 to PyTorch tensors, and return them.
        """
        # file_path = self.files[idx]
        # data = np.load(file_path)
        
        data = np.load(self.file, allow_pickle=True)
        pairs = data["pairs"]
        labels = data["labels"]
        
        pairs = pairs[:,:,:,:,3]
        pairs = pairs.reshape(-1, 2, 125, 125, 1)
        x1 = pairs[:,0]
        x2 = pairs[:,1,]

        def crop_center(img,cropx,cropy):
            x,y = img.shape[1:3]
            startx = x//2-(cropx//2)
            starty = y//2-(cropy//2)    
            return img[:,startx:startx+cropx,starty:starty+cropy,:]
        
        x1 = crop_center(x1,40,40)
        x2 = crop_center(x2,40,40)

        # Reshape the input tensors to add the channel dimension
        # x1 = x1.reshape(-1, 2, 125, 125, 1)
        # x2 = x2.reshape(-1, 2, 125, 125, 1)
        
        # Apply transforms if any
        if self.transform is not None:
            x1 = self.transform(x1)
            x2 = self.transform(x2)

        # Convert numpy arrays to PyTorch tensors
        x1_tensor = torch.tensor(x1, dtype=torch.float32).permute(0, 3, 1, 2)
        x2_tensor = torch.tensor(x2, dtype=torch.float32).permute(0, 3, 1, 2)
        labels = torch.Tensor(labels)

        return {"x1": x1_tensor, "x2": x2_tensor, 'labels':labels}
    
def preprocess_fixed_nodes(x_data,y_data,nodes_per_graph=10): 
    '''Preprocesses graph dataset (courtesy to Roy's open-source code)'''
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
    
    for jet in tqdm(range(N)):
        stop_indx = end_multiplicity_indx_per_jet[jet] #connections # stop finding edges once we hit zeros -> when we hit 10
        if end_multiplicity_indx_per_jet[jet]>=connections:
            for m in range(connections):
                edge_tensor[jet,m,:] = torch.sqrt( (phi[jet,m]-phi[jet,:connections])**2 + (rapidity[jet,m]-rapidity[jet,:connections])**2 )
            edges_exist_at = torch.where(edge_tensor[jet,:,:].abs()>0)
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
        
    return x_data[keep_inds,:nodes_per_graph], y_data[keep_inds].long(), edge_tensor[keep_inds], edge_indx_tensor[keep_inds].long(), edge_attr_matrix[keep_inds], graph_help[keep_inds]


def create_contrastive_graph_pairs(dataset):
    pairs = []
    labels = []
    # Group graphs by their labels (0 or 1)
    label_to_graphs = defaultdict(list)
    for data in dataset:
        label_to_graphs[data.y.item()].append(data)

    # Create pairs for positive class
    for data in label_to_graphs[1]:  # For each positive graph
        data_aug = graph_augment(data)  # Create an augmented version
        pairs.append((data, data_aug))  # Add the original and augmented graph as a pair
        labels.append(1)  # Label 1 for positive pair (similar)

    # Create pairs for negative class
    for data in label_to_graphs[0]:  # For each negative graph
        data_aug = graph_augment(data)  # Create an augmented version
        pairs.append((data, data_aug))  # Add the original and augmented graph as a pair
        labels.append(0)  # Label 0 for negative pair (dissimilar)

    return pairs, labels

