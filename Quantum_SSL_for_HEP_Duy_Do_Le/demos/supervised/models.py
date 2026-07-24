import torch

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

class QG_Images(pyg_data.InMemoryDataset):
    def __init__(self, images, labels, channel=None, root='../data/QG_Images', transform=None, pre_transform=None, force_reload=True):
        self.images = images
        self.labels = labels
        self.channel = channel if channel else "x"
            
        super().__init__(root, transform, pre_transform, force_reload=force_reload)

        self.load(self.processed_paths[0])
        
    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return [f'data_{self.channel}.pt']

    def download(self):
        pass

    def process(self):
        data_list = []
        for index, img in enumerate(self.images):
            data = self.image_to_graph(img, self.labels[index])
            data_list.append(data)

        # self.data, self.slices = self.collate(data_list)
        # torch.save((self.data, self.slices), self.processed_paths[0])

        self.save(data_list, self.processed_paths[0])
        return data_list

    def image_to_graph(self, image, label):
        # Find non-zero pixels (nodes)
        y_coords, x_coords = np.nonzero(image)
        intensities = image[y_coords, x_coords]
        assert len(intensities != 0)

        # Create node features (intensity, x-coord, y-coord)
        # node_features = 
        coords = np.stack((x_coords, y_coords), axis=1)

        # Convert to PyTorch tensors
        node_features = torch.tensor(intensities, dtype=torch.float).unsqueeze(1)
        coords = torch.tensor(coords, dtype=torch.float)

        # Create PyTorch Geometric Data object with node features
        data = pyg_data.Data(x=node_features, pos=coords, num_nodes=node_features.shape[0], 
                             y=torch.tensor([label], dtype=torch.float))

        return data
    
    
class Custom_GCN(pyg_nn.MessagePassing):
    def __init__(self, out_channels, in_channels=16):
        super().__init__(aggr='add')

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
        )
        
        self.pixel_embedding = nn.Sequential(
            nn.Linear(1, 4),
            nn.ReLU(),
            nn.Linear(4, 4),
        )
        
        self.dist_embedding = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU(),
            nn.Linear(8, 8),
        )

    def forward(self, x, pos, edge_index):
        # Start propagating messages.
        return self.propagate(edge_index, x=x, pos=pos)

    def message(self, x_i, x_j, pos_i, pos_j):

        # print(torch.norm(pos_i - pos_j, p=2, dim=1).shape )
        edge_feat = torch.cat([self.pixel_embedding(x_i), self.pixel_embedding(x_j), self.dist_embedding(torch.norm(pos_i - pos_j, p=2, dim=1).unsqueeze(1))], dim=-1)
        return self.mlp(edge_feat)
    
class Custom_GCN2(pyg_nn.MessagePassing):
    def __init__(self, out_channels, in_channels=24):
        super().__init__(aggr='add')

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
        )
        
        self.pixel_embedding = nn.Sequential(
            nn.Linear(8, 8),
            nn.ReLU(),
            nn.Linear(8, 8),
        )
        
        self.dist_embedding = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU(),
            nn.Linear(8, 8),
        )

    def forward(self, x, pos, edge_index):
        # Start propagating messages.
        return self.propagate(edge_index, x=x, pos=pos)

    def message(self, x_i, x_j, pos_i, pos_j):

        edge_feat = torch.cat([self.pixel_embedding(x_i), self.pixel_embedding(x_j), self.dist_embedding(torch.norm(pos_i - pos_j, p=2, dim=1).unsqueeze(1))], dim=-1)
        return self.mlp(edge_feat)
    
class GCN_Encoder(nn.Module):
    def __init__(self, hidden_dim=8):
        super().__init__()

        self.conv1 = Custom_GCN(hidden_dim)
        self.conv2 = Custom_GCN2(hidden_dim*2)
        self.output_dim = 8
        # self.classifier = pyg_nn.MLP([hidden_dim, hidden_dim, output_dim], bias=[False, True])
        
        self.readout = pyg_nn.MLP([hidden_dim*2, self.output_dim, self.output_dim], bias=[False, True])

    def forward(self, data):
        x, pos, edge_index, batch = data.x, data.pos, data.edge_index, data.batch
        
        # First Custom_GCN layer
        x = self.conv1(x=x, pos=pos, edge_index=edge_index)
        x = x.relu()
        # x = self.dropout(x)
        
        # Second Custom_GCN layer
        x = self.conv2(x=x, pos=pos, edge_index=edge_index)
        x = x.relu()
        # x = self.dropout(h)
        
        # Global Pooling:
        x = pyg_nn.global_add_pool(x, batch)
        
        # Classifier:
        # return self.classifier(x)
        
        return self.readout(x)

import pytorch_lightning as pl    
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("Device:", device)
batch_size = 64
from pytorch_metric_learning import losses
import torchmetrics

class ModelPL_Classify(pl.LightningModule):
    def __init__(self, model, learning_rate=0.001):
        super().__init__()
        self.model = model
        self.classifier = pyg_nn.MLP([model.output_dim, 16, 2], bias=[False, True])
        self.learning_rate = learning_rate
        self.criterion = torch.nn.CrossEntropyLoss()
        
        from torchmetrics import AUROC, Accuracy 
        self.train_auc = AUROC(task='binary')
        self.val_auc = AUROC(task='binary')
        self.test_auc = AUROC(task='binary')
        
        self.train_acc = Accuracy(task='binary')
        self.val_acc = Accuracy(task='binary')
        self.test_acc = Accuracy(task='binary')

    def forward(self, data):
        embeddings = self.model(data)
        return self.classifier(embeddings)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        lr_scheduler = {
            'scheduler': optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                        mode='min', factor=0.25, patience=1),
            'monitor': 'val_loss', 
            'interval': 'epoch',
            'frequency': 1
        }
        return [optimizer], [lr_scheduler]

    def training_step(self, data, batch_idx):
        lr = self.optimizers().param_groups[0]['lr']
        self.log('learning_rate', lr, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        
        logits = self(data)
        loss = self.criterion(logits.squeeze(), data.y)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        
        self.train_auc(F.softmax(logits.squeeze(), dim=1)[:, 1], data.y)
        self.log("train_auc", self.train_auc, on_step=False, on_epoch=True, prog_bar=False, batch_size=batch_size)
        
        self.train_acc(logits.argmax(dim=-1), data.y)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True, prog_bar=False, batch_size=batch_size)
        
        return loss

    def validation_step(self, data, batch_idx):
        logits = self(data)
        loss = self.criterion(logits.squeeze(), data.y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        
        self.val_auc(F.softmax(logits.squeeze(), dim=1)[:, 1], data.y)
        self.log("val_auc", self.val_auc, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        
        self.val_acc(logits.argmax(dim=-1), data.y)
        self.log('val_acc', self.val_acc, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)

    def test_step(self, data, batch_idx):
        logits = self(data)
        
        self.test_auc(F.softmax(logits.squeeze(), dim=1)[:, 1], data.y)
        self.log("test_auc", self.test_auc, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        
        self.test_acc(logits.argmax(dim=-1), data.y)
        self.log('test_acc', self.test_acc, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        
        
class TopKIntensity(T.BaseTransform):
    def __init__(self, k: int):
        self.k = k
    
    def __call__(self, data):
        if data.num_nodes > self.k:
            _, top_k_indices = torch.topk(data.x[:, 0], self.k)
            data.x = data.x[top_k_indices]
            data.pos = data.pos[top_k_indices]
            data.num_nodes = self.k
        return data


class EdgesToTopK(T.BaseTransform):
    def __init__(self, k: int):
        self.k = k

    def __call__(self, data):
        
        if self.k > len(data.x[:, 0]):
            raise ValueError(f"Requested top-k ({self.k}) is larger than available data ({data.x}:{len(data.x[:, 0])})")
        _, top_k_indices = torch.topk(data.x[:, 0], self.k)

        # Create edges from all nodes to the top k nodes
        edges = []
        for i in range(data.num_nodes):
            for j in top_k_indices:
                edges.append([i, j.item()])
        
        # Convert edges to tensor
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        data.edge_index = edge_index
        
        return data


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


class ModelPL_Contrastive(pl.LightningModule):
    def __init__(self, model, learning_rate=0.01):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        # self.criterion = losses.ContrastiveLoss(pos_margin=0.1, neg_margin=1.0)
        # try:
        self.criterion_alt = losses.NTXentLoss(temperature=0.5)
        # except:
        from qml_ssl.losses import NTXentLoss
        self.criterion = NTXentLoss(temperature=0.5)

        self.train_loss = torchmetrics.MeanMetric()
        self.train_loss_alt = torchmetrics.MeanMetric()
        self.val_loss = torchmetrics.MeanMetric()

    def forward(self, data):
        return self.model(data)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        lr_scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.25, patience=1),
            'monitor': 'val_loss',
            'interval': 'epoch',
            'frequency': 1
        }
        return [optimizer], [lr_scheduler]

    def training_step(self, data, batch_idx):
        embeddings = self(data)
        loss = self.criterion(embeddings, data.y)
        loss_alt = self.criterion_alt(embeddings, data.y)
        self.train_loss.update(loss)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.train_loss_alt.update(loss_alt)
        self.log('train_loss_alt', loss_alt, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        return loss

    def validation_step(self, data, batch_idx):
        embeddings = self(data)
        loss = self.criterion(embeddings, data.y)
        self.val_loss.update(loss)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        return loss

    def test_step(self, data, batch_idx):
        embeddings = self(data)
        loss = self.criterion(embeddings, data.y)
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        return loss

class LinearProbePL(pl.LightningModule):
    def __init__(self, pretrained_model, num_classes, learning_rate=0.001):
        super().__init__()
        self.pretrained_model = pretrained_model
        # self.classifier = pyg_nn.MLP([pretrained_model.output_dim, 16, num_classes], bias=[False, True])
        self.classifier = nn.Sequential(
            nn.Linear(pretrained_model.output_dim, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes),
        )
        self.learning_rate = learning_rate
        self.criterion = nn.CrossEntropyLoss()

        from torchmetrics import AUROC, Accuracy 
        self.train_auc = AUROC(task='binary')
        self.val_auc = AUROC(task='binary')
        self.test_auc = AUROC(task='binary')
        
        self.train_acc = Accuracy(task='binary')
        self.val_acc = Accuracy(task='binary')
        self.test_acc = Accuracy(task='binary')
        
        for param in self.pretrained_model.parameters():
            param.requires_grad = False

    def forward(self, x):
        embeddings = self.pretrained_model(x)
        logits = self.classifier(embeddings)
        return logits
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        lr_scheduler = {
            'scheduler': optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                        mode='min', factor=0.25, patience=1),
            'monitor': 'val_loss', 
            'interval': 'epoch',
            'frequency': 1
        }
        return [optimizer], [lr_scheduler]

    def training_step(self, data, batch_idx):
        lr = self.optimizers().param_groups[0]['lr']
        self.log('learning_rate', lr, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        
        logits = self(data)
        loss = self.criterion(logits.squeeze(), data.y.long())
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        
        self.train_auc(F.softmax(logits.squeeze(), dim=1)[:, 1], data.y)
        self.log("train_auc", self.train_auc, on_step=False, on_epoch=True, prog_bar=False, batch_size=batch_size)
        
        self.train_acc(logits.argmax(dim=-1), data.y)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True, prog_bar=False, batch_size=batch_size)
        
        return loss

    def validation_step(self, data, batch_idx):
        logits = self(data)
        loss = self.criterion(logits.squeeze(), data.y.long())
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        
        self.val_auc(F.softmax(logits.squeeze(), dim=1)[:, 1], data.y)
        self.log("val_auc", self.val_auc, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        
        self.val_acc(logits.argmax(dim=-1), data.y)
        self.log('val_acc', self.val_acc, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)

    def test_step(self, data, batch_idx):
        logits = self(data)
        
        self.test_auc(F.softmax(logits.squeeze(), dim=1)[:, 1], data.y)
        self.log("test_auc", self.test_auc, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        
        self.test_acc(logits.argmax(dim=-1), data.y)
        self.log('test_acc', self.test_acc, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)


def generate_embeddings(model, data_loader):
    """
    Generate embeddings for the given data using the provided model.

    Args:
        model (nn.Module): Trained model.
        data_loader (DataLoader): Data loader for the dataset.

    Returns:
        tuple: Embeddings and labels as numpy arrays.
    """
    model.eval()
    embeddings = []
    labels = []
    
    with torch.no_grad():
        for data in data_loader:
            data = data.to(model.device)
            emb = model.model(data)
            embeddings.append(emb)
            labels.append(data.y)
    
    embeddings = torch.cat(embeddings).cpu().numpy()
    labels = torch.cat(labels).cpu().numpy()
    
    return embeddings, labels