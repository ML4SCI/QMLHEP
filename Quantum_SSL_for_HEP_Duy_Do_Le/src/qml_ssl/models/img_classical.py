import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
from pytorch_metric_learning import losses

from .mods import get_mnist_augmentations

_ACTIVATIONS = {"relu": nn.ReLU, "gelu": nn.GELU, "swish": lambda: nn.SiLU(), "leaky_relu": nn.LeakyReLU}
_POOLING = {"max": nn.MaxPool2d, "avg": nn.AvgPool2d}

class ConvUnit(nn.Module):
    """
    A convolutional unit consisting of a convolutional layer, batch normalization, activation, and pooling.
    """
    def __init__(self, input_channels, output_channels, kernel_size, stride, padding, 
                 pool_type, pool_kernel_size, pool_stride, activ_type):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(output_channels)
        self.activ = _ACTIVATIONS[activ_type]()
        self.pool = _POOLING[pool_type](kernel_size=pool_kernel_size, stride=pool_stride)

    def forward(self, x):
        return self.pool(self.activ(self.bn(self.conv(x))))

class ConvEncoder(nn.Module):
    """
    A convolutional encoder for MNIST images using nn.ModuleList.
    """
    
    def __init__(self, activ_type, pool_type, layer_num, hidden_channel_num, input_channel_num, out_channel_num):
        super().__init__()

        self.layers = nn.ModuleList()

        # First convolutional layer
        self.layers.append(ConvUnit(input_channel_num, hidden_channel_num, 3, 1, 1, pool_type, 2, 2, activ_type))

        # Intermediate layers
        for _ in range(layer_num - 2):
            self.layers.append(ConvUnit(hidden_channel_num, hidden_channel_num, 3, 1, 1, pool_type, 2, 2, activ_type))

        # Final convolutional layer
        self.layers.append(ConvUnit(hidden_channel_num, out_channel_num, 3, 1, 1, pool_type, 2, 2, activ_type))

        # Pooling layer
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        
        x = self.global_pool(x)
        x = x.flatten(start_dim=1)
        return x


class Conv_UnSupContrastive(pl.LightningModule):
    """
    A PyTorch Lightning module for unsupervised contrastive learning using pytorch-metric-learning.
    """
    def __init__(self, lr, activ_type="relu", pool_type="max", layer_num=2, hidden_channel_num=8, input_channel_num=1, out_channel_num=8, proj_dim=2, augmentations=None, loss="NTXentLoss", loss_kwargs={}):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.encoder = ConvEncoder(activ_type, pool_type, layer_num, hidden_channel_num, input_channel_num, out_channel_num)
        self.proj = nn.Linear(out_channel_num, proj_dim)

        self.output_dim = proj_dim

        # Use the specified loss type from pytorch-metric-learning
        if loss == "NTXentLoss":
            self.loss_fn = losses.NTXentLoss(**loss_kwargs)
        elif loss == "ContrastiveLoss":
            self.loss_fn = losses.ContrastiveLoss(**loss_kwargs)
        else:
            raise ValueError("Unsupported loss type.")

        # Metrics for loss logging
        self.train_loss = torchmetrics.MeanMetric()

        # Augmentation options
        self.augmentations = augmentations

    def forward(self, x):
        x = self.encoder(x)
        x = self.proj(x)
        return x

    def training_step(self, batch):
        self.train()  
        x, _ = batch  # Ignore the actual labels since this is unsupervised

        # Generate two augmented views of the same batch
        if self.augmentations is None:
            view1 = x
            view2 = x
        else:
            view1 = self.augmentations(x)
            view2 = self.augmentations(x)

        # Get embeddings for both views
        embeddings_view1 = self.forward(view1)
        embeddings_view2 = self.forward(view2)

        # Combine the embeddings (double the batch size) and create pseudo labels
        embeddings = torch.cat([embeddings_view1, embeddings_view2], dim=0)
        labels = torch.arange(embeddings_view1.size(0)).to(self.device)  # Instance-level discrimination
        labels = torch.cat([labels, labels], dim=0)  # Duplicate the labels for both views

        # Compute loss
        loss = self.loss_fn(embeddings, labels)

        # Log loss
        self.train_loss.update(loss)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch):
        pass
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


class Conv_SupContrastive(pl.LightningModule):
    """
    A PyTorch Lightning module for supervised contrastive learning.
    """
    def __init__(self, lr, activ_type="relu", pool_type="max", layer_num=2, hidden_channel_num=8, input_channel_num=1, out_channel_num=8, proj_dim=2, preprocess=None, loss="NTXentLoss", loss_kwargs={}):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        if preprocess is not None:
            self.preprocessing = get_mnist_augmentations()
        else: 
            self.preprocessing = None
        self.encoder = ConvEncoder(activ_type, pool_type, layer_num,hidden_channel_num, input_channel_num, out_channel_num)
        self.proj = nn.Linear(out_channel_num, proj_dim)

        self.output_dim = proj_dim

        if loss == "NTXentLoss":
            self.loss = losses.NTXentLoss(**loss_kwargs)
        elif loss == "ContrastiveLoss":
            self.loss = losses.ContrastiveLoss(**loss_kwargs)
        else:
            raise ValueError("Unsupported loss type.")

        self.train_loss = torchmetrics.MeanMetric()
        self.valid_loss = torchmetrics.MeanMetric()

    def forward(self, x):
        x = self.encoder(x)
        x = self.proj(x)
        return x

    def training_step(self, batch):
        x, y = batch
        if self.preprocessing:
            x = self.preprocessing(x)
        embeddings = self.forward(x)
        loss = self.loss(embeddings, y)
        self.train_loss.update(loss)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch):
        x, y = batch
        if self.preprocessing:
            x = self.preprocessing(x)
        embeddings = self.forward(x)
        loss = self.loss(embeddings, y)
        self.log("valid_loss", loss, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


class LinearProbe(pl.LightningModule):
    def __init__(self, pretrained_model, classes, lr=0.001, num_layers=1, hidden_dim=16, lr_scheduler_metric=None):
        super().__init__()
        self.save_hyperparameters(ignore="pretrained_model")
        self.pretrained_model = pretrained_model
        
        layers = []
        input_dim = pretrained_model.output_dim

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim  # Update input_dim for the next layer

        layers.append(nn.Linear(input_dim, len(classes)))
        
        self.classifier = nn.Sequential(*layers)
        self.lr = lr
        self.criterion = nn.CrossEntropyLoss()

        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=len(classes))
        self.valid_acc = torchmetrics.Accuracy(task="multiclass", num_classes=len(classes))

        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    
        # Freeze the pretrained model
        for param in self.pretrained_model.parameters():
            param.requires_grad = False

    def forward(self, x):
        with torch.no_grad():  # No gradients for pretrained model
            x = self.pretrained_model(x)
        return self.classifier(x)

    def map_labels(self, labels):
        mapped_labels = torch.tensor([self.class_to_idx[int(label)] for label in labels], device=self.device)
        return mapped_labels

    def training_step(self, batch):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, self.map_labels(y))
        self.train_acc(logits, self.map_labels(y))

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, self.map_labels(y))
        self.valid_acc(logits, self.map_labels(y))

        self.log("valid_loss", loss, on_epoch=True, prog_bar=True)
        self.log("valid_acc", self.valid_acc, on_epoch=True, prog_bar=True)

        if self.hparams.lr_scheduler_metric:
            opt = self.optimizers()
            lr = opt.param_groups[0]['lr']
            self.log("lr", lr, on_epoch=True, prog_bar=True)


    def test_step(self, batch):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, self.map_labels(y))
        self.valid_acc(logits, self.map_labels(y))

        self.log("test_loss", loss, on_epoch=True, prog_bar=True)
        self.log("test_acc", self.valid_acc, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        if not self.hparams.lr_scheduler_metric:
            return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
            lr_scheduler = {
                'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=1),
                'monitor': self.hparams.lr_scheduler_metric, 
                'interval': 'epoch',
            }
            return [optimizer], [lr_scheduler]


class Conv_Classifier(pl.LightningModule):
    """
    A PyTorch Lightning module for supervised classification.
    """
    def __init__(self, classes, lr, activ_type="relu", pool_type="max", layer_num=2, hidden_channel_num=8, input_channel_num=1, out_channel_num=8, proj_dim=2, preprocess=None, lr_scheduler_metric=None):
        super().__init__()
        self.save_hyperparameters()

        if preprocess is not None:
            self.preprocessing = get_mnist_augmentations()
        else: 
            self.preprocessing = None
        self.encoder = ConvEncoder(activ_type, pool_type, layer_num,hidden_channel_num, input_channel_num, out_channel_num)
        self.proj = nn.Linear(out_channel_num, proj_dim)
        self.classifier = nn.Linear(proj_dim, len(classes))
        self.loss = nn.CrossEntropyLoss()

        # Metrics
        self.train_loss = torchmetrics.MeanMetric()
        self.valid_loss = torchmetrics.MeanMetric()
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=len(classes))
        self.valid_acc = torchmetrics.Accuracy(task="multiclass", num_classes=len(classes))

        # Handle custom classes
        self.custom_classes = classes  # (e.g., (0, 3, 6))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

    def forward(self, x):
        x = self.encoder(x)
        x = self.proj(x)
        return self.classifier(x)

    def map_labels(self, labels):
        mapped_labels = torch.tensor([self.class_to_idx[int(label)] for label in labels], device=self.device)
        return mapped_labels

    def training_step(self, batch):
        x, y = batch
        if self.preprocessing:
            x = self.preprocessing(x)
        logits = self.forward(x)
        loss = self.loss(logits, self.map_labels(y))
        self.train_loss.update(loss)
        self.train_acc(logits, self.map_labels(y))

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch):
        x, y = batch
        if self.preprocessing:
            x = self.preprocessing(x)
        logits = self.forward(x)
        loss = self.loss(logits, self.map_labels(y))
        self.valid_loss.update(loss)
        self.valid_acc(logits, self.map_labels(y))

        self.log("valid_loss", loss, on_epoch=True, prog_bar=True)
        self.log("valid_acc", self.valid_acc, on_step=False, on_epoch=True, prog_bar=True)

        if self.hparams.lr_scheduler_metric:
            opt = self.optimizers()
            lr = opt.param_groups[0]['lr']
            self.log("lr", lr, on_epoch=True, prog_bar=True)

    def test_step(self, batch):
        x, y = batch
        if self.preprocessing:
            x = self.preprocessing(x)
        logits = self.forward(x)
        loss = self.loss(logits, self.map_labels(y))
        self.valid_loss.update(loss)
        self.valid_acc(logits, self.map_labels(y))

        self.log("test_loss", loss, on_epoch=True, prog_bar=True)
        self.log("test_acc", self.valid_acc, on_step=False, on_epoch=True, prog_bar=True)


    def configure_optimizers(self):
        if not self.hparams.lr_scheduler_metric:
            return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
            lr_scheduler = {
                'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2),
                'monitor': self.hparams.lr_scheduler_metric, 
                'interval': 'epoch',
                'frequency': 1
            }
            return [optimizer], [lr_scheduler]
            
    

import torch_geometric.nn as pyg_nn

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
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                        mode='min', factor=0.25, patience=1),
            'monitor': 'val_loss', 
            'interval': 'epoch',
            'frequency': 1
        }
        return [optimizer], [lr_scheduler]

    def training_step(self, data, batch_idx):
        lr = self.optimizers().param_groups[0]['lr']
        self.log('learning_rate', lr, on_step=False, on_epoch=True, prog_bar=True)
        
        logits = self(data)
        loss = self.criterion(logits.squeeze(), data.y)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        self.train_auc(F.softmax(logits.squeeze(), dim=1)[:, 1], data.y)
        self.log("train_auc", self.train_auc, on_step=False, on_epoch=True, prog_bar=False)
        
        self.train_acc(logits.argmax(dim=-1), data.y)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True, prog_bar=False)
        
        return loss

    def validation_step(self, data, batch_idx):
        logits = self(data)
        loss = self.criterion(logits.squeeze(), data.y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        self.val_auc(F.softmax(logits.squeeze(), dim=1)[:, 1], data.y)
        self.log("val_auc", self.val_auc, on_step=False, on_epoch=True, prog_bar=True)
        
        self.val_acc(logits.argmax(dim=-1), data.y)
        self.log('val_acc', self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, data, batch_idx):
        logits = self(data)
        
        self.test_auc(F.softmax(logits.squeeze(), dim=1)[:, 1], data.y)
        self.log("test_auc", self.test_auc, on_step=False, on_epoch=True, prog_bar=True)
        
        self.test_acc(logits.argmax(dim=-1), data.y)
        self.log('test_acc', self.test_acc, on_step=False, on_epoch=True, prog_bar=True)



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
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.train_loss_alt.update(loss_alt)
        self.log('train_loss_alt', loss_alt, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, data, batch_idx):
        embeddings = self(data)
        loss = self.criterion(embeddings, data.y)
        self.val_loss.update(loss)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, data, batch_idx):
        embeddings = self(data)
        loss = self.criterion(embeddings, data.y)
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss


# class LinearProbe_(pl.LightningModule):
#     def __init__(self, pretrained_model, num_classes, learning_rate=0.001):
#         super().__init__()
#         self.pretrained_model = pretrained_model
#         # self.classifier = pyg_nn.MLP([pretrained_model.output_dim, 16, num_classes], bias=[False, True])
#         self.classifier = nn.Sequential(
#             nn.Linear(pretrained_model.output_dim, 16),
#             nn.ReLU(),
#             nn.Linear(16, num_classes),
#         )
#         self.learning_rate = learning_rate
#         self.criterion = nn.CrossEntropyLoss()

#         from torchmetrics import AUROC, Accuracy 
#         self.train_auc = AUROC(task='binary')
#         self.val_auc = AUROC(task='binary')
#         self.test_auc = AUROC(task='binary')
        
#         self.train_acc = Accuracy(task='binary')
#         self.val_acc = Accuracy(task='binary')
#         self.test_acc = Accuracy(task='binary')
        
#         for param in self.pretrained_model.parameters():
#             param.requires_grad = False

#     def forward(self, x):
#         embeddings = self.pretrained_model(x)
#         logits = self.classifier(embeddings)
#         return logits
    
#     def configure_optimizers(self):
#         optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
#         lr_scheduler = {
#             'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
#                                         mode='min', factor=0.25, patience=1),
#             'monitor': 'val_loss', 
#             'interval': 'epoch',
#             'frequency': 1
#         }
#         return [optimizer], [lr_scheduler]

#     def training_step(self, data, batch_idx):
#         lr = self.optimizers().param_groups[0]['lr']
#         self.log('learning_rate', lr, on_step=False, on_epoch=True, prog_bar=True)
        
#         logits = self(data)
#         loss = self.criterion(logits.squeeze(), data.y.long())
#         self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
#         self.train_auc(F.softmax(logits.squeeze(), dim=1)[:, 1], data.y)
#         self.log("train_auc", self.train_auc, on_step=False, on_epoch=True, prog_bar=False)
        
#         self.train_acc(logits.argmax(dim=-1), data.y)
#         self.log('train_acc', self.train_acc, on_step=False, on_epoch=True, prog_bar=False)
        
#         return loss

#     def validation_step(self, data, batch_idx):
#         logits = self(data)
#         loss = self.criterion(logits.squeeze(), data.y.long())
#         self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
#         self.val_auc(F.softmax(logits.squeeze(), dim=1)[:, 1], data.y)
#         self.log("val_auc", self.val_auc, on_step=False, on_epoch=True, prog_bar=True)
        
#         self.val_acc(logits.argmax(dim=-1), data.y)
#         self.log('val_acc', self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

#     def test_step(self, data, batch_idx):
#         logits = self(data)
        
#         self.test_auc(F.softmax(logits.squeeze(), dim=1)[:, 1], data.y)
#         self.log("test_auc", self.test_auc, on_step=False, on_epoch=True, prog_bar=True)
        
#         self.test_acc(logits.argmax(dim=-1), data.y)
#         self.log('test_acc', self.test_acc, on_step=False, on_epoch=True, prog_bar=True)

