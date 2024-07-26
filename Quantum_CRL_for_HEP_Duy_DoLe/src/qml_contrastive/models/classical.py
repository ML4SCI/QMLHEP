import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
from pytorch_metric_learning import losses

from ..utils import get_preprocessing

_ACTIVATIONS = {"relu": nn.ReLU, "gelu": nn.GELU}
_POOLING = {"max": nn.MaxPool2d, "avg": nn.AvgPool2d}

class ConvUnit(nn.Module):
    """
    A convolutional unit consisting of a convolutional layer, batch normalization, activation, and pooling.
    """
    def __init__(self, input_channels, output_channels, kernel_size, stride, padding, pool_type, pool_kernel_size, pool_stride, activ_type):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(output_channels)
        self.activ = _ACTIVATIONS[activ_type]()
        self.pool = _POOLING[pool_type](kernel_size=pool_kernel_size, stride=pool_stride)

    def forward(self, x):
        return self.pool(self.activ(self.bn(self.conv(x))))

class ConvEncoder(nn.Module):
    """
    A convolutional encoder for MNIST images.
    """
    backbone_output_size = 100
    def __init__(self, activ_type, pool_type):
        super().__init__()
        self.conv_unit1 = ConvUnit(1, 4, 3, 1, 1, pool_type, 1, 1, activ_type)
        self.conv_unit2 = ConvUnit(4, 4, 3, 1, 1, pool_type, 2, 2, activ_type)

    def forward(self, x):
        x = self.conv_unit1(x)
        x = self.conv_unit2(x)
        return x.view(-1, self.backbone_output_size)

class LinearHead(nn.Module):
    """
    A linear head for projecting the encoded features to the desired output dimension.
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.head = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.head(x)

class Conv_Siamese(pl.LightningModule):
    """
    A PyTorch Lightning module for supervised contrastive learning.
    """
    def __init__(self, activ_type, pool_type, head_output, lr, pos_margin=0.25, neg_margin=1.5, preprocess=None):
        super().__init__()
        self.save_hyperparameters()
        self.preprocessing = get_preprocessing(preprocess)
        self.encoder = ConvEncoder(activ_type, pool_type)
        self.head = LinearHead(ConvEncoder.backbone_output_size, head_output)
        self.loss = losses.ContrastiveLoss(pos_margin=pos_margin, neg_margin=neg_margin)
        self.train_loss = torchmetrics.MeanMetric()
        self.valid_loss = torchmetrics.MeanMetric()

    def forward(self, x):
        x = self.encoder(x)
        x = self.head(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        if self.preprocessing:
            x = self.preprocessing(x)
        embeddings = self.forward(x)
        loss = self.loss(embeddings, y)
        self.train_loss.update(loss)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        if self.preprocessing:
            x = self.preprocessing(x)
        embeddings = self.forward(x)
        loss = self.loss(embeddings, y)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


class Conv_Classifier(pl.LightningModule):
    """
    A PyTorch Lightning module for supervised classification.
    """
    def __init__(self, activ_type, pool_type, head_output, classes, lr, preprocess=None):
        super().__init__()
        self.save_hyperparameters()
        self.preprocessing = get_preprocessing(preprocess)
        self.encoder = ConvEncoder(activ_type, pool_type)
        self.head = LinearHead(ConvEncoder.backbone_output_size, head_output)

        self.classify = nn.Linear(head_output, len(classes) )

        self.loss = nn.CrossEntropyLoss()
        self.train_loss = torchmetrics.MeanMetric()
        self.valid_loss = torchmetrics.MeanMetric()
        
        self.custom_classes = classes  # (e.g., (0, 3, 6))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

    def forward(self, x):
        x = self.encoder(x)
        x = self.head(x)
        return self.classify(x)

    def map_labels(self, labels):
        mapped_labels = torch.tensor([self.class_to_idx[int(label)] for label in labels], device=self.device)
        return mapped_labels

    def training_step(self, batch, batch_idx):
        x, y = batch
        if self.preprocessing:
            x = self.preprocessing(x)
        logits = self.forward(x)
        loss = self.loss(logits, self.map_labels(y))
        self.train_loss.update(loss)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        if self.preprocessing:
            x = self.preprocessing(x)
        logits = self.forward(x)
        loss = self.loss(logits, self.map_labels(y))
        self.valid_loss.update(loss)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)