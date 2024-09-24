from sklearn.metrics import roc_curve, auc, confusion_matrix
import tensorflow as tf
from sklearn.metrics import confusion_matrix as cmatrix
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools
import wandb
import torch

def run_model_lct(model, epoch, dataloader, lossFn, optimizer=None, train=True, return_embeddings=False):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0
    correct = 0
    num_samples = 0
    all_embeddings = []
    all_labels = []

    for batch in dataloader:
        optimizer.zero_grad() if train else None

        # Extract the pairs and labels from the dataloader batch
        data1, data2, labels = batch[0], batch[1], batch[2]
        
        # Get the embeddings for both graphs
        emb1 = model(data1.x.float(), data1.edge_index, data1.batch)
        emb2 = model(data2.x.float(), data2.edge_index, data2.batch)
        
        if return_embeddings:
            # Save embeddings and labels for later use in classification
            all_embeddings.append(emb1.detach().cpu())
            all_labels.append(labels.detach().cpu())

        # Compute contrastive loss
        loss = lossFn(emb1, emb2, labels)
        
        if train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * data1.num_graphs
        num_samples += data1.num_graphs

        # Compute accuracy
        with torch.no_grad():
            cos_sim = torch.nn.functional.cosine_similarity(emb1, emb2)
            predictions = (cos_sim > 0.5).long()  # You can adjust the threshold
            correct += (predictions == labels).sum().item()

    avg_loss = total_loss / num_samples
    accuracy = correct / num_samples

    if return_embeddings:
        return torch.cat(all_embeddings, dim=0), torch.cat(all_labels, dim=0)
    return avg_loss, accuracy


def evaluate_precision_recall_accuracy(y_true, y_pred, threshold=0.5):
    '''Returns Precision, Recall and Accuracy'''
    y_pred_binary = (y_pred >= threshold).astype(int)
    
    true_positives = np.sum((y_true == 1) & (y_pred_binary == 1))
    false_positives = np.sum((y_true == 0) & (y_pred_binary == 1))
    false_negatives = np.sum((y_true == 1) & (y_pred_binary == 0))
    true_negatives = np.sum((y_true == 0) & (y_pred_binary == 0))
    
    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
    accuracy = (true_positives + true_negatives) / len(y_true)
    
    return precision, recall, accuracy


def confusion_matrix(y_true, y_pred, threshold=0.5):
    '''Creates confusion matrix'''
    y_pred_binary = (y_pred >= threshold).astype(int)
    
    true_positives = np.sum((y_true == 1) & (y_pred_binary == 1))
    false_positives = np.sum((y_true == 0) & (y_pred_binary == 1))
    false_negatives = np.sum((y_true == 1) & (y_pred_binary == 0))
    true_negatives = np.sum((y_true == 0) & (y_pred_binary == 0))
    
    return np.array([[true_negatives, false_positives],
                     [false_negatives, true_positives]])


def make_cm(y_true,y_pred,classes=None,figsize=(10,10),text_size=15):
    '''Creates a pretty confusion matrix'''
    cm = cmatrix(y_true,tf.round(y_pred))
    cm_norm = cm.astype("float")/cm.sum(axis=1)[:,np.newaxis] # normalise confusion matrix
    n_class = cm.shape[0]

    fig, ax = plt.subplots(figsize=figsize)
    cax = ax.matshow(cm,cmap=plt.cm.Blues)
    fig.colorbar(cax)

    if classes:
      labels=classes
    else:
      labels=np.arange(cm.shape[0])

    ax.set(title="Confusion Matrix", 
          xlabel="Predicted label",
          ylabel="True label",
          xticks=np.arange(n_class),
          yticks=np.arange(n_class),
          xticklabels=labels,
          yticklabels=labels)

    ax.xaxis.set_label_position("bottom")
    ax.xaxis.tick_bottom()

    ax.yaxis.label.set_size(text_size)
    ax.xaxis.label.set_size(text_size)
    ax.title.set_size(text_size)
    
    threshold = (cm.max()+cm.min())/2

    for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
      plt.text(j,i,f"{cm[i,j]} ({cm_norm[i,j]*100:.1f})%",
              horizontalalignment="center",
              color="white" if cm[i,j]>threshold else "black",
              size=text_size)


def plot_auc(y_true, y_pred):
    '''Plots AUC-ROC curve'''
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.show()


# wandb enabled
def plot_auc(labels, preds):
    auc = roc_auc_score(labels, preds)
    fpr, tpr, _ = roc_curve(labels, preds)
    plt.plot(fpr, tpr, label="AUC = {0}".format(auc))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    wandb.log({"Confusion Matrix": wandb.Image(plt)})
    plt.show()

def make_cm(y_true,y_pred,classes=None,figsize=(10,10),text_size=15):
    cm = cmatrix(y_true,y_pred)
    cm_norm = cm.astype("float")/cm.sum(axis=1)[:,np.newaxis] # normalise confusion matrix
    n_class = cm.shape[0]

    fig, ax = plt.subplots(figsize=figsize)
    cax = ax.matshow(cm,cmap=plt.cm.Blues)
    fig.colorbar(cax)

    if classes:
        labels=classes
    else:
        labels=np.arange(cm.shape[0])

    ax.set(title="Confusion Matrix", 
        xlabel="Predicted label",
        ylabel="True label",
        xticks=np.arange(n_class),
        yticks=np.arange(n_class),
        xticklabels=labels,
        yticklabels=labels)

    ax.xaxis.set_label_position("bottom")
    ax.xaxis.tick_bottom()

    ax.yaxis.label.set_size(text_size)
    ax.xaxis.label.set_size(text_size)
    ax.title.set_size(text_size)


    threshold = (cm.max()+cm.min())/2

    for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j,i,f"{cm[i,j]} ({cm_norm[i,j]*100:.1f})%",
            horizontalalignment="center",
            color="white" if cm[i,j]>threshold else "black",
            size=text_size)
    wandb.log({"Confusion Matrix": wandb.Image(fig)})
