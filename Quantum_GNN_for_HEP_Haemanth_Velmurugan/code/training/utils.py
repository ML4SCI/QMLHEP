import json
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve


def read_configurations(config_file):
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config


config = read_configurations("../config.json")
epochs = config["EPOCHS"]


def plot_loss(history, step=2):
    x = range(epochs)
    plt.plot(x, history['train_loss'], label='Train loss')
    plt.plot(x, history['val_loss'], label='Val loss')
    plt.plot(x, history['train_acc'], label='Train acc')
    plt.plot(x, history['val_acc'], label='Val acc')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.xticks(range(0, epochs, step), range(1, epochs+1, step))

    plt.legend()
    plt.show()


def plot_auc(labels, preds):
    auc = roc_auc_score(labels, preds)
    fpr, tpr, _ = roc_curve(labels, preds)
    plt.plot(fpr, tpr, label="AUC = {0}".format(auc))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()
