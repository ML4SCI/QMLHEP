
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 14})

def plot_metrics_from_csv(metrics_file, metrics={'valid_loss', 'valid_acc', 'valid_auc'}):
    df = pd.read_csv(metrics_file)

    required_columns = metrics
    if not required_columns.issubset(df.columns):
        raise ValueError("The CSV file does not contain the required metrics.")

    df = df.sort_values('epoch')

    df = df.fillna(method='ffill')

    epochs = df['epoch']
    valid_loss = df['valid_loss']
    valid_acc = df['valid_acc']
    valid_auc = df['valid_auc']

    plt.figure(figsize=(5*len(metrics), 5))

    plt.subplot(1, len(metrics), 1)
    plt.plot(epochs, valid_loss, marker='o', linestyle='-', color='b', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Validation Loss')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(epochs, valid_acc, marker='o', linestyle='-', color='r', label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(epochs, valid_auc, marker='o', linestyle='-', color='g', label='Validation AUC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.title('Validation AUC')
    plt.legend()

    plt.tight_layout()
    plt.show()
    