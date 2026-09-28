import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, auc
import seaborn as sns


def compute_efficiency_metrics(y_true, y_probs, signal_efficiency_points=(0.3, 0.5, 0.7, 0.9)):
    """
    Background efficiency/rejection at fixed signal-efficiency (TPR) working
    points, read off the ROC curve -- the standard HEP convention, instead of
    the raw (and possibly miscalibrated) 0.5 classifier threshold.

    For each target signal efficiency, finds the smallest achieved TPR that is
    >= the target and reports the FPR (background efficiency) at that point,
    plus its reciprocal (background rejection; +inf when FPR is exactly 0).

    Returns a flat dict of scalars (e.g. {"Sig Eff 0.3 - Bkg Eff": ...,
    "Sig Eff 0.3 - Bkg Rejection": ...}) so it merges directly into an
    existing metrics dict/JSON without needing list/dict JSON-stringification.
    """
    fpr, tpr, _ = roc_curve(y_true, y_probs)

    metrics = {}
    for target in signal_efficiency_points:
        candidates = np.flatnonzero(tpr >= target)
        idx = candidates[0] if candidates.size else len(tpr) - 1
        bkg_eff = float(fpr[idx])
        bkg_rejection = float("inf") if bkg_eff == 0.0 else 1.0 / bkg_eff
        label = f"Sig Eff {target:g}"
        metrics[f"{label} - Achieved Sig Eff"] = float(tpr[idx])
        metrics[f"{label} - Bkg Eff"] = bkg_eff
        metrics[f"{label} - Bkg Rejection"] = bkg_rejection

    return metrics

# Matplotlib parameters for consistent styling
FONT_PARAMS = {'fontsize': 16, 'fontweight': 'bold'}
TICK_PARAMS = {'fontsize': 12}
LEGEND_FONT = {'size': 12}


def plot_loss_history(history, save_path=None):
    """
    Plots the training and validation loss history.
    args:
        history (dict): A dictionary containing 'train_loss' and 'val_loss' lists.
        save_path (str): Path to save the plot image.
    """
    plt.figure(figsize=(10, 6)) 
    plt.plot(history['train_loss'], label='Training Loss', linewidth=2)
    plt.plot(history['val_loss'], label='Validation Loss', linewidth=2)

    plt.title('Loss History', **FONT_PARAMS)
    plt.xlabel('Epoch', **FONT_PARAMS)
    plt.ylabel('Loss', **FONT_PARAMS)
    plt.xticks(**TICK_PARAMS)
    plt.yticks(**TICK_PARAMS)
    plt.legend(fontsize=LEGEND_FONT['size'])
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Loss history fig save in: '{save_path}'.")
        plt.close()
    else:
        plt.show()

def plot_auc_history(history, save_path=None):
    """
    Plots the training and validation AUC history.
    args:
        history (dict): A dictionary containing 'train_auc' and 'val_auc' lists.
        save_path (str): Path to save the plot image.
    """
    if 'train_auc' not in history or 'val_auc' not in history:
        print(f"Warning: No training or validation AUC data found in history. Skipping AUC plot.")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(history['train_auc'], label='Training AUC', color='green', linewidth=2) if history['train_auc'] else None
    plt.plot(history['val_auc'], label='Validation AUC', color='red', linewidth=2) if history['val_auc'] else None

    plt.title('AUC History', **FONT_PARAMS)
    plt.xlabel('Epoch', **FONT_PARAMS)
    plt.ylabel('AUC', **FONT_PARAMS)
    plt.xticks(**TICK_PARAMS)
    plt.yticks(**TICK_PARAMS)
    plt.ylim(0.5, 1.0)
    plt.legend(fontsize=LEGEND_FONT['size'])
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"AUC history fig save in: '{save_path}'.")
        plt.close()
    else:
        plt.show()

def plot_roc_curve(y_true, y_probs, save_path=None):
    """
    Plots the ROC curve.

    Args:
        y_true (array-like): True binary labels.
        y_probs (array-like): Target scores, can either be probability estimates of the positive class.
        save_path(str): Path to save the plot image.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    auc_score = roc_auc_score(y_true, y_probs)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.5f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing')

    plt.title('Receiver Operating Characteristic (ROC) Curve', **FONT_PARAMS)
    plt.xlabel('False Positive Rate', **FONT_PARAMS)
    plt.ylabel('True Positive Rate', **FONT_PARAMS)
    plt.xticks(**TICK_PARAMS)
    plt.yticks(**TICK_PARAMS)

    plt.legend(fontsize=LEGEND_FONT['size'])
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)
        print(f"ROC curve plot saved to '{save_path}'.")
        plt.close()
    else:
        plt.show()

def plot_precision_recall_curve(y_true, y_probs, save_path=None):
    """
    Plots the Precision-Recall curve.
    Args:

    """
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    pr_auc = auc(recall, precision)

    
    plt.figure(figsize=(8, 6))
    plt.title('Precision-Recall Curve', **FONT_PARAMS)
    plt.plot(recall, precision, label=f'Curve PR (AUC = {pr_auc:.4f})')
    plt.xlabel('Recall', **FONT_PARAMS)
    plt.ylabel('Precision', **FONT_PARAMS)

    plt.legend(fontsize=LEGEND_FONT['size'])
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        print(f"Precision-Recall curve plot saved to '{save_path}'.")
        plt.close()
    else:
        plt.show()

def plot_confusion_matrix(conf_matrix, save_path=None):
    # Plot Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', annot_kws={"size": 14})
    
    plt.title('Confusion Matrix', **FONT_PARAMS)
    plt.xlabel('Predicted Value', **FONT_PARAMS)
    plt.ylabel('Real Value', **FONT_PARAMS)
    
    tick_labels = ['bkg (0)', 'top (1)']
    plt.yticks(ticks=[0.5, 1.5], labels=tick_labels, **TICK_PARAMS)
    plt.xticks(ticks=[0.5, 1.5], labels=tick_labels, **TICK_PARAMS)

    if save_path:
        plt.savefig(save_path)
        print(f"Confusion matrix plot saved to '{save_path}'.")
        plt.close()
    else:
        plt.show()

def plot_confusion_matrix_normalized(conf_matrix, save_path=None):
    """Row-normalized twin of plot_confusion_matrix: each true-class row sums
    to 1, so cells read as per-class recall/error rates instead of raw counts."""
    conf_matrix = np.asarray(conf_matrix, dtype=float)
    row_sums = conf_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    normalized = conf_matrix / row_sums

    # Plot Normalized Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(normalized, annot=True, fmt='.2%', cmap='Blues', annot_kws={"size": 14}, vmin=0.0, vmax=1.0)

    plt.title('Confusion Matrix (Normalized)', **FONT_PARAMS)
    plt.xlabel('Predicted Value', **FONT_PARAMS)
    plt.ylabel('Real Value', **FONT_PARAMS)

    tick_labels = ['bkg (0)', 'top (1)']
    plt.yticks(ticks=[0.5, 1.5], labels=tick_labels, **TICK_PARAMS)
    plt.xticks(ticks=[0.5, 1.5], labels=tick_labels, **TICK_PARAMS)

    if save_path:
        plt.savefig(save_path)
        print(f"Normalized confusion matrix plot saved to '{save_path}'.")
        plt.close()
    else:
        plt.show()
