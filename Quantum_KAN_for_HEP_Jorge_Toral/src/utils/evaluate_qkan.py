# evaluate_qkan.py

import os
import json
import torch.nn as nn
import torch
import numpy as np
import time
from sklearn.metrics import (
    roc_auc_score, accuracy_score, confusion_matrix,
    f1_score, precision_score, recall_score
)

import src.preprocessing.processor_qg as processor
from src.architectures.qkan_model import QKANModel#, backend_mode
import src.utils.metrics as viz
import argparse
from src.utils import workspace

def evaluate_qkan(CONFIG, train_backend, eval_backend, seed=42):

    print(f"\n" + "="*50)
    print(f"Evaluating QKAN on the test set using backend mode: '{eval_backend}'")
    print(f"="*50 + "\n")

    # 1. Load test data
    _, _, _, _, X_test, y_test, _, _ = processor.load_and_preprocess_data(data_dir="data", processed_dir=f"data/seed_{seed}")
    
    # Random subsampling for quick evaluation
    n_test_samples = 1000  # Evaluate only 1000 jets to speed up
    torch.manual_seed(seed) # For reproducibility
    test_indices = torch.randperm(len(X_test))[:n_test_samples]
    X_test = X_test[test_indices]
    y_test = y_test[test_indices]

    # 2. Load the model with fine-tuned weights
    if train_backend == "noisy":
        save_path = CONFIG["qkan_noisy_path"]
        
    elif train_backend == "ideal":
        save_path = CONFIG["qkan_ideal_path"]
        
    elif train_backend == "shots":
        save_path = CONFIG["qkan_shots_path"]

    model = QKANModel(init_weights_path=CONFIG["polynomial_weights_dir"])
    model.load_state_dict(torch.load(save_path))
    print(f"Fine-tuned weights loaded from '{save_path}'")

    model.eval()

    # 3. Evaluation by batches to avoid memory overflow
    batch_size = 64
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_test, y_test), 
        batch_size=batch_size, shuffle=False
    )

    criterion = nn.BCEWithLogitsLoss()
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_test, y_test), 
        batch_size=batch_size, shuffle=False
    )
    test_loss = 0.0
    all_probs = []
    all_true = []

    print(f"Evaluating {len(X_test)} jets from the test set using backend mode: '{eval_backend}'...")
    start_time = time.time()

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X)

            loss = criterion(outputs.squeeze(), batch_y.squeeze())
            test_loss += loss.item() * batch_X.size(0)
            
            probs = torch.sigmoid(outputs).numpy()
            all_probs.extend(probs)
            all_true.extend(batch_y.numpy())

    eval_time = time.time() - start_time
    test_loss /= len(X_test)

    # Metrics from clasic evaluation
    test_true = np.array(all_true)
    test_probs = np.array(all_probs)
    test_preds_binary = (test_probs > 0.5).astype(int)

    test_acc = accuracy_score(test_true, test_preds_binary)
    test_f1 = f1_score(test_true, test_preds_binary)
    test_auc = roc_auc_score(test_true, test_probs)
    test_precision = precision_score(test_true, test_preds_binary)
    test_recall = recall_score(test_true, test_preds_binary)
    cm = confusion_matrix(test_true, test_preds_binary)

    # Print results
    print("\n" + "="*40)
    print(f"FINAL RESULTS QKAN, Backend: '{eval_backend}'")
    print(f"Time evaluation: {eval_time} s")
    print(f"Test AUC: {test_auc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test F1 Score: {test_f1:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("="*40)

    # Graphical evaluation
    if eval_backend == "noisy":
        viz.plot_roc_curve(test_true, test_probs, save_path=CONFIG['roc_qkan_noisy'])
        viz.plot_confusion_matrix(cm, save_path=CONFIG['cm_qkan_noisy'])
        viz.plot_confusion_matrix_normalized(cm, save_path=CONFIG['cm_qkan_noisy_normalized'])
        viz.plot_precision_recall_curve(test_true, test_probs, save_path=CONFIG['pr_qkan_noisy'])
        metrics_path = CONFIG['metrics_qkan_noisy']
    elif eval_backend == "ideal":
        viz.plot_roc_curve(test_true, test_probs, save_path=CONFIG['roc_qkan_ideal'])
        viz.plot_confusion_matrix(cm, save_path=CONFIG['cm_qkan_ideal'])
        viz.plot_confusion_matrix_normalized(cm, save_path=CONFIG['cm_qkan_ideal_normalized'])
        viz.plot_precision_recall_curve(test_true, test_probs, save_path=CONFIG['pr_qkan_ideal'])
        metrics_path = CONFIG['metrics_qkan_ideal']
    elif eval_backend == "shots":
        viz.plot_roc_curve(test_true, test_probs, save_path=CONFIG['roc_qkan_shots'])
        viz.plot_confusion_matrix(cm, save_path=CONFIG['cm_qkan_shots'])
        viz.plot_confusion_matrix_normalized(cm, save_path=CONFIG['cm_qkan_shots_normalized'])
        viz.plot_precision_recall_curve(test_true, test_probs, save_path=CONFIG['pr_qkan_shots'])
        metrics_path = CONFIG['metrics_qkan_shots']

    # Metadata
    metrics_dic = {
        "Backend": eval_backend,
        "Eval Time (s)": eval_time,
        "Test AUC": test_auc,
        "Test Accuracy": test_acc,
        "Test F1 Score": test_f1,
        "Test Precision": test_precision,
        "Test Recall": test_recall,
        "Test Loss": test_loss,
        "Confusion Matrix": cm.tolist()
    }

    # Save metrics to JSON

    with open(metrics_path, 'w') as f:
        json.dump(metrics_dic, f, indent=4)

    print(f"\nEvaluation complete. Metrics saved to '{metrics_path}'")