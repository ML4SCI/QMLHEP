# src/architectures/random_forest.py
import os
import sys
import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, accuracy_score,
    f1_score, precision_score,
    recall_score, confusion_matrix, log_loss
)
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent.parent.resolve()))
import src.utils.metrics as viz


class RandomForestTrainer:
    """
    Classical Random Forest baseline, trained/evaluated on the same
    processed-feature tensors and single-fold (--seed % n_subsets) selection
    the KAN pipeline uses (see ClassicKANTrainer in classic_kan.py). Unlike
    the KAN pipeline, this is a one-shot fit -- no epochs, no pruning/symbolic
    stages, no checkpoint history loop.
    """

    def __init__(self, config):
        """Initializes trainer with global CONFIG."""
        self.config = config
        self.model = None

    def train_rf_model(self, X_train, y_train, model_save_path):
        """
        Fits a RandomForestClassifier on the full processed feature tensor
        (no pruning/input filtering -- RF has no need for a quantum-friendly
        subset) using fixed, config-driven hyperparameters, and saves it via
        joblib.

        args:
            - X_train (np.ndarray): Training features, shape [N, n_features].
            - y_train (np.ndarray): Training labels, shape [N].
            - model_save_path (str): Path to save the fitted model (.joblib).
        return:
            - self.model (RandomForestClassifier): The fitted model.
        """
        self.model = RandomForestClassifier(
            n_estimators=self.config["rf_n_estimators"],
            max_depth=self.config["rf_max_depth"],
            min_samples_split=self.config["rf_min_samples_split"],
            min_samples_leaf=self.config["rf_min_samples_leaf"],
            max_features=self.config["rf_max_features"],
            class_weight=self.config["rf_class_weight"],
            n_jobs=self.config["rf_n_jobs"],
            random_state=self.config["seed"],
        )
        self.model.fit(X_train, y_train)

        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        joblib.dump(self.model, model_save_path)
        print(f"[RandomForestTrainer] Model saved to: {model_save_path}")
        return self.model

    def load_checkpoint(self, model_save_path):
        """Loads a fitted RandomForestClassifier from a joblib checkpoint."""
        self.model = joblib.load(model_save_path)
        return self.model

    def evaluate_rf_model(self, model_save_path, X_test, y_test,
                           save_path_roc_curve=None, conf_matrix_save_path=None,
                           save_path_pr_curve=None, conf_matrix_normalized_save_path=None):
        """
        Loads a Random Forest checkpoint and evaluates it on the test set.
        Mirrors ClassicKANTrainer.evaluate_kan_model's metric dict shape
        exactly, so both architectures' rows are directly comparable in the
        shared metrics table (src/utils/reporting.py).

        args:
            - model_save_path (str): Path to the saved model checkpoint (.joblib file).
            - X_test (np.ndarray): Test features, shape [N, n_features].
            - y_test (np.ndarray): Test labels, shape [N].
            - save_path_roc_curve (str): Path to save the ROC curve plot.
            - conf_matrix_save_path (str): Path to save the confusion matrix plot.
            - save_path_pr_curve (str): Path to save the Precision-Recall curve plot.
            - conf_matrix_normalized_save_path (str): Path to save the row-normalized confusion matrix plot.
        return:
            - model (RandomForestClassifier): The loaded model.
            - test_results (tuple): (true labels, predicted probabilities, predicted binary classes).
            - metrics (dict): Evaluation metrics -- loss (log_loss), accuracy, F1, AUC,
                precision, recall, and confusion matrix.
        """
        self.load_checkpoint(model_save_path)

        test_true = np.asarray(y_test).reshape(-1)
        test_preds_probs = self.model.predict_proba(X_test)[:, 1]
        test_preds_binary = self.model.predict(X_test)

        test_loss = log_loss(test_true, test_preds_probs)
        test_accuracy = accuracy_score(test_true, test_preds_binary)
        test_f1 = f1_score(test_true, test_preds_binary)
        test_auc = roc_auc_score(test_true, test_preds_probs)
        test_precision = precision_score(test_true, test_preds_binary)
        test_recall = recall_score(test_true, test_preds_binary)
        conf_matrix = confusion_matrix(test_true, test_preds_binary)
        efficiency_metrics = viz.compute_efficiency_metrics(test_true, test_preds_probs)

        print("\n--- Metrics on the Test Set (Random Forest) ---")
        print(f"Test Loss: {test_loss:.5f}")
        print(f"Test Accuracy: {test_accuracy:.5f}")
        print(f"Test F1 Score: {test_f1:.5f}")
        print(f"Test AUC: {test_auc:.5f}")
        print(f"Test Precision: {test_precision:.5f}")
        print(f"Test Recall: {test_recall:.5f}")
        print("\nConfusion Matrix:")
        print(conf_matrix)

        if save_path_roc_curve: viz.plot_roc_curve(test_true, test_preds_probs, save_path=save_path_roc_curve)
        if save_path_pr_curve: viz.plot_precision_recall_curve(test_true, test_preds_probs, save_path=save_path_pr_curve)
        if conf_matrix_save_path: viz.plot_confusion_matrix(conf_matrix, save_path=conf_matrix_save_path)
        if conf_matrix_normalized_save_path: viz.plot_confusion_matrix_normalized(conf_matrix, save_path=conf_matrix_normalized_save_path)

        metrics = {
            "Test Loss": test_loss, "Test Accuracy": test_accuracy, "Test F1 Score": test_f1,
            "Test AUC": test_auc, "Test Precision": test_precision, "Test Recall": test_recall,
            "Confusion Matrix": conf_matrix.tolist()
        }
        metrics.update(efficiency_metrics)

        return self.model, (test_true, test_preds_probs, test_preds_binary), metrics

    def feature_importance_report(self, feature_names, save_path_plot=None, save_path_json=None):
        """
        Reads self.model.feature_importances_, pairs them with feature_names,
        and writes a sorted horizontal bar plot + a {feature: importance}
        JSON dict -- RF's interpretability counterpart to the KAN pipeline's
        spline plots.

        args:
            - feature_names (list[str]): Names for each input feature column,
                in the same order as the training data's columns.
            - save_path_plot (str): Path to save the bar plot image.
            - save_path_json (str): Path to save the {feature: importance} JSON.
        return:
            - importances (dict): {feature_name: importance}, sorted descending.
        """
        raw_importances = self.model.feature_importances_
        pairs = sorted(zip(feature_names, raw_importances), key=lambda p: p[1], reverse=True)
        importances = {name: float(value) for name, value in pairs}

        if save_path_json:
            os.makedirs(os.path.dirname(save_path_json), exist_ok=True)
            import json
            with open(save_path_json, 'w') as f:
                json.dump(importances, f, indent=4)

        if save_path_plot:
            names = [p[0] for p in pairs]
            values = [p[1] for p in pairs]
            plt.figure(figsize=(10, max(6, 0.3 * len(names))))
            plt.barh(names[::-1], values[::-1], color='steelblue')
            plt.title('Random Forest Feature Importances', fontsize=16, fontweight='bold')
            plt.xlabel('Importance', fontsize=16, fontweight='bold')
            plt.tight_layout()
            os.makedirs(os.path.dirname(save_path_plot), exist_ok=True)
            plt.savefig(save_path_plot)
            plt.close()
            print(f"Feature importance plot saved to: '{save_path_plot}'.")

        return importances
