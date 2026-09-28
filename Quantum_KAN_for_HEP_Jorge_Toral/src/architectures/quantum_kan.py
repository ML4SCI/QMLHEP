# src/architectures/quantum_kan.py
import os
import time
import json
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.resolve()))
from src.architectures.qkan_model import QKANModel

class QuantumKANTrainer:
    def __init__(self, config, train_backend="ideal", random_init=False, graph_filename="quantum_weights.pt"):
        self.config = config
        self.train_backend = train_backend
        self.random_init = random_init
        # Use all available CPU cores for the classical-side work (loss,
        # optimizer, grad clipping) instead of an artificial fixed cap.
        torch.set_num_threads(max(1, os.cpu_count() or 1))

        # Initialize the model pointing to the unified .pt file
        weights_path = os.path.join(self.config["polynomial_weights_dir"], graph_filename)
        self.model = QKANModel(graph_path=weights_path, backend_mode=train_backend, random_init=random_init)
        
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.get("qkan_learning_rate", 5e-3))
        self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=6, threshold=1e-3
        )
        self._setup_backend_paths()

    def _setup_backend_paths(self):
        prefix = "random_" if self.random_init else ""
        if self.train_backend == "noisy":
            self.save_path = self.config[f"qkan_{prefix}noisy_path"]
            self.history_path = self.config[f"history_{prefix}noisy_loss"]
        elif self.train_backend == "shots":
            self.save_path = self.config[f"qkan_{prefix}shots_path"]
            self.history_path = self.config[f"history_{prefix}shots_loss"]
        else:
            self.save_path = self.config[f"qkan_{prefix}ideal_path"]
            self.history_path = self.config[f"history_{prefix}ideal_loss"]

    def fit(self, X_train, y_train, X_val, y_val, resume=True, force=False):
        if os.path.exists(self.save_path) and resume and not force:
            print(f"\n[Q-Trainer] Quantum model ({self.train_backend}) found at '{self.save_path}'. Loading...")
            self.model.load_state_dict(torch.load(self.save_path))
            if os.path.exists(self.history_path):
                with open(self.history_path, 'r') as f: 
                    return json.load(f)
            return {"train_loss": [], "val_loss": [], "val_auc": []}

        print(f"\n[Q-Trainer] Starting Quantum Fine-Tuning. Backend: {self.train_backend}")
        
        n_val = min(self.config.get("n_val_samples", 1000), len(X_val))
        val_gen = torch.Generator().manual_seed(self.config["seed"])
        val_indices = torch.randperm(len(X_val), generator=val_gen)[:n_val]
        X_val_sub, y_val_sub = X_val[val_indices], y_val[val_indices]
        
        val_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_val_sub, y_val_sub), 
            batch_size=self.config.get("qkan_batch_size", 64), shuffle=False
        )

        # early stopping and history tracking
        patience = self.config.get("qkan_patience", 5)
        early_stop_delta = self.config.get("qkan_early_stop_delta", 1e-3)

        best_val_auc = 0.0
        best_val_loss = float('inf')
        patience_counter = 0
        history = {"train_loss": [], "val_loss": [], "val_auc": []}

        for epoch in range(self.config.get("qkan_epochs", 20)):
            sampling_gen = torch.Generator().manual_seed(self.config["seed"] + epoch)
            train_indices = torch.randperm(len(X_train), generator=sampling_gen)[:self.config.get("n_train_samples_for_epoch", 500)]
            
            train_loader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(X_train[train_indices], y_train[train_indices]),
                batch_size=self.config.get("qkan_batch_size", 64), shuffle=True
            )

            self.model.train()
            train_loss = 0.0
            for batch_X, batch_y in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs.squeeze(), batch_y.squeeze())
                
                if torch.isnan(loss): continue
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                train_loss += loss.item() * batch_X.size(0)
                
            train_loss /= len(train_indices)

            self.model.eval()
            val_loss = 0.0
            all_val_probs, all_val_true = [], []
            
            with torch.no_grad():
                for batch_X_val, batch_y_val in val_loader:
                    outputs_val = self.model(batch_X_val)
                    loss_val = self.criterion(outputs_val.squeeze(), batch_y_val.squeeze())
                    val_loss += loss_val.item() * batch_X_val.size(0)
                    
                    all_val_probs.extend(torch.sigmoid(outputs_val).cpu().numpy())
                    all_val_true.extend(batch_y_val.cpu().numpy())
                    
            val_loss /= n_val
            val_auc = 0.5 if np.isnan(all_val_probs).any() else roc_auc_score(all_val_true, all_val_probs)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_auc"].append(val_auc)

            self.lr_scheduler.step(val_loss)
            
            if epoch % 5 == 0:
                print(f"Epoch [{epoch+1}/{self.config.get('qkan_epochs')}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f}")

            if (val_auc > best_val_auc + early_stop_delta) or\
                (val_loss < best_val_loss - early_stop_delta):
                best_val_auc = val_auc
                os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
                torch.save(copy.deepcopy(self.model.state_dict()), self.save_path)
                print(f" -> Saving quantum model at epoch {epoch+1} (Loss: {val_loss:.4f} - AUC: {val_auc:.4f})")

            if (val_loss < best_val_loss - early_stop_delta) or\
                (val_auc > best_val_auc + early_stop_delta):
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f" Early Stopping activated at epoch {epoch+1}.")
                    break

        os.makedirs(os.path.dirname(self.history_path), exist_ok=True)
        with open(self.history_path, 'w') as f: 
            json.dump(history, f, indent=4)
        return history

    def evaluate(self, X_test, y_test, eval_backend="noisy", baseline=False, random_init=False, sine=False):
        """
        Evaluate the quantum model on the test set.
        Allows changing the simulation backend specifically for evaluation.

        Args:
            - baseline (bool): if True, routes plots/metrics to the
              *_baseline_{eval_backend} config paths instead of the plain
              *_{eval_backend} ones, and tags the saved metrics JSON with
              "Baseline": True. Used by evaluate_baseline() to keep pre-training
              (warm-start-only) metrics separate from post-training ones, so
              both can be compared side by side.
            - random_init (bool): if True, routes plots/metrics to the
              *_random_{eval_backend} config paths instead, for the
              random-VQC-init ablation (see QKANModel's random_init flag).
            - sine (bool): if True, routes to the *_sine_* config paths (SineKAN-basis
              warm start), combinable with baseline=True.
        """
        print(f"\n" + "="*50)
        print(f"[Q-Trainer] {'Baseline ' if baseline else ''}Evaluating QKAN on the test set. Backend: '{eval_backend}'")
        print(f"="*50 + "\n")

        # 1. Dynamically change the backend if it is different from the training one
        if self.model.backend_mode != eval_backend:
            print(f"[Q-Trainer] Changing circuit backend to {eval_backend} for evaluation...")
            self.model.backend_mode = eval_backend
            self.model.dev = self.model._initialize_device()
            import pennylane as qml
            self.model.qnode = qml.QNode(self.model._circuit, self.model.dev, interface="torch")

        self.model.eval()

        # 2. Batch evaluation
        batch_size = self.config.get("qkan_batch_size", 64)
        test_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_test, y_test), 
            batch_size=batch_size, shuffle=False
        )

        test_loss = 0.0
        all_probs = []
        all_true = []
        start_time = time.time()

        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                outputs = self.model(batch_X)
                loss = self.criterion(outputs.squeeze(), batch_y.squeeze())
                test_loss += loss.item() * batch_X.size(0)
                
                probs = torch.sigmoid(outputs).cpu().numpy()
                all_probs.extend(probs)
                all_true.extend(batch_y.cpu().numpy())

        eval_time = time.time() - start_time
        test_loss /= len(X_test)

        # 3. Metrics calculation
        import numpy as np
        from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
        
        test_true = np.array(all_true)
        test_probs = np.array(all_probs)
        test_preds_binary = (test_probs > 0.5).astype(int)

        test_acc = accuracy_score(test_true, test_preds_binary)
        test_f1 = f1_score(test_true, test_preds_binary)
        test_auc = roc_auc_score(test_true, test_probs)
        test_precision = precision_score(test_true, test_preds_binary)
        test_recall = recall_score(test_true, test_preds_binary)
        cm = confusion_matrix(test_true, test_preds_binary)

        print("\n" + "="*40)
        print("\n--- Metrics on the Test Set ---")
        print(f"Test Loss: {test_loss:.5f}")
        print(f"Test Accuracy: {test_acc:.5f}")
        print(f"Test F1 Score: {test_f1:.5f}")
        print(f"Test AUC: {test_auc:.5f}")
        print(f"Test Precision: {test_precision:.5f}")
        print(f"Test Recall: {test_recall:.5f}")

        # Print Confusion Matrix
        print("\nConfusion Matrix:")
        print(cm)
        print("\n" + "="*40)

        # 4. Dynamic routing for saving artifacts. baseline=True routes to the
        # *_baseline_{eval_backend} config keys instead of the plain ones (see
        # evaluate_baseline()), so pre- and post-training metrics never collide.
        import src.utils.metrics as viz
        efficiency_metrics = viz.compute_efficiency_metrics(test_true, test_probs)
        suffix = ("_sine" if sine else "") + ("_baseline" if baseline else "") + ("_random" if random_init else "")
        viz.plot_roc_curve(test_true, test_probs, save_path=self.config[f"roc_qkan{suffix}_{eval_backend}"])
        viz.plot_confusion_matrix(cm, save_path=self.config[f"cm_qkan{suffix}_{eval_backend}"])
        viz.plot_confusion_matrix_normalized(cm, save_path=self.config[f"cm_qkan{suffix}_{eval_backend}_normalized"])
        viz.plot_precision_recall_curve(test_true, test_probs, save_path=self.config[f"pr_qkan{suffix}_{eval_backend}"])
        metrics_path = self.config[f"metrics_qkan{suffix}_{eval_backend}"]

        import os
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)

        # Persist raw probabilities/labels too, mirroring the {stage}_eval_data_*
        # convention ClassicKANTrainer/RandomForestTrainer already use -- needed
        # for any downstream analysis (e.g. re-deriving efficiency/rejection at
        # other working points) beyond what's baked into metrics_dic.
        np.save(self.config[f"qkan_eval_data_true{suffix}_{eval_backend}"], test_true)
        np.save(self.config[f"qkan_eval_data_probs{suffix}_{eval_backend}"], test_probs)
        np.save(self.config[f"qkan_eval_data_binary{suffix}_{eval_backend}"], test_preds_binary)

        metrics_dic = {
            "Backend": eval_backend,
            "Baseline": baseline,
            "Random Init": random_init,
            "Sine Basis": sine,
            "Eval Time (s)": eval_time,
            "Test AUC": test_auc,
            "Test Accuracy": test_acc,
            "Test F1 Score": test_f1,
            "Test Precision": test_precision,
            "Test Recall": test_recall,
            "Test Loss": test_loss,
            "Confusion Matrix": cm.tolist()
        }
        metrics_dic.update(efficiency_metrics)

        import json
        with open(metrics_path, 'w') as f:
            json.dump(metrics_dic, f, indent=4)

        return metrics_dic

    def evaluate_baseline(self, X_test, y_test, eval_backend="noisy", random_init=False, sine=False):
        """
        Evaluate the freshly warm-started (untrained) QKAN, before any quantum
        fine-tuning, using the exact same metrics/plot pipeline as evaluate()
        (F1, accuracy, AUC, precision, recall, confusion matrix, ROC/PR plots),
        saved to *_baseline_{eval_backend} paths so they are directly comparable
        to the post-training evaluate() call at the same backend. Measures how
        much predictive signal survives the classical->quantum extraction alone,
        before any quantum optimization.

        Must be called before fit(): fit()'s resume path just loads a
        state_dict without evaluating, and its from-scratch path trains
        starting from self.model's CURRENT weights, so this has to run while
        self.model still holds the untouched warm-start weights.

        evaluate() may switch self.model.backend_mode/dev/qnode as a side
        effect if eval_backend != self.train_backend (see evaluate() above);
        this restores it back to self.train_backend afterwards so a subsequent
        fit() call trains on the intended backend instead of silently training
        under eval_backend.
        """
        print(f"\n[Q-Trainer] Baseline evaluation (untrained, warm-start only). Backend: '{eval_backend}'")
        metrics = self.evaluate(X_test, y_test, eval_backend=eval_backend, baseline=True,
                                random_init=random_init, sine=sine)

        if self.model.backend_mode != self.train_backend:
            print(f"[Q-Trainer] Restoring backend to '{self.train_backend}' for training...")
            self.model.backend_mode = self.train_backend
            self.model.dev = self.model._initialize_device()
            import pennylane as qml
            self.model.qnode = qml.QNode(self.model._circuit, self.model.dev, interface="torch")

        return metrics