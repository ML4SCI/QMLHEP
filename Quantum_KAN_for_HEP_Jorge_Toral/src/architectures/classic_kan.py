# src/architectures/classic_kan.py
import os
import sys
import time
import json
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    roc_auc_score, accuracy_score,
    f1_score, precision_score,
    recall_score, confusion_matrix
)
import gc
from kan.utils import SYMBOLIC_LIB
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.resolve()))
from src.architectures.hep_kan import HEPKAN, prune_fanin
import src.utils.metrics as viz


def clean_memory(*args):
    """Utility function to clean up memory by deleting objects and clearing GPU cache."""
    for obj in args:
        if obj is not None:
            del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class ClassicKANTrainer:
    def __init__(self, config):
        """Initializes trainer with global CONFIG."""
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = HEPKAN(
            width=self.config["width"], 
            grid=self.config["grid"], 
            k=self.config["k"], 
            symbolic_enabled=True, 
            auto_save=False
        ).to(self.device)
        
        self.criterion = nn.BCEWithLogitsLoss()
        self.best_val_auc = 0.0
        self.best_val_loss = float('inf')

    def load_checkpoint(self, model_save_path):
        """Loads a checkpoint maintaining HEPKAN overridden methods."""
        base_kan = HEPKAN.loadckpt(model_save_path)
        self.model = HEPKAN.__new__(HEPKAN)
        self.model.__dict__.update(base_kan.__dict__)
        self.model.to(self.device)
        return self.model

    def train_kan_model(self,
                        width,
                        grid,
                        k,
                        learning_rate,
                        num_epochs,
                        batch_size,
                        early_stop_patience,
                        early_stop_min_delta,
                        model_save_path,
                        X_train_tensor,
                        y_train_tensor,
                        X_val_tensor,
                        y_val_tensor,
                        lamb=0.01,
                        lamb_l1=1.0,
                        lamb_entropy=2.0,
                        lamb_coef=0.0,
                        lamb_coefdiff=0.0,
                        reg_metric='edge_forward_spline_n',
                        update_grid_freq=5,
                        num_workers=0):
        """
        Trains a KAN model using a custom loop that includes key pykan features:
        regularization and adaptive grid updates.

        Args:
            width (list): The architecture of the KAN model.
            grid (int): The number of grid points for the splines.
            k (int): The order of the splines.
            learning_rate (float): The learning rate for the Adam optimizer.
            num_epochs (int): The maximum number of epochs for training.
            batch_size (int): The size of the batches for training.
            early_stop_patience (int): Number of epochs with no improvement to wait before stopping.
            early_stop_min_delta (float): Minimum change in validation loss to be considered an improvement.
            model_save_path (str): Path to save the best model checkpoint.
            X_train_tensor, y_train_tensor: Training data and labels as PyTorch tensors.
            X_val_tensor, y_val_tensor: Validation data and labels as PyTorch tensors.
            lamb (float): Overall regularization strength.
            lamb_l1 (float): L1 regularization strength for sparsity.
            lamb_entropy (float): Entropy regularization strength.
            update_grid_freq (int): The frequency (in epochs) to update the spline grid.
            num_workers (int): Number of worker processes for data loading.
        Returns:
            dict: A dictionary containing the training history (losses and AUCs).

        ---
        Example usage:
            trainer = ClassicKANTrainer(config)
            history = trainer.train_kan_model(
                width=[10, 20, 10, 1],
                grid=5,
                k=3,
                learning_rate=0.001,
                num_epochs=50,
                batch_size=64,
                early_stop_patience=5,
                early_stop_min_delta=1e-4,
                model_save_path='best_kan_model.ckpt',
                X_train_tensor=X_train_tensor,
                y_train_tensor=y_train_tensor,
                X_val_tensor=X_val_tensor,
                y_val_tensor=y_val_tensor
            )

        """
        print(f"\n--- Starting Training for KAN Model ---")
        print(f"Parameters: width={width}, grid={grid}, k={k}, lr={learning_rate}")

        # Reinitialize the model to ensure a fresh start for training
        self.model = HEPKAN(width, grid, k, symbolic_enabled=True, auto_save=False).to(self.device)
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"KAN model created with {num_params} parameters.")

        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                                   num_workers=num_workers, persistent_workers=False)
        val_dataset = torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                                 num_workers=num_workers, persistent_workers=False)

        best_val_auc = 0.0
        epochs_no_improve = 0
        best_val_loss = float('inf')
        start_time = time.time()

        history = {'train_loss': [], 'val_loss': [], 'train_auc': [], 'val_auc': []}

        for epoch in range(num_epochs):
            if epoch > 0 and epoch % update_grid_freq == 0:
                new_grid_size = self.model.grid + 3
                new_model = HEPKAN(width, new_grid_size, k, symbolic_enabled=True, auto_save=False).to(self.device)
                new_model.initialize_from_another_model(self.model, X_train_tensor.to(self.device))
                self.model = new_model
                
                early_stop_min_delta = max(early_stop_min_delta * 0.6, 1e-6)
                epochs_no_improve = 0
                best_val_loss = float('inf')

                print(f"\nUpdating grid at epoch {epoch+1}...")
                print(f"  -> Current grid: {self.model.grid}")
                with torch.no_grad():
                    optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

            self.model.train()
            train_loss = 0.0
            all_train_probs_list, all_train_true_list = [], []

            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                
                bce_loss = self.criterion(outputs, batch_y)
                reg_value = self.model.get_reg(reg_metric, lamb_l1, lamb_entropy, lamb_coef, lamb_coefdiff)
                total_loss = bce_loss + lamb * reg_value

                total_loss.backward()
                optimizer.step()

                train_loss += bce_loss.item() * batch_X.size(0)
                all_train_probs_list.append(torch.sigmoid(outputs).cpu().detach())
                all_train_true_list.append(batch_y.cpu().detach())
                
            train_loss /= len(train_dataset)

            self.model.eval()
            val_loss = 0.0
            all_val_probs_list, all_val_true_list = [], []
            
            with torch.no_grad():
                for batch_X_val, batch_y_val in val_loader:
                    batch_X_val, batch_y_val = batch_X_val.to(self.device), batch_y_val.to(self.device)
                    outputs_val = self.model(batch_X_val)
                    loss_val = self.criterion(outputs_val, batch_y_val)
                    val_loss += loss_val.item() * batch_X_val.size(0)

                    all_val_probs_list.append(torch.sigmoid(outputs_val).cpu().detach())
                    all_val_true_list.append(batch_y_val.cpu().detach())
            val_loss /= len(val_dataset)

            all_val_probs = torch.cat(all_val_probs_list, dim=0).numpy()
            all_val_true = torch.cat(all_val_true_list, dim=0).numpy()
            all_train_probs = torch.cat(all_train_probs_list, dim=0).numpy()
            all_train_true = torch.cat(all_train_true_list, dim=0).numpy()

            val_auc = roc_auc_score(all_val_true, all_val_probs)
            train_auc = roc_auc_score(all_train_true, all_train_probs)

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] |"
                      f"Training Loss: {train_loss:.5f} | Training AUC: {train_auc:.5f} | "
                      f"Validation Loss: {val_loss:.5f} | Validation AUC: {val_auc:.5f}")

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_auc'].append(train_auc)
            history['val_auc'].append(val_auc)

            if ((val_auc > best_val_auc + early_stop_min_delta) or (val_loss < best_val_loss - early_stop_min_delta)):
                best_val_auc = val_auc
                best_val_loss = val_loss
                epochs_no_improve = 0
                
                os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
                self.model.saveckpt(model_save_path)

                training_time_seconds = time.time() - start_time
                metadata = {
                    'num_params': num_params,
                    'training_time_seconds': training_time_seconds,
                    'final_val_auc': best_val_auc,
                    'final_val_loss': best_val_loss,
                    'hyperparameters': {
                        'width': width, 'grid': grid, 'k': k, 'learning_rate': learning_rate
                    }
                }
                try:
                    with open(f"{model_save_path}_metadata.json", "w") as f:
                        json.dump(metadata, f, indent=4)
                except Exception as e:
                    print(f"Warning: Could not save metadata to JSON. Error: {e}")
                print(f"Epoch [{epoch+1}/{num_epochs}] | Checkpoint saved (val_loss: {best_val_loss:.5f} - val_auc: {best_val_auc:.5f}).")    
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= early_stop_patience:
                print("Early Stopping triggered!")
                break
                
        print(f"\nTraining finished in {time.time() - start_time:.2f} seconds!")
        return history

    def evaluate_kan_model(self, model_save_path, X_test_tensor, y_test_tensor,
                           save_path_roc_curve=None, conf_matrix_save_path=None,
                           save_path_pr_curve=None, conf_matrix_normalized_save_path=None):
        """
        Loads a KAN model and its metadata from a checkpoint file and evaluates 
        its performance on the test set.
        args:
            - model_save_path (str): Path to the saved model checkpoint (.pth file).
            - X_test_tensor (torch.Tensor): Test features as a PyTorch tensor.
            - y_test_tensor (torch.Tensor): Test labels as a PyTorch tensor.
            - save_path_roc_curve (str): Path to save the ROC curve plot.
            - conf_matrix_save_path (str): Path to save the confusion matrix plot.
            - save_path_pr_curve (str): Path to save the Precision-Recall curve plot.
            - conf_matrix_normalized_save_path (str): Path to save the row-normalized confusion matrix plot.
        return:
            - model (HEPKAN): The loaded KAN model.
            - test_results (tuple): A tuple containing true labels, 
                predicted probabilities, and predicted binary classes.
            - metrics (dict): A dictionary containing evaluation metrics such as loss, 
                accuracy, F1 score, AUC, precision, recall, and confusion matrix.

        ---
        Example usage:
        >>> trainer = ClassicKANTrainer(config)
        >>> model, test_results, metrics = trainer.evaluate_kan_model(
            model_save_path='best_kan_model.ckpt',
            X_test_tensor=X_test_tensor,
            y_test_tensor=y_test_tensor,
            save_path_roc_curve='roc_curve.png',
            conf_matrix_save_path='confusion_matrix.png',
            save_path_pr_curve='pr_curve.png'
        )
        """
        self.load_checkpoint(model_save_path)
        self.model.eval()

        X_test = X_test_tensor.to(self.device)
        y_test = y_test_tensor.to(self.device)
        test_true = y_test.cpu().numpy()

        with torch.no_grad():
            outputs_test = self.model(X_test)
            loss_test = self.criterion(outputs_test, y_test)
            probs_test = torch.sigmoid(outputs_test)
            predicted_classes_test = (probs_test > 0.5).float()
            
            test_preds_probs = probs_test.cpu().numpy()
            test_preds_binary = predicted_classes_test.cpu().numpy()

        test_loss = loss_test.item()
        test_accuracy = accuracy_score(test_true, test_preds_binary)
        test_f1 = f1_score(test_true, test_preds_binary)
        test_auc = roc_auc_score(test_true, test_preds_probs)
        test_precision = precision_score(test_true, test_preds_binary)
        test_recall = recall_score(test_true, test_preds_binary)
        conf_matrix = confusion_matrix(test_true, test_preds_binary)
        efficiency_metrics = viz.compute_efficiency_metrics(test_true, test_preds_probs)

        # --- Print Final Metrics ---
        # Now using the metadata loaded from the file
        print("\n--- Metrics on the Test Set ---")
        print(f"Test Loss: {test_loss:.5f}")
        print(f"Test Accuracy: {test_accuracy:.5f}")
        print(f"Test F1 Score: {test_f1:.5f}")
        print(f"Test AUC: {test_auc:.5f}")
        print(f"Test Precision: {test_precision:.5f}")
        print(f"Test Recall: {test_recall:.5f}")

        # Print Confusion Matrix
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

    def prune_and_save_kan(self, X_sample, save_path, input_th=1e-2, node_th=1e-2, edge_th=1e-2, max_fanin=2):
        """
        Load a trained KAN model, prune it, and save a new checkpoint
        compatible with the evaluation function.

        Args:
            - X_sample (torch.Tensor or np.ndarray): Sample input data for computing activations.
            - save_path (str): Path where the new checkpoint of the pruned model will be saved.
            - input_th (float): Threshold for pruning input features.
            - node_th (float): Threshold for pruning hidden nodes.
            - edge_th (float): Threshold for pruning edges.
            - max_fanin (int or None): Hard cap on active input->hidden edges per hidden
              neuron, applied ADDITIONALLY on top of node_th/edge_th threshold pruning
              (keeps only the `max_fanin` edges with the greatest attribution score per
              hidden neuron). Pass None to disable.

        Returns:
            - HEPKAN: The pruned model object, ready for retraining.

        ---
        Example usage:
        >>> trainer = ClassicKANTrainer(config)
        >>> pruned_model = trainer.prune_and_save_kan(
            X_sample=X_sample_tensor,
            save_path='pruned_kan_model.ckpt',
            input_th=1e-2,
            node_th=1e-2,
            edge_th=1e-2
        )
        """
        print("\n" + "="*40)
        print("Starting KAN Pruning Phase (Input + General)...")
        print("="*40)

        # Directory for saving the pruned model
        self.model.ckpt_path = os.path.dirname(save_path)

        # Set the model to evaluation mode and compute activations for the sample input
        self.model.eval()
        with torch.no_grad():
            if isinstance(X_sample, np.ndarray):
                X_sample = torch.tensor(X_sample, dtype=torch.float32)
            X_sample = X_sample.to(self.device if hasattr(self, 'device') else 'cpu')
            self.model.get_act(X_sample)

        # Phase 1: Pruning Input Features
        print(f"Pruning inputs with threshold {input_th}...")
        self.model = self.model.prune_input(threshold=input_th)

        # It's necessary to regenerate activations after cloning the network in prune_input
        with torch.no_grad():
            self.model.get_act(X_sample)

        # Phase 2: Standard General Pruning (Hidden Nodes and Edges)
        print(f"Pruning hidden nodes and edges with node_th={node_th}, edge_th={edge_th}...")
        self.model = self.model.prune(node_th=node_th, edge_th=edge_th)

        # Phase 2.5: Hard fan-in cap (ADDITIVE on top of node_th/edge_th threshold
        # pruning), applied ONLY to the input->hidden layer (act_fun[0]). Keeps the
        # `max_fanin` edges with the greatest attribution contribution per hidden
        # neuron. model.prune()'s last internal step is attribute()+prune_edge(), so
        # model.edge_scores is already fresh for this topology - no extra
        # get_act()/attribute() call needed here.
        if max_fanin is not None:
            print(f"Capping input->hidden fan-in to top-{max_fanin} edges per hidden neuron...")
            self.model = prune_fanin(self.model, max_inputs=max_fanin, layer_index=0)

        # Extracting the record of surviving variables
        active_input_indices = self.model.input_id.cpu().tolist()
        print(f"\nActive input features retained: {active_input_indices}")
        print(f"New input dimension for Quantum phase: {len(active_input_indices)}")

        # Save checkpoint
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        self.model.saveckpt(save_path)

        # Save metadata
        metadata = {
            'num_params_pruned': sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            'active_input_indices': active_input_indices,
            'hyperparameters': {
                'width': self.model.width,
                'grid': self.model.grid,
                'k': self.model.k,
            }
        }
        
        try:
            with open(f"{save_path}_metadata.json", "w") as f:
                json.dump(metadata, f, indent=4)
        except Exception as e:
            print(f"Warning: Could not save pruning metadata. Error: {e}")

        print(f"[ClassicKANTrainer] Pruned model saved successfully at: '{save_path}'")
        
        return self.model

    def retrain_pruned_kan(self,
                            pruned_model,
                            learning_rate,
                            num_epochs,
                            batch_size,
                            early_stop_patience,
                            early_stop_min_delta,
                            model_save_path,
                            X_train_tensor,
                            y_train_tensor,
                            X_val_tensor,
                            y_val_tensor,
                            lamb=0.0,
                            lamb_l1=0.0,
                            lamb_entropy=0.0,
                            lamb_coef=0.0,
                            lamb_coefdiff=0.0,
                            num_workers=0):
        """
        Retrain (fine-tuning) a KAN model pruned and save the best model.
        The pruned_model should be an instance of KAN already pruned.

        Args:
            - pruned_model (KAN): The pruned KAN model instance.
            - learning_rate (float): Learning rate for the Adam optimizer.
            - num_epochs (int): Maximum number of epochs for retraining.
            - batch_size (int): Batch size for training.
            - early_stop_patience (int): Number of epochs with no improvement to wait before stopping.
            - early_stop_min_delta (float): Minimum change in validation loss to be considered an improvement.
            - model_save_path (str): Path to save the best retrained model.
            - X_train_tensor, y_train_tensor: Training data and labels as PyTorch tensors.
            - X_val_tensor, y_val_tensor: Validation data and labels as PyTorch tensors.
            - lamb (float): Overall regularization strength for retraining.
            - lamb_l1 (float): L1 regularization strength for sparsity during retraining.
            - lamb_entropy (float): Entropy regularization strength for retraining.
            - lamb_coef (float): Coefficient regularization strength for retraining.
            - lamb_coefdiff (float): Coefficient difference regularization strength for retraining.
            - num_workers (int): Number of worker processes for data loading.
        Return:
            - HEPKAN: The best retrained KAN model after fine-tuning.
            - dict: A dictionary containing the training history (losses and AUCs) during retraining.
        
        ---
        Example usage:
        >>> trainer = ClassicKANTrainer(config)
        >>> best_model, history = trainer.retrain_pruned_kan(
            pruned_model=pruned_kan_model,
            learning_rate=0.001,
            num_epochs=50,
            batch_size=64,
            early_stop_patience=5,
            early_stop_min_delta=1e-4,
            model_save_path='best_retrained_kan_model.ckpt',
            X_train_tensor=X_train_tensor,
            y_train_tensor=y_train_tensor,
            X_val_tensor=X_val_tensor,
            y_val_tensor=y_val_tensor
        )
        """
        print(f"\n--- Start Retraining (Fine-Tuning) the Pruned Model ---")
        self.model = pruned_model
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_dataset = torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        best_val_auc = float('-inf')
        best_val_loss = float('inf')
        epochs_no_improve = 0
        start_time = time.time()

        history = {'train_loss': [], 'val_loss': [], 'train_auc': [], 'val_auc': []}

        for epoch in range(num_epochs):
            self.model.train()
            train_loss = 0.0
            all_train_probs_list, all_train_true_list = [], []

            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                reg_value = self.model.get_reg('edge_forward_spline_n', lamb_l1, lamb_entropy, lamb_coef, lamb_coefdiff)
                loss = loss + lamb * reg_value
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * batch_X.size(0)
                all_train_probs_list.append(torch.sigmoid(outputs).cpu().detach())
                all_train_true_list.append(batch_y.cpu().detach())
            train_loss /= len(train_dataset)

            self.model.eval()
            val_loss = 0.0
            all_val_probs_list, all_val_true_list = [], []

            with torch.no_grad():
                for batch_X_val, batch_y_val in val_loader:
                    batch_X_val, batch_y_val = batch_X_val.to(self.device), batch_y_val.to(self.device)
                    outputs_val = self.model(batch_X_val)
                    loss_val = self.criterion(outputs_val, batch_y_val)
                    val_loss += loss_val.item() * batch_X_val.size(0)
                    all_val_probs_list.append(torch.sigmoid(outputs_val).cpu().detach())
                    all_val_true_list.append(batch_y_val.cpu().detach())
            val_loss /= len(val_dataset)

            all_val_probs = torch.cat(all_val_probs_list, dim=0).numpy()
            all_val_true = torch.cat(all_val_true_list, dim=0).numpy()
            all_train_probs = torch.cat(all_train_probs_list, dim=0).numpy()
            all_train_true = torch.cat(all_train_true_list, dim=0).numpy()

            val_auc = roc_auc_score(all_val_true, all_val_probs)
            train_auc = roc_auc_score(all_train_true, all_train_probs)

            print(f"Retraining Epoch [{epoch+1}/{num_epochs}] | Loss Train: {train_loss:.5f} | Train AUC: {train_auc:.5f} | Loss Val: {val_loss:.5f} | Val AUC: {val_auc:.5f}")

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_auc'].append(train_auc)
            history['val_auc'].append(val_auc)

            if (val_loss < best_val_loss - early_stop_min_delta) or (val_auc > best_val_auc + early_stop_min_delta):
                best_val_loss = val_loss
                best_val_auc = val_auc
                epochs_no_improve = 0

                os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
                self.model.saveckpt(model_save_path)

                retraining_time = time.time() - start_time
                num_params_pruned = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                metadata = {
                    'num_params': num_params_pruned,
                    'training_time_seconds': retraining_time,
                    'final_val_loss': best_val_loss,
                    'final_val_auc': best_val_auc,
                    'hyperparameters': {
                        'width': self.model.width, 'grid': self.model.grid, 'k': self.model.k, 'learning_rate': learning_rate
                    }
                }
                try:
                    with open(f"{model_save_path}_metadata.json", "w") as f:
                        json.dump(metadata, f, indent=4)
                except Exception as e:
                    print(f"Warning: Could not save metadata. Error: {e}")
            else:
                epochs_no_improve += 1
            
            if epochs_no_improve >= early_stop_patience:
                print("Early Stopping activated for retraining!")
                break

        final_model = self.load_checkpoint(model_save_path)
        return final_model, history

    def find_best_symbolic_globally(self, model, all_function_names):
        """
        First step of symbolic adjustment.
        Performs a global search over all edges and all functions in the SYMBOLIC_LIB.
        Logs the R2 and complexity for each function tested on each edge.
        Arg:
            model: KAN model and pruned (KAN instance).
            all_function_names: List of all function names in SYMBOLIC_LIB.
            weight_simple: Weighting factor for complexity vs R2 in cost function.
        Return:
            edge_results: Dictionary with results for each edge.
        ---
        Example usage:
        >>> trainer = ClassicKANTrainer(config)
        >>> edge_results = trainer.find_best_symbolic_globally(model, all_function_names)
        """
        print("Starting global search. Thank you for your patience.")
        edge_results = {}
        for l in range(len(model.width_in) - 1):
            for i in range(model.width_in[l]):
                for j in range(model.width_out[l + 1]):
                    edge = (l, i, j)
                    if model.symbolic_fun[l].mask[j, i] > 0. and model.act_fun[l].mask[i][j] == 0.:
                        print(f"Skipping edge {edge}: already symbolic.")
                        continue
                    elif model.symbolic_fun[l].mask[j, i] == 0. and model.act_fun[l].mask[i][j] == 0.:
                        print(f"Skipping edge {edge}: associated edge (value 0).")
                        continue
                    
                    print(f"--- Testing edge {edge} ---")
                    edge_results[edge] = []
                    for fun_name in all_function_names:
                        try:
                            r2 = model.fix_symbolic(l, i, j, fun_name, verbose=False, log_history=False)
                            if r2 != -1e8:
                                c = SYMBOLIC_LIB[fun_name][2]
                                edge_results[edge].append((fun_name, r2.item(), c))
                                gc.collect()
                            model.unfix_symbolic(l, i, j, log_history=False)
                        except Exception:
                            model.unfix_symbolic(l, i, j, log_history=False)
                            gc.collect()
                            continue
        print("Global search completed. Edge results collected.")
        return edge_results

    def apply_best_symbolic_from_log(self, model, edge_results, r2_threshold=0.85, weight_simple=0.4):
        """
        Step 2: Analysis and Adjustment of symbolic functions based on the global search results.
        Reviews the results log and fixes the BEST function for each edge.
        Args:
            - model: KAN model to modify (KAN instance).
            - edge_results: Dictionary with results for each edge.
            - r2_threshold: Minimum R2 to consider a function acceptable.
            - weight_simple: Weighting factor for complexity vs R2 in cost function.
        Returns:
            - fix_log: List of strings describing the adjustments made for each edge.
        ---
        Example usage:
        >>> trainer = ClassicKANTrainer(config)
        >>> fix_log = trainer.apply_best_symbolic_from_log(model, edge_results, r2_threshold=0.85, weight_simple=0.4)
        """
        print("\nStarting Step 2: Analysis and Adjustment.")
        fix_log = []
        r2_loss_fun = lambda x: np.log2(1+1e-5-x)
        c_loss_fun = lambda x: x

        for edge, results in edge_results.items():
            if not results:
                print(f"Skipping edge {edge}: no valid symbolic functions found.")
                continue

            candidate_functions = []
            for (name, r2, c) in results:
                if r2 >= r2_threshold:
                    r2_loss = r2_loss_fun(r2)
                    c_loss = c_loss_fun(c)
                    total_loss = weight_simple * c_loss + (1 - weight_simple) * r2_loss
                    candidate_functions.append((name, r2, c, total_loss))
            
            if not candidate_functions:
                print(f"Skipping {edge}: No function met the R² threshold of {r2_threshold}")
                continue

            candidate_functions.sort(key=lambda x: x[3])
            best_fun_name, best_r2, best_c, best_loss = candidate_functions[0]
            print(f"ADJUSTING {edge}: {best_fun_name} (R2={best_r2:.5f}, c={best_c}, total_loss={best_loss:.5f})")
            model.fix_symbolic(edge[0], edge[1], edge[2], best_fun_name, verbose=False, log_history=True)
            fix_log.append(f"Edge {edge}: {best_fun_name} (R2={best_r2:.5f})")
        print("Step 2 completed. Adjustments applied based on the global search log.")
        return fix_log

    def simplify_and_save(self, source_checkpoint_path, symbolic_model_path, x_train_sample, r2_threshold=0.8, weight_simple=0.5):
        """
        Simplifies a KAN model by fitting symbolic functions to its edges
        based on R2 threshold and complexity weight. Saves the final symbolic
        model to the specified path.
        Args:
            - source_checkpoint_path (str): Path to the checkpoint from which the model will be loaded.
            - symbolic_model_path (str): Path where the new checkpoint for the symbolic model will be saved.
            - x_train_sample (torch.Tensor): Sample of the training set for calculating activations.
            - r2_threshold (float): R² threshold for accepting a symbolic function.
            - weight_simple (float): Weight given to simplicity vs. R² fit.
        Returns:
            - HEPKAN: The final model object with symbolic adjustments applied.
        ---
        Example usage:
        >>> trainer = ClassicKANTrainer(config)
        >>> final_model = trainer.simplify_and_save(
                source_checkpoint_path='best_kan_model.ckpt',
                symbolic_model_path='symbolic_kan_model.ckpt',
                x_train_sample=X_train_tensor,
                r2_threshold=0.8,
                weight_simple=0.5
            )
        """
        print("--- Starting Simplification Process ---")
        start_time = time.time()
        self.load_checkpoint(source_checkpoint_path)
        self.model.eval()

        print("Computing activations for the sample training data...")
        self.model.get_act(x_train_sample.to(self.device))
        fix_log = []
        all_function_names = list(SYMBOLIC_LIB.keys())

        r2_loss_fun = lambda x: np.log2(1 + 1e-5 - x)
        c_loss_fun = lambda x: x

        for l in range(len(self.model.width_in) - 1):
            for i in range(self.model.width_in[l]):
                for j in range(self.model.width_out[l+1]):
                    edge = (l, i, j)
                    if self.model.act_fun[l].mask[i][j] == 0:
                        continue
                    
                    print(f"\n--- Evaluating edge {edge} ---")
                    edge_results = []
                    for fun_name in all_function_names:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                        try:
                            r2 = self.model.fix_symbolic(l, i, j, fun_name, fit_params_bool=True, verbose=False, log_history=False)
                            if r2 is not None and not np.isnan(r2.item()) and r2.item() > -1e7:
                                print(f" -> Trying {fun_name}: R2={r2.item():.5f}")
                                c = SYMBOLIC_LIB[fun_name][2]
                                total_loss = weight_simple * c_loss_fun(c) + (1 - weight_simple) * r2_loss_fun(r2.item())
                                edge_results.append((fun_name, r2.item(), c, total_loss))
                            self.model.unfix_symbolic(l, i, j, log_history=False)
                        except Exception:
                            print(" -> Exception occurred, skipping this function.")
                            self.model.unfix_symbolic(l, i, j, log_history=False)
                            continue
                    
                    if not edge_results:
                        print("  -> No valid symbolic functions found for this edge skipping to next edge.")
                        continue
                        
                    edge_results.sort(key=lambda x: x[3])
                    best_fun_name, best_r2, best_c, best_loss = edge_results[0]

                    if best_r2 >= r2_threshold:
                        print(f"  -> Fitting '{best_fun_name}' (R2={best_r2:.5f})")
                        self.model.fix_symbolic(l, i, j, best_fun_name, fit_params_bool=True, verbose=False, log_history=True)
                        fix_log.append(f"Edge {edge}: {best_fun_name} (R2={best_r2:.5f})")
                    else:
                        print(f"  -> Skipping. Best R2={best_r2:.5f} < {r2_threshold}. Adjusting to identity, ('x'), by default.")
                        self.model.fix_symbolic(l, i, j, 'x', fit_params_bool=True, verbose=False, log_history=True)
                        fix_log.append(f"Edge {edge}: 'x' (fallback, R2={best_r2:.5f})")
                    gc.collect()

        print("\n--- Simplification Process Completed ---")
        print(f"Saving the final symbolic model to '{symbolic_model_path}'...")
        os.makedirs(os.path.dirname(symbolic_model_path), exist_ok=True)

        self.model.saveckpt(symbolic_model_path)

        simplification_time = time.time() - start_time
        print(f"Simplification completed in {simplification_time:.2f} seconds.")
        meta_data = {
            'num_params': sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            'simplification_time_seconds': simplification_time,
            'r2_threshold': r2_threshold,
            'weight_simple': weight_simple,
            'hyperparameters': {'width': self.model.width, 'grid': self.model.grid, 'k': self.model.k}
        }
        try:
            with open(f"{symbolic_model_path}_metadata.json", "w") as f:
                json.dump(meta_data, f, indent=4)
        except Exception as e:
            print(f"Warning: Could not save simplification metadata. Error: {e}")

        print("\n" + "=" * 40 + "\nSummary of Fixes:")
        for line in fix_log: print(line)
        return self.model

    def finetune_symbolic_model(self,
                                source_symbolic_path,
                                final_model_path,
                                X_train_tensor,
                                y_train_tensor,
                                X_val_tensor,
                                y_val_tensor,
                                learning_rate=1e-3,
                                num_epochs=20,
                                batch_size=64,
                                early_stop_patience=5,
                                early_stop_min_delta=5e-4,
                                num_workers=0,
                                only_last_layer=True):
        """
        Fine-tunes a symbolic KAN model on the provided training and validation datasets.
        The method also provides an option to fine-tune only the last layer of the model or 
        the entire model.
        Args:
            - source_symbolic_path (str): Path to the .pth file of the symbolic model.
            - final_model_path (str): Path where the final fine-tuned model will be saved.
            - X_train_tensor, y_train_tensor: Training data and labels as PyTorch tensors.
            - X_val_tensor, y_val_tensor: Validation data and labels as PyTorch tensors.
            - learning_rate (float): Learning rate for the Adam optimizer.
            - num_epochs (int): Maximum number of epochs for fine-tuning.
            - batch_size (int): Batch size for training.
            - early_stop_patience (int): Number of epochs with no improvement to wait before stopping.
            - early_stop_min_delta (float): Minimum change in validation loss to be considered an improvement.
            - only_last_layer (bool): If True, only the last layer of the model will be fine-tuned.
        Returns:
            - KAN: The fine-tuned model object.
        ---
        Example usage:
        >>> trainer = ClassicKANTrainer(config)
        >>> fine_tuned_model = trainer.finetune_symbolic_model(
                source_symbolic_path='symbolic_kan_model.ckpt',
                final_model_path='fine_tuned_kan_model.ckpt',
                X_train_tensor=X_train_tensor,
                y_train_tensor=y_train_tensor,
                X_val_tensor=X_val_tensor,
                y_val_tensor=y_val_tensor,
                learning_rate=1e-3,
                num_epochs=20,
                batch_size=64,
                early_stop_patience=5,
                early_stop_min_delta=5e-4,
                only_last_layer=True
            )
        """
        print("\n--- Starting Fine-Tuning ---")
        self.load_checkpoint(source_symbolic_path)

        if only_last_layer:
            for param in self.model.parameters():
                param.requires_grad = False
            
            last_layer_index = len(self.model.symbolic_fun) - 1
            for param in self.model.symbolic_fun[last_layer_index].parameters():
                param.requires_grad = True
                
            if self.model.node_bias and last_layer_index < len(self.model.node_bias):
                self.model.node_bias[last_layer_index].requires_grad = True
            if self.model.node_scale and last_layer_index < len(self.model.node_scale):
                self.model.node_scale[last_layer_index].requires_grad = True

            optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=learning_rate)
        else:
            for param in self.model.parameters():
                param.requires_grad = True

            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_dataset = torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        best_val_auc = float('-inf')
        epochs_no_improve = 0
        start_time = time.time()
        history = {'train_loss': [], 'val_loss': [], 'train_auc': [], 'val_auc': []}

        for epoch in range(num_epochs):
            self.model.train()
            train_loss = 0.0
            all_train_probs_list, all_train_true_list = [], []

            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                if ~(only_last_layer):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item() * batch_X.size(0)
                all_train_probs_list.append(torch.sigmoid(outputs).cpu().detach())
                all_train_true_list.append(batch_y.cpu().detach())
            train_loss /= len(train_dataset)

            self.model.eval()
            val_loss = 0.0
            all_val_probs_list, all_val_true_list = [], []

            with torch.no_grad():
                for batch_X_val, batch_y_val in val_loader:
                    batch_X_val, batch_y_val = batch_X_val.to(self.device), batch_y_val.to(self.device)
                    outputs_val = self.model(batch_X_val)
                    loss_val = self.criterion(outputs_val, batch_y_val)
                    val_loss += loss_val.item() * batch_X_val.size(0)
                    all_val_probs_list.append(torch.sigmoid(outputs_val).cpu().detach())
                    all_val_true_list.append(batch_y_val.cpu().detach())
            val_loss /= len(val_dataset)

            all_val_probs = torch.cat(all_val_probs_list, dim=0).numpy()
            all_val_true = torch.cat(all_val_true_list, dim=0).numpy()
            all_train_probs = torch.cat(all_train_probs_list, dim=0).numpy()
            all_train_true = torch.cat(all_train_true_list, dim=0).numpy()

            val_auc = roc_auc_score(all_val_true, all_val_probs)
            train_auc = roc_auc_score(all_train_true, all_train_probs)

            print(f"Fitting epoch [{epoch+1}/{num_epochs}] | Train Loss: {train_loss:.5f} | Train AUC: {train_auc:.5f} | Val Loss: {val_loss:.5f} | Val AUC: {val_auc:.5f}")

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_auc'].append(train_auc)
            history['val_auc'].append(val_auc)

            if val_auc > best_val_auc + early_stop_min_delta:
                best_val_auc = val_auc
                epochs_no_improve = 0

                os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
                self.model.saveckpt(final_model_path)
                
                finetune_time = time.time() - start_time
                metadata = {
                    'num_params': sum(p.numel() for p in self.model.parameters() if p.requires_grad),
                    'finetuning_time_seconds': finetune_time,
                    'hyperparameters': {
                        'width': self.model.width,
                        'grid': self.model.grid,
                        'k': self.model.k,
                        'learning_rate': learning_rate
                    }
                }
                try:
                    with open(f"{final_model_path}_metadata.json", "w") as f:
                        json.dump(metadata, f, indent=4)
                except Exception as e:
                    print(f"Warning: Could not save metadata. Error: {e}")
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= early_stop_patience:
                print("Early Stopping activated during fine-tuning!")
                break

        best_model = self.load_checkpoint(final_model_path)
        for param in best_model.parameters():
            param.requires_grad = True

        return best_model, history
