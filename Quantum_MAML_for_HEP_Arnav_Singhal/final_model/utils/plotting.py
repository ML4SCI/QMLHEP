import matplotlib.pyplot as plt
from typing import Dict, List

def plot_training_results(results: Dict[str, List[float]], init_type: str):
    epochs = range(1, len(results["meta_loss"]) + 1)

    # Meta-loss
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, results["meta_loss"], label="Meta-loss")
    plt.xlabel("Epoch"); plt.ylabel("Meta-loss")
    plt.title(f"Meta-loss over Epochs ({init_type.capitalize()} Initialization)")
    plt.legend(); plt.grid(True); plt.show()

    # Training loss
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, results["training_loss"], label="Training Loss")
    plt.xlabel("Epoch"); plt.ylabel("Training Loss")
    plt.title(f"Training Loss over Epochs ({init_type.capitalize()} Initialization)")
    plt.legend(); plt.grid(True); plt.show()

    # Validation loss
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, results["validation_loss"], label="Validation Loss")
    plt.xlabel("Epoch"); plt.ylabel("Validation Loss")
    plt.title(f"Validation Loss over Epochs ({init_type.capitalize()} Initialization)")
    plt.legend(); plt.grid(True); plt.show()

    # Gradient norms
    if "gradient_norms" in results and len(results["gradient_norms"]) == len(results["meta_loss"]):
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, results["gradient_norms"], label="Gradient Norms")
        plt.xlabel("Epoch"); plt.ylabel("||grad||₂")
        plt.title(f"Gradient Norms over Epochs ({init_type.capitalize()} Initialization)")
        plt.legend(); plt.grid(True); plt.show()

    # Prediction variance
    has_train_var = "train_pvar" in results and len(results["train_pvar"]) == len(results["meta_loss"])
    has_val_var   = "val_pvar" in results and len(results["val_pvar"]) == len(results["meta_loss"])
    if has_train_var or has_val_var:
        plt.figure(figsize=(10, 6))
        if has_train_var:
            plt.plot(epochs, results["train_pvar"], label="Train p(class=1) variance")
        if has_val_var:
            plt.plot(epochs, results["val_pvar"], label="Val p(class=1) variance")
        plt.xlabel("Epoch"); plt.ylabel("Variance")
        plt.title(f"Prediction Variance over Epochs ({init_type.capitalize()} Initialization)")
        plt.legend(); plt.grid(True); plt.show()

    # Accuracy (train vs val)
    has_train = "train_accuracy" in results and len(results["train_accuracy"]) == len(results["meta_loss"])
    has_val   = "val_accuracy" in results and len(results["val_accuracy"]) == len(results["meta_loss"])
    if has_train or has_val:
        plt.figure(figsize=(10, 6))
        if has_train:
            plt.plot(epochs, results["train_accuracy"], label="Train Accuracy")
        if has_val:
            plt.plot(epochs, results["val_accuracy"], label="Val Accuracy")
        plt.xlabel("Epoch"); plt.ylabel("Accuracy")
        plt.title(f"Accuracy over Epochs ({init_type.capitalize()} Initialization)")
        plt.legend(); plt.grid(True); plt.show()


def plot_comparison(results: Dict[str, Dict[str, List[float]]]):
    epochs = range(1, len(next(iter(results.values()))["meta_loss"]) + 1)

    # Meta-loss
    plt.figure(figsize=(12, 8))
    for init_type, result in results.items():
        plt.plot(epochs, result["meta_loss"], label=f"{init_type.capitalize()} Meta-loss")
    plt.xlabel("Epoch"); plt.ylabel("Meta-loss")
    plt.title("Meta-loss Comparison Across Initializations")
    plt.legend(); plt.grid(True); plt.show()

    # Training loss
    plt.figure(figsize=(12, 8))
    for init_type, result in results.items():
        plt.plot(epochs, result["training_loss"], label=f"{init_type.capitalize()} Training Loss")
    plt.xlabel("Epoch"); plt.ylabel("Training Loss")
    plt.title("Training Loss Comparison Across Initializations")
    plt.legend(); plt.grid(True); plt.show()

    # Validation loss
    plt.figure(figsize=(12, 8))
    for init_type, result in results.items():
        plt.plot(epochs, result["validation_loss"], label=f"{init_type.capitalize()} Validation Loss")
    plt.xlabel("Epoch"); plt.ylabel("Validation Loss")
    plt.title("Validation Loss Comparison Across Initializations")
    plt.legend(); plt.grid(True); plt.show()

    # Gradient norms
    if all("gradient_norms" in r for r in results.values()):
        plt.figure(figsize=(12, 8))
        for init_type, result in results.items():
            if len(result["gradient_norms"]) == len(next(iter(results.values()))["meta_loss"]):
                plt.plot(epochs, result["gradient_norms"], label=f"{init_type.capitalize()} ||grad||₂")
        plt.xlabel("Epoch"); plt.ylabel("||grad||₂")
        plt.title("Gradient Norms Comparison Across Initializations")
        plt.legend(); plt.grid(True); plt.show()

    # Train prediction variance
    if all("train_pvar" in r for r in results.values()):
        plt.figure(figsize=(12, 8))
        for init_type, result in results.items():
            plt.plot(epochs, result["train_pvar"], label=f"{init_type.capitalize()} Train Var")
        plt.xlabel("Epoch"); plt.ylabel("Variance p(class=1)")
        plt.title("Train Prediction Variance Comparison")
        plt.legend(); plt.grid(True); plt.show()

    # Val prediction variance
    if all("val_pvar" in r for r in results.values()):
        plt.figure(figsize=(12, 8))
        for init_type, result in results.items():
            plt.plot(epochs, result["val_pvar"], label=f"{init_type.capitalize()} Val Var")
        plt.xlabel("Epoch"); plt.ylabel("Variance p(class=1)")
        plt.title("Validation Prediction Variance Comparison")
        plt.legend(); plt.grid(True); plt.show()

    # Training accuracy
    if all("train_accuracy" in r for r in results.values()):
        plt.figure(figsize=(12, 8))
        for init_type, result in results.items():
            plt.plot(epochs, result["train_accuracy"], label=f"{init_type.capitalize()} Train Acc")
        plt.xlabel("Epoch"); plt.ylabel("Accuracy")
        plt.title("Training Accuracy Comparison Across Initializations")
        plt.legend(); plt.grid(True); plt.show()

    # Validation accuracy
    if all("val_accuracy" in r for r in results.values()):
        plt.figure(figsize=(12, 8))
        for init_type, result in results.items():
            plt.plot(epochs, result["val_accuracy"], label=f"{init_type.capitalize()} Val Acc")
        plt.xlabel("Epoch"); plt.ylabel("Accuracy")
        plt.title("Validation Accuracy Comparison Across Initializations")
        plt.legend(); plt.grid(True); plt.show()

__all__ = ["plot_training_results", "plot_comparison"]
