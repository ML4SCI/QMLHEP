import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger, CometLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, ModelSummary

API_KEY = None
API_KEY = "qagcCboPOVUgd06fVzpoc5rly"

def get_trainer(
    experiment_name: str,
    project_name: str = "Quantum Contrastive Representation Learning",
    max_epochs: int = 20,
    patience: int = 20,
    monitor_metric: str = "valid_loss",
    mode: str = "min",
    save_dir: str = "logs/",
    model_summary_depth: int = 8,
) -> pl.Trainer:
    """
    Get a PyTorch Lightning Trainer instance with logging, checkpointing, early stopping, and other callbacks.
    
    Args:
        experiment_name (str): Name for the experiment (used in logging).
        project_name (str): Name of the project for logging (used for CometLogger).
        max_epochs (int): Maximum number of epochs for training.
        patience (int): Number of epochs to wait for improvement before stopping early.
        monitor_metric (str): The metric to monitor for early stopping and checkpointing.
        mode (str): Whether to minimize or maximize the monitored metric ('min' or 'max').
        save_dir (str): Directory to save logs and checkpoints.
        logger_types (list): List of loggers to use ('csv', 'comet').
        comet_api_key (str): API key for CometLogger (optional, can be None).
        model_summary_depth (int): Depth of model summary for ModelSummary callback.
        save_top_k (int): Number of top models to save based on the monitored metric.
    
    Returns:
        pl.Trainer: Configured PyTorch Lightning Trainer instance.
    """

    # Create loggers
    loggers = []
    
    # CSV Logger
    csv_logger = CSVLogger(save_dir=save_dir, name=experiment_name)
    loggers.append(csv_logger)

    # Comet Logger
    # Create the save directory for logs and checkpoints
    experiment_save_dir = os.path.join("logs", experiment_name, f"version_{csv_logger.version}")
    os.makedirs(experiment_save_dir, exist_ok=True)

    # if API_KEY  is None, will log to local Comet
    comet_logger = CometLogger(
        api_key=API_KEY,
        project_name=project_name,
        experiment_name=experiment_name,
        save_dir=experiment_save_dir,
    )
    loggers.append(comet_logger)

    # Create callbacks
    summary_callback = ModelSummary(max_depth=model_summary_depth)

    early_stopping_callback = EarlyStopping(
        monitor=monitor_metric,
        patience=patience,
        mode=mode,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=experiment_save_dir,
        filename="{epoch}-{" + monitor_metric + "}",
        save_top_k=1,
        monitor=monitor_metric,
        mode=mode
    )

    callbacks = [summary_callback, early_stopping_callback, checkpoint_callback]

    # Create the trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        logger=loggers,
        devices="auto",  # Automatically select available devices
        callbacks=callbacks
    )

    print(experiment_save_dir)

    return trainer, csv_logger.log_dir
