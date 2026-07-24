"""GCN Mutag
"""

import ml_collections


def get_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    # Optimizer.
    config.optimizer_hparams = ml_collections.ConfigDict()
    config.optimizer_hparams.optimizer = "adam"
    config.optimizer_hparams.learning_rate = 1e-4

    # Training hyperparameters.
    config.train_hparams = ml_collections.ConfigDict()
    config.train_hparams.model_name = "GCN"
    config.train_hparams.dataset_name = "mutag"
    config.train_hparams.dataset_config_name = "mutag"
    config.train_hparams.batch_size = 16
    config.train_hparams.num_train_steps = 15_000
    config.train_hparams.log_every_steps = 1000
    config.train_hparams.eval_every_steps = 1000
    config.train_hparams.checkpoint_every_steps = 10_000
    config.train_hparams.add_virtual_node = False
    config.train_hparams.add_undirected_edges = True
    config.train_hparams.add_self_loops = True
    config.train_hparams.load_checkpoint = False

    # GNN hyperparameters.
    config.model_hparams = ml_collections.ConfigDict()
    config.model_hparams.latent_size = 32
    config.model_hparams.message_passing_steps = 5
    config.model_hparams.output_globals_size = 2
    return config
