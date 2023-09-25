"""Defines the default hyperparameters and training configuration.

Uses a Graph Convolutional Network model (https://arxiv.org/abs/1609.02907).
"""

import ml_collections


def get_config():
	"""Get the default hyperparameter configuration."""
	config = ml_collections.ConfigDict()

	# Optimizer.
	config.optimizer_hparams = ml_collections.ConfigDict()
	config.optimizer_hparams.optimizer = "adam"
	config.optimizer_hparams.learning_rate = 1e-3

	# Training hyperparameters.
	config.train_hparams = ml_collections.ConfigDict()
	config.train_hparams.model_name = "QGraphConvNet"
	config.train_hparams.dataset_name = "mutag"
	config.train_hparams.dataset_config_name = "mutag"
	config.train_hparams.batch_size = 32
	config.train_hparams.num_train_steps = 15_000
	config.train_hparams.log_every_steps = 1000
	config.train_hparams.eval_every_steps = 1_000
	config.train_hparams.checkpoint_every_steps = 10_000
	config.train_hparams.add_virtual_node = False
	config.train_hparams.add_undirected_edges = True
	config.train_hparams.add_self_loops = True
	config.train_hparams.load_checkpoint = False

	# GNN hyperparameters.
	config.model_hparams = ml_collections.ConfigDict()
	config.model_hparams.latent_size = 64
	config.model_hparams.num_mlp_layers = 1
	config.model_hparams.message_passing_steps = 2
	config.model_hparams.output_globals_size = 2
	config.model_hparams.dropout_rate = 0.1
	config.model_hparams.skip_connections = True
	config.model_hparams.layer_norm = True

	# Quantum circuit specific hyperparameters
	config.model_hparams.num_qubits = 4
	config.model_hparams.num_layers = 1
	config.model_hparams.num_features = 6
	config.model_hparams.entanglement_gate = "cz" # "cz" or "cx"

	return config
