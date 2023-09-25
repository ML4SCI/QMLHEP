import ml_collections

def get_config():
	"""Get the default hyperparameter configuration."""
	config = ml_collections.ConfigDict()

	config.n_hops = 3
	config.learning_rate = 0.001
	config.batch_size = 16
	config.num_epochs = 50
	config.split = 150
	return config