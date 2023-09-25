import torch
import jax
from torch.utils.data import random_split, DataLoader
from ego_net.config import get_config
from ego_net.data import MUTAGDataset
from pathlib import Path
import optax
import tensorcircuit as tc
import jax.numpy as jnp
from tqdm import tqdm
from time import time
import numpy as np
tc.set_backend("jax")

DATASET_PATH = Path(__file__).parents[1] / 'data/downloaded/'

def my_collate_fn(batch):
    x = []
    y = []
    ego_graphs = []
    for data in batch:
        x += [data['x']]
        y += [data['y']]
        ego_graphs += [data['ego_graphs']]

    x = torch.stack(x)
    y = torch.stack(y)
    ego_graphs = torch.stack(ego_graphs)
    return x, y, ego_graphs


def get_loaders(config):
    split = config.split
    mutag_ds = MUTAGDataset(DATASET_PATH, config.n_hops)

    split_a_size = split
    split_b_size = len(mutag_ds) - split_a_size

    train_dataset, test_dataset = random_split(
        mutag_ds, [split_a_size, split_b_size],
        generator=torch.Generator().manual_seed(42))

    batch_size = config.batch_size
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              collate_fn=my_collate_fn,
                              pin_memory=True,
                              num_workers=8)

    val_loader = DataLoader(test_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            collate_fn=my_collate_fn,
                            pin_memory=True,
                            num_workers=8)

    return train_loader, val_loader

def graph_circ(ego_graphs, x, lmbd, theta):
    n_qubits = ego_graphs.shape[-1]
    n_features = x.shape[-1]
    n_hops = ego_graphs.shape[-2]
    steps = n_features // 3
    readout = n_qubits
    c = tc.Circuit(n_qubits + 1)

    # Paper's implementation
    for hop in range(n_hops):
        inputs = jnp.take(x, ego_graphs[hop], axis=0)
        inputs = jnp.nan_to_num(inputs)
        inputs = jnp.multiply(inputs, lmbd[hop])

        for q in range(n_qubits):
            for i in range(steps):
                c.rx(q, theta=inputs[q, 3*i])
                c.ry(q, theta=inputs[q, 3*i + 1])
                c.rz(q, theta=inputs[q, 3*i + 2])

            c.rx(q, theta=theta[hop, q, 0])
            c.ry(q, theta=theta[hop, q, 1])
            c.rz(q, theta=theta[hop, q, 2])

        for q in range(n_qubits - 1):
            c.cz(i, i + 1)
    return tc.backend.real(jnp.array([c.expectation_ps(z=[i]) for i in range(n_qubits)]))

# Virtual map the quantum circuit
qpred_vmap = tc.backend.vmap(tc.backend.jit(graph_circ), vectorized_argnums=(0, ))

def loss_fn(params, x, y, ego_graphs):
    res = qpred_vmap(ego_graphs, x, params['lmbd'], params['theta'])
    res = jnp.mean(res, axis=0) # paper's implementation
    res = jnp.dot(params['w'], res) + params['b']
    logits = res
    one_hot = jax.nn.one_hot(y, 2).reshape(-1, )
    loss = optax.softmax_cross_entropy(logits=logits, labels=one_hot)
    return loss, logits

if __name__ == "__main__":

	configuration = get_config()
	train_loader, val_loader = get_loaders(configuration)

	# configure parameters
	dummy = next(iter(train_loader))
	print(dummy[-1].shape)
	max_nodes, _, n_qubits = dummy[-1].shape[1:]
	n_features = dummy[0].shape[-1]
	n_hops = configuration.n_hops

	print(max_nodes, n_qubits, n_features)

	key = jax.random.PRNGKey(0)
	key, *subkeys = jax.random.split(key, num=5)

	lmbd = jax.random.uniform(subkeys[0], (n_hops, n_qubits, n_features))
	theta = jax.random.uniform(subkeys[1], (n_hops, n_qubits, 3)) # paper's implementation
	w = jax.random.uniform(subkeys[2], (2, n_qubits)) # paper's implementation
	b = jax.random.uniform(subkeys[3], (2,)) # paper's implementation

	params = {'lmbd': lmbd, 'theta': theta, 'w': w, 'b': b}
	optimizer = optax.adam(learning_rate=configuration.learning_rate)
	opt_state = optimizer.init(params)

	# Preparing VVAG using
	qml_vvag = tc.backend.vectorized_value_and_grad(
		loss_fn, argnums=0, vectorized_argnums=(1, 2, 3), has_aux=True
	)
	qml_vvag = tc.backend.jit(qml_vvag)

	# dummy input check
	print("Checking model with dummy input")

	dummy_x = jnp.ones([1, max_nodes, 7])
	dummy_y = jnp.ones([1, 1])
	dummy_ego_graphs = jnp.ones([1, max_nodes, n_hops, n_qubits]).astype(jnp.int32)

	s = time()
	qml_vvag(params, dummy_x, dummy_y, dummy_ego_graphs)
	e = time()

	print(f"Done in {e-s} s")

	losses = []
	accs = []

	for epoch in range(1, configuration.num_epochs + 1):

		epoch_loss = []
		epoch_accuracy = []

		with tqdm(train_loader, unit='batch') as tepoch:

			s = time()
			for x, y, ego_graphs in tepoch:
				tepoch.set_description(f"Epoch {epoch}")

				x = x.numpy()
				y = y.numpy()
				ego_graphs = ego_graphs.numpy().astype(np.int32)

				(loss, logits), grads = qml_vvag(params, x, y, ego_graphs)

				accuracy = jnp.mean(jnp.argmax(logits, -1) == y)

				updates, opt_state = optimizer.update(grads, opt_state)
				params = optax.apply_updates(params, updates)

				epoch_loss.append(jnp.mean(loss))
				epoch_accuracy.append(accuracy)

			e = time()

			train_loss = np.mean(epoch_loss)
			train_accuracy = np.mean(epoch_accuracy)

			print(
			f'epoch: {epoch:3d}',
			f'train_loss: {train_loss:.4f}, train_acc: {train_accuracy:.4f}',
			f'epoch time: {e-s:.4f}')

			losses.append(train_loss)
			accs.append(train_accuracy)

	# Testing
	val_loss = []
	val_accuracy = []

	s = time()
	with tqdm(val_loader, unit='batch') as tepoch:
		for x, y, ego_graphs in tepoch:
			tepoch.set_description(f"Val")

			x = x.numpy()
			y = y.numpy()
			ego_graphs = ego_graphs.numpy().astype(np.int32)

			(loss, logits), grads = qml_vvag(params, x, y, ego_graphs)

			accuracy = jnp.mean(jnp.argmax(logits, -1) == y)

			val_loss.append(jnp.mean(loss))
			val_accuracy.append(accuracy)
	e = time()

	print(f"val_loss: {np.mean(val_loss):.4f}, val_acc: {np.mean(val_accuracy):.4f}")
	print(f"val time: {e-s:.4f}")