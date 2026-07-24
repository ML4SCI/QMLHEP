"""Definition of the GCN model for MUTAG dataset.
"""

from typing import Callable

from flax import linen as nn
import jax.numpy as jnp
import jraph
from qgnn_hep.models.util import add_graphs_tuples

class MLP(nn.Module):
	"""A multi-layer perceptron."""

	hidden_dim: int
	output_dim: int
	enhance: bool = False
	deterministic: bool = True

	@nn.compact
	def __call__(self, x):
		x = nn.Dense(features=self.hidden_dim)(x)
		x = nn.relu(x)

		if self.enhance:
			x = nn.LayerNorm()(x)
			x = nn.Dropout(rate = 0.2, deterministic=self.deterministic)(x)

		x = nn.Dense(features=self.hidden_dim)(x)
		x = nn.relu(x)

		if self.enhance:
			x = nn.LayerNorm()(x)
			x = nn.Dropout(rate= 0.2, deterministic=self.deterministic)(x)

		x = nn.Dense(features=self.output_dim)(x)
		return nn.relu(x)

class GCN(nn.Module):
	"""A Graph Convolution Network + Pooling model defined with Jraph."""

	# input_dim: int
	latent_size: int
	# num_mlp_layers: int
	message_passing_steps: int
	output_globals_size: int
	# dropout_rate: float = 0
	# skip_connections: bool = True
	# layer_norm: bool = True
	deterministic: bool = True
	pooling_fn: Callable[
		[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray  # pytype: disable=annotation-type-mismatch  # jax-ndarray
	] = jraph.segment_mean

	def pool(self, graphs: jraph.GraphsTuple) -> jraph.GraphsTuple:
		"""Pooling operation, taken from Jraph."""

		# Equivalent to jnp.sum(n_node), but JIT-able.
		sum_n_node = graphs.nodes.shape[0]  # pytype: disable=attribute-error  # jax-ndarray
		# To aggregate nodes from each graph to global features,
		# we first construct tensors that map the node to the corresponding graph.
		# Example: if you have `n_node=[1,2]`, we construct the tensor [0, 1, 1].
		n_graph = graphs.n_node.shape[0]
		node_graph_indices = jnp.repeat(jnp.arange(n_graph), graphs.n_node, axis=0, total_repeat_length=sum_n_node)
		# We use the aggregation function to pool the nodes per graph.
		pooled = self.pooling_fn(
			graphs.nodes, node_graph_indices, n_graph
		)  # pytype: disable=wrong-arg-types  # jax-ndarray
		return graphs._replace(globals=pooled)

	@nn.compact
	def __call__(self, graphs: jraph.GraphsTuple) -> jraph.GraphsTuple:

		print("Graphs: ", graphs.nodes.shape, graphs.edges.shape, graphs.globals.shape)

		# We will first linearly project the original node features as 'embeddings'.
		embedder = jraph.GraphMapFeatures(embed_node_fn=nn.Dense(self.latent_size))
		processed_graphs = embedder(graphs)

		print("After embedder", processed_graphs.nodes.shape, processed_graphs.edges.shape, processed_graphs.globals.shape)

		# Now, we will apply the GCN once for each message-passing round.
		for i in range(self.message_passing_steps):
			update_node_fn = jraph.concatenated_args(
				MLP(self.latent_size, self.latent_size, enhance=True, deterministic=self.deterministic)
			)
			graph_conv = jraph.GraphConvolution(update_node_fn=update_node_fn, add_self_edges=True)
			processed_graphs = graph_conv(processed_graphs)
			print(
				f"After message passing {i}",
				processed_graphs.nodes.shape,
				processed_graphs.edges.shape,
				processed_graphs.globals.shape,
			)

		# We apply the pooling operation to get a 'global' embedding.
		processed_graphs = self.pool(processed_graphs)
		print(
			"After pooling", processed_graphs.nodes.shape, processed_graphs.edges.shape, processed_graphs.globals.shape
		)

		# Now, we decode this to get the required output logits.
		decoder = jraph.GraphMapFeatures(embed_global_fn=nn.Dense(self.output_globals_size))
		processed_graphs = decoder(processed_graphs)
		print(
			"After decoder", processed_graphs.nodes.shape, processed_graphs.edges.shape, processed_graphs.globals.shape
		)

		return processed_graphs
