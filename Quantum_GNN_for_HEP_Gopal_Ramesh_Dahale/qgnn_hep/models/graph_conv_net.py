"""Definition of the GraphConvNet model."""

from typing import Callable

from flax import linen as nn
import jax.numpy as jnp
import jraph
from qgnn_hep.models.util import add_graphs_tuples, MLP

class GraphConvNet(nn.Module):
    """A Graph Convolution Network + Pooling model defined with Jraph."""

    latent_size: int
    num_mlp_layers: int
    message_passing_steps: int
    output_globals_size: int
    dropout_rate: float = 0
    skip_connections: bool = True
    layer_norm: bool = True
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
        # We will first linearly project the original node features as 'embeddings'.

        print("Input graphs", graphs.nodes.shape, graphs.edges.shape, graphs.globals.shape)
        print("Latent size", self.latent_size)
        embedder = jraph.GraphMapFeatures(embed_node_fn=nn.Dense(self.latent_size))
        processed_graphs = embedder(graphs)
        print(
            "Embedder output",
            processed_graphs.nodes.shape,
            processed_graphs.edges.shape,
            processed_graphs.globals.shape,
        )

        # Now, we will apply the GCN once for each message-passing round.
        for _ in range(self.message_passing_steps):
            mlp_feature_sizes = [self.latent_size] * self.num_mlp_layers
            update_node_fn = jraph.concatenated_args(
                MLP(mlp_feature_sizes, dropout_rate=self.dropout_rate, deterministic=self.deterministic)
            )
            graph_conv = jraph.GraphConvolution(update_node_fn=update_node_fn, add_self_edges=True)

            if self.skip_connections:
                processed_graphs = add_graphs_tuples(graph_conv(processed_graphs), processed_graphs)
            else:
                processed_graphs = graph_conv(processed_graphs)

            if self.layer_norm:
                processed_graphs = processed_graphs._replace(
                    nodes=nn.LayerNorm()(processed_graphs.nodes),
                )
            print(
                "After message passing",
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
