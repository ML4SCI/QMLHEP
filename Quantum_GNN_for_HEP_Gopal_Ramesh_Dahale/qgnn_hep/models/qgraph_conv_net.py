"""Definition of the GraphConvNet model."""

from typing import Callable

from flax import linen as nn
import jax.numpy as jnp
import jraph
from qgnn_hep.models.util import add_graphs_tuples, QMLP

class QGraphConvNet(nn.Module):
    """A Graph Convolution Network + Pooling model defined with Jraph."""

    latent_size: int
    num_mlp_layers: int
    message_passing_steps: int
    output_globals_size: int
    dropout_rate: float = 0
    skip_connections: bool = True
    layer_norm: bool = True
    num_qubits: int = 2
    num_layers: int = 1
    num_features: int = 3
    entanglement_gate: str = "cz"
    deterministic: bool = True
    pooling_fn: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray  # pytype: disable=annotation-type-mismatch  # jax-ndarray
    ] = jraph.segment_mean

    def setup(self):
        self.embed_layer = nn.Dense(self.latent_size)
        self.embed_global_layer = nn.Dense(self.output_globals_size)

        mlp_feature_sizes = [self.latent_size] * self.num_mlp_layers
        self.node_fn = QMLP(mlp_feature_sizes,
                            dropout_rate=self.dropout_rate,
                            num_qubits = self.num_qubits,
                            num_layers = self.num_layers,
                            num_features = self.num_features,
                            entanglement_gate = self.entanglement_gate,
                            deterministic=self.deterministic)
        self.norm_layer = nn.LayerNorm() if self.layer_norm else None


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


    def __call__(self, graphs: jraph.GraphsTuple) -> jraph.GraphsTuple:
        # We will first linearly project the original node features as 'embeddings'.

        print("Input graphs", graphs.nodes.shape, graphs.edges.shape, graphs.globals.shape)
        print("Latent size", self.latent_size)
        embedder = jraph.GraphMapFeatures(embed_node_fn=self.embed_layer)
        processed_graphs = embedder(graphs)
        print(
            "Embedder output",
            processed_graphs.nodes.shape,
            processed_graphs.edges.shape,
            processed_graphs.globals.shape,
        )

        # Now, we will apply the GCN once for each message-passing round.
        for _ in range(self.message_passing_steps):

            update_node_fn = jraph.concatenated_args(self.node_fn)
            graph_conv = jraph.GraphConvolution(update_node_fn=update_node_fn, add_self_edges=True)

            if self.skip_connections:
                processed_graphs = add_graphs_tuples(graph_conv(processed_graphs), processed_graphs)
            else:
                processed_graphs = graph_conv(processed_graphs)

            if self.layer_norm:
                processed_graphs = processed_graphs._replace(
                    nodes=self.norm_layer(processed_graphs.nodes),
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
        decoder = jraph.GraphMapFeatures(embed_global_fn=self.embed_global_layer)
        processed_graphs = decoder(processed_graphs)
        print(
            "After decoder", processed_graphs.nodes.shape, processed_graphs.edges.shape, processed_graphs.globals.shape
        )

        return processed_graphs
