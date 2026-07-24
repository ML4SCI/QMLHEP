from typing import Callable, Sequence

from flax import linen as nn
import jax.numpy as jnp
import jraph
from jax import jit, vmap
import pennylane as qml

def add_graphs_tuples(graphs: jraph.GraphsTuple, other_graphs: jraph.GraphsTuple) -> jraph.GraphsTuple:
    """Adds the nodes, edges and global features from other_graphs to graphs."""
    return graphs._replace(
        nodes=graphs.nodes + other_graphs.nodes,
        edges=graphs.edges + other_graphs.edges,
        globals=graphs.globals + other_graphs.globals,
    )


class MLP(nn.Module):
    """A multi-layer perceptron."""

    feature_sizes: Sequence[int]
    dropout_rate: float = 0
    deterministic: bool = True
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    @nn.compact
    def __call__(self, inputs):
        x = inputs
        for size in self.feature_sizes:
            x = nn.Dense(features=size)(x)
            x = self.activation(x)
            x = nn.Dropout(rate=self.dropout_rate, deterministic=self.deterministic)(x)
        return x



class DRQNN(nn.Module):
    """Data Re-uploading Quantum Neural Network"""
    num_qubits: int = 2
    num_layers: int = 1
    num_features: int = 3
    entanglement_gate: str = "cz"
    theta_init: Callable = nn.initializers.uniform(scale=jnp.pi)

    def setup(self):
        self.dev = qml.device('default.qubit', wires=self.num_qubits)
        self.theta = self.param('θ', self.theta_init, (self.num_layers, self.num_qubits, self.num_features, 2))
        qnode = jit(qml.QNode(self.circuit, device=self.dev, diff_method='backprop'))
        self.drc = vmap(qnode, in_axes=(None, 0))

    def rotation_layer(self, qubit, params, x):
        # print("rotation_layer:", params.shape, x.shape)
        z = params[:, 0]*x + params[:, 1]
        qml.RX(z[0], qubit)
        qml.RY(z[1], qubit)
        qml.RZ(z[2], qubit)

    def entangling_layer(self, num_qubits):
        for q in range(num_qubits - 1):
            qml.CZ((q, q + 1))

        if num_qubits != 2:
            qml.CZ((num_qubits - 1, 0))

    def circuit(self, params, x):
        for l in range(self.num_layers):
            for f in range(self.num_features//3):
                for q in range(self.num_qubits):
                    self.rotation_layer(q, params[l][q][3*f:3*(f+1)], x[3*f:3*(f+1)])

            self.entangling_layer(self.num_qubits)
        return [qml.expval(qml.PauliZ(q)) for q in range(self.num_qubits)]

    def __call__(self, inputs):
        out = self.drc(self.theta, inputs)
        return jnp.array(out).T


class QMLP(nn.Module):
    """A multi-layer perceptron."""

    feature_sizes: Sequence[int]
    dropout_rate: float = 0
    num_qubits: int = 2
    num_layers: int = 1
    num_features: int = 3
    entanglement_gate: str = "cz"
    deterministic: bool = True
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    @nn.compact
    def __call__(self, inputs):
        # print("QMLP: inputs", inputs.shape)
        x = inputs
        for size in self.feature_sizes:
            x = nn.Dense(features=size)(x)
            x = self.activation(x)
            x = nn.Dropout(rate=self.dropout_rate, deterministic=self.deterministic)(x)

        x = nn.Dense(self.num_features)(x) # output shape (batch_size, num_features)

        # print("QMLP: before qnn", x.shape)
        x = DRQNN(self.num_qubits, self.num_layers, self.num_features, self.entanglement_gate)(x)
        # print("QMLP: after qnn", x.shape)

        for size in self.feature_sizes:
            x = nn.Dense(features=size)(x)
            x = self.activation(x)
            x = nn.Dropout(rate=self.dropout_rate, deterministic=self.deterministic)(x)

        return x