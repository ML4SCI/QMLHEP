import torch_geometric
from torch_geometric.utils import to_networkx
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from pathlib import Path
from dhce.utils import get_dhce_data
import tensorcircuit as tc
tc.set_backend("tensorflow")

DATASET_PATH = Path(__file__).parents[1] / 'data/downloaded/'

def circ(x, weights):
	c = tc.Circuit(1)
	z = x * weights[0] + weights[1]

	c.rx(0, theta=z[0])
	c.ry(0, theta=z[1])
	c.rz(0, theta=z[2])
	c.ry(0, theta=z[3])

	return tc.backend.real(c.expectation_ps(z=[0]))

def hinge_accuracy(y_true, y_pred):
	y_true = tf.squeeze(y_true) > 0.0
	y_pred = tf.squeeze(y_pred) > 0.0
	result = tf.cast(y_true == y_pred, tf.float32)
	return tf.reduce_mean(result)

class DHCEModel(tf.keras.Model):

	def __init__(self, in_shape):
		super().__init__()

		self.qml_layer = tc.keras.QuantumLayer(circ, weights_shape=[2, in_shape])
		self.qml_layer.compute_output_shape = lambda input_shape: (input_shape[0], 1)

	def call(self, input_tensor):
		x = self.qml_layer(input_tensor)
		return x

	def build_graph(self, raw_shape):
		x = tf.keras.Input(shape=raw_shape)
		return tf.keras.Model(inputs=[x], outputs=self.call(x))


if __name__ == "__main__":
	tu_dataset = torch_geometric.datasets.TUDataset(root=DATASET_PATH, name="MUTAG")
	tu_dataset.shuffle()

	# Generate DHCE data
	temp_en_graphs = []
	data = tu_dataset[0]
	graph = to_networkx(data).to_undirected()

	smax = 0
	y = []
	for data in tu_dataset:
		graph = to_networkx(data).to_undirected()
		y.append(int(data.y[0]))
		temp_en_graphs.append(get_dhce_data(graph))
		smax = max(smax, len(temp_en_graphs[-1]))

	en_graphs = []
	for data in temp_en_graphs:
		en_graphs.append(data + [data[-1]]*(smax - len(data)))
	en_graphs = np.array(en_graphs)
	y = 2*np.array(y) - 1

	# train test split
	X_train, X_test, y_train, y_test = train_test_split(en_graphs, y, test_size=0.1, random_state=42, stratify=y)

	in_shape = en_graphs.shape[-1]
	model = DHCEModel(in_shape)
	model.compile(
		loss=tf.keras.losses.Hinge(),
		optimizer=tf.keras.optimizers.Adam(0.01),
		metrics=[hinge_accuracy],
	)

	print(model.build_graph(in_shape).summary())

	EPOCHS = 50
	BATCH_SIZE = 16

	qnn_history = model.fit(
		X_train, y_train,
		batch_size=BATCH_SIZE,
		epochs=EPOCHS,
		verbose=1,
		validation_data=(X_test, y_test)
	)

	qnn_results = model.evaluate(X_test, y_test)
