import pennylane as qml
import tensorflow as tf
from tensorflow import keras

class InfoNCELoss(keras.losses.Loss):
    def __init__(self, n_qubits, n_ancillas, q_depth, q_params, temperature=0.1, epsilon=1e-4, negative_mode='unpaired'):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature
        self.epsilon = epsilon
        self.negative_mode = negative_mode
        self.q_params = q_params
        self.q_depth = q_depth
        self.n_qubits = n_qubits
        self.n_ancillas = n_ancillas

    def call(self, query, positive_key, negative_keys):
        q_in_query = tf.tanh(query) * np.pi / 2.0
        q_in_pos = tf.tanh(positive_key) * np.pi / 2.0

        positive_logit = self.compute_similarity(q_in_query, q_in_pos)

        if self.negative_mode == 'unpaired':
            q_in_neg = tf.tanh(negative_keys) * np.pi / 2.0
            negative_logits = self.compute_similarity(q_in_query, q_in_neg)
        else:
            q_in_neg = tf.tanh(negative_keys) * np.pi / 2.0
            negative_logits = self.compute_similarity(q_in_query, q_in_neg, mode='paired')

        logits = tf.concat([positive_logit, negative_logits], axis=1)
        labels = tf.zeros(len(logits), dtype=tf.int64)
        return tf.keras.losses.sparse_categorical_crossentropy(labels, logits / self.temperature, from_logits=True)

    def compute_similarity(self, query, key, mode='paired'):
        if mode == 'paired':
            similarity = []
            for h1, h2 in zip(query, key):
                q_pair = tf.concat([h1, h2], axis=0)
                sim = quantum_circuit(q_pair, self.q_params, self.q_depth, self.n_qubits, self.n_ancillas, training=True)
                similarity.append(tf.reduce_sum(sim)**2)
            return tf.stack(similarity, axis=0)[:, tf.newaxis]
        else:
            similarity = []
            for h_query in query:
                row_aux = []
                for h_key in key:
                    q_pair = tf.concat([h_query, h_key], axis=0)
                    sim = quantum_circuit(q_pair, self.q_params, self.q_depth, self.n_qubits, self.n_ancillas, training=True)
                    row_aux.append(tf.reduce_sum(sim)**2)
                similarity.append(tf.stack(row_aux, axis=0))
            return tf.stack(similarity, axis=0)