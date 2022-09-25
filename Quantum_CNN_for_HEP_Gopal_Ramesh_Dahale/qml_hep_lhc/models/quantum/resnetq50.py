from tensorflow.keras.applications import ResNet50
from qml_hep_lhc.models import QCNN
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout
from tensorflow.keras import Model
import tensorflow as tf
import numpy as np
from qml_hep_lhc.layers.utils import get_count_of_qubits, get_num_in_symbols
from qml_hep_lhc.utils import _import_class
from qml_hep_lhc.layers import TwoLayerPQC


class ResnetQ50(QCNN):
    def __init__(self, data_config, args=None):
        super(ResnetQ50, self).__init__(data_config, args)
        self.args = vars(args) if args is not None else {}

        input_shape = [None] + list(self.input_dim)

        self.base_model = ResNet50(include_top=False,
                                   weights='imagenet',
                                   input_shape=(self.input_dim))

        self.base_model.trainable = False
        input_shape = self.base_model.compute_output_shape(input_shape)

        self.flatten = Flatten()
        input_shape = self.flatten.compute_output_shape(input_shape)

        self.dropout = Dropout(0.25)
        self.dense1 = Dense(512, activation='relu')
        input_shape = self.dense1.compute_output_shape(input_shape)

        self.dense2 = Dense(16, activation='relu')
        input_shape = self.dense2.compute_output_shape(input_shape)

        n_qubits = get_count_of_qubits(self.fm_class, np.prod(input_shape[1:]))
        n_inputs = get_num_in_symbols(self.fm_class, np.prod(input_shape[1:]))

        feature_map = _import_class(f"qml_hep_lhc.encodings.{self.fm_class}")()
        ansatz = _import_class(f"qml_hep_lhc.ansatzes.{self.ansatz_class}")()

        self.vqc = TwoLayerPQC(
            n_qubits,
            n_inputs,
            feature_map,
            ansatz,
            self.cluster_state,
            None,
            self.n_layers,
            self.drc,
        )

    def call(self, input_tensor):
        x = self.base_model(input_tensor)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return self.vqc(x)

    def build_graph(self):
        x = Input(shape=self.input_dim)
        return Model(inputs=[x], outputs=self.call(x), name="ResnetQ50")
