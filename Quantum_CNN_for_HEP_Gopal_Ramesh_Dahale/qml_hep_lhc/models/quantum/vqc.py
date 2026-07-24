from email.policy import default
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Flatten, Dense, MaxPool2D
from qml_hep_lhc.models import QCNN
from qml_hep_lhc.layers import TwoLayerPQC, NQubitPQC
from qml_hep_lhc.layers.utils import get_count_of_qubits, get_num_in_symbols
import numpy as np
from qml_hep_lhc.utils import _import_class


class VQC(QCNN):
    """
	General Quantum Convolutional Neural Network
	"""
    def __init__(self, data_config, args=None):
        super(VQC, self).__init__(data_config, args)
        self.args = vars(args) if args is not None else {}

        if self.ansatz_class == 'NQubit':
            self.vqc = NQubitPQC(
                n_qubits=self.n_qubits,
                cluster_state=self.cluster_state,
                n_layers=self.n_layers,
                sparse=self.sparse,
            )
        else:
            n_qubits = get_count_of_qubits(self.fm_class,
                                           np.prod(self.input_dim))
            n_inputs = get_num_in_symbols(self.fm_class,
                                          np.prod(self.input_dim))

            feature_map = _import_class(
                f"qml_hep_lhc.encodings.{self.fm_class}")()
            ansatz = _import_class(
                f"qml_hep_lhc.ansatzes.{self.ansatz_class}")()

            self.vqc = TwoLayerPQC(
                n_qubits=n_qubits,
                n_inputs=n_inputs,
                feature_map=feature_map,
                ansatz=ansatz,
                cluster_state=self.cluster_state,
                n_layers=self.n_layers,
                drc=self.drc,
            )

    def call(self, input_tensor):
        return self.vqc(input_tensor)

    def build_graph(self):
        x = Input(shape=self.input_dim)
        return Model(inputs=[x],
                     outputs=self.call(x),
                     name=f"VQC-{self.fm_class}-{self.ansatz_class}")

    @staticmethod
    def add_to_argparse(parser):
        return parser
