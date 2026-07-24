from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Flatten
from qml_hep_lhc.models import QCNN
from qml_hep_lhc.layers import QConv2D, TwoLayerPQC, NQubitPQC
import numpy as np
from qml_hep_lhc.layers.utils import get_count_of_qubits, get_num_in_symbols
from qml_hep_lhc.utils import _import_class


class FQCNN(QCNN):
    """
	General Quantum Convolutional Neural Network
	"""
    def __init__(self, data_config, args=None):
        super(FQCNN, self).__init__(data_config, args)
        self.args = vars(args) if args is not None else {}

        input_shape = [None] + list(self.input_dim)

        self.num_qconv_layers = self.args.get('num_qconv_layers', 1)
        self.qconv_dims = self.args.get('qconv_dims', [1])

        assert len(
            self.qconv_dims
        ) == self.num_qconv_layers, 'qconv_dims must be a list of length num_qconv_layers'

        self.qconvs = []
        for i, num_filters in enumerate(self.qconv_dims):
            self.qconvs.append(
                QConv2D(
                    filters=1,
                    kernel_size=3,
                    strides=1,
                    n_layers=self.n_layers,
                    padding="valid",
                    cluster_state=self.cluster_state,
                    fm_class=self.fm_class,
                    ansatz_class=self.ansatz_class,
                    drc=self.drc,
                    name=f'qconv2d_{i}',
                ))
            input_shape = self.qconvs[-1].compute_output_shape(input_shape)

        if self.ansatz_class == 'NQubit':
            self.vqc = NQubitPQC(
                n_qubits=self.n_qubits,
                cluster_state=self.cluster_state,
                n_layers=self.n_layers,
                sparse=self.sparse,
            )
        else:
            n_qubits = get_count_of_qubits(self.fm_class,
                                           np.prod(input_shape[1:]))
            n_inputs = get_num_in_symbols(self.fm_class,
                                          np.prod(input_shape[1:]))

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
        x = input_tensor
        for qconv in self.qconvs:
            x = qconv(x)
        x = Flatten()(x)
        x = self.vqc(x)
        return x

    def build_graph(self):
        x = Input(shape=self.input_dim)
        return Model(inputs=[x],
                     outputs=self.call(x),
                     name=f"FQCNN-{self.fm_class}-{self.ansatz_class}")
