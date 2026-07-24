from tensorflow.keras.layers import Layer, Add, Activation, Concatenate
from qml_hep_lhc.layers.utils import normalize_padding, normalize_tuple, convolution_iters, get_count_of_qubits, get_num_in_symbols
from qml_hep_lhc.utils import _import_class
import numpy as np
from qml_hep_lhc.layers import TwoLayerPQC
from qml_hep_lhc.layers import NQubitPQC
import tensorflow as tf
import warnings


class QConv2D(Layer):
    """
    2D Quantum convolution layer (e.g. spatial convolution over images).
    This layer creates a convolution kernel that is convolved 
    with the layer input to produce a tensor of outputs. Finally,
    `activation` is applied to the outputs as well.
    """
    def __init__(
            self,
            filters=1,
            kernel_size=(3, 3),
            strides=(1, 1),
            n_qubits=1,
            n_layers=1,
            sparse=False,
            padding='valid',
            activation='tanh',
            cluster_state=False,
            fm_class='AngleMap',
            ansatz_class='Chen',
            observable=None,
            drc=False,
            name='QConv2D',
    ):

        super(QConv2D, self).__init__(name=name)

        # Filters
        if isinstance(filters, float):
            filters = int(filters)
        if filters is not None and filters <= 0:
            raise ValueError('Invalid value for argument `filters`. '
                             'Expected a strictly positive value. '
                             f'Received filters={filters}.')
        self.filters = filters

        # Num layers
        if isinstance(n_layers, float):
            n_layers = int(n_layers)
        if n_layers is not None and n_layers <= 0:
            raise ValueError('Invalid value for argument `n_layers`. '
                             'Expected a strictly positive value. '
                             f'Received n_layers={n_layers}.')
        self.n_layers = n_layers

        if ansatz_class != 'NQubit':
            if sparse:
                warnings.warn(
                    'Sparse mode is only supported for NQubit ansatz.')
            if n_qubits:
                warnings.warn('n_qubits is only used for NQubit ansatz.')

        if ansatz_class == 'NQubit':
            if fm_class:
                warnings.warn('fm_class is only used for TwoLayerPQC.')
            if drc:
                warnings.warn('drc is only used for TwoLayerPQC.')
            if ansatz_class:
                warnings.warn('ansatz_class is only used for TwoLayerPQC.')

        self.observable = observable
        self.kernel_size = normalize_tuple(kernel_size, 'kernel_size')
        self.strides = normalize_tuple(strides, 'strides')
        self.padding = normalize_padding(padding)
        self.activation = Activation(activation)
        self.cluster_state = cluster_state
        self.fm_class = fm_class
        self.ansatz_class = ansatz_class
        self.drc = drc
        self.n_qubits = n_qubits
        self.sparse = sparse

        if observable is None:
            self.len_obs = 1
        else:
            self.len_obs = len(observable)

    def build(self, input_shape):
        self.n_channels = input_shape[3]

        self.conv_pqcs = [[(filter, channel)
                           for channel in range(self.n_channels)]
                          for filter in range(self.filters)]

        if self.ansatz_class == 'NQubit':
            for filter in range(self.filters):
                for channel in range(self.n_channels):
                    name = f"{self.name}_{filter}_{channel}"
                    self.conv_pqcs[filter][channel] = NQubitPQC(
                        self.n_qubits, self.cluster_state, self.observable,
                        self.n_layers, self.sparse, name)
        else:
            self.n_qubits = get_count_of_qubits(self.fm_class,
                                                np.prod(self.kernel_size))
            self.n_inputs = get_num_in_symbols(self.fm_class,
                                               np.prod(self.kernel_size))

            self.feature_map = _import_class(
                f"qml_hep_lhc.encodings.{self.fm_class}")()
            self.ansatz = _import_class(
                f"qml_hep_lhc.ansatzes.{self.ansatz_class}")()

            for filter in range(self.filters):
                for channel in range(self.n_channels):
                    name = f"{self.name}_{filter}_{channel}"
                    self.conv_pqcs[filter][channel] = TwoLayerPQC(
                        self.n_qubits, self.n_inputs, self.feature_map,
                        self.ansatz, self.cluster_state, self.observable,
                        self.n_layers, self.drc, name)

    def _convolution(self, x, filter, channel):
        return self.conv_pqcs[filter][channel](x)

    def call(self, x):
        if len(x.shape) == 4:
            x = tf.expand_dims(x, -1)  # NHWCD
        depth = x.shape[-1]
        x = tf.transpose(x, [0, 3, 1, 2, 4])  # NCHWD
        x = tf.extract_volume_patches(
            x,
            ksizes=(1, 1, self.kernel_size[0], self.kernel_size[1], 1),
            strides=(1, 1, self.strides[0], self.strides[1], 1),
            padding=self.padding.upper(),
        )
        x = tf.transpose(x, [1, 0, 2, 3, 4])  # CNHWD
        channels, _, h, w, _ = x.shape
        x = tf.reshape(x, [channels, -1, np.prod(self.kernel_size)])

        if channels == 1:
            conv_out = [
                self._convolution(x[0, :, :], filter, 0)
                for filter in range(self.filters)
            ]
        else:
            conv_out = [
                Add()([
                    self._convolution(x[c, :, :], filter, c)
                    for c in range(self.n_channels)
                ]) for filter in range(self.filters)
            ]
        conv_out = Concatenate(axis=-1)(conv_out)
        conv_out = tf.reshape(conv_out,
                              [-1, h, w, self.filters, depth * self.len_obs])
        return self.activation(conv_out)
