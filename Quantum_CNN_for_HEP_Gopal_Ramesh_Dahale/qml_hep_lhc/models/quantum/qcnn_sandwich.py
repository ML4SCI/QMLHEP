from email.policy import default
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Flatten, Dense, BatchNormalization, Conv2D
from qml_hep_lhc.models import QCNN
from qml_hep_lhc.layers import QConv2D
from qml_hep_lhc.utils import ParseAction


class QCNNSandwich(QCNN):
    """
	General Quantum Convolutional Neural Network
	"""
    def __init__(self, data_config, args=None):
        super(QCNNSandwich, self).__init__(data_config, args)
        self.args = vars(args) if args is not None else {}

        # Model Configuration
        self.num_conv_layers = self.args.get('num_conv_layers', 1)
        self.conv_dims = self.args.get('conv_dims', [64])

        assert len(
            self.conv_dims
        ) == self.num_conv_layers, 'conv_dims must be a list of length num_conv_layers'

        self.num_fc_layers = self.args.get('num_fc_layers', 1)
        self.fc_dims = self.args.get('fc_dims', [128])

        assert len(
            self.fc_dims
        ) == self.num_fc_layers, 'fc_dims must be a list of length num_fc_layers'

        self.convs = [
            Conv2D(self.conv_dims[0], (3, 3), (1, 1),
                   'same',
                   activation='relu',
                   input_shape=self.input_dim)
        ]

        for num_filters in self.conv_dims[1:]:
            self.convs.append(
                Conv2D(num_filters, (3, 3), (1, 1), 'same', activation='relu'))

        self.fcs = []
        for units in self.fc_dims:
            self.fcs.append(Dense(units, activation='relu'))

        self.fcs.append(Dense(self.num_classes, activation='softmax'))

        self.num_qconv_layers = self.args.get('num_qconv_layers', 1)
        self.qconv_dims = self.args.get('qconv_dims', [1])

        assert len(
            self.qconv_dims
        ) == self.num_qconv_layers, 'qconv_dims must be a list of length num_qconv_layers'

        self.batch_norm = BatchNormalization()

        self.qconvs = []
        for i, num_filters in enumerate(self.qconv_dims):
            self.qconvs.append(
                QConv2D(
                    filters=1,
                    kernel_size=3,
                    strides=1,
                    n_layers=self.n_layers,
                    padding="same",
                    cluster_state=self.cluster_state,
                    fm_class=self.fm_class,
                    ansatz_class=self.ansatz_class,
                    drc=self.drc,
                    name=f'qconv2d_{i}',
                ))

    def call(self, input_tensor):
        x = input_tensor
        for conv in self.convs:
            x = conv(x)
        x = self.batch_norm(x)
        for qconv in self.qconvs:
            x = qconv(x)
        x = Flatten()(x)
        for fc in self.fcs:
            x = fc(x)
        return x

    def build_graph(self):
        x = Input(shape=self.input_dim)
        return Model(inputs=[x],
                     outputs=self.call(x),
                     name=f"QCNNSandwich-{self.fm_class}-{self.ansatz_class}")

    @staticmethod
    def add_to_argparse(parser):
        return parser
