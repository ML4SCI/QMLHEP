from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Flatten, Dense, Dropout
from qml_hep_lhc.models import QCNN
from qml_hep_lhc.layers import QConv2D


class QCNNHybrid(QCNN):
    """
	General Quantum Convolutional Neural Network
	"""
    def __init__(self, data_config, args=None):
        super(QCNNHybrid, self).__init__(data_config, args)
        self.args = vars(args) if args is not None else {}

        self.qconv2d_1 = QConv2D(
            filters=1,
            kernel_size=3,
            strides=1,
            n_qubits=self.n_qubits,
            n_layers=self.n_layers,
            sparse=self.sparse,
            padding="same",
            cluster_state=self.cluster_state,
            fm_class=self.fm_class,
            ansatz_class=self.ansatz_class,
            drc=self.drc,
            name='qconv2d_1',
        )

        self.qconv2d_2 = QConv2D(
            filters=1,
            kernel_size=2,
            strides=1,
            n_qubits=self.n_qubits,
            n_layers=self.n_layers,
            sparse=self.sparse,
            padding="same",
            cluster_state=self.cluster_state,
            fm_class=self.fm_class,
            ansatz_class=self.ansatz_class,
            drc=self.drc,
            name='qconv2d_2',
        )

        self.dense1 = Dense(8, activation='elu')
        self.dropout = Dropout(0.25)
        self.dense2 = Dense(2, activation='softmax')

    def call(self, input_tensor):
        x = self.qconv2d_1(input_tensor)
        x = self.qconv2d_2(x)
        x = Flatten()(x)
        x = self.dense1(x)
        x = self.dropout(x)
        x = self.dense2(x)
        return x

    def build_graph(self):
        x = Input(shape=self.input_dim)
        return Model(inputs=[x],
                     outputs=self.call(x),
                     name=f"QCNNHybrid-{self.fm_class}-{self.ansatz_class}")
