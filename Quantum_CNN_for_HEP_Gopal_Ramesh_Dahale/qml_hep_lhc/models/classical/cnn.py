from qml_hep_lhc.models.base_model import BaseModel
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input
from tensorflow.keras import Model
from qml_hep_lhc.utils import ParseAction


class CNN(BaseModel):
    def __init__(self, data_config, args=None):
        super(CNN, self).__init__(args)
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

        # Data config
        self.input_dim = data_config["input_dims"]
        self.num_classes = len(data_config["mapping"])

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

        # self.pooling = MaxPool2D(pool_size=(2, 2))
        self.flatten = Flatten()
        # self.dropout = Dropout(0.25)

    def call(self, input_tensor):
        x = input_tensor
        for conv in self.convs:
            x = conv(x)

        # x = self.pooling(x)
        # x = self.dropout(x)
        x = self.flatten(x)
        for fc in self.fcs:
            x = fc(x)

        return x

    def build_graph(self):
        x = Input(shape=self.input_dim)
        return Model(inputs=[x], outputs=self.call(x), name="CNN")

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument('--num-conv-layers', type=int, default=1)
        parser.add_argument('--conv-dims', action=ParseAction, default=[64])
        parser.add_argument('--num-fc-layers', type=int, default=1)
        parser.add_argument('--fc-dims', action=ParseAction, default=[128])
        return parser
