from qml_hep_lhc.models.base_model import BaseModel
from tensorflow.keras.layers import Dense, Flatten, Dropout, Input
from tensorflow.keras import Model
from qml_hep_lhc.utils import ParseAction


class MLP(BaseModel):
    def __init__(self, data_config, args=None):
        super(MLP, self).__init__(args)
        self.args = vars(args) if args is not None else {}

        # Moel Configuration
        self.num_fc_layers = self.args.get('num_fc_layers', 1)
        self.fc_dims = self.args.get('fc_dims', [128])

        assert len(
            self.fc_dims
        ) == self.num_fc_layers, 'fc_dims must be a list of length num_fc_layers'

        # Data config
        self.input_dim = data_config["input_dims"]
        self.num_classes = len(data_config["mapping"])

        self.fcs = []
        for units in self.fc_dims:
            self.fcs.append(Dense(units, activation='relu'))

        self.fcs.append(Dense(self.num_classes, activation='softmax'))

        self.flatten = Flatten(input_shape=self.input_dim)
        self.dropout = Dropout(0.5)

    def call(self, input_tensor):
        x = input_tensor
        x = self.flatten(x)
        for fc in self.fcs[:-1]:
            x = fc(x)
            x = self.dropout(x)
        x = self.fcs[-1](x)
        return x

    def build_graph(self):
        x = Input(shape=self.input_dim)
        return Model(inputs=[x], outputs=self.call(x), name="MLP")

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument('--num-fc-layers', type=int, default=1)
        parser.add_argument('--fc-dims', action=ParseAction, default=[128])
        return parser
