from tensorflow.keras.applications import ResNet50
from qml_hep_lhc.models.base_model import BaseModel
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout
from tensorflow.keras import Model
import tensorflow as tf


class Resnet50(BaseModel):
    def __init__(self, data_config, args=None):
        super(Resnet50, self).__init__(args)
        self.args = vars(args) if args is not None else {}

        # Data config
        self.input_dim = data_config["input_dims"]
        self.num_classes = len(data_config["mapping"])
        self.base_model = ResNet50(include_top=False,
                                   weights='imagenet',
                                   input_shape=(self.input_dim))
        self.base_model.trainable = False

        self.flatten = Flatten()
        self.droput = Dropout(0.25)
        self.dense1 = Dense(512, activation='relu')
        self.dense2 = Dense(128, activation='relu')
        self.dense3 = Dense(self.num_classes, activation='softmax')

    def call(self, input_tensor):
        x = self.base_model(input_tensor)
        x = self.flatten(x)
        x = self.droput(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return x

    def build_graph(self):
        x = Input(shape=self.input_dim)
        return Model(inputs=[x], outputs=self.call(x), name="Resnet50")

    @staticmethod
    def add_to_argparse(parser):
        return parser
