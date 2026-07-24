from qml_hep_lhc.models.classical.resnet.bottleneck import BottleneckResidual
from tensorflow.keras.layers import Dense, Activation, AveragePooling2D, Flatten, Input, add, BatchNormalization
from qml_hep_lhc.models.base_model import BaseModel
from tensorflow.keras import Model


class ResnetV2(BaseModel):
    """
    Resent v2 model. Paper: https://arxiv.org/abs/1603.05027
    This implementation is based on https://www.geeksforgeeks.org/residual-networks-resnet-deep-learning/
    """
    def __init__(self, data_config, args=None):
        super(ResnetV2, self).__init__(args)
        self.args = vars(args) if args is not None else {}

        # Model configuration
        self.depth = self.args.get("resnet_depth", 56)
        if (self.depth - 2) % 9 != 0:
            raise ValueError('depth should be 9n + 2 (eg 56 or 110 in [b])')

        self.num_res_blocks = int((self.depth - 2) / 9)

        # Data config
        self.input_dim = data_config["input_dims"]
        self.num_classes = len(data_config["mapping"])

        # Layers
        num_filters_in = 16
        self.res_block1 = BottleneckResidual(num_filters=num_filters_in,
                                             conv_first=True)
        self.res_blocks = []

        for stage in range(3):
            for res_block in range(self.num_res_blocks):
                block = []
                activation = 'relu'
                batch_normalization = True
                strides = 1

                if stage == 0:
                    num_filters_out = num_filters_in * 4
                    if res_block == 0:  # first layer and first stage
                        activation = None
                        batch_normalization = False
                else:
                    num_filters_out = num_filters_in * 2
                    if res_block == 0:  # first layer but not first stage
                        strides = 2  # downsample

                block.append(
                    BottleneckResidual(num_filters=num_filters_in,
                                       kernel_size=1,
                                       strides=strides,
                                       activation=activation,
                                       batch_normalization=batch_normalization,
                                       conv_first=False))
                block.append(
                    BottleneckResidual(num_filters=num_filters_in,
                                       conv_first=False))
                block.append(
                    BottleneckResidual(num_filters=num_filters_out,
                                       kernel_size=1,
                                       conv_first=False))

                if res_block == 0:
                    block.append(
                        BottleneckResidual(num_filters=num_filters_out,
                                           kernel_size=1,
                                           strides=strides,
                                           activation=None,
                                           batch_normalization=False))

                self.res_blocks.append(block.copy())
                del block

            num_filters_in = num_filters_out

        self.batch_norm = BatchNormalization()
        self.activation_layer = Activation('relu')
        self.pooling = AveragePooling2D(pool_size=(8, 8))
        self.flatten = Flatten()
        self.dense = Dense(self.num_classes,
                           activation='softmax',
                           kernel_initializer='he_normal')

    def call(self, input_tensor):
        """
        The function takes in an input tensor and returns the output tensor
        
        Args:
          input_tensor: The input tensor to the network.
        
        Returns:
          The output of the last layer of the model.
        """
        num_filters_in = 16
        x = self.res_block1(input_tensor)

        for stage in range(3):

            for res_block in range(self.num_res_blocks):

                if stage == 0:
                    num_filters_out = num_filters_in * 4
                else:
                    num_filters_out = num_filters_in * 2
                y = self.res_blocks[stage * self.num_res_blocks +
                                    res_block][0](x)

                y = self.res_blocks[stage * self.num_res_blocks +
                                    res_block][1](y)

                y = self.res_blocks[stage * self.num_res_blocks +
                                    res_block][2](y)

                if res_block == 0:
                    x = self.res_blocks[stage * self.num_res_blocks +
                                        res_block][3](x)

                x = add([x, y])
            num_filters_in = num_filters_out
        x = self.batch_norm(x)
        x = self.activation_layer(x)
        x = self.pooling(x)
        y = self.flatten(x)
        y = self.dense(y)
        return y

    def build_graph(self):
        x = Input(shape=self.input_dim)
        return Model(inputs=[x], outputs=self.call(x), name="ResnetV2")

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--resnet-depth", "-rd", type=int, default=56)
        return parser
