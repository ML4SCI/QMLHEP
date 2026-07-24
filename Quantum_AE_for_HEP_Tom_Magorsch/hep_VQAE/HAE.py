import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
import pennylane as qml

class HybridAE(Model):

    def __init__(self, latent_dim, input_dim, q_input_dim, DRCs, kernel_size, stride, device, diff_method="adjoint"):

        super().__init__()

        tf.keras.backend.set_floatx('float64')

        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.q_input_dim = q_input_dim
        self.dev = device
        self.DRCs = DRCs
        self.kernel_size = kernel_size
        self.stride = stride
        self.circuit_node = qml.QNode(self.circuit, device, diff_method=diff_method)
        self.number_of_kernel_uploads = len(list(range(0, input_dim - kernel_size + 1, stride)))**2
        self.num_upload_params = self.number_of_kernel_uploads * 2 * kernel_size ** 2

        self.weight_shapes = {"weights": (DRCs*self.num_upload_params,)}

        self.qlayer = qml.qnn.KerasLayer(self.circuit_node, self.weight_shapes, output_dim=latent_dim)

        self.build_model()



    def call(self, x):
        return self.model(x)

    def build_model(self):
        """
        Build the keras model with convolutional and quantum layer. This is an example, any other choise of classical layers works.
        """
        inputs = tf.keras.layers.Input(shape=(self.input_dim, self.input_dim, 1))

        convlayer_1 = tf.keras.layers.Conv2D(12, 4, strides=1, padding='valid', activation="relu")
        convlayer_2 = tf.keras.layers.Conv2D(10, 2, strides=1, padding='same', activation="relu")
        convlayer_3 = tf.keras.layers.Conv2D(1, 2, strides=1, padding='same', activation="relu")

        # currently input to keras layer needs to be 1D thats why we have to flatten first
        dress_1 = tf.keras.layers.Flatten()

        dress_2 = tf.keras.layers.Dense(81)
        reshape_1 = tf.keras.layers.Reshape((9, 9, 1))
        convlayer_4 = tf.keras.layers.Conv2DTranspose(12, 4, strides=1, padding='valid', activation="relu")
        out_layer = tf.keras.layers.Conv2D(1, 2, strides=1, padding='same', activation="sigmoid")


        self.model = tf.keras.models.Sequential([inputs,
                                                 convlayer_1,
                                                 convlayer_2,
                                                 convlayer_3,
                                                 dress_1,
                                                 self.qlayer,
                                                 dress_2,
                                                 reshape_1,
                                                 convlayer_4,
                                                 out_layer])

    def single_upload(self, params, data, wire):
        for i, d in enumerate(tf.reshape(data, [-1])):
            if i % 3 == 0:
                qml.RZ(params[i * 2] + params[i * 2 + 1] * d, wires=wire)
            if i % 3 == 1:
                qml.RY(params[i * 2] + params[i * 2 + 1] * d, wires=wire)
            if i % 3 == 2:
                qml.RZ(params[i * 2] + params[i * 2 + 1] * d, wires=wire)

    def conv_upload(self, params, img, wires):
        number_of_kernel_uploads = len(list(range(0,self.q_input_dim-self.kernel_size+1,self.stride)))*len(list(range(0,self.q_input_dim-self.kernel_size+1,self.stride)))
        params_per_upload = len(params) // number_of_kernel_uploads
        upload_counter = 0
        wire = 0
        for y in range(0,self.q_input_dim-self.kernel_size+1,self.stride):
            for x in range(0,self.q_input_dim-self.kernel_size+1,self.stride):
                self.single_upload(params[upload_counter * params_per_upload: (upload_counter + 1) * params_per_upload],
                                   img[y:y+self.kernel_size, x:x+self.kernel_size], wires[wire])
                upload_counter = upload_counter + 1
                wire = wire + 1

    def circular_entanglement(self, wires):
        qml.CNOT(wires=[wires[-1], 0])
        for i in range(len(wires)-1):
            qml.CNOT(wires=[i, i+1])

    def circuit(self, inputs, weights):

        # inputs need to be 1D tensor but we can reshape now
        inputs = tf.reshape(inputs, [self.q_input_dim, self.q_input_dim])

        number_of_kernel_uploads = len(list(range(0,self.q_input_dim-self.kernel_size+1,self.stride)))**2
        num_upload_params = number_of_kernel_uploads*2*self.kernel_size**2

        for i in range(self.DRCs):
            self.conv_upload(weights[i*num_upload_params:(i+1)*num_upload_params], inputs, list(range(number_of_kernel_uploads)))
            self.circular_entanglement(list(range(number_of_kernel_uploads)))

        return [qml.expval(qml.PauliZ(i)) for i in range(self.latent_dim)]

    def plot_circuit(self):
        data = np.random.rand(self.q_input_dim*self.q_input_dim)

        fig, ax = qml.draw_mpl(self.circuit_node)(data, np.ones(self.DRCs*self.num_upload_params))
        fig.show()
