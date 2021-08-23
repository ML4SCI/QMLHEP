# MIT License
#
# Copyright (c) 2021 Eraraya Ricardo Muten
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import math
import numpy as np
import tensorflow as tf
from qcnn_drc.data_reuploading import ReUploadingPQC


class QConv2D_DRC:
    def __init__(self, filters, kernel_size, strides, drc_hyperparameters, layer_id, padding=None):
        # Parameter initialization
        self.filters = filters
        self.layer_id = layer_id
        self.padding = padding

        self.n_qubits = drc_hyperparameters["n_qubits"]
        self.n_layers = drc_hyperparameters["n_layers"]

        if any(np.array(self.n_qubits) != 1):
            self.use_ent = drc_hyperparameters["use_ent"]
            self.use_terminal_ent = drc_hyperparameters["use_terminal_ent"]
        else:
            # Dummy parameters for compatibility
            # (won't be used since no circuit has more than 1 qubit)
            self.use_ent = list(np.ones(len(self.n_qubits), dtype=bool))
            self.use_terminal_ent = list(np.ones(len(self.n_qubits), dtype=bool))

        if type(kernel_size) == int:
            self.h_kernel_size = kernel_size
            self.w_kernel_size = kernel_size
        else:
            self.h_kernel_size = kernel_size[0]
            self.w_kernel_size = kernel_size[1]

        if type(strides) == int:
            self.h_stride = strides
            self.w_stride = strides
        else:
            self.h_stride = strides[0]
            self.w_stride = strides[1]

    def qconv_drc_operation(self, h_iter, w_iter, n_qubits, n_layers, use_entanglement, use_terminal_entanglement,
                            inputs, filter_id, channel_id, name="QConv_Operation"):

        # Quantum circuit as the convolutional filter
        pqc = ReUploadingPQC(n_qubits, n_layers, self.h_kernel_size * self.w_kernel_size,
                             use_entanglement=use_entanglement, use_terminal_entanglement=use_terminal_entanglement,
                             name=name + '_' + str(self.layer_id) + '_' + str(filter_id) + '_' + str(channel_id))

        # Do the convolutional operation
        conv = []
        for i in range(h_iter):
            for j in range(w_iter):
                temp = pqc(inputs[:, self.h_stride * i:self.h_stride * i + self.h_kernel_size,
                           self.w_stride * j:self.w_stride * j + self.w_kernel_size])
                conv += [temp]
        output = tf.keras.layers.Concatenate(axis=1)(conv)
        output = tf.keras.layers.Reshape((h_iter, w_iter, 1))(output)

        return output

    def call(self, inputs):
        # Calculate the number of convolution iterations
        h_iter = 1 + (inputs.shape[1] - self.h_kernel_size) / self.h_stride
        w_iter = 1 + (inputs.shape[2] - self.w_kernel_size) / self.w_stride

        if self.padding is None:
            h_iter = int(h_iter)
            w_iter = int(w_iter)

        if self.padding == "valid":
            h_iter = math.ceil(h_iter)
            w_iter = math.ceil(w_iter)
            h_pad_size = (h_iter - 1) * self.h_stride + self.h_kernel_size - inputs.shape[1]
            w_pad_size = (w_iter - 1) * self.w_stride + self.w_kernel_size - inputs.shape[2]
            # Pad the inputs
            padding_constant = tf.constant([[0, 0], [0, h_pad_size], [0, w_pad_size], [0, 0]])
            inputs = tf.pad(inputs, padding_constant)

        # Do the convolutional operation for all filters
        # Filter iterations
        filter_output = []
        for filter in range(self.filters):
            # input channel iteration
            channel_outputs = []
            for channel in range(inputs.shape[-1]):
                channel_outputs += [
                    self.qconv_drc_operation(h_iter, w_iter,
                                             self.n_qubits[channel],
                                             self.n_layers[channel],
                                             self.use_ent[channel],
                                             self.use_terminal_ent[channel],
                                             inputs[:, :, :, channel],
                                             filter_id=filter + 1, channel_id=channel + 1)
                ]

            # Add the convolution outputs between input channels
            if inputs.shape[-1] > 1:
                filter_output += [tf.keras.layers.Add()(channel_outputs)]
            else:
                filter_output += channel_outputs

        # Concatenate the filter outputs
        if self.filters > 1:
            layer_output = tf.keras.layers.Concatenate(axis=-1)(filter_output)
        else:
            layer_output = filter_output[0]

        return layer_output
