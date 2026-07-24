import torch
import pennylane as qml
import numpy as np

# Define the device globally
num_qubits = 3
head_circuit_qubits = 1
dev = qml.device("default.qubit", wires=num_qubits)
devh = qml.device("default.qubit", wires=head_circuit_qubits)

class QuantumHead(torch.nn.Module):
    def __init__(self, input_size, n_qubits, device):
        super(QuantumHead, self).__init__()
        self.input_size = input_size
        self.n_qubits = n_qubits
        self.drc_layers = int(self.input_size / self.n_qubits)
        self.params_size = qml.StronglyEntanglingLayers.shape(n_layers=self.drc_layers, n_wires=self.n_qubits)
        self.device = device

        # Define parameters for the circuit
        # self.input_params = torch.nn.Parameter(torch.randn(self.drc_layers, self.n_qubits))
        # self.weights = torch.nn.Parameter(torch.randn(*self.params_size))

        # Define QNode for the circuit
        self.circuit_node = qml.QNode(self.circuit, self.device, interface="torch")

        self.add_module('quantum_head', qml.qnn.TorchLayer(self.circuit_node, {'input_params': (self.drc_layers, self.n_qubits), 'weights': self.params_size}, init_method = torch.nn.init.normal_))

    def circuit(self, inputs, input_params, weights):
        # Encoding of classical input values using RY gates
        drc_layers = input_params.shape[0]
        inputs = torch.reshape(inputs, (drc_layers, int(inputs.shape[0] / drc_layers)))
        
        for layer in range(drc_layers):
            for j in range(self.n_qubits):
                qml.RY(input_params[layer, j] * inputs[layer, j], wires=j)
            
            qml.StronglyEntanglingLayers(weights=weights[layer,:,:].reshape((1, weights.shape[1], weights.shape[2])),
                                         wires=range(self.n_qubits))

        return qml.state()

    def forward(self, x):
        # Apply the circuit using the explicitly defined QNode
        out = torch.vmap(self.get_submodule('quantum_head'))(x.type(torch.complex64))
        return out 


class QuantumConvolution(torch.nn.Module):
    def __init__(self, input_size, in_filters, out_filters, kernel_size, stride, num_qubits, device):
        super(QuantumConvolution, self).__init__()
        self.in_filters = in_filters
        self.out_filters = out_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.num_qubits = num_qubits
        self.iter = int(np.ceil(1 + (input_size[0] - self.kernel_size[0]) / self.stride))

        self.drc_layers = int((kernel_size[0] * kernel_size[1]) / num_qubits)

        self.params_per_kernel_shape = qml.StronglyEntanglingLayers.shape(n_layers=self.drc_layers, n_wires=num_qubits)
        self.device = device

        # Initialize quantum circuits for each filter
        for o_f in range(self.out_filters):
            for i_f in range(self.in_filters):
                self.add_module('torch_window_'+str(o_f)+str(i_f), qml.qnn.TorchLayer(qml.QNode(self.drc_circuit, self.device, interface="torch"), {'input_params': (self.drc_layers, num_qubits), 'weights':self.params_per_kernel_shape}, init_method = torch.nn.init.normal_))

    def drc_circuit(self, inputs, input_params, weights):
        drc_layers = input_params.shape[0]
        inputs = torch.reshape(inputs, (self.drc_layers, int(inputs.shape[0] / self.drc_layers)))
        
        for layer in range(drc_layers):
            for j in range(self.num_qubits):
                qml.RY(input_params[layer, j] * inputs[layer, j], wires=j)
            
            qml.StronglyEntanglingLayers(weights=weights[layer,:,:].reshape((1, weights.shape[1], weights.shape[2])), 
                                         wires=range(self.num_qubits))
        return qml.expval(qml.PauliZ(self.num_qubits - 1))

    def forward(self, inputs):
        output = torch.zeros((inputs.shape[0], self.out_filters, self.iter, self.iter))

        for o_f in range(self.out_filters):
          for i_f in range(self.in_filters):
            for l in range(self.iter):
              for b in range(self.iter):

                flattened_inputs_window = torch.nn.Flatten()(inputs[:, i_f, l*self.stride : l*self.stride + self.kernel_size[0], b*self.stride : b*self.stride + self.kernel_size[0]])
                out_i_f = torch.vmap(self.get_submodule('torch_window_'+str(o_f)+str(i_f)))(flattened_inputs_window)
                output[:,o_f,l,b] += out_i_f

        return output


class QuantumModel(torch.nn.Module):
    def __init__(self):
        super(QuantumModel, self).__init__()
        # Define the layers in the model using the refactored QuantumConvolution and QuantumHead classes
        self.layer_3 = QuantumConvolution(input_size=(18+1, 18+1), in_filters=1, out_filters=3, kernel_size=(3, 3), stride=2, num_qubits=num_qubits, device=dev)
        self.layer_4 = QuantumConvolution(input_size=(9, 9), in_filters=3, out_filters=1, kernel_size=(3, 3), stride=2, num_qubits=num_qubits, device=dev)
        self.layer_7 = torch.nn.Flatten()
        self.layer_8 = QuantumHead(input_size=16, n_qubits=head_circuit_qubits, device=devh)

    def forward(self, x):
        x = torch.nn.functional.pad(x, (0, 1, 0, 1), 'constant', value=0)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.layer_7(x)
        x = self.layer_8(x)
        x = torch.nn.functional.normalize(x)
        return x
