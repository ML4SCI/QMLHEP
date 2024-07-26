"""
Quantum autoencoder implementations using pennylane
"""
import pennylane as qml
import torch
import qutip

class ConvEncoderCircuit:
    def __init__(self, input_qbits, latent_qbits, device, img_dim, kernel_size, stride, DRCs, diff_method="best"):
        """Create basic SQAE

        Args:
            input_qbits (int): number of qbits to upload input and use as encoder
            latent_qbits (int): number of latent qbits
            device (pennylane device): pennylane device to use for circuit evaluation
            img_dim (int): dimension of the images (width)
            kernel_size (int): size of the kernel to use when uploading the input
            stride (int): stride to use when uploading the input
            DRCs (int): number of times to repeat the encoding upload circuit in the encoder
            diff_method (str): method to differentiate quantum circuit, usually "adjoint" is best.
        """

        self.dev = device
        self.input_qbits = input_qbits
        self.latent_qbits = latent_qbits
        self.trash_qbits = self.input_qbits - self.latent_qbits
        self.total_qbits = input_qbits + self.trash_qbits + 1
        self.circuit_node = qml.QNode(self.circuit, device, interface="torch", diff_method=diff_method)
        self.latent_node = qml.QNode(self.visualize_latent_circuit, device, interface="torch")

        self.auc_hist = []
        self.animation_filenames = []
        self.bloch_animation_filenames = []

        # self.parameters_shape = (2 * input_qbits,)
        # self.input_shape = (input_qbits,)

        self.kernel_size = kernel_size
        self.stride = stride
        self.DRCs = DRCs

        self.input_shape = (img_dim, img_dim)
        self.number_of_kernel_uploads = len(list(range(0, img_dim - kernel_size + 1, stride)))**2
        self.parameters_shape = (DRCs * 2 * self.number_of_kernel_uploads * kernel_size ** 2,)
        self.params_per_layer = self.parameters_shape[0] // DRCs


    def _conv_upload(self, params, img, kernel_size, stride, wires):
        """Upload image using the convolution like method

        Args:
            params (list): parameters to use
            img (2d list): the actual image
            kernel size (int): kernel size for upload
            stride (int): stride for upload
            wires (list): list of integers to use as qbit index for upload
        """
        def single_upload(params, data, wire):
            """Upload data on a single qbit

            Args:
                params (list): parameters to use for upload, must be twice as long as data
                data (list): data to upload
                wire (int): on which wire to upload
            """
            data_flat = data.flatten()
            for i, d in enumerate(data_flat):
                if i % 3 == 0:
                    qml.RZ(params[i * 2] + params[i * 2 + 1] * d, wires=wire)
                elif i % 3 == 1:
                    qml.RY(params[i * 2] + params[i * 2 + 1] * d, wires=wire)
                elif i % 3 == 2:
                    qml.RZ(params[i * 2] + params[i * 2 + 1] * d, wires=wire)
        
        # number_of_kernel_uploads = len(list(range(0, img.shape[1]-kernel_size+1, stride))) * len(list(range(0, img.shape[0]-kernel_size+1, stride)))
        number_of_kernel_uploads = 9 ## Error in calculate number_of_kernel_uploads
        params_per_upload = len(params) // number_of_kernel_uploads

        upload_counter = 0
        wire = 0
        # print("HERE")
        # print(img.shape, kernel_size, stride)
        for y in range(0, img.shape[1]-kernel_size+1, stride):
            for x in range(0, img.shape[0]-kernel_size+1, stride):
                # print(y, y+kernel_size, x, x+kernel_size)
                # print("imgae slide", img[y:y+kernel_size, x:x+kernel_size])
                # print( wires[wire])
                single_upload(params[upload_counter * params_per_upload: (upload_counter + 1) * params_per_upload],
                                   img[x:x+kernel_size, y:y+kernel_size], wires[wire])
                upload_counter = upload_counter + 1
                wire = wire + 1

    def encoder(self, params, inputs):
        """The encoder circuit for the SQAE

        Args:
            params (list): parameters to use
            inputs (list): inputs to upload
        """
        def circular_entanglement(wires):
            qml.CNOT(wires=[wires[-1], 0])
            for i in range(len(wires)-1):
                qml.CNOT(wires=[i, i+1])

        for i in range(self.DRCs):
            self._conv_upload(
                params[i * self.params_per_layer:(i + 1) * self.params_per_layer], 
                inputs, self.kernel_size, self.stride, 
                list(range(self.number_of_kernel_uploads))
                )
            circular_entanglement(list(range(self.number_of_kernel_uploads)))

    def circuit(self, params, inputs):
        """Full circuit to be used as SQAE

        includes encoder and SWAP test

        Args:
            params (list): parameters to be used for PQC
            inputs (list): inputs for the circuit

        Returns:
            expectation value of readout bit

        """
        # print("2 circuit", inputs.shape)
        self.encoder(params, inputs)

        # swap test
        qml.Hadamard(wires=self.total_qbits-1)
        for i in range(self.trash_qbits):
            qml.CSWAP(wires=[self.total_qbits - 1, self.latent_qbits + i, self.input_qbits + i])
        qml.Hadamard(wires=self.total_qbits-1)

        return qml.expval(qml.PauliZ(self.total_qbits-1))

    def visualize_latent_circuit(self, params, inputs, bit, measure_gate):
        """encoder circuit with measurement on chosen qbit

        can be used to measure arbitrary qbit to visualize bloch sphere etc..

        Args:
            params (list): parameters to be used for PQC
            inputs (list): inputs for the circuit
            bit (int): qbit index to be measured
            measure_gate (function): function of gate to be measured (qml.PauliX, qml.PauliY, qml.PauliZ)

        Returns:
            expectation value of chosen bit

        """
        self.encoder(params, inputs)

        return qml.expval(measure_gate(bit))

    def plot_circuit(self):
        """Plots the circuit with dummy inputs
        """

        inputs = torch.rand(self.input_shape)
        params = torch.rand(self.parameters_shape)
        fig, _ = qml.draw_mpl(self.circuit_node)(params, inputs)
        fig.show()
        
    
    # def plot_latent_space(self, latent_bit, x_test_bg, x_test_signal, save_fig=None):
    #     """Plots the bloch sphere of a chosen qbit for given signal and background inputs

    #     Args:
    #         latent_bit (int): bit to visualize
    #         x_test_bg (list): background inputs
    #         x_test_signal (list): signal inputs
    #         save_fig (str): if string is given, plot will be saved with given name

    #     """

    #     b = qutip.Bloch()
    #     points_bg_x = [self.latent_node(self.params, i, latent_bit, qml.PauliX) for i in x_test_bg]
    #     points_bg_y = [self.latent_node(self.params, i, latent_bit, qml.PauliY) for i in x_test_bg]
    #     points_bg_z = [self.latent_node(self.params, i, latent_bit, qml.PauliZ) for i in x_test_bg]
    #     x = [points_bg_x[i] for i in range(len(points_bg_x))]
    #     y = [points_bg_y[i] for i in range(len(points_bg_y))]
    #     z = [points_bg_z[i] for i in range(len(points_bg_z))]
    #     bloch_points_bg = [x,y,z]
    #     b.add_points(bloch_points_bg)

    #     points_sig_x = [self.latent_node(self.params, i, latent_bit, qml.PauliX) for i in x_test_signal]
    #     points_sig_y = [self.latent_node(self.params, i, latent_bit, qml.PauliY) for i in x_test_signal]
    #     points_sig_z = [self.latent_node(self.params, i, latent_bit, qml.PauliZ) for i in x_test_signal]
    #     xs = [points_sig_x[i] for i in range(len(points_sig_x))]
    #     ys = [points_sig_y[i] for i in range(len(points_sig_y))]
    #     zs = [points_sig_z[i] for i in range(len(points_sig_z))]
    #     bloch_points_sig = [xs, ys, zs]
    #     b.add_points(bloch_points_sig)

    #     if save_fig:
    #         b.save(name=save_fig)
    #     else:
    #         b.show()
    #     b.clear()

import pytorch_lightning as pl

class CustomTorchLayer(qml.qnn.TorchLayer):
    def __init__(
        self,
        qnode,
        weight_shapes,
    ):
        super().__init__(qnode, weight_shapes)
        
    def forward(self, inputs):  # pylint: disable=arguments-differ
        """Evaluates a forward pass through the QNode based upon input data and the initialized
        weights.

        Args:
            inputs (tensor): data to be processed

        Returns:
            tensor: output data
        """
        # has_batch_dim = len(inputs.shape) > 1
        has_batch_dim = False

        # in case the input has more than one batch dimension
        if has_batch_dim:
            batch_dims = inputs.shape[:-1]
            # inputs = torch.reshape(inputs, (-1, inputs.shape[-1]))

        # calculate the forward pass as usual
        results = self._evaluate_qnode(inputs)

        if isinstance(results, tuple):
            if has_batch_dim:
                results = [torch.reshape(r, (*batch_dims, *r.shape[1:])) for r in results]
            return torch.stack(results, dim=0)

        # reshape to the correct number of batch dims
        if has_batch_dim:
            results = torch.reshape(results, (*batch_dims, *results.shape[1:]))

        return results


class QuantumAutoencoder(pl.LightningModule):
    def __init__(self, input_qbits, latent_qbits, device, img_dim, kernel_size, stride, DRCs, lr=0.01):
        super(QuantumAutoencoder, self).__init__()
        self.quantum_circuit = ConvEncoderCircuit(input_qbits, latent_qbits, device, img_dim, kernel_size, stride, DRCs)
        quantum_layer = CustomTorchLayer(self.quantum_circuit.circuit_node, {"params": self.quantum_circuit.parameters_shape})
        # self.params = torch.nn.Parameter(torch.rand(self.quantum_circuit.parameters_shape, requires_grad=True))
        # self.register_parameter('params', self.params)
        self.lr = lr
        self.quantum_layer = torch.nn.Sequential(
            quantum_layer
        )

    def forward(self, batch):
        batch_size = batch.size(0)
        results = []
        # print("forward", batch.shape)
        import time
        
        for i in range(batch_size):
            start = time.time()
            sample = batch[i]
            # sample = sample.unsqueeze(0)  # Ensure it has a batch dimension of 1
            result = self.quantum_layer(sample)
            results.append(result)
            print(time.time() - start)
        return torch.stack(results)

    def training_step(self, batch, batch_idx):
        # Average loss (1 - fidelity) of batch
        # batch = batch[0]  # Extract the first element from the tuple
        results = self(batch)
        # print(results.shape)
        loss = ((1 - results) ** 2).mean()
        self.log('train_loss', loss)
        return loss
    
    # def forward(self, batch, batch_idx):
    #     print("1. forward", batch.shape)
    #     return self.quantum_layer(batch[batch_idx])

    # def training_step(self, batch):
    #     # Average loss (1 - fidelity) of batch
    #     loss = (1 - self(batch))**2
    #     self.log('train_loss', loss)
    #     return loss

    def validation_step(self, batch, batch_idx):
        val_loss = ((1 - self(batch))**2).mean()
        self.log('val_loss', val_loss)
        return val_loss


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

        





