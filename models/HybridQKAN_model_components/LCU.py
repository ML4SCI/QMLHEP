import pennylane as qml
import torch
import math

def quantum_lcu_block(qsvt_vals, weight_vals):
    """
    qsvt_vals: tensor of shape (P,)
    weight_vals: tensor of shape (P,)
    Output: tensor scalar (Z expectation after quantum weighting)
    """
    P = len(qsvt_vals)
    n_ctrl = math.ceil(math.log2(P))
    wires = list(range(n_ctrl + 1))  # control + 1 target
    dev = qml.device("default.qubit", wires=len(wires))

    alpha = torch.sqrt(torch.abs(weight_vals) + 1e-8)
    alpha = alpha / torch.norm(alpha + 1e-8)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit():
        qml.StatePrep(alpha, wires=wires[:-1])

        for i in range(P):
            ctrl_bin = [int(b) for b in f"{i:0{n_ctrl}b}"]
            qml.ctrl(qml.RY, control=wires[:-1], control_values=ctrl_bin)(2 * qsvt_vals[i], wires=wires[-1])

        qml.adjoint(qml.StatePrep(alpha, wires=wires[:-1]))

        return qml.expval(qml.PauliZ(wires[-1]))

    return circuit()
