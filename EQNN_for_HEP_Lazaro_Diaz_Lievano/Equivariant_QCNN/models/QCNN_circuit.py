# In this module we define the EQCNN and QCNN architecture
import pennylane as qml
import Equivariant_QCNN.models.utils.unitary as unitary
import Equivariant_QCNN.models.utils.embedding as embedding

def conv_layer_equiv_U2(U, params): # apply a layer of U2 to all the system
    for i in range(int(8 / 2)):
        U(params, [2 * i, 2 * i + 1])
    U(params, [1, 2])
    U(params, [5, 6])
    U(params, [0, 3])
    U(params, [4, 7])

def conv_layer_equiv_U2_pair(U, params):
    U(params, [0, 2])
    U(params, [4,6])

def conv_layer_equiv_U2_single(U, params):
    U(params, [0, 4])

#---------
def conv_layer_equiv_U4(U, params):
    U(params, [0,1,2,3])
    U(params, [4,5,6,7])
    U(params, [2,3,4,5])
    U(params, [0, 1, 6, 7])

def conv_layer_equiv_U4_single(U,params):
    U(params, [0,2,4,6])

# Quantum Circuits for Convolutional layers
def conv_layer1(U, params):
    U(params, wires=[0, 7])
    for i in range(0, 8, 2):
        U(params, wires=[i, i + 1])
    for i in range(1, 7, 2):
        U(params, wires=[i, i + 1])

def conv_layer2(U, params):
    U(params, wires=[0, 6])
    U(params, wires=[0, 2])
    U(params, wires=[4, 6])
    U(params, wires=[2, 4])

def conv_layer3(U, params):
    U(params, wires=[0,4])

# Quantum Circuits for Pooling layers
def pooling_layer1(V, params):
    for i in range(0, 8, 2):
        V(params, wires=[i+1, i])

def pooling_layer2(V, params):
    V(params, wires=[2,0])
    V(params, wires=[6,4])

def pooling_layer3(V, params):
    V(params, wires=[0,4])

## ---- equiv

def p4m_QCNN_structure(U, params, U_params = 6, layers=3):
    param1 = params[0:U_params] #conv1 U2 6
    param2 = params[U_params: U_params+5] # pooling1 5 
    param3 = params[U_params+5:  U_params+5+6] # conv2 pair U2 6 
    param4 = params[5+6+ U_params: 5+6+ U_params + 5] #pooling2 5
    param5 = params[5+6+5+ U_params: 5+6+5+U_params + 6] # conv3 single U2 6
    param6 = params[5+6+5+6+ U_params: 5+6+5+6+ U_params + 5] #pooling3 5

    # Pooling Ansatz1 is used by default
    #conv1_U2
    conv_layer_equiv_U2(unitary.U2_equiv, param1) # 6 params
    # conv2_U4
    #conv_layer_equiv_U4(U, param2) # 3 params
    # pooling 1
    pooling_layer1(unitary.Pooling_ansatz_equiv, param2) # 5 params
    # conv3_U4_single 
    #conv_layer_equiv_U4_single(U, param4) # 3 params
    conv_layer_equiv_U2_pair(unitary.U2_equiv, param3) # 6 params
    # pooling 2
    pooling_layer2(unitary.Pooling_ansatz_equiv, param4) # 5 params
    # conv4_U2_single
    conv_layer_equiv_U2_single(unitary.U2_equiv, param5) # 6 params
    pooling_layer3(unitary.Pooling_ansatz_equiv, param6) # 5 params

    qml.Hadamard(4)


def reflection_QCNN_structure_without_pooling(U, params, U_params, layers):
    param_layers = [params[i * U_params:(i + 1) * U_params] for i in range(layers)]

    for i in range(layers):
        conv_layer_equiv_U2(U, param_layers[i])


## ------ normal 
def QCNN_structure(U, params, U_params, n_qubits, layers):
    param_layers = [params[i * U_params:(i + 1) * U_params] for i in range(layers)]
    pooling_params = params[U_params * layers: U_params * layers + layers * 2] 
    
    for i in range(layers):
        conv_layer1(U, param_layers[i], n_qubits)
        if i < len(pooling_params) // 2: 
            pooling_layer1(unitary.Pooling_ansatz1, pooling_params[i * 2:(i + 1) * 2], n_qubits)



def QCNN_structure_without_pooling(U, params, U_params):
    param1 = params[0:U_params]
    param2 = params[U_params: 2 * U_params]
    param3 = params[2 * U_params: 3 * U_params]

    conv_layer1(U, param1)
    conv_layer2(U, param2)
    conv_layer3(U, param3)

def QCNN_1D_circuit(U, params, U_params):
    param1 = params[0: U_params]
    param2 = params[U_params: 2*U_params]
    param3 = params[2*U_params: 3*U_params]

    for i in range(0, 8, 2):
        U(param1, wires=[i, i + 1])
    for i in range(1, 7, 2):
        U(param1, wires=[i, i + 1])

    U(param2, wires=[2,3])
    U(param2, wires=[4,5])
    U(param3, wires=[3,4])


dev = qml.device('default.qubit', wires = 8)
@qml.qnode(dev)
def QCNN(X, params, U, U_params, embedding_type='Equivariant-Amplitude', cost_fn='cross_entropy', layers = 3):

    # Data Embedding
    embedding.data_embedding(X, embedding_type=embedding_type)

    # Quantum Convolutional Neural Network
    if U == 'U_TTN':
        QCNN_structure(unitary.U_TTN, params, U_params, layers)
    elif U == "U2_equiv":
        reflection_QCNN_structure_without_pooling(unitary.U2_equiv, params, U_params, layers)
    elif U == "U4_equiv":
        p4m_QCNN_structure(unitary.U4_equiv, params, U_params, layers)
    elif U == 'U_5':
        QCNN_structure(unitary.U_5, params, U_params, layers)
    elif U == 'U_6':
        QCNN_structure(unitary.U_6, params, U_params, layers)
    elif U == 'U_9':
        QCNN_structure(unitary.U_9, params, U_params)
    elif U == 'U_13':
        QCNN_structure(unitary.U_13, params, U_params, layers)
    elif U == 'U_14':
        QCNN_structure(unitary.U_14, params, U_params, layers)
    elif U == 'U_15':
        QCNN_structure(unitary.U_15, params, U_params, layers)
    elif U == 'U_SO4':
        QCNN_structure(unitary.U_SO4, params, U_params, layers)
    elif U == 'U_SU4':
        QCNN_structure(unitary.U_SU4, params, U_params, layers)
    elif U == 'U_SU4_no_pooling':
        QCNN_structure_without_pooling(unitary.U_SU4, params, U_params)
    elif U == 'U_SU4_1D':
        QCNN_1D_circuit(unitary.U_SU4, params, U_params)
    elif U == 'U_9_1D':
        QCNN_1D_circuit(unitary.U_9, params, U_params)
    else:
        print("Invalid Unitary Ansatze")
        return False

    if cost_fn == 'cross_entropy':
        result = qml.probs(wires=4)

    elif cost_fn == 'mse':
        if U == "U2_equiv":
            result = qml.expval(qml.PauliZ(0) @ qml.PauliZ(1) @ qml.PauliZ(2) @ qml.PauliZ(3) @ qml.PauliZ(4) @ qml.PauliZ(5) @ qml.PauliZ(6) @ qml.PauliZ(7))
        else:
            result = qml.expval(qml.PauliZ(4))

    return result
