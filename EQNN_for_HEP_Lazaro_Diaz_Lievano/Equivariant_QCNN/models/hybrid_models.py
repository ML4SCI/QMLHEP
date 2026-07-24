import pennylane as qml
import numpy as np 


def U2(phi, wires):
    qml.RX(phi[0], wires=wires[0])
    qml.RX(phi[1], wires=wires[1])
    qml.IsingZZ(phi[2], wires=wires)
    qml.RX(phi[3], wires=wires[0])
    qml.RX(phi[4], wires=wires[1])
    qml.IsingYY(phi[5], wires=wires)

def Pooling_ansatz(phi, wires):
    qml.RX(phi[0], wires=wires[0])
    qml.RX(phi[1], wires=wires[1])
    qml.RY(phi[2], wires=wires[1])
    qml.RZ(phi[3], wires=wires[1])
    qml.CRX(phi[4], wires=[wires[1], wires[0]])

def qcnn_full(params, wires):
  # 14 params
  for i in range(int(len(wires) / 2)):
    U2(params[:6], [wires[2 * i], wires[2 * i + 1]])
  U2(params[:6], [wires[1], wires[2]])
  U2(params[:6], [wires[5], wires[6]])
  U2(params[:6], [wires[0], wires[3]])
  U2(params[:6], [wires[4], wires[7]])

  qml.Barrier()
  for i in range(int(len(wires) / 2)):
    Pooling_ansatz(params[9:14], [wires[2 * i], wires[2 * i + 1]])

  qml.Barrier()


## HYBRID MODELS


def hybrid_models( model = 1, num_layers=4):
    num_qubits = 8
    dev = qml.device("default.qubit", wires=num_qubits)
    

    if model == 0:  ## equiv
        num_layers = num_layers
        @qml.qnode(dev)
        def hybrid_circuit(inputs, params):
            qml.AmplitudeEmbedding(inputs, wires=range(num_qubits), normalize=True)
            qcnn_full(params, [0,1,2,3,4,5,6,7])
            #for layer in range(num_layers):
                #qcnn_full(params[15*layer: 15*(layer+1)], [0,1,2,3,4,5,6,7])
            qml.Hadamard(0)
            qml.Hadamard(2)
            qml.Hadamard(4)
            qml.Hadamard(6)
            return qml.probs(wires = [0,2,4,6])


        weight_shapes = {"params": (15*num_layers,)}
        qlayer = qml.qnn.TorchLayer(hybrid_circuit, weight_shapes) 

        clayer_1 = torch.nn.Linear(16, 2) 
        #clayer_2 = torch.nn.Linear(8, 2) 
        softmax = torch.nn.Softmax(dim=1)

        layers = [qlayer, clayer_1, softmax]
        model = torch.nn.Sequential(*layers)

    
    elif model == 1: # no equiv. classic + quantum + classic 
        @qml.qnode(dev)
        def no_hybrid_circuit(inputs, weights):
            qml.AmplitudeEmbedding(inputs, wires=range(num_qubits), normalize=True)
            qml.BasicEntanglerLayers(weights, wires=range(num_qubits))
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(num_qubits)]
        weight_shapes = {"weights": (num_layers, num_qubits)}
        qlayer = qml.qnn.TorchLayer(no_hybrid_circuit, weight_shapes) 
        clayer_1 = torch.nn.Linear(256, 256) 
        clayer_2 = torch.nn.Linear(8, 2)
        softmax = torch.nn.Softmax(dim=1) 

        layers = [clayer_1, qlayer, clayer_2, softmax]
        model = torch.nn.Sequential(*layers)

    else: # no equiv. quantum + classic 
        ## model 2 
        @qml.qnode(dev)
        def no_hybrid_circuit(inputs, weights):
            qml.AmplitudeEmbedding(inputs, wires=range(num_qubits), normalize=True)
            qml.BasicEntanglerLayers(weights, wires=range(num_qubits))
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(num_qubits)]
        weight_shapes = {"weights": (num_layers, num_qubits)}
        qlayer = qml.qnn.TorchLayer(no_hybrid_circuit, weight_shapes) 

        clayer_1 = torch.nn.Linear(8, 2) 
        softmax = torch.nn.Softmax(dim=1) 

        layers = [qlayer, clayer_1, softmax]
        model = torch.nn.Sequential(*layers)

    return model

### TRAINING 

import torch
import torch.nn.functional as F


def training_hybrid(X_train, y_train, epochs = 20, batch_size= 64, samples= 10000, model = 0, num_layers = 4, lr = 0.1):
    # one-hot encoding
    loss_history = []

    y_train_one_hot = F.one_hot(y_train[:samples].to(torch.int64), num_classes=2) # Convert y_train to an integer tensor

    X = X_train[:samples].reshape(samples, 16*16).float().requires_grad_(True)
    y_hot = y_train_one_hot.long()

    batches = samples // batch_size
    
    data_loader = torch.utils.data.DataLoader(
        list(zip(X, y_hot)), batch_size=batch_size, shuffle=True, drop_last=True
    )

    model = hybrid_models( model, num_layers)

    opt = torch.optim.SGD(model.parameters(), lr=lr)
    loss = torch.nn.L1Loss()

    for epoch in range(epochs):
        running_loss = 0
        for xs, ys in data_loader:
            opt.zero_grad()

            y_pred = model(xs)
            loss_evaluated = loss(y_pred, ys)
            loss_evaluated.backward()
            opt.step()
            running_loss += loss_evaluated.item()
        avg_loss = running_loss / batches
        loss_history.append(avg_loss)
        print("Average loss over epoch {}: {:.4f}".format(epoch + 1, avg_loss))
    
    return model, loss_history