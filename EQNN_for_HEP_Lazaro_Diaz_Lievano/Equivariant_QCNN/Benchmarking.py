import data
import Training
import QCNN_circuit
import numpy as np

def accuracy_test(predictions, labels, cost_fn, binary = True):
    if cost_fn == 'mse':
        if binary == True:
            acc = 0
            for l, p in zip(labels, predictions):
                if np.abs(l - p) < 1:
                    acc = acc + 1
            return acc / len(labels)

        else:
            acc = 0
            for l, p in zip(labels, predictions):
                if np.abs(l - p) < 0.5:
                    acc = acc + 1
            return acc / len(labels)

    elif cost_fn == 'cross_entropy':
        acc = 0
        for l,p in zip(labels, predictions):
            if p[0] > p[1]:
                P = 0
            else:
                P = 1
            if P == l:
                acc = acc + 1
        return acc / len(labels)

def round_predictions_f(predictions, cost_fn):
    round_predictions = []
    if cost_fn == "mse": #consider binary = True
        for p in predictions:
            if p<0:
                round_predictions.append(-1)
            elif p>0:
                round_predictions.append(1) 
    elif cost_fn == "cross_entropy": #consider binary = False
        for p in predictions:
            if p[0] > p[1]:
                round_predictions.append(0)
            else:
                round_predictions.append(1)
    return round_predictions

def Encoding_to_Embedding(Encoding):
    if Encoding == 'img16x16x1':
        Embedding = "Equivariant-Amplitude"
    elif Encoding == 'resize256':
        Embedding = 'Amplitude'
    elif Encoding == 'pca8':
        Embedding = 'Angle'
    
    return Embedding


def Benchmarking(dataset, classes, Unitaries, U_num_params, Encodings, circuit, cost_fn, binary=True):
    I = len(Unitaries)
    J = len(Encodings)

    for i in range(I):
        for j in range(J):
            f = open('QCNN/Result/result.txt', 'a')
            U = Unitaries[i]
            U_params = U_num_params[i]
            Encoding = Encodings[j]
            Embedding = Encoding_to_Embedding(Encoding)

            X_train, X_test, Y_train, Y_test = data.data_load_and_process(dataset, classes=classes,
                                                                          feature_reduction=Encoding, binary=binary)

            print("\n")
            print("Loss History for " + circuit + " circuits, " + U + " " + Encoding + " with " + cost_fn)
            loss_history, trained_params = Training.circuit_training(X_train, Y_train, U, U_params, Embedding, circuit, cost_fn)

            predictions = [QCNN_circuit.QCNN(x, trained_params, U, U_params, Embedding, cost_fn) for x in X_test]
          
            accuracy = accuracy_test(predictions, Y_test, cost_fn, binary)
            print("Accuracy for " + U + " " + Encoding + " :" + str(accuracy))

            f.write("Loss History for " + circuit + " circuits, " + U + " " + Encoding + " with " + cost_fn)
            f.write("\n")
            f.write(str(loss_history))
            f.write("\n")
            f.write("Accuracy for " + U + " " + Encoding + " :" + str(accuracy))
            f.write("\n")
            f.write("\n")
    f.close()
