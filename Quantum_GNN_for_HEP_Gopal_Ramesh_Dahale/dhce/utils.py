import numpy as np

def node_hindex_centrality(graph):
    numNode = graph.number_of_nodes()
    TotalDeg = np.array(list(graph.degree(graph.nodes())))[:, 1]
    Nei = [list(iter(graph[i])) for i in graph.nodes()]

    Hn = []
    Hi = [0] * numNode
    Hn.append(TotalDeg)

    for inter in range(1, numNode):
        Hitmp = np.array(Hn[inter - 1])

        for iNode in range(0, numNode):
            index = Nei[iNode]
            iDegree = np.sort(Hitmp[index])[::-1]
            Hi[iNode] = len(
                np.where(iDegree - np.arange(1,
                                             len(iDegree) + 1) >= 0)[0])

        Hn.append(Hi.copy())

        if list(Hn[-1]) == list(Hn[-2]):
            break

    return np.array(Hn[:-1])

def Entropy_Shannon(X):
    # For discrete data
    data = np.unique(X)
    numData = len(data)
    Frequency = [0] * numData

    for index in range(numData):
        Frequency[index] = np.sum(X == data[index])

    P = Frequency / np.sum(Frequency)
    H = -np.sum(P * np.log2(P))

    return H


def get_dhce_data(graph):
    Hn = node_hindex_centrality(graph)
    maxDim = Hn.shape[0]
    EnGraph = [0] * maxDim

    for ih in range(maxDim):
        EnGraph[ih] = Entropy_Shannon(Hn[ih])

    return EnGraph