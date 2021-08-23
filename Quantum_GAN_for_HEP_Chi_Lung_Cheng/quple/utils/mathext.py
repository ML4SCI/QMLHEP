from enum import Enum
import numpy as np
import numba

class DataPrecision(Enum):
    single = 1
    double = 2

@numba.vectorize([numba.float64(numba.complex128),numba.float32(numba.complex64)])
def abs2(x):
    # element-wise modulus square
    return x.real**2 + x.imag**2

def gramian_matrix(A:np.ndarray, B:np.ndarray):
    # The Gramian matrix
    # G = AB*
    return A @ B.conjugate().T

def split_gramian_matrix(A:np.ndarray, B:np.ndarray, n_split:int=1):
    if n_split == 1:
        return gramian_matrix(A, B)
    # output dimension
    dimension = A.shape[0]
    if A.shape != B.shape:
        raise ValueError('Input matrices must have the same shape')
    if A.dtype != B.dtype:
        raise ValueError('Input matrices must have the same data type')
    if dimension % n_split != 0:
        raise ValueError('Dimension of input matrix must divide the number of splits')
    # size of each split
    k = dimension//n_split 
    G = np.zeros((dimension, dimension), dtype=A.dtype)
    for i in range(n_split):
        for j in range(n_split):
            G[i*k:(i+1)*k, j*k:(j+1)*k] = gramian_matrix(A[i*k:(i+1)*k], B[j*k:(j+1)*k])
    return G