import numpy as np
import numba
from sklearn import svm
import quple
from sklearn.model_selection import GridSearchCV
from .qsvm_logger import QSVMLogger

@numba.vectorize([numba.float64(numba.complex128),numba.float32(numba.complex64)])
def abs2(x):
    return x.real**2 + x.imag**2


class QSVM:
    def __init__(self):
        pass
    
    @staticmethod
    def get_kernel_matrix(state_vectors_left, state_vectors_right):
        return abs2(state_vectors_left.conjugate() @ state_vectors_right.T)
    
    @staticmethod
    def run(encoding_circuit, x_train, y_train, x_test, y_test, 
            random_seed = 0, backend=None, **kwargs):
        from qiskit.aqua.algorithms import QSVM as QSVM_qiskit
        attributes = {}
        attributes['encoding_circuit'] = type(encoding_circuit).__name__
        attributes['feature_dimension'] = x_train.shape[1]
        attributes['train_size'] = x_train.shape[0]
        attributes['test_size'] = x_test.shape[0]
        attributes['n_qubit'] = encoding_circuit.num_qubits
        attributes['encoding_map'] = encoding_circuit._data_map_func.__name__ 
        logger = QSVMLogger(attributes)      
        logger.info('Evaluating kernel matrix for train data...')
        train_kernel_matrix = QSVM_qiskit.get_kernel_matrix(backend, encoding_circuit, x_train)
        logger.info('Evaluating kernel matrix for test data...')        
        train_test_kernel_matrix = QSVM_qiskit.get_kernel_matrix(backend, encoding_circuit, x_test, x_train)
        logger.info('Training started')
        svc = svm.SVC(kernel='precomputed', probability=True, random_state=random_seed)

        tune_params = {
                       'C': [1,10,100,200,400,800],
                       'gamma': [0.01,0.1,1]
                      }

        clf = GridSearchCV(estimator=svc, param_grid=tune_params, n_jobs=-1, cv=3, scoring='roc_auc')

        clf.fit(train_kernel_matrix,y_train)

        logger.on_train_end(clf, train_kernel_matrix, train_test_kernel_matrix, y_test)
        return clf