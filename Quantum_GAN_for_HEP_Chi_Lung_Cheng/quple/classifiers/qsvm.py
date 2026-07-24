from typing import Optional, Sequence, List, Dict, Union
import numpy as np
import numba


from sklearn import svm
from sklearn.model_selection import GridSearchCV

import quple
from .qsvm_logger import QSVMLogger

from quple.utils.mathext import abs2, DataPrecision, split_gramian_matrix

class DefaultDict(dict):
    def __missing__(self, key):
        return key


class QSVM:
    __DEFAULT_SVC_PARAM_GRID__ = {
        'C': [1,2,3,4,5,8,9,10,15,20,30,50,100,200,400,800,1000],
    }
    
    def __init__(self, encoding_circuit:Union["quple.ParameterisedCircuit", "qiskit.QuantumCircuit"], 
                 precision:Union[str,DataPrecision]='double', 
                 kernel_matrix_split:int=1, 
                 state_vector_event_split=None, **kwargs):
        
        self._attributes = DefaultDict({})
        self.encoding_circuit = encoding_circuit
        self.precision = precision
        if self.circuit_type == 'qiskit':
            if 'backend' not in kwargs:
                raise ValueError('Backend must be given for a qiskit circuit')
            else:
                self.backend = kwargs['backend']
        self._kernel_matrix_split = kernel_matrix_split
        self._state_vector_event_split = state_vector_event_split
        
    @property
    def precision(self):
        return self._precision
    
    @precision.setter
    def precision(self, val):
        if isinstance(val, str):
            self._precision = DataPrecision[val]
        elif isinstance(val, DataPrecision):
            self._precision = val
        else:
            raise Value('Invalid type for numeric precision: {}'.format(type(val)))
            
        
    @property
    def entangle_strategy(self):
        return self._attributes['entangle_strategy']
    
    @property
    def attributes(self):
        return self._attributes
    
    @property
    def kernel_matrix_split(self):
        return self._kernel_matrix_split
    
    @property
    def backend(self):
        return self._backend
    
    @backend.setter
    def backend(self, val):
        if self._precision == DataPrecision['single']:
            val.backend.set_options(precision='single')
        elif self._precision == DataPrecision['double']:
            val.backend.set_options(precision='double')
        else:
            pass
            #raise ValueError('Invalid value for numerical precision: {}'.format(self._precision))
        self._backend = val
    
    @property
    def n_qubit(self):
        return self._attributes['n_qubit']
    
    @property
    def encoding_map(self):
        return self._attributes['encoding_map']
    
    @property
    def circuit_type(self):
        return self._attributes['circuit_type']
    
    @property
    def circuit_name(self):
        return self._attributes['circuit_name']    
        
    @property
    def encoding_circuit(self):
        return self._encoding_circuit
  
    @encoding_circuit.setter
    def encoding_circuit(self, val):
        import qiskit
        if isinstance(val, quple.ParameterisedCircuit):
            self._attributes['circuit_type'] = 'quple'
            self._attributes['n_qubit'] = val.n_qubit
            self._attributes['encoding_map'] = val.encoding_map.__name__ if \
                isinstance(val, quple.data_encoding.EncodingCircuit) else '' 
            self._attributes['entangle_strategy'] = val.entangle_strategy if \
                isinstance(val, quple.ParameterisedCircuit) and \
                isinstance(val.entangle_strategy, str) else ''
            self._attributes['circuit_name'] = val.name
            self._encoding_circuit = val
        elif isinstance(val, qiskit.QuantumCircuit):
            self._attributes['circuit_type'] = 'qiskit'
            self._attributes['n_qubit'] = val.num_qubits
            self._attributes['encoding_map'] = val._data_map_func.__name__ 
            self._attributes['entangle_strategy'] = val._entanglement if \
                isinstance(val, qiskit.circuit.library.NLocal) else '' 
            self._attributes['circuit_name'] = type(val).__name__
            self._encoding_circuit = val
        elif isinstance(val, qiskit.aqua.components.feature_maps.FeatureMap):
            self._attributes['circuit_type'] = 'qiskit'
            self._attributes['n_qubit'] = val._num_qubits
            self._attributes['encoding_map'] = ''
            self._attributes['entangle_strategy'] = ''
            self._attributes['circuit_name'] = type(val).__name__
            self._encoding_circuit = val
        else:
            raise ValueError('Circuit type currently not supported: {}'.format(type(val))) 
    
    @staticmethod
    def get_kernel_matrix(state_vectors_left, state_vectors_right, n_split=1):
        return abs2(split_gramian_matrix(state_vectors_left, state_vectors_right, n_split))

    def get_state_vectors(self, x):
        if self.circuit_type == 'quple':
            if self._state_vector_event_split is not None:
                return self.encoding_circuit.get_state_vectors_test(x, self._state_vector_event_split)
            return self.encoding_circuit.get_state_vectors(x)
        elif self.circuit_type == 'qiskit':
            from quple.qiskit_interface.tools import get_qiskit_state_vectors
            return get_qiskit_state_vectors(self.backend, self.encoding_circuit, x)
        else:
            return None
    
    @staticmethod 
    def train_svc(train_kernel_matrix, y_train, random_seed=0,
                  param_grid=None, n_jobs=-1, cv=3, scoring='roc_auc', **kwargs):
        param_grid = param_grid or QSVM.__DEFAULT_SVC_PARAM_GRID__
        
        svc = svm.SVC(kernel='precomputed', probability=True, random_state=random_seed)
        clf = GridSearchCV(estimator=svc, param_grid=param_grid, n_jobs=n_jobs,
                           cv=cv, scoring=scoring)

        clf.fit(train_kernel_matrix, y_train)
        return clf
    # deprecated
    def run_with_tune(self, x_train, y_train, x_val, y_val, 
                      x_test, y_test, random_seed = 0, logger=None, **kwargs):
                             
        # setup logger
        if logger:
            if isinstance(logger, QSVMLogger):
                attributes = {}
                attributes['feature_dimension'] = x_train.shape[1]
                attributes['train_size'] = x_train.shape[0]
                attributes['val_size'] = x_val.shape[0]
                attributes['test_size'] = x_test.shape[0]
                attributes.update(self.attributes)
                logger.set_attributes(attributes)
            else:
                raise ValueError('Only quple.QSVMLogger is supported')

        if logger:
            logger.on_train_start()

        if logger:
            logger.info('Evaluating state vectors for train data...')             
        train_state_vectors = self.get_state_vectors(x_train)
        if logger:
            logger.info('Evaluating state vectors for validation data...')             
        val_state_vectors = self.get_state_vectors(x_val)
        if logger:
            logger.info('Evaluating state vectors for test data...')             
        test_state_vectors = self.get_state_vectors(x_test)
        if logger:
            logger.info('Evaluating kernel matrix for train data...')
        train_kernel_matrix = self.get_kernel_matrix(train_state_vectors, train_state_vectors, self._kernel_matrix_split)
        if logger:
            logger.info('Evaluating kernel matrix for validation data...')
        val_kernel_matrix = self.get_kernel_matrix(val_state_vectors, train_state_vectors, self._kernel_matrix_split)
        if logger:
            logger.info('Evaluating kernel matrix for test data...')
        test_kernel_matrix = self.get_kernel_matrix(test_state_vectors, train_state_vectors, self._kernel_matrix_split)
        
        train_state_vectors = None # Free memory
        val_kernel_matrix = None # Free memory
        test_state_vectors = None  # Free memory 

        clf = self.train_svc(y_train, random_seed=random_seed)    
    

        logger.on_train_end(clf, train_kernel_matrix, test_kernel_matrix, y_test, val_kernel_matrix, y_val)
        return clf
    
    def run(self, x_train, y_train, x_test, y_test, 
            random_seed = 0, logger=None, **kwargs):
                             
        # setup logger
        if logger:
            if isinstance(logger, QSVMLogger):
                attributes = {}
                attributes['feature_dimension'] = x_train.shape[1]
                attributes['train_size'] = x_train.shape[0]
                attributes['test_size'] = x_test.shape[0]
                attributes.update(self.attributes)
                logger.set_attributes(attributes)
            else:
                raise ValueError('Only quple.QSVMLogger is supported')
        if logger:
            logger.on_train_start()
        
        if logger:
            logger.info('Evaluating state vectors for train data...')             
        train_state_vectors = self.get_state_vectors(x_train)
        if logger:
            logger.info('Evaluating kernel matrix for train data...')
        train_kernel_matrix = self.get_kernel_matrix(train_state_vectors, train_state_vectors, self._kernel_matrix_split) 
        if logger:
            logger.info('Fitting SVM model...')
        clf = self.train_svc(train_kernel_matrix, y_train, random_seed=random_seed, n_jobs=quple.CV_NJOBS)   
        train_kernel_matrix = None # Free memory
        if logger:
            logger.info('Evaluating state vectors for test data...')             
        test_state_vectors = self.get_state_vectors(x_test)        
        if logger:
            logger.info('Evaluating kernel matrix for test data...')
        test_kernel_matrix = self.get_kernel_matrix(test_state_vectors, train_state_vectors, self._kernel_matrix_split)
        
        train_state_vectors = None # Free memory
        test_state_vectors = None  # Free memory 

        metric = logger.on_train_end(clf, test_kernel_matrix, y_test)
        return clf, metric
    
    def test_run(self, x_train, y_train, x_test, y_test, 
            random_seed = 0, logger=None, **kwargs):
                             
        # setup logger
        if logger:
            if isinstance(logger, QSVMLogger):
                attributes = {}
                attributes['feature_dimension'] = x_train.shape[1]
                attributes['train_size'] = x_train.shape[0]
                attributes['test_size'] = x_test.shape[0]
                attributes.update(self.attributes)
                logger.set_attributes(attributes)
            else:
                raise ValueError('Only quple.QSVMLogger is supported')
        if logger:
            logger.on_train_start()
        
        if logger:
            logger.info('Evaluating state vectors for train data...')      
        train_state_vectors = np.memmap('temp.dat', dtype='complex64', shape=(x_train.shape[0], 2**x_train.shape[1]), mode='w+')    
        train_state_vectors[:] = self.get_state_vectors(x_train)
        if logger:
            logger.info('Evaluating kernel matrix for train data...')
        
        train_kernel_matrix = self.get_kernel_matrix(train_state_vectors, train_state_vectors, self._kernel_matrix_split) 
        if logger:
            logger.info('Fitting SVM model...')
        clf = self.train_svc(train_kernel_matrix, y_train, random_seed=random_seed, n_jobs=quple.CV_NJOBS)   
        train_kernel_matrix = None # Free memory
        if logger:
            logger.info('Evaluating state vectors for test data...')   
        
        test_state_vectors = self.get_state_vectors(x_test)        
        if logger:
            logger.info('Evaluating kernel matrix for test data...')
        test_kernel_matrix = self.get_kernel_matrix(train_state_vectors, test_state_vectors, self._kernel_matrix_split).T
        
        train_state_vectors = None # Free memory
        test_state_vectors = None  # Free memory 

        metric = logger.on_train_end(clf, test_kernel_matrix, y_test)
        return clf, metric   