from typing import Optional, Sequence, List, Dict, Union, Callable
from abc import ABC
import logging

import numpy as np
import six

import tensorflow as tf
import tensorflow_quantum as tfq

import cirq
import quple
from quple import ParameterisedCircuit
from quple.data_encoding import EncodingCircuit

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)

class DefaultDict(dict):
    def __missing__(self, key):
        return key

class VQC(tf.keras.Sequential):
    """Variational Quantum Classifier (VQC)
    
    The variational quantum classifier (VQC) is a hybrid quantum-classical algorithm that 
    utilizes both quantum and classical resources for solving classification problems. The
    VQC is a kind of quantum neural network (QNN) where the model representation of the 
    is based entirely on the quantum processor, with classical heuristics participating 
    only as optimizers for the trainable parameters of the quantum model.
    (Ref: https://arxiv.org/pdf/2003.02989.pdf)
    
    The core components of a VQC involves the construction of two quantum circuits: a data 
    encoding circuit for mapping input features into the quantum state of the circuit 
    qubits, and a variational circuit with tunable parameters which are optimized in an 
    iterative manner by a classical computer. The required circuit depth for a VQC is
    somewhat flexible and typically works well with low circuit depth which are suitable
    for running on Noisy Intermediate-Scale Quantum (NISQ) devices.
    
    More specifically, the VQC defines a quantum model 
                :math: f(x, θ) = <0|U†(x, θ)MU(x, θ)|0> :math: 
    as the expectation value of some observable :math: M :math: with respect to a state
    prepared via the unitary operation :math: U(x, θ) :math: to the initial state
    :math: |0> = |0>^{⊗n} :math: of the quantum computer with an :math: n :math: qubit
    register. Here :math: U(x, θ) :math: is a quantum circuit that depends on the data
    input x and a set of parameters θ which can be seen as the weights in a neural 
    network. :math: U(x, θ) :math: can be written as a composition of a data encoding
    circuit :math: S(x) :math: and a variational circuit :math: W(θ) :math:, 
    i.e. :math: U(x, θ) = S(x)W(θ) :meth:. The circuit is run multiple times and the
    measurement results are averaged to obtain the expectation value. Optionally a 
    classical activation function may be used to map the results from f(x, θ) to the
    predicted class label, i.e. :math: y = ϕ(f(x, θ)) :math:. The parameters θ are then 
    trained in an iterative manner by a classical optimizer to optimize over some
    cost function to obtain the best model.

    The model implementation is based on the tensorflow and tensorflow quantum library.
    """    
    def __init__(self, encoding_circuit:'cirq.Circuit', variational_circuit:'cirq.Circuit', 
                 optimizer:Optional[Union[str,tf.keras.optimizers.Optimizer]]='adam', 
                 differentiator:Optional[tfq.differentiators.Differentiator]=None,
                 regularizer=None,
                 repetitions=None,
                 loss='mse', 
                 activation='sigmoid',
                 metrics=['binary_accuracy', 'qAUC'],
                 readout=None,
                 classical_layer:bool=False,
                 random_state:int=None,                 
                 name:str=None, *arg, **args):
        """ Creates a variational quantum classifier
        
        Args:
            encoding_circuit: cirq.Circuit or quple.QuantumCircuit instace
                A parameterised quantum circuit used for data encoding.
            variational_circuit: cirq.Circuit or quple.QuantumCircuit instace
                A parameterised quantum circuit which the parameters
                are to be tuned by a classical optimizer.
            optimizer: string (name of optimizer)or tf.keras.optimizers.Optimizer instance; default='adam'
                The optimizer to use for tuning the parameters in the
                variational circuit. 
            differentiator: Optional `tfq.differentiator` object 
                To specify how gradients of variational circuit should be calculated.
            regularizer: Optional `tf.keras.regularizer` object
                Regularizer applied to the parameters of the variational circuit.
            repetitions: int; default=None
                Number of repetitions for measurement
            loss: string (name of objective function), objective function or tf.keras.losses.Loss instance
                An objective function of the form f_n(y_true,y_pred) which maps truth labels and the predicted
                labels to some loss values. 
            activation: string (name of activation function) or activation function instance
                The activation function for the output layer.
            metrics: List of string (name of a built-in function), function or a tf.keras.metrics.Metric instance. 
                List of metrics to be evaluated by the model during training and testing. 
            readout: `cirq.PauliSum` or Python `list` of `cirq.PauliSum` objects
                Measurement operators (observables) for the variational circuit layer.
            classical_layer: boolean; default=False
                Whether to add a classical output layer.
            random_state: Optional int
                The random state for reproducible result.
            name: Optional str
                Name given to the classifier.
        """
        super(VQC, self).__init__()
        self._attributes = DefaultDict({})
        if random_state:
            tf.random.set_seed(random_state)
        self._readout = readout
        self.encoding_circuit = encoding_circuit
        self.variational_circuit = variational_circuit
        if isinstance(variational_circuit, ParameterisedCircuit):
            circuit_qubits = variational_circuit.qubits
        else:
            circuit_qubits = quple.get_circuit_qubits(variational_circuit)
        self._attributes['qubits'] = circuit_qubits
        self._attributes['n_qubit'] = len(circuit_qubits)
        self.differentiator = differentiator        
        self.regularizer = regularizer
        self.repetitions = repetitions
        self.activation = activation        
        self.random_state = random_state
        self.classical_layer = classical_layer
        layers = self._get_sequential_layers()   
        super(VQC, self).__init__(layers, name)
        self.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        
    @tf.function
    def custom_auc(self, y_true, y_pred):
        y_pred_ = (y_pred+1)/2
        return self.auc_metric(y_true, y_pred_)        
        
    def compile(self, optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy', 'qAUC'], *args, **kwargs):
        _metrics = []
        for metric in metrics:
            if metric.lower() == 'qauc':
                from quple.interface.tfq.metrics.metrics import qAUC
                _metrics.append(qAUC())
            else:
                _metrics.append(metric)
        super(VQC, self).compile(optimizer, loss, _metrics, *args, **kwargs)
        # attribute checks for compatibility with older version of tensorflow
        if hasattr(self.optimizer, '_name'):
            optimizer_name = self.optimizer._name
        elif hasattr(self.optimizer, '__name__'):
            optimizer_name = self.optimizer.__name__
        else:
            raise ValueError('unable to extract optimizer name, please check tensorflow version')
        if hasattr(self, 'compiled_metrics'):
            metrics_name = [metric if isinstance(metric, str) else metric.name for metric in self.compiled_metrics._metrics]
        elif hasattr(self, 'metrics'):
            metrics_name = [metric.__name__ for metric in self.metrics]
        else:
            raise ValueError('unable to extract metrics name, please check tensorflow version')
        if hasattr(self, 'compiled_loss'):
            loss_name = self.compiled_loss._losses
        elif hasattr(self, 'loss_functions'):
            loss_name = [loss.__name__ for loss in self.loss_functions]
        else:
            raise ValueError('unable to extract loss name, please check tensorflow version')  

        self._attributes['optimizer'] = optimizer_name
        self._attributes['metrics']   = metrics_name
        self._attributes['loss']      = loss_name
        
    @property
    def attributes(self):
        return self._attributes

    @property
    def encoding_circuit(self):
        return self._encoding_circuit
    
    @encoding_circuit.setter
    def encoding_circuit(self, val):
        feature_dimension = len(quple.get_circuit_unflattened_symbols(val))
        if not feature_dimension:
            raise ValueError('Encoding circuit must be a parameterised circuit with '
                             'number of parameters matching the feature dimension')
        self._check_circuit_qubits()
        self._encoding_circuit = val
        self._attributes['feature_dimension'] = feature_dimension
        self._attributes['encoding_circuit'] = val.name if isinstance(val, quple.QuantumCircuit) else ''
        encoding_map_name = ''
        if isinstance(val, EncodingCircuit):
            encoding_map = val.encoding_map
            if hasattr(encoding_map, '__name__'):
                encoding_map_name = val.encoding_map.__name__
            elif hasattr(type(encoding_map), '__name__'):
                encoding_map_name = type(val.encoding_map).__name__
        self._attributes['encoding_map'] = encoding_map_name
        logger.info('Registered encoding circuit with feature dimension: {}'.format(feature_dimension))
    
    @property
    def qubits(self):
        return self.attributes['qubits']
    
    @property
    def n_qubit(self) -> int:
        return self.attributes['n_qubit']
        
    @property
    def symbols(self) -> List[str]:
        return self.attributes['symbols']
    
    @property
    def num_parameters(self) -> int:
        return self.attributes['num_parameters']
    
    @property
    def feature_dimension(self) -> int:
        return self.attributes['feature_dimension']
    
    @property
    def variational_circuit(self):
        return self._variational_circuit
    
    @variational_circuit.setter
    def variational_circuit(self, val):
        from pdb import set_trace
        circuit_parameters = quple.get_circuit_symbols(val)
        num_parameters = len(circuit_parameters)
        if not num_parameters:
            raise ValueError('Variational circuit must be a parameterised circuit which'
                             ' the parameters are to be optimized by the optimizer.')
        self._check_circuit_qubits()
        self._variational_circuit = val
        # if readout is not provided, default to measure all qubits
        if not self._readout:
            self._readout = [cirq.Z(qubit) for qubit in quple.get_circuit_qubits(val)]
        self._attributes['circuit_parameters'] = circuit_parameters
        self._attributes['num_parameters'] = num_parameters
        self._attributes['variational_circuit'] = val.name if isinstance(val, quple.QuantumCircuit) else ''
        logger.info('Registered variational circuit with number of parameters: {}'.format(num_parameters))
        
    def _check_circuit_qubits(self):
        # make sure both encoding and variational circuits are initialized
        if self.__dict__.get('_encoding_circuit', None) and \
        self.__dict__.get('_variational_circuit', None):
            encoding_circuit_qubits = quple.get_circuit_qubits(self.encoding_circuit)
            variational_circuit_qubits = quple.get_circuit_qubits(self.variational_circuit)
            if set(encoding_circuit_qubits) != set(variational_circuit_qubits):
                raise ValueError('encoding circuit and variational circuit must'
                                 ' have the same qubit layout')
    
    @property
    def readout(self):
        return self._readout
    
    @property
    def differentiator(self):
        return self._differentiator
    
    @differentiator.setter
    def differentiator(self, val):
        if (val != None) and (not isinstance(val, tfq.differentiators.Differentiator)):
            raise ValueError('Only tensorflow quantum differentiator is allowed')
        self._attributes['differentiator'] = val.__name__ if val else ''
        self._differentiator = val
    
    @property
    def activation(self):
        return self._activation
    
    @activation.setter
    def activation(self, val):
        if isinstance(val, six.string_types):
            val = tf.keras.activations.get(val)
        self._attributes['activation'] = val.__name__ if val else ''
        self._activation = val
        
    def _get_sequential_layers(self):
        
        input_layer = tf.keras.layers.Input(shape=(), dtype=tf.string)
        pqc_layer = tfq.layers.PQC(self.variational_circuit,
                                   self.readout,
                                   repetitions=self.repetitions,
                                   differentiator=self.differentiator,
                                   regularizer=self.regularizer)
        layers = [input_layer, pqc_layer]
        if self.classical_layer:
            #output_layer = tf.keras.layers.Dense(1, activation=self.activation, 
            #                kernel_initializer=tf.keras.initializers.Constant(value=1)) 
            #output_layer.trainable = False
            output_layer = tf.keras.layers.Dense(1, activation=self.activation)
            layers.append(output_layer)
        return layers
    
    def _reset_layers(self):
        self._layers = None
        for layer in self._get_sequential_layers():
            self.add(layer)
        
    def _check_data(self, x):
        if isinstance(x, np.ndarray):
            num_dim = x.ndim
            if x.ndim != 2:
                raise ValueError('Data in numpy array format must be of dimension 2')
            num_var = x.shape[1]
            if num_var != self.feature_dimension:
                raise ValueError('Data has feature dimension {} but the encoding'
                ' circuit has feature dimension {}'.format(num_var, self.feature_dimension))
    
    def convert_to_tensor(self, x:np.ndarray):
        self._check_data(x)
        logger.info('Converting circuits to tensors...')
        return tfq.convert_to_tensor(self.encoding_circuit.resolve_parameters(x))
    
    def run(self, x_train, y_train, x_val, y_val, 
            x_test, y_test, 
            batch_size:Optional[int]=None,
            epochs:int=1, callbacks=None,
            roc_curve=True,
            weights=None):

        if weights is None:
            weights = (None, None, None)
        validation_data = (x_val, y_val) if weights is None else (x_val, y_val, weights[1])

        for callback in callbacks:
            if isinstance(callback, quple.classifiers.VQCLogger):
                attributes = {}
                attributes['feature_dimension'] = len(x_train)
                attributes['train_size'] = len(x_val)
                attributes['val_size'] =  len(x_val)
                attributes['test_size'] = len(x_test)
                attributes['batch_size'] = batch_size
                attributes['epochs'] = epochs   
                attributes.update(self.attributes)
                callback.set_attributes(attributes)
        self.fit(x_train, y_train, batch_size, epochs,
                 validation_data=validation_data,
                 callbacks=callbacks,
                 sample_weight=weights[0])
        if isinstance(x_test, np.ndarray):
            x_test = self.convert_to_tensor(x_test)        
        self.evaluate(x_test, y_test, callbacks=callbacks, sample_weight=weights[2]) 
        if roc_curve:
            self.roc_curve(x_test, y_test, callbacks=callbacks)
        for callback in callbacks:
            if isinstance(callback, quple.classifiers.VQCLogger):
                callback.reset_logger()
    def fit(self, x, y,
            batch_size:Optional[int]=None,
            epochs:int=1, 
            validation_data=None,
            *args, **kwargs):
        self._attributes['train_size'] = len(x)
        if isinstance(x, np.ndarray):
            x = self.convert_to_tensor(x)
        if validation_data and isinstance(validation_data, tuple):
            self._attributes['val_size'] = len(validation_data[0])
            if isinstance(validation_data[0], np.ndarray):
                x_val = self.convert_to_tensor(validation_data[0])
                validation_data = (x_val, *validation_data[1:])
        self._attributes['batch_size'] = batch_size
        self._attributes['epochs'] = epochs
        return super(VQC, self).fit(x, y,
                             batch_size=batch_size,
                             epochs=epochs,
                             validation_data=validation_data,
                             *args, **kwargs)
    
    def evaluate(self, x, y, *args, **kwargs):
        self._attributes['test_size'] = len(x)
        if isinstance(x, np.ndarray):
            x = self.convert_to_tensor(x)
        return super(VQC, self).evaluate(x, y, *args, **kwargs)
    
    def predict(self, x, *args, **kwargs):
        self._attributes['predict_size'] = len(x)
        if isinstance(x, np.ndarray):
            x = self.convert_to_tensor(x)
        return super(VQC, self).predict(x, *args, **kwargs)    
    
    def roc_curve(self, x, y, callbacks=None):
        y_true = y
        y_score = self.predict(x)
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        if callbacks is not None:
            for callback in callbacks:
                if isinstance(callback, quple.classifiers.VQCLogger):
                    callback.log_roc_curve(fpr, tpr, roc_auc)
        return fpr, tpr, thresholds, roc_auc
        