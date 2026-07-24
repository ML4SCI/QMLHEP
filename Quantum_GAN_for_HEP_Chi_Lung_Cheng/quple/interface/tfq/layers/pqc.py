import numbers
import numpy as np
import tensorflow as tf

import cirq
import sympy
from tensorflow_quantum.python.layers.circuit_executors import \
    expectation, sampled_expectation
from tensorflow_quantum.python.layers.circuit_construction import elementary
from tensorflow_quantum.python import util

import quple
from quple import QuantumCircuit
from quple.interface.tfq.tf_resolvers import resolve_formulas

class PQC(tf.keras.layers.Layer):
    """Parametrized Quantum Circuit (PQC) Layer.
    This layer is for training parameterized quantum models.
    Given a parameterized circuit, this layer initializes the parameters
    and manages them in a Keras native way.
    We start by defining a simple quantum circuit on one qubit.
    This circuit parameterizes an arbitrary rotation on the Bloch sphere in
    terms of the three angles a, b, and c:
    >>> q = cirq.GridQubit(0, 0)
    >>> (a, b, c) = sympy.symbols("a b c")
    >>> circuit = cirq.Circuit(
    ...     cirq.rz(a)(q),
    ...     cirq.rx(b)(q),
    ...     cirq.rz(c)(q),
    ...     cirq.rx(-b)(q),
    ...     cirq.rz(-a)(q)
    ... )
    In order to extract information from our circuit, we must apply measurement
    operators.  For now we choose to make a Z measurement.  In order to observe
    an output, we must also feed our model quantum data (NOTE: quantum data
    means quantum circuits with no free parameters).  Though the output values
    will depend on the default random initialization of the angles in our model,
    one will be the negative of the other since `cirq.X(q)` causes a bit flip:
    >>> outputs = tfq.layers.PQC(circuit, cirq.Z(q))
    >>> quantum_data = tfq.convert_to_tensor([
    ...     cirq.Circuit(),
    ...     cirq.Circuit(cirq.X(q))
    ... ])
    >>> res = outputs(quantum_data)
    >>> res
    <tf.Tensor: id=577, shape=(2, 1), dtype=float32, numpy=
    array([[ 0.8722095],
           [-0.8722095]], dtype=float32)>
    We can also choose to measure the three pauli matrices, sufficient to
    fully characterize the operation of our model, or choose to simulate
    sampled expectation values by specifying a number of measurement shots
    (repetitions) to average over.  Notice that using only 200 repetitions
    introduces variation between the two rows of data, due to the
    probabilistic nature of measurement.
    >>> measurement = [cirq.X(q), cirq.Y(q), cirq.Z(q)]
    >>> outputs = tfq.layers.PQC(circuit, measurement, repetitions=200)
    >>> quantum_data = tfq.convert_to_tensor([
    ...     cirq.Circuit(),
    ...     cirq.Circuit(cirq.X(q))
    ... ])
    >>> res = outputs(quantum_data)
    >>> res
    <tf.Tensor: id=808, shape=(2, 3), dtype=float32, numpy=
    array([[-0.38,  0.9 ,  0.14],
           [ 0.19, -0.95, -0.35]], dtype=float32)>
    A value for `backend` can also be supplied in the layer constructor
    arguments to indicate which supported backend you would like to use.
    A value for `differentiator` can also be supplied in the constructor
    to indicate the differentiation scheme this `PQC` layer should use.
    Here's how you would take the gradients of the above example using a
    `cirq.Simulator` backend (which is slower than the default
    `backend=None` which uses C++):
    >>> q = cirq.GridQubit(0, 0)
    >>> (a, b, c) = sympy.symbols("a b c")
    >>> circuit = cirq.Circuit(
    ...     cirq.rz(a)(q),
    ...     cirq.rx(b)(q),
    ...     cirq.rz(c)(q),
    ...     cirq.rx(-b)(q),
    ...     cirq.rz(-a)(q)
    ... )
    >>> measurement = [cirq.X(q), cirq.Y(q), cirq.Z(q)]
    >>> outputs = tfq.layers.PQC(
    ...     circuit,
    ...     measurement,
    ...     repetitions=5000,
    ...     backend=cirq.Simulator(),
    ...     differentiator=tfq.differentiators.ParameterShift())
    >>> quantum_data = tfq.convert_to_tensor([
    ...     cirq.Circuit(),
    ...     cirq.Circuit(cirq.X(q))
    ... ])
    >>> res = outputs(quantum_data)
    >>> res
    <tf.Tensor: id=891, shape=(2, 3), dtype=float32, numpy=
    array([[-0.5956, -0.2152,  0.7756],
           [ 0.5728,  0.1944, -0.7848]], dtype=float32)>
    Lastly, like all layers in TensorFlow the `PQC` layer can be called on any
    `tf.Tensor` as long as it is the right shape. This means you could replace
    `quantum_data` with values fed in from a `tf.keras.Input`.
    """

    def __init__(
            self,
            model_circuit,
            data_circuit,
            operators,
            *,
            repetitions=None,
            backend=None,
            differentiator=None,
            initializer=None,
            regularizer=None,
            constraint=None,
            trainable=True,
            seed=None,
            name=None,
            **kwargs,
    ):
        """Instantiate this layer.
        Create a layer that will output expectation values of the given
        operators when fed classical data to it's input layer. This layer will
        accept one input tensor representing a classical data source which is used
        to resolve the values of the parameter expressions in data_circuit. The resolved
        data_circuit will be appended to the model_circuit which is then executed and
        finally output the expectation values.
        
        Arguments:
            model_circuit: `cirq.Circuit` or `quple.QuantumCircuit` containing 
                `sympy.Symbols` that will be used as the model which will be fed with 
                quantum data from the data circuit.
            data_circuit: `cirq.Circuit` or `quple.QuantumCircuit` containing 
                `sympy.Symbols` that maps the inputs in this layer to the quantum
                data of the circuit which will be fed to the model circuit.
            operators: `cirq.PauliSum` or Python `list` of `cirq.PauliSum` objects
                used as observables at the end of the model circuit.
            repetitions: Optional python `int` indicating how many samples to use
                when estimating expectation values.  If `None` analytic expectation
                calculation is used.
            backend: Optional backend to use to simulate states. Defaults to
                the native TensorFlow simulator (None), however users may also
                specify a preconfigured cirq simulation object to use instead.
                If a cirq object is given it must inherit either
                `cirq.SimulatesFinalState` if analytic expectations are desired or
                `cirq.Sampler` if sampled expectations are desired.
            differentiator: Optional `tfq.differentiator` object to specify how
                gradients of `model_circuit` should be calculated.
            initializer: Optional `tf.keras.initializer` object to specify how the
                symbols in `model_circuit` should be initialized when creating
                the managed variables. If None, defaults to initialize from a
                random uniform distribution between 0 and 2*pi. 
            regularizer: Optional `tf.keras.regularizer` object applied to the
                managed variables parameterizing `model_circuit`.
            constraint: Optional `tf.keras.constraint` object applied to the
                managed variables parameterizing `model_circuit`.
            trainable: Boolean, whether the layer's variables should be trainable.
            seed: Optional integer. Controls the random seed of the weight initializer.
            name: Optional string. Name of the layer.
        """
        super().__init__(trainable=trainable, name=name, **kwargs)

        # Ingest model_circuit.
        if not isinstance(model_circuit, cirq.Circuit):
            raise TypeError("model_circuit must be a cirq.Circuit object."
                            " Given: {}".format(model_circuit))
        # Ingest data_circuit.
        if not isinstance(data_circuit, cirq.Circuit):
            raise TypeError("data_circuit must be a cirq.Circuit object."
                            " Given: {}".format(data_circuit))
            
        if not isinstance(data_circuit, QuantumCircuit):
            data_circuit = QuantumCircuit.from_cirq(data_circuit)
        if not isinstance(model_circuit, QuantumCircuit):
            model_circuit = QuantumCircuit.from_cirq(model_circuit)
            
        if quple.has_composite_symbols(model_circuit):
            raise ValueError("model_circuit symbols must not be composite")            
        
        data_circuit.flatten()
        data_circuit_qubits = data_circuit.qubits
        model_circuit_qubits = model_circuit.qubits
        unaccounted_qubits = set(data_circuit_qubits) - set(model_circuit_qubits)
        if len(unaccounted_qubits) != 0:
            raise ValueError("model_circuit missing the following input qubits from"
                             " the data_circuit: ".format(unaccounted_qubits))
        self._data_qubits = data_circuit_qubits
        self._n_qubit = len(data_circuit_qubits)
        self._input_symbols_list = [v.name for v in data_circuit.expr_map.values()]
        self._base_input_symbols_list = data_circuit.raw_symbols
        self._data_circuit_formulas = list(data_circuit.expr_map)
        self._num_formulas = tf.constant(len(self._data_circuit_formulas))
        self._symbols_list = model_circuit.symbols
        
        if not set(self._base_input_symbols_list).isdisjoint(set(self._symbols_list)):
            raise ValueError("data_circuit and model circuit must not share symbols of the same name")

        self._input_symbols = tf.constant(self._input_symbols_list)
        self._symbols = tf.constant(self._symbols_list)

        self._all_symbols_list = self._input_symbols_list + self._symbols_list
        self._all_symbols = tf.constant(self._all_symbols_list)

        self._model_circuit = util.convert_to_tensor([model_circuit])
        self._data_circuit = util.convert_to_tensor([data_circuit])
        if len(self._symbols_list) == 0:
            raise ValueError("model_circuit has no sympy.Symbols. Please "
                             "provide a circuit that contains symbols so "
                             "that their values can be trained.")
            
        if len(self._input_symbols_list) == 0:
            raise ValueError("data_circuit has no sympy.Symbols. Please "
                             "provide a circuit that contains symbols so "
                             "that their values can be passed as input.")            

        # Ingest operators.
        if isinstance(operators, (cirq.PauliString, cirq.PauliSum)):
            operators = [operators]
        if not isinstance(operators, (list, np.ndarray, tuple)):
            raise TypeError("operators must be a cirq.PauliSum or "
                            "cirq.PauliString, or a list, tuple, "
                            "or np.array containing them. "
                            "Got {}.".format(type(operators)))
        if not all([
                isinstance(op, (cirq.PauliString, cirq.PauliSum))
                for op in operators
        ]):
            raise TypeError("Each element in operators to measure "
                            "must be a cirq.PauliString"
                            " or cirq.PauliSum")
        self._operators = util.convert_to_tensor([operators])

        # Ingest and promote repetitions.
        self._analytic = False
        if repetitions is None:
            self._analytic = True
        if not self._analytic and not isinstance(repetitions, numbers.Integral):
            raise TypeError("repetitions must be a positive integer value."
                            " Given: ".format(repetitions))
        if not self._analytic and repetitions <= 0:
            raise ValueError("Repetitions must be greater than zero.")
        if not self._analytic:
            self._repetitions = tf.constant(
                [[repetitions for _ in range(len(operators))]],
                dtype=tf.dtypes.int32)

        # Set backend and differentiator.
        if not isinstance(backend, cirq.Sampler
                         ) and repetitions is not None and backend is not None:
            raise TypeError("provided backend does not inherit cirq.Sampler "
                            "and repetitions!=None. Please provide a backend "
                            "that inherits cirq.Sampler or set "
                            "repetitions=None.")
        if not isinstance(backend, cirq.SimulatesFinalState
                         ) and repetitions is None and backend is not None:
            raise TypeError("provided backend does not inherit "
                            "cirq.SimulatesFinalState and repetitions=None. "
                            "Please provide a backend that inherits "
                            "cirq.SimulatesFinalState or choose a positive "
                            "number of repetitions.")
        if self._analytic:
            self._executor = expectation.Expectation(
                backend=backend, differentiator=differentiator)
        else:
            self._executor = sampled_expectation.SampledExpectation(
                backend=backend, differentiator=differentiator)

        self._append_layer = elementary.AddCircuit()

        # Set additional parameter controls.
        if initializer is None:
            initializer = tf.keras.initializers.RandomUniform(0, 2 * np.pi, seed=seed)
        self.initializer = tf.keras.initializers.get(initializer)
        self.regularizer = tf.keras.regularizers.get(regularizer)
        self.constraint = tf.keras.constraints.get(constraint)

        self._init_weights()
        self._validate_init()
        
    def _validate_init(self):
        pass
    
    def _validate_build(self, input_shape):
        num_input_symbols = len(self._base_input_symbols_list)
        sum_input_shape = tf.reduce_sum(input_shape[1:])
        if num_input_symbols != sum_input_shape:
            raise ValueError(f"sum of input shape (={sum_input_shape}, excluding batch dimension) must match the"
                             f" number of input symbols in data circuit (={num_input_symbols})")
        
    def _get_input_resolver(self):
        return resolve_formulas(self._data_circuit_formulas,
                                self._base_input_symbols_list)
        
    def _init_weights(self):
        """Initialize trainable weights.
        """
        # Weight creation is not placed in a build function because the number
        # of weights is independent of the input shape.
        self.parameters = self.add_weight('parameters',
                                          shape=self._symbols.shape,
                                          initializer=self.initializer,
                                          regularizer=self.regularizer,
                                          constraint=self.constraint,
                                          dtype=tf.float32,
                                          trainable=True)

    @property
    def symbols(self):
        """The model symbols that are managed by this layer (in-order).
        
        Note: `symbols[i]` indicates what symbol name the managed weights corresponding to 
        the model circuit map to.
        """
        return [sympy.Symbol(x) for x in self._symbols_list]

    def symbol_values(self):
        """Returns a Python `dict` containing the model symbol name, value pairs.
        Returns:
            Python `dict` with `str` keys and `float` values representing
                the current symbol values.
        """
        return dict(zip(self.symbols, self.get_weights()[0]))
    
    @property
    def input_symbols(self):
        """The flattened input symbols that are managed by this layer (in-order).
        """
        return [sympy.Symbol(x) for x in self._input_symbols_list]
    
    @property
    def base_input_symbols(self):
        """The base (non-flattened) input symbols that are managed by this layer (in-order).
        
        Note: `symbols[i]` indicates what symbol name the inputs in this layer map to.
        """
        return [sympy.Symbol(x) for x in self._base_input_symbols_list]     
    
    @property
    def all_symbols(self):
        """The flattened input and model symbols that are managed by this layer (in-order).
        """
        return [sympy.Symbol(x) for x in self._all_symbols_list]     

    def build(self, input_shape):
        """Keras build function."""
        self._input_resolver = self._get_input_resolver()
        self._validate_build(input_shape)
        self.built = True

    def call(self, inputs):
        """Keras call function."""
        circuit_batch_dim = tf.gather(tf.shape(inputs), 0)
        tiled_up_data_circuit = tf.tile(self._data_circuit, [circuit_batch_dim])
        tiled_up_model = tf.tile(self._model_circuit, [circuit_batch_dim])
        model_appended = self._append_layer(tiled_up_data_circuit, append=tiled_up_model)
        tiled_up_parameters_ = tf.tile([self.parameters], [circuit_batch_dim, 1])
        resolved_inputs = self._input_resolver(inputs)     
        tiled_up_parameters = tf.concat([resolved_inputs, tiled_up_parameters_], 1)
        tiled_up_operators = tf.tile(self._operators, [circuit_batch_dim, 1])

        # this is disabled to make autograph compilation easier.
        # pylint: disable=no-else-return
        if self._analytic:
            return self._executor(model_appended,
                                  symbol_names=self._all_symbols,
                                  symbol_values=tiled_up_parameters,
                                  operators=tiled_up_operators)
        else:
            tiled_up_repetitions = tf.tile(self._repetitions,
                                           [circuit_batch_dim, 1])
            return self._executor(model_appended,
                                  symbol_names=self._all_symbols,
                                  symbol_values=tiled_up_parameters,
                                  operators=tiled_up_operators,
                                  repetitions=tiled_up_repetitions)
        # pylint: enable=no-else-return