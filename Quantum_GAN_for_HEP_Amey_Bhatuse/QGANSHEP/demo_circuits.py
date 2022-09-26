import sympy as sp
import numpy as np
import cirq

def one_qubit_unitary(qubit, symbols):
  return cirq.Circuit(
    [cirq.rz(symbols[0])(qubit),
      cirq.ry(symbols[1])(qubit),
      cirq.rz(symbols[2])(qubit)]
  )

def two_qubit_unitary(qubits):
  cx_ops = [cirq.CX(q0, q1) for q0, q1 in zip(qubits, qubits[1:])]
  cx_ops += ([cirq.CX(qubits[-1], qubits[0])] if len(qubits) != 2 else [])
  return cx_ops

def pqc_circuit_for_conv(qubits,layers):
  """
  Arguments:
    qubits(cirq.GridQubit)
    layers(number of layers)

  Returns:
    cirq.Circuit(parameterised circuit)
    sympy symbols for gates having input data
    sympy symbols for gates having parameters

  """
  circuit = cirq.Circuit()
  num_qubits = len(qubits)
  input_symbols = sp.symbols('x_:'+str(num_qubits))
  param_symbols = sp.symbols('theta_:'+str(3*num_qubits*layers))
  param_symbols = np.reshape(param_symbols,(layers,num_qubits,3))
  for i in range(num_qubits):
    circuit += cirq.ry(input_symbols[i])(qubits[i])
  
  for layer in range(layers):
    for i,q in enumerate(qubits):
      circuit += one_qubit_unitary(q,param_symbols[layer,i,:])
    circuit += two_qubit_unitary(qubits)
  
  return circuit,input_symbols,list(param_symbols.flat)