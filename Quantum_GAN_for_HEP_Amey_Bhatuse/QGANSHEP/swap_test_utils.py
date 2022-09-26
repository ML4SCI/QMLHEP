import numpy as np
import sympy as sp
import cirq
import tensorflow as tf
import tensorflow_quantum as tfq

def one_proj(a):

    """
    returns : projection operator for a given state
    """
    return 0.5 * (1 - cirq.Z(a))

def count_set_bits(n):

    """
    returns : count of the set bits in a decimal number
    """
    count=0
    while(n>0):
        count += n&1
        n >>= 1
        
    return count

def swap_test_op(qubits_a,qubits_b):

    """
    returns: swap test operator for the variational swap test
    """

    ret_op = 0
    for i in range(1<<len(qubits_a)):
        if count_set_bits(i)%2 == 0:
            tmp_op = 1
            for j,ch in enumerate(bin(i)[2:].zfill(len(qubits_a))):
                intermediate = one_proj(qubits_a[j]) * one_proj(qubits_b[j])
                if ch =='0':
                    intermediate = 1 - intermediate
                tmp_op *= intermediate
            ret_op += tmp_op

    return 1-(2*ret_op - 1)

def data_encoding_circuit(qubits,rotations):

    """
    returns: data encoding circuit for encoding data from images into quantum cirucit gates
    """

    circuit =  cirq.Circuit()
    for i,q in enumerate(qubits):
        circuit += cirq.ry(rotations[i])(q)
        
    return circuit,rotations

def variational_swap_test_circuit(qubits_a,qubits_b,rotations):

    """
    returns: variational circuit for the swap test 
    """

    circuit = cirq.Circuit()
    for q0,q1 in zip(qubits_a,qubits_b):
        circuit += (cirq.CNOT(q0,q1))
    rotations = np.reshape(rotations,(-1,2))
    for i, q in enumerate(qubits_a):
        circuit += cirq.Y(q)**(rotations[i][0])
        circuit += cirq.X(q)**(rotations[i][1])
  
    return circuit,np.ndarray.flatten(rotations)

def quantum_state_overlap(generator_model_,real_data,random_data):
    """
    Function for calculating how much two states overlap each other using dot product

    Arguments: 
      generator_model(generator model)
      real_data(original data)
      random_data(fake data/noise)

    Returns: 
      overlap between two states created using two data entries
      
    """
    intermediate_output_real = generator_model_.get_layer('Swap_Test_Layer').input[0]
    intermediate_output_gen = generator_model_.get_layer('Swap_Test_Layer').input[1]
    generator_model_1 = tf.keras.models.Model(inputs=[generator_model_.input],outputs=[intermediate_output_real,intermediate_output_gen])
    random_data_samples = generator_model_1.predict([real_data,random_data])
    rotations_r = sp.symbols('real_x_:'+str(intermediate_output_real.shape[1]))
    rotations_g = sp.symbols('gen_x_:'+str(intermediate_output_gen.shape[1]))
    circuit1 =  cirq.Circuit()
    circuit2 = cirq.Circuit()
    qubits1 = cirq.GridQubit.rect(1,4)
    qubits2 = cirq.GridQubit.rect(1,4)
    for i,q in enumerate(qubits1):
        circuit1 += cirq.ry(rotations_r[i])(q)
    for i,q in enumerate(qubits2):
        circuit2 += cirq.ry(rotations_g[i])(q)
    real_state = tfq.layers.State()(circuit1,symbol_names=rotations_r,symbol_values=np.mean(random_data_samples[0],axis=0).reshape(1,len(random_data_samples[0][0])))
    real_state = real_state.to_tensor(default_value=0).numpy()
    gen_state = tfq.layers.State()(circuit2,symbol_names=rotations_g,symbol_values=np.mean(random_data_samples[1],axis=0).reshape(1,len(random_data_samples[0][0])))
    gen_state = gen_state.to_tensor(default_value=0).numpy()
    state_overlap = np.dot(real_state[0],gen_state[0])
    return state_overlap

