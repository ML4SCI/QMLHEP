# This generates the results of the bechmarking code

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import benchmarking.Benchmarking as Benchmarking


"""
Here are possible combinations of benchmarking user could try.
Unitaries: ['U_TTN', 'U2_equiv', 'U4_equiv', 'U_5', 'U_6', 'U_9', 'U_13', 'U_14', 'U_15', 'U_SO4', 'U_SU4', 'U_SU4_no_pooling', 'U_SU4_1D', 'U_9_1D']
U_num_params: [2, 6, 6, 10, 10, 2, 6, 6, 4, 6, 15, 15, 15, 2]
feature_reduction: ['img16x16x1','resize256', 'pca8']
embedding_type: ["Equivariant-Amplitude", "Amplitude", "Angle"]
dataset: 'mnist' or 'fashion_mnist' or "quark_gluon" (in process)
circuit: 'QCNN' 
cost_fn: 'mse' or 'cross_entropy'
Note: when using 'mse' as cost_fn binary="True" is recommended, when using 'cross_entropy' as cost_fn must be binary="False".
"""


# invariant testings


# U2_equiv is invariant under p4m using MSE
# U2_equiv is invariant under reflections over X using cross_entropy 

Unitaries = ["U2_equiv", "U_5", "U_6"]#,"U_9", "U_15", "U_5"] 
U_num_params = [6,10,10]
Encodings = ["img16x16x1"]
dataset = 'quark_gluon'
classes = [0,1]
binary = True
cost_fn = 'mse'



Benchmarking.Benchmarking(dataset, classes, Unitaries, U_num_params, Encodings, circuit='QCNN', cost_fn=cost_fn, binary=binary)

