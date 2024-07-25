from QCNN_circuit import QCNN
import data
import embedding
import matplotlib.pyplot as plt
import numpy as np
from Training import circuit_training

"""
Here are possible combinations of benchmarking user could try.
Unitaries: ['U_TTN', 'U2_equiv', 'U4_equiv', 'U_5', 'U_6', 'U_9', 'U_13', 'U_14', 'U_15', 'U_SO4', 'U_SU4', 'U_SU4_no_pooling', 'U_SU4_1D', 'U_9_1D']
U_num_params: [2, 6, 6, 10, 10, 2, 6, 6, 4, 6, 15, 15, 15, 2]
feature_reduction: ['img16x16x1','resize256', 'pca8']
embedding_type: ["Equivariant-Amplitude", "Amplitude", "Angle"]
dataset: 'mnist' or 'fashion_mnist'
circuit: 'QCNN' 
cost_fn: 'mse' or 'cross_entropy'
Note: when using 'mse' as cost_fn binary="True" is recommended, when using 'cross_entropy' as cost_fn must be binary="False".
"""

# invariant testings
# U2_equiv is invariant under p4m using MSE
# U2_equiv is invariant only under reflections over X using cross_entropy 


U_params = 10
total_params = U_params * 3 + 2 * 3
params = np.random.random(total_params)

U = "U_5"
feature_reduction = "pca8"
circuit = "QCNN"
cost_fn = "mse"
binary = True

X_train, X_test, Y_train, Y_test = data.data_load_and_process( "mnist", [0,1], feature_reduction, binary)


#print(Y_test)


#data and transformed data
matrix = np.array(X_train[1])
#matrix_reflected_y = np.fliplr(matrix) # Image Reflected over Y-axis
#matrix_reflected_x = np.flipud(matrix) # Image Reflected over X-axis
#matrix_rotated_90 = np.rot90(matrix) # Image rotated 90 degrees

#plt.imshow(matrix)
#plt.show()

#plt.imshow(matrix_reflected_y)
#plt.show()

#plt.imshow(matrix_reflected_x)
#plt.show()

#plt.imshow(matrix_rotated_90)
#plt.show()

result1 = QCNN(matrix, params, U, U_params, feature_reduction, cost_fn)
#result2 = QCNN(matrix_reflected_y, params,U, U_params, feature_reduction, cost_fn)
#result3 = QCNN(matrix_reflected_x, params, U, U_params, feature_reduction, cost_fn)
#result4 = QCNN(matrix_rotated_90, params, U, U_params, feature_reduction, cost_fn)



print(f" Measurement for Original Image: {result1}")
#print(f" Measurement for Image Reflected over Y-axis: {result2}")
#print(f" Measurement for Image Reflected over X-axis: {result3}")
#print(f" Measurement for Image Rotated 90 Degrees: {result4}")


circuit_training(X_train, Y_train, U, U_params, feature_reduction, circuit, cost_fn, steps = 10)



