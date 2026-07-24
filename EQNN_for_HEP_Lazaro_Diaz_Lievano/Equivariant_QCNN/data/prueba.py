from data import data_load_and_process
import matplotlib.pyplot as plt
import numpy as np 

dataset = "electron_photon"
 
X_train, X_test, y_train, y_test = data_load_and_process(dataset, classes=[0,1], feature_reduction= "img16x16x1", binary=True)

plt.imshow(X_train[0])
plt.show()
print(y_train[0])
print(np.unique(y_train, return_counts = True))

