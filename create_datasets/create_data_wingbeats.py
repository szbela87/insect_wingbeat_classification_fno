from functions import *
from timeit import default_timer
import numpy as np

"""
This script creates numpy files for training and testing corresponding
to https://github.com/xutong30/WbNet-ResNet-Attention
"""

# Creating the data file used during the training (training and validation set)
t_s = default_timer()
target_names = ['Ae. aegypti', 'Ae. albopictus', 'An. gambiae',
                'An. arabiensis', 'C. pipiens', 'C. quinquefasciatus']

X, y = get_data(target_names,'trainData_Wingbeats.csv')
t_e = default_timer()

data = np.hstack((X,y.reshape(-1,1)))

print(f"Elapsed time: {t_e-t_s:.2f}s")
print(X.shape, y.shape, data.shape)

with open("train_wingbeats.npy","wb") as f:
	np.save(f,data)

# Creating the test data file
t_s = default_timer()

X, y = get_data(target_names,'testData_Wingbeats.csv')
t_e = default_timer()

data = np.hstack((X,y.reshape(-1,1)))

print(f"Elapsed time: {t_e-t_s:.2f}s")
print(X.shape, y.shape, data.shape)

with open("test_wingbeats.npy","wb") as f:
	np.save(f,data)
