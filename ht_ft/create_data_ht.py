# creating small dataset for hyperparameter tuning

import numpy as np
import pandas as pd
from timeit import default_timer
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import os
import random
import argparse

SEED = 2024
TRAIN_SIZE = 0.1
TRAIN_DATASET = "../data/train_wingbeats.npy"
target_names = ['Ae. aegypti', 'Ae. albopictus', 'An. gambiae',
                'An. arabiensis', 'C. pipiens', 'C. quinquefasciatus']

t_s = default_timer()

# Loading the data used for the training
X = np.load(TRAIN_DATASET, mmap_mode='r') # training set
y = X[:,-1].reshape(-1).astype(int)
X = X[:,:-1]

# Splitting into two pieces (for training and validation set)
X_train, _, y_train, _ = train_test_split(X, y, stratify = y, train_size = TRAIN_SIZE, random_state = SEED)

t_e = default_timer()

print(f"Data loading - Elapsed time: {t_e-t_s:.2f}s")
data = np.hstack((X_train,y_train.reshape(-1,1)))
with open("wingbeats_ht.npy","wb") as f:
	np.save(f,data)
