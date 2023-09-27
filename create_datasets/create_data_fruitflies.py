from __future__ import division
import numpy as np
import pandas as pd
from timeit import default_timer
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import os
import random

TEST_SPLIT_RATIO = 0.25 # Training/Testing splitting ration

random.seed(SEED)
np.random.seed(SEED)

# Loading the data
t_s = default_timer()
target_names = ['melanogaster','suzukii','zaprionus']
target_names_dict = {target_names[i]: i for i in range(len(target_names))}

# Loading the training data
X = pd.read_csv("../data/trainData_FruitFlies.csv",sep=",")
columns = X.columns
X = X.values

X_test = pd.read_csv("../data/testData_FruitFlies.csv",sep=",")
X_test = X_test.values

t_e = default_timer()

print(f"Data loading - Elapsed time: {t_e-t_s:.2f}s")
print(f"shapes: {X.shape}, {X_test.shape}")

X = np.concatenate((X,X_test),axis=0)
print(f"X new: {X.shape}")

df = pd.DataFrame(X)
df = df.sample(frac=1).reset_index(drop=True)

df1 = df[:int(len(df)*0.8)].reset_index(drop=True)
df2 = df[int(len(df)*0.8):].reset_index(drop=True)

columns = df1.columns
df1[columns[-1]] = df1[columns[-1]].replace(target_names_dict)
X = df1.values
y = X[:,-1].reshape(-1)
y = y.astype(int)
X = X[:,:-1]
X_train, X_valid, y_train, y_valid = train_test_split(X, y, stratify = y, test_size = TEST_SPLIT_RATIO, random_state = SEED)

df2[columns[-1]]=df2[columns[-1]].replace(target_names_dict)
X_test = df2.values
y_test = X_test[:,-1].reshape(-1)
y_test = y_test.astype(int)

counter_train = {0: 0, 1:0, 2:0}
for i in y_train:
    counter_train[i]+=1
    
counter_valid = {0: 0, 1:0, 2:0}
for i in y_valid:
    counter_valid[i]+=1
    
counter_test = {0: 0, 1:0, 2:0}
for i in y_test:
    counter_test[i]+=1

print(f"target_names_dict {target_names_dict}")
print(f"train: {len(X_train)} valid: {len(X_valid)} test: {len(X_test)}")
print(f"counter train: {counter_train}")
print(f"counter valid: {counter_valid}")
print(f"counter test: {counter_test}")

df1.to_csv("trainData_Fruitflies_new.csv",sep=",",index=False)
df2.to_csv("testData_Fruitflies_new.csv",sep=",",index=False)

print(f"sizes | train: {len(df1)} test: {len(df2)}")

