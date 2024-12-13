#############################################
#                                           #
# Hyperparameter searching with grid search #
#                                           #
#############################################

from __future__ import division
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import pandas as pd
from timeit import default_timer
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torch
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from model import ResNet9_small, ResNet9_medium, ResNet9_large, ResNet9_FNO, ResNet9_FNO_medium
from functions import *
import os
import random
import argparse
import optuna
from itertools import product

from tqdm import tqdm



os.environ['CUDA_VISIBLE_DEVICES'] ='0,1'

###################
#                 #
# Input arguments #
#                 #
###################

parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, required=True)
parser.add_argument('--test_split_ratio', type=float, required=True)
parser.add_argument('--epochs',type=int,required=True)
parser.add_argument('--train_batch_size',type=int,required=True)
parser.add_argument('--valid_batch_size',type=int,required=True)
parser.add_argument('--eval_freq',type=int,required=True)
parser.add_argument('--pool_size',type=int,required=True)
parser.add_argument('--n_trials',type=int,required=True)
parser.add_argument('--modes', type=int, default=16, help='Modes in the Fourier layers')
args = parser.parse_args()

TRAIN_DATASET = "../data/wingbeats_ht.npy"

    
if args.model != "small" and args.model != "large" and args.model != "medium" and args.model != "fno" and args.model != "fno_med":
    print("Please, choose between the 'small'/'medium'/'large'/'fno'/'fno_med' models")
    exit()

TEST_SPLIT_RATIO = args.test_split_ratio # Training/Validation splitting ration #0.25
EPOCHS = args.epochs
TRAIN_BATCH_SIZE = args.train_batch_size  # Batch size in the training set # 32
VALID_BATCH_SIZE = args.valid_batch_size  # Batch size in the validation set # 8*32
SEED = 2023
OUT_FEATURES = 6

# Creating the results directory
if not os.path.exists('results'):
    os.makedirs('results')

# Fixing the seeds
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
random.seed(SEED)
np.random.seed(SEED)

print(f"Cuda is available: {torch.cuda.is_available()}")
dev_names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
print(f"Device: {dev_names}")
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

####################
#                  #
# Loading the data #
#                  #
####################

t_s = default_timer()
    
target_names = ['Ae. aegypti', 'Ae. albopictus', 'An. gambiae',
            'An. arabiensis', 'C. pipiens', 'C. quinquefasciatus']

# Loading the data used for the training
X = np.load(TRAIN_DATASET, mmap_mode='r') # training set
y = X[:,-1].reshape(-1).astype(int)
X = X[:,:-1]

# Splitting into two pieces (for training and validation set)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, stratify = y, test_size = TEST_SPLIT_RATIO, random_state = SEED)
  

t_e = default_timer()

print(f"Data loading - Elapsed time: {t_e-t_s:.2f}s")

# Standardizing
std_ = np.std(X_train)
mean_ = np.mean(X_train)
X_train = (X_train - mean_) / std_
X_valid = (X_valid - mean_) / std_

# Creating the dataloaders
train_input = torch.FloatTensor(X_train)
train_target = torch.LongTensor(y_train)
valid_input = torch.FloatTensor(X_valid)
valid_target = torch.LongTensor(y_valid)
train_dataset = TensorDataset(train_input, train_target)
valid_dataset = TensorDataset(valid_input, valid_target)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=VALID_BATCH_SIZE, shuffle=False)

EVAL_FREQ = len(train_loader)//args.eval_freq

DIV_FACTOR = 50
FINAL_DIV_FACTOR = 1000
MODES = args.modes
LEARNING_RATE_VALUES = [0.0025,0.0005,0.0001]
WEIGHT_DECAY_VALUES = [5e-5, 5e-4, 5e-3]
KERNEL_SIZE_VALUES = [7, 11, 15]

all_combinations = list(product(LEARNING_RATE_VALUES, WEIGHT_DECAY_VALUES, KERNEL_SIZE_VALUES))


# Objective function
def objective(LEARNING_RATE = 1.0e-4, WEIGHT_DECAY = 1.0e-5, KERNEL_SIZE = 5):
    
    print(40*"-")
    print(f"Lr: {LEARNING_RATE} Wd: {WEIGHT_DECAY} Ks: {KERNEL_SIZE}")
    
    DIV_FACTOR = LEARNING_RATE/1.0e-5 # learning_rate / div_factor = 1.0e-5
    
    # Model
    if args.model == "small":
        model = ResNet9_small(out_features=OUT_FEATURES,kernel_size=KERNEL_SIZE,pool_size=args.pool_size)
    elif args.model == "medium":
        model = ResNet9_medium(out_features=OUT_FEATURES,kernel_size=KERNEL_SIZE,pool_size=args.pool_size)
    elif args.model == "large":
        model = ResNet9_large(out_features=OUT_FEATURES,kernel_size=KERNEL_SIZE,pool_size=args.pool_size)
    elif args.model == "fno":
        model = ResNet9_FNO(out_features=OUT_FEATURES,kernel_size=KERNEL_SIZE,pool_size=args.pool_size,modes=args.modes)
    elif args.model == "fno_med":
        model = ResNet9_FNO_medium(out_features=OUT_FEATURES,kernel_size=KERNEL_SIZE,pool_size=args.pool_size,modes=MODES)
    if len(dev_names)>1:
        model = torch.nn.DataParallel(model)
    model.to(device)

    # Optimizer and loss function
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=WEIGHT_DECAY, nesterov=True)
    criterion = torch.nn.CrossEntropyLoss(reduction="sum").to(device)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LEARNING_RATE,
        div_factor=DIV_FACTOR,
        final_div_factor=FINAL_DIV_FACTOR,
        steps_per_epoch=len(train_loader),
        epochs=EPOCHS,
        verbose=0
    )

    # Training
    b_val_acc = 0.0
    for epoch in range(EPOCHS):
        model.train()
        for batch_id, batch in enumerate(train_loader):
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
        
            if batch_id % EVAL_FREQ == 0:
                # Validation
                valid_loss, valid_acc = evaluate(model, valid_loader, criterion, device)
                if valid_acc > b_val_acc:
                    b_val_acc = valid_acc
                    
        valid_loss, valid_acc = evaluate(model, valid_loader, criterion, device)
        if valid_acc > b_val_acc:
            b_val_acc = valid_acc

    return b_val_acc

best_valid_acc = 0.0
all_results = []    

t_s = default_timer()

# Best parameters - initialization
B_LEARNING_RATE = 0.0
B_WEIGHT_DECAY = 0.0
B_KERNEL_SIZE = 0

# Hyperparameter searching with grid search
with open("ht_results.dat", "w", buffering=1) as result_file:  # Line-buffered writing


    header = f"{'LEARNING_RATE':<15}{'WEIGHT_DECAY':<15}{'KERNEL_SIZE':<15}{'Validation_Accuracy':<20}\n"
    result_file.write(header)
    result_file.write("=" * len(header) + "\n")  
    for idx in range(len(all_combinations)):
        

        LEARNING_RATE, WEIGHT_DECAY, KERNEL_SIZE = all_combinations[idx]
        v_acc = objective(LEARNING_RATE, WEIGHT_DECAY, KERNEL_SIZE)
        if v_acc > best_valid_acc:
            best_valid_acc = v_acc
            B_LEARNING_RATE = LEARNING_RATE
            B_WEIGHT_DECAY = WEIGHT_DECAY
            B_KERNEL_SIZE = KERNEL_SIZE
        all_results.append(v_acc)
        t_e = default_timer()
        
        print(f"{(idx+1)/len(all_combinations)*100:.2f}% | V. Acc: {v_acc*100:.4f}% | Best. V. Acc: {best_valid_acc*100:.4f}% | E. Time: {(t_e-t_s)/60.:.2f}m | R. Time: {(t_e-t_s)/60./(idx+1)*(len(all_combinations)-idx-1):.2f}m")
        formatted_line = f"{LEARNING_RATE:<15.6e}{WEIGHT_DECAY:<15.6e}{KERNEL_SIZE:<15d}{v_acc:<20.6f}\n"
        result_file.write(formatted_line)
        
    result_file.write("=" * len(header) + "\n")  
    formatted_line = f"{B_LEARNING_RATE:<15.6e}{B_WEIGHT_DECAY:<15.6e}{B_KERNEL_SIZE:<15d}\n"
    result_file.write(formatted_line)

     
