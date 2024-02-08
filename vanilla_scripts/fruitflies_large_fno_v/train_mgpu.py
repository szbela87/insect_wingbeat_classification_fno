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

os.environ['CUDA_VISIBLE_DEVICES'] ='0'

###################
#                 #
# Input arguments #
#                 #
###################

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--test_split_ratio', type=float, required=True)
parser.add_argument('--epochs',type=int,required=True)
parser.add_argument('--weight_decay', type=float, required=True)
parser.add_argument('--lr', type=float, required=True)
parser.add_argument('--div_factor', type=float, required=True)
parser.add_argument('--final_div_factor', type=float, required=True)
parser.add_argument('--out_features',type=int,required=True)
parser.add_argument('--train_batch_size',type=int,required=True)
parser.add_argument('--valid_batch_size',type=int,required=True)
parser.add_argument('--eval_freq',type=int,required=True)
parser.add_argument('--kernel_size',type=int,required=True)
parser.add_argument('--pool_size',type=int,required=True)
parser.add_argument('--modes', type=int, default=64, help='Modes in the Fourier layers')
args = parser.parse_args()
if args.dataset == "Abuzz":
    print('Dataset: ', args.dataset)
    TRAIN_DATASET = "../data/train_abuzz.npy"
    TEST_DATASET = "../data/test_abuzz.npy"
elif args.dataset == "Wingbeats":
    print('Dataset: ', args.dataset)
    TRAIN_DATASET = "../data/train_wingbeats.npy"
    TEST_DATASET = "../data/test_wingbeats.npy"
elif args.dataset == "Fruitflies":
    print('Dataset: ', args.dataset)
    TRAIN_DATASET = "../data/train_fruitflies.csv"
    TEST_DATASET = "../data/test_fruitflies.csv"
elif args.dataset == "Insects":
    print('Dataset: ', args.dataset)
    TRAIN_DATASET = "../data/train_insects.csv"
    TEST_DATASET = "../data/test_insects.csv"
else:
    print("Please, choose between 'Abuzz'/'Wingbeats'/'Fruitflies'/'Insects'")
    exit()
if args.model != "small" and args.model != "large" and args.model != "medium" and args.model != "fno" and args.model != "fno_med":
    print("Please, choose between the 'small'/'medium'/'large'/'fno'/'fno_med' models")
    exit()

TEST_SPLIT_RATIO = args.test_split_ratio # Training/Validation splitting ration #0.25
RESULTS_FILENAME = "./results/inrun_results" # _x.csv
VALID_RESULTS_FILENAME = "./results/valid_results" # _x.csv
TRAIN_RESULTS_FILENAME = "./results/train_results" # _x.csv
BEST_MODEL_FILENAME = "./results/best-model" # _x.pt
EPOCHS = args.epochs
TRAIN_BATCH_SIZE = args.train_batch_size  # Batch size in the training set # 32
VALID_BATCH_SIZE = args.valid_batch_size  # Batch size in the validation set # 8*32
SEED = 2023
WEIGHT_DECAY = args.weight_decay#0.005
LEARNING_RATE = args.lr#0.0001
DIV_FACTOR = args.div_factor#10
FINAL_DIV_FACTOR = args.final_div_factor#1000
OUT_FEATURES = args.out_features

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
if args.dataset == "Abuzz" or args.dataset == "Wingbeats":
    
    target_names = ['Ae. aegypti', 'Ae. albopictus', 'An. gambiae',
                'An. arabiensis', 'C. pipiens', 'C. quinquefasciatus']

    # Loading the data used for the training
    X = np.load(TRAIN_DATASET, mmap_mode='r') # training set
    y = X[:,-1].reshape(-1).astype(int)
    X = X[:,:-1]

    # Splitting into two pieces (for training and validation set)
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, stratify = y, test_size = TEST_SPLIT_RATIO, random_state = SEED)
elif args.dataset == "Fruitflies":
    
    target_names = ['melanogaster','suzukii','zaprionus']
    target_names_dict = {target_names[i]: i for i in range(len(target_names))}
    
    # Loading the training data
    X = pd.read_csv(TRAIN_DATASET,sep=",")
    columns = X.columns
    X[columns[-1]]=X[columns[-1]].replace(target_names_dict)
    X = X.values
    y = X[:,-1].reshape(-1)
    y = y.astype(int)
    X = X[:,:-1]
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, stratify = y, test_size = TEST_SPLIT_RATIO, random_state = SEED)
    
    target_names = ['Dr. melanogaster','Dr. suzukii','Zaprionus']
elif args.dataset == "Insects":
    target_names = ["Aedes_female","Aedes_male","Fruit_flies","House_flies","Quinx_female","Quinx_male","Stigma_female","Stigma_male","Tarsalis_female","Tarsalis_male"]
    
    X = pd.read_csv(TRAIN_DATASET,sep=",")
    
    columns = X.columns
    #X[columns[-1]]=X[columns[-1]].replace(target_names_dict)
    X = X.values
    y = X[:,-1].reshape(-1)
    y = y.astype(int)
    X = X[:,:-1]
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

# Trainings
for sim_num in range(0,5):
    if args.model == "small":
        model = ResNet9_small(out_features=OUT_FEATURES,kernel_size=args.kernel_size,pool_size=args.pool_size)
    elif args.model == "medium":
        model = ResNet9_medium(out_features=OUT_FEATURES,kernel_size=args.kernel_size,pool_size=args.pool_size)
    elif args.model == "large":
        model = ResNet9_large(out_features=OUT_FEATURES,kernel_size=args.kernel_size,pool_size=args.pool_size)
    elif args.model == "fno":
        model = ResNet9_FNO(out_features=OUT_FEATURES,kernel_size=args.kernel_size,pool_size=args.pool_size,modes=args.modes)
    elif args.model == "fno_med":
        model = ResNet9_FNO_medium(out_features=OUT_FEATURES,kernel_size=args.kernel_size,pool_size=args.pool_size,modes=args.modes)
    if len(dev_names)>1:
        model = torch.nn.DataParallel(model)
    model.to(device)
    print(f"Number of the parameters: {count_parameters(model)}\n")
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=WEIGHT_DECAY, nesterov=True)
    criterion = torch.nn.CrossEntropyLoss(reduction="sum").to(device)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LEARNING_RATE, div_factor=DIV_FACTOR, final_div_factor=FINAL_DIV_FACTOR, steps_per_epoch=len(train_loader), epochs = EPOCHS, verbose=0)

    train_accs = []
    train_losses = []
    valid_accs = []
    valid_losses = []



    f = open(f"{RESULTS_FILENAME}_{sim_num}.csv", "w")
    f.write(160*"-"+"\n")
    f.write(f"Device: {dev_names[0]} | Number: {len(dev_names)}\n")
    f.write(f"Epochs: {EPOCHS}\n")
    f.write(f"Optimizer: {type (optimizer).__name__}\n") 
    f.write(f"Scheduler: {type (scheduler).__name__}\n") 
    f.write(f"Div factor: {DIV_FACTOR}\n") 
    f.write(f"Final div factor: {FINAL_DIV_FACTOR}\n") 
    f.write(f"Weight decay: {WEIGHT_DECAY}\n") 
    f.write(f"Learning rate: {LEARNING_RATE}\n") 
    f.write(f"Number of the parameters: {count_parameters(model)}\n")
    f.write(f"Model: {model}\n")
    f.write(160*"-"+"\n")
    f.close()
    print("Training")
    print(5 * "-" + f"{sim_num:5}" + 4*" "+ 160 * "-")

    best_valid_loss = float('inf')
    best_valid_acc = -1.0
    valid_acc = 0.0

    all_time_s = 0.0
    lr = 0.0

    train_accs = []
    train_losses = []
    valid_accs = []
    valid_losses = []
    valid_indices = []

    # Training the `sim_num`-th model
    for epoch in range(EPOCHS):

        start_time = default_timer()

        epoch_loss = 0.0
        epoch_acc = 0.0

        model.train()

        batch_id = 0
        number_of_training_elements = 0

        valid_accs_temp = []
        valid_losses_temp = []
        valid_indices_temp = []

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()

            y_pred = model(x)

            loss = criterion(y_pred, y)
            batch_size = x.shape[0]
            number_of_training_elements += batch_size

            loss.backward()
            optimizer.step()

            end_time = default_timer()

            # Evaluating the model
            if (batch_id+1)%args.eval_freq==0:

                valid_indices_temp.append(batch_id+1)
                valid_loss, valid_acc = evaluate(model, valid_loader, criterion, device)

                valid_losses_temp.append(valid_loss)
                valid_accs_temp.append(valid_acc)

                if valid_acc > best_valid_acc:
                    best_valid_acc = valid_acc
                    torch.save(model.state_dict(), f"{BEST_MODEL_FILENAME}_{sim_num}.pt")

                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss

                lr = scheduler.get_last_lr()[0]

                line = f'\t | Epoch: {epoch+1:03} | Batch Id: {batch_id+1:05} | ET: {end_time-start_time:.2f}s | lr: {lr:.2e} | Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}% | B. Val. Loss: {best_valid_loss:.3f} |  B. Val. Acc: {best_valid_acc*100:.2f}%'
                print(line)
                f = open(f"{RESULTS_FILENAME}_{sim_num}.csv", "a")
                f.write(line+"\n")
                f.close()



            batch_id+=1
            scheduler.step()

        valid_indices_temp.append(batch_id)
        valid_loss, valid_acc = evaluate(model, valid_loader, criterion, device)

        valid_losses_temp.append(valid_loss)
        valid_accs_temp.append(valid_acc)

        valid_losses.append(valid_losses_temp)
        valid_accs.append(valid_accs_temp)

        valid_indices.append(valid_indices_temp)

        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            torch.save(model.state_dict(), f"{BEST_MODEL_FILENAME}_{sim_num}.pt")

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss

        train_loss, train_acc = evaluate(model, train_loader, criterion, device)

        end_time = default_timer()

        all_time_s += end_time - start_time

        train_losses.append(train_loss)
        train_accs.append(train_acc)

        line = f'Epoch: {epoch+1:03} | ET: {end_time-start_time:.2f}s | \t Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}% \t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}% \t | B. Val. Loss: {best_valid_loss:.3f} |  B. Val. Acc: {best_valid_acc*100:.2f}%'
        print(line)
        print(160*"-")

        f = open(f"{RESULTS_FILENAME}_{sim_num}.csv", "a")
        f.write(line+"\n")
        f.write(160*"-"+"\n")
        f.close()

    line = f"\nDuration: {all_time_s:.2f}s\n"
    f = open(f"{RESULTS_FILENAME}_{sim_num}.csv", "a")
    f.write(line+"\n")
    f.write(80*"-"+"\n")
    f.close()

    # Saving the results for analyzing them later in the evaluation part
    valid_losses_plot = []
    valid_accs_plot = []
    epoch_plot = []
    for epoch in range(len(valid_accs)):
        valid_accs_temp = valid_accs[epoch]
        valid_losses_temp = valid_losses[epoch]
        valid_indices_temp = valid_indices[epoch]
        ind = 0
        for mini_batch_id in valid_indices_temp:
            epoch_plot.append(epoch + mini_batch_id/len(train_loader))
            valid_accs_plot.append(valid_accs_temp[ind]*100)
            valid_losses_plot.append(valid_losses_temp[ind])
            ind += 1

    valid_results = pd.DataFrame({"epoch":epoch_plot,
                  "valid_loss":valid_losses_plot,
                  "valid_acc":valid_accs_plot
                  })

    valid_results.to_csv(f"{VALID_RESULTS_FILENAME}_{sim_num}.csv",sep=";",index=False)
    train_accs = [acc*100 for acc in train_accs]
    train_results = pd.DataFrame({"epoch":list(np.arange(1,EPOCHS+1,1)),
                  "train_loss":train_losses,
                  "train_acc":train_accs
                  })
    train_results.to_csv(f"{TRAIN_RESULTS_FILENAME}_{sim_num}.csv",sep=";",index=False)
