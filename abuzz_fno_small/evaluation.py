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
from model import ResNet9_small, ResNet9_large, ResNet9_FNO
from functions import *
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt   
import os
import random
from matplotlib.ticker import MaxNLocator
import argparse

#########################
#                       #
#    Input arguments    #
#                       #
#########################

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--test_split_ratio', type=float, required=True)
parser.add_argument('--batch_size', type=int, required=True)
parser.add_argument('--kernel_size', type=int, required=True)
parser.add_argument('--pool_size', type=int, required=True)
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
else:
    print("Please, choose between 'Abuzz' or 'Wingbeats' or 'FruitFlies'.")
    exit()
if args.model != "small" and args.model != "large" and args.model != "fno":
    print("Please, choose between the 'small'/'large'/'fno' models")
    exit()


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

TEST_SPLIT_RATIO = args.test_split_ratio # Training/Validation splitting ration
RESULTS_FILENAME = "./results/inrun_results" # _x.csv
VALID_RESULTS_FILENAME = "./results/valid_results" # _x.csv
TRAIN_RESULTS_FILENAME = "./results/train_results" # _x.csv
BEST_MODEL_FILENAME = "./results/best-model" # _x.pt
BATCH_SIZE = args.batch_size  # Batch size in the training set
SEED = 2023

random.seed(SEED)
np.random.seed(SEED)

print(f"Cuda is available: {torch.cuda.is_available()}")
dev_names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
print(f"Device: {dev_names}")
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#########################
#                       #
# Accuracies and losses #
#                       #
#########################
#"""
epoch_df = pd.read_csv(f"{VALID_RESULTS_FILENAME}_{0}.csv",sep=";")
epoch_df = epoch_df.drop(["valid_loss","valid_acc"], axis=1)
epoch_df["epoch"] = epoch_df["epoch"].astype(str)
epochs = list(epoch_df["epoch"])
epochs_new =  [item[:10] for item in epochs]
epoch_df["epoch"] = epochs_new
df_new = epoch_df.set_index("epoch")
for i in range(5):
    df_valid = pd.read_csv(f"{VALID_RESULTS_FILENAME}_{i}.csv",sep=";")
    df_valid = df_valid.rename(columns={"valid_loss":f"valid_loss_{i}",
                         "valid_acc":f"valid_acc_{i}"})
    df_valid["epoch"] = df_valid["epoch"].astype(str)
    df_valid_epoch = list(df_valid["epoch"])
    df_valid_epoch_new = [item[:10] for item in df_valid_epoch]
    df_valid["epoch"] = df_valid_epoch_new
    df_valid = df_valid.set_index("epoch")
    df_new = df_new.join(df_valid)
df_new = df_new.reset_index()
df_new["epoch"] = df_new["epoch"].astype(float)

df_new["valid_loss_max"]=df_new[[f"valid_loss_{i}" for i in range(5)]].max(axis=1)
df_new["valid_loss_min"]=df_new[[f"valid_loss_{i}" for i in range(5)]].min(axis=1)
df_new["valid_loss_mean"]=df_new[[f"valid_loss_{i}" for i in range(5)]].mean(axis=1)

df_new["valid_acc_max"]=df_new[[f"valid_acc_{i}" for i in range(5)]].max(axis=1)
df_new["valid_acc_min"]=df_new[[f"valid_acc_{i}" for i in range(5)]].min(axis=1)
df_new["valid_acc_mean"]=df_new[[f"valid_acc_{i}" for i in range(5)]].mean(axis=1)

EPOCHS = np.max(df_new["epoch"].values).astype(int)
epochs_train = np.arange(1,EPOCHS+1)
df_new_train = pd.DataFrame({"epoch":epochs_train})
df_new_train = df_new_train.set_index("epoch")

for i in range(5):
    df_train = pd.read_csv(f"{TRAIN_RESULTS_FILENAME}_{i}.csv",sep=";")
    df_train = df_train.rename(columns={"train_loss":f"train_loss_{i}",
                         "train_acc":f"train_acc_{i}"})
    
    df_train = df_train.set_index("epoch")
    df_new_train = df_new_train.join(df_train)

df_new_train["train_loss_max"]=df_new_train[[f"train_loss_{i}" for i in range(5)]].max(axis=1)
df_new_train["train_loss_min"]=df_new_train[[f"train_loss_{i}" for i in range(5)]].min(axis=1)
df_new_train["train_loss_mean"]=df_new_train[[f"train_loss_{i}" for i in range(5)]].mean(axis=1)

df_new_train["train_acc_max"]=df_new_train[[f"train_acc_{i}" for i in range(5)]].max(axis=1)
df_new_train["train_acc_min"]=df_new_train[[f"train_acc_{i}" for i in range(5)]].min(axis=1)
df_new_train["train_acc_mean"]=df_new_train[[f"train_acc_{i}" for i in range(5)]].mean(axis=1)
df_new_train = df_new_train.reset_index()

SMALL_SIZE = 15
MEDIUM_SIZE = 22
BIGGER_SIZE = 22

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

#########################
#                       #
#     Creating plots    #
#                       #
#########################

# Accuracies
fig, ax = plt.subplots(1,1,figsize=(10,8))
plot_exp = ax.plot(df_new["epoch"], df_new["valid_acc_mean"], 'orange',label="Validation accuracies")
ax.fill_between(df_new["epoch"], df_new["valid_acc_min"],df_new["valid_acc_max"], color='gold', alpha=0.8)
plot_exp = ax.plot(df_new_train["epoch"], df_new_train["train_acc_mean"], 'b-',label="Training accuracies")
ax.fill_between(df_new_train["epoch"], df_new_train["train_acc_min"],df_new_train["train_acc_max"], color='b', alpha=0.2)
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.xlabel("epoch",weight="bold")
plt.ylabel("accuracy",weight="bold")
plt.legend()
plt.grid()
plt.title(f"{args.dataset} dataset",fontsize=22,weight="bold")
plt.show(block=False)
fig.savefig('accuracies.svg', format='svg')

# Losses
fig, ax = plt.subplots(1,1,figsize=(10,8))
plot_exp = ax.plot(df_new["epoch"], df_new["valid_loss_mean"], 'orange',label="Validation losses")
ax.fill_between(df_new["epoch"], df_new["valid_loss_min"],df_new["valid_loss_max"], color='gold', alpha=0.8)
plot_exp = ax.plot(df_new_train["epoch"], df_new_train["train_loss_mean"], 'b-',label="Training losses")
ax.fill_between(df_new_train["epoch"], df_new_train["train_loss_min"],df_new_train["train_loss_max"], color='b', alpha=0.2)
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.xlabel("epoch",weight="bold")
plt.ylabel("loss",weight="bold")
plt.legend()
plt.grid()
plt.title(f"{args.dataset} dataset",fontsize=22,weight="bold")
plt.show(block=False)
fig.savefig('losses.svg', format='svg')
#"""
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
    X_test = np.load(TEST_DATASET, mmap_mode='r')

    # Loading the test set
    y_test = X_test[:,-1].reshape(-1).astype(int)
    X_test = X_test[:,:-1]
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

    X_test = pd.read_csv(TEST_DATASET,sep=",")
    X_test[columns[-1]]=X_test[columns[-1]].replace(target_names_dict)
    X_test = X_test.values
    y_test = X_test[:,-1].reshape(-1)
    y_test = y_test.astype(int)
    
    target_names = ['Dr. melanogaster','Dr. suzukii','Zaprionus']

t_e = default_timer()

print(f"Data loading - Elapsed time: {t_e-t_s:.2f}s")

# Standardizing
std_ = np.std(X_train)
mean_ = np.mean(X_train)
X_train = (X_train - mean_) / std_
X_valid = (X_valid - mean_) / std_
X_test = (X_test - mean_) / std_

# Creating the dataloaders
train_input = torch.FloatTensor(X_train)
train_target = torch.LongTensor(y_train)
valid_input = torch.FloatTensor(X_valid)
valid_target = torch.LongTensor(y_valid)
test_input = torch.FloatTensor(X_test)
test_target = torch.LongTensor(y_test)
train_dataset = TensorDataset(train_input, train_target)
valid_dataset = TensorDataset(valid_input, valid_target)
test_dataset = TensorDataset(test_input, test_target)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

###############################################
#                                             #
# The model with the best validation accuracy #
#                                             #
###############################################

IND = np.argmax(df_new[[f"valid_acc_{i}" for i in range(5)]].max(axis=0).values)
#IND = 0 

out_features = len(target_names)

if args.model == "small":
    model = ResNet9_small(out_features=out_features,kernel_size=args.kernel_size,pool_size=args.pool_size)
elif args.model == "fno":
    model = ResNet9_FNO(out_features=out_features,kernel_size=args.kernel_size,pool_size=args.pool_size,modes=args.modes)
else:
    model = ResNet9_large(out_features=out_features,kernel_size=args.kernel_size,pool_size=args.pool_size)
state_dict = torch.load(f"{BEST_MODEL_FILENAME}_{IND}.pt")
new_state_dict = {}
for key in state_dict:
    new_key = key.replace('module.','')
    new_state_dict[new_key] = state_dict[key]

model.load_state_dict(new_state_dict)
model.to(device)
print(f"Number of the parameters: {count_parameters(model)}\n")
criterion = torch.nn.CrossEntropyLoss(reduction="sum").to(device)

train_loss, train_acc = evaluate(model, train_loader, criterion, device)
valid_loss, valid_acc = evaluate(model, valid_loader, criterion, device)
test_loss, test_acc = evaluate(model, test_loader, criterion, device)
print(160*"-"+"\n")
print(f"The model with the best validation accuracy:")
print(f'Train Loss: {train_loss:.3f} -  Train Acc: {train_acc*100:.2f}% | Valid Loss: {valid_loss:.3f} - Valid Acc: {valid_acc*100:.2f}% | Test Loss: {test_loss:.3f} - Test Acc: {test_acc*100:.2f}% \t\n')

f = open("results.dat", "w")
f.write(f"Dataset: {args.dataset}\n")
f.write(f"Device: {dev_names[0]} | Number: {len(dev_names)}\n")
f.write(f"Number of the parameters: {count_parameters(model)}\n")
f.write(160*"-"+"\n")
f.write(f"The model with the best validation accuracy:\n")
f.write(f'Train Loss: {train_loss:.3f} -  Train Acc: {train_acc*100:.2f}% | Valid Loss: {valid_loss:.3f} - Valid Acc: {valid_acc*100:.2f}% | Test Loss: {test_loss:.3f} - Test Acc: {test_acc*100:.2f}% \t\n\n')



y_preds = []
y_true = []
with torch.no_grad():
    for x, y in test_loader:
        y_true.append(y.view(-1,1))

        x = x.to(device)
        y = y.to(device)
                
        y_pred = model(x)
            
        top_pred = y_pred.argmax(1, keepdim=True)
        y_preds.append(top_pred.detach().cpu().view(-1,1))

y_preds = torch.cat(y_preds)
y_true = torch.cat(y_true)
results = classification_report(y_true, y_preds, target_names=target_names,output_dict=True)
print(pd.DataFrame(results).T,"\n")
f.write(pd.DataFrame(results).T.to_string())
f.write("\n"+160*"-"+"\n")
f.close()

####################
#                  #
# Confusion matrix #
#                  #
####################

cm = confusion_matrix(y_true, y_preds, labels=list(range(len(target_names))))
SMALL_SIZE = 50
MEDIUM_SIZE = 40
BIGGER_SIZE = 40

target_names_shorter = [target_names[i][:6] for i in range(len(target_names))]

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=0)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=40)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=0)    # legend fontsize
plt.rc('figure', titlesize=0)  # fontsize of the figure title

fig, ax = plt.subplots(figsize=(36,30))
sns.heatmap(cm, annot=True, fmt='g', ax=ax);  #annot=True to annotate cells, ftm='g' to disable scientific notation
ax.set_xlabel('Predicted labels',weight="bold",fontsize=50)
ax.set_ylabel('True labels',weight="bold",fontsize=50); 
tick_marks_x = np.arange(len(target_names))+0.5
tick_marks_y = np.arange(len(target_names))+0.5
plt.xticks(tick_marks_x, target_names_shorter, rotation=45, fontsize=40)
plt.yticks(tick_marks_y, target_names_shorter, rotation=0, fontsize=40)
plt.title(f"{args.dataset} dataset",fontsize=60,weight="bold")
plt.show(block=False)
fig.savefig('cm.svg', format='svg')

#################################
#                               #
# Accuracies among the 5 models #
#                               #
#################################

print(160*"-"+"\n")
print("Accuracies among the 5 models:")
f = open("results.dat", "a")
f.write("Accuracies among the 5 models:\n")
final_results = {"train_losses":[],"train_accuracies":[],
                 "valid_losses":[],"valid_accuracies":[],
                 "test_losses":[],"test_accuracies":[]}
f.close()
for ind in range(5):
    if args.model == "small":
        model = ResNet9_small(out_features=out_features,kernel_size=args.kernel_size,pool_size=args.pool_size)
    elif args.model == "fno":
        model = ResNet9_FNO(out_features=out_features,kernel_size=args.kernel_size,pool_size=args.pool_size,modes=args.modes)
    else:
        model = ResNet9_large(out_features=out_features,kernel_size=args.kernel_size,pool_size=args.pool_size)
    state_dict = torch.load(f"{BEST_MODEL_FILENAME}_{ind}.pt")
    new_state_dict = {}
    for key in state_dict:
        new_key = key.replace('module.','')
        new_state_dict[new_key] = state_dict[key]

    model.load_state_dict(new_state_dict)
    model.to(device)
    
    train_loss, train_acc = evaluate(model, train_loader, criterion, device)
    valid_loss, valid_acc = evaluate(model, valid_loader, criterion, device)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    
    final_results["train_losses"].append(train_loss)
    final_results["train_accuracies"].append(train_acc)
    
    final_results["valid_losses"].append(valid_loss)
    final_results["valid_accuracies"].append(valid_acc)
    
    final_results["test_losses"].append(test_loss)
    final_results["test_accuracies"].append(test_acc)
    
    print(f'Model {ind} | Train Loss: {train_loss:.3f} -  Train Acc: {train_acc*100:.2f}% | Valid Loss: {valid_loss:.3f} - Valid Acc: {valid_acc*100:.2f}% | Test Loss: {test_loss:.3f} - Test Acc: {test_acc*100:.2f}% \t')
    f = open("results.dat", "a")
    f.write(f'Model {ind} | Train Loss: {train_loss:.3f} -  Train Acc: {train_acc*100:.2f}% | Valid Loss: {valid_loss:.3f} - Valid Acc: {valid_acc*100:.2f}% | Test Loss: {test_loss:.3f} - Test Acc: {test_acc*100:.2f}% \t\n')
    f.close()
    
f = open("results.dat", "a")
print(f"\nMean test accuracy: {np.mean(final_results['test_accuracies']):.4f}")
f.write(f"\nMean test accuracy: {np.mean(final_results['test_accuracies']):.4f}\n")
print(f"Max test accuracy: {np.max(final_results['test_accuracies']):.4f}")
f.write(f"Max test accuracy: {np.max(final_results['test_accuracies']):.4f}\n")
print(f"Mean test loss: {np.mean(final_results['test_losses']):.4f}")
f.write(f"Mean test loss: {np.mean(final_results['test_losses']):.4f}\n")
print(f"Min test loss: {np.min(final_results['test_losses']):.4f}\n")
f.write(f"Min test loss: {np.min(final_results['test_losses']):.4f}\n")
f.close()

print(160*"-"+"\n")

input("Press the Enter key to continue: ")

