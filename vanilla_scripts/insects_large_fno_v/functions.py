import os
import soundfile as sf
import librosa
import seaborn as sn
from scipy import signal
import numpy as np
import torch
import pandas as pd

def seed_worker(worker_id):
    """
    https://pytorch.org/docs/stable/notes/randomness.html
    """
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.cuda.manual_seed_all(seed)
   
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def calculate_accuracy(y_pred, y):
    top_pred = y_pred.argmax(1, keepdim=True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc
    
def evaluate(model, iterator, criterion, device):

    epoch_loss = 0.0
    epoch_acc = 0.0

    model.eval()
    number_of_elements = 0

    with torch.no_grad():

        for x, y in iterator:

            x = x.to(device)
            y = y.to(device)
            
            batch_size = x.shape[0]
            number_of_elements += batch_size
            
            y_pred = model(x)
            loss = criterion(y_pred, y)
            
            top_pred = y_pred.argmax(1, keepdim=True)
            acc = top_pred.eq(y.view_as(top_pred)).sum()

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / number_of_elements, epoch_acc / number_of_elements
    
def get_data(target_names,files):

    """
    Loading the data corresponding to https://github.com/xutong30/WbNet-ResNet-Attention
    ** Used only for creating the train.npy and the test.npy files. **
    """
    
    df = pd.read_csv(files)
    df["target"]=df["Genera"]+". "+df["Species"]
    df.drop(["Genera","Species"],axis=1,inplace=True)
    
    targets = {}
    for i in range(len(target_names)):
        targets[target_names[i]] = i
        
    bad_files = 0
    
    num = 0
    X = []
    y = []
    for i in range(len(df)):
        line = df.iloc[i]
        target = line["target"]
        filename = line["Fname"].replace("\\","/")
        print(f"#{num+1} | {filename}")
        try:
            #data1, fs = librosa.load(filename)
            #data1, fs = sf.read(filename)
            data, fs = librosa.load(filename,sr=8000) # not every file is in 8KHz
            X.append(data)
            y.append(targets[target])
            #sprint(f"{data.shape} {filename} | {fs} | min: {np.min(data):.2f} | max: {np.max(data):.2f} | diff: {np.max(np.abs(abs(data1-data))):.2f}")
            
            num += 1
            #if num > 3:
            #    break
        except:
            bad_files += 1
            
    
    X = np.array(X).astype("float32")
    y = np.array(y).astype("int")

    #print(target, '#recs = ', num)
    print("")
    print('# of classes: %d' % len(np.unique(y)))
    print('total dataset size: %d' % X.shape[0])
    print('Sampling frequency = %d Hz' % fs)
    print("n_samples: %d" % X.shape[1])
    print(f"bad files: {bad_files}")
    print("duration (sec): %f" % (X.shape[1]/fs))

    return X, y

