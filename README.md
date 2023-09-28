# CF-ResNet-1D-9 Models for Insect Wingbeat Sound Classification

## Getting Started
### Environment Requirements

First, please make sure you have installed Conda. Then, our environment can be installed by:
```
conda create -n insect_classification python=3.8
conda activate insect_classification
pip install -r requirements.txt
```

### Data Preparation
 
You can obtain all the four datasets from [Google Drive](---temporarily removed due to the double-blind review process---).

```
mkdir data
```
**Please put them in the `./data` directory.**

**Also copy the `evaluation.py`, `train_mgpu.py` and `functions.py` files in the corresponding directory.**

### Training Example
In the different directories, we provide the training scripts, i.e. for the *Wingbeats* dataset and the large *CF-ResNet-1D-9* model can be found
at `./wingbeats_large_fno`. These directories are:
* `wingbeats_small_fno`
* `wingbeats_large_fno`
* `fruitflies_small_fno`
* `fruitflies_large_fno`
* `abuzz_small_fno`
* `abuzz_large_fno`
* `insects_small_fno`
* `insects_large_fno`

The `fno`/`fno_med` words in the folder names indicate that the small/large *CF-ResNet-1D-9* models were used on the corresponding dataset.

Each training script produces 5 training sessions.

To train the model use the command 
```
./train_wingbeats_fno.sh
```

The output files, i.e. the best models and logs are in the `./wingbeats_large/results` directory after the training, these are:
* `best_model_{i}.pt` - the model with best validation accuracy from the i-th training,
* `inrun_results_{i}.csv` - the results during the training by the i-th training (just for logging),
* `train_results_{i}.csv` - the training results corresponding to the i-th training process,
* `valid_results_{i}.csv` - the validation results corresponding to the i-th training process.

Here, $i=0...4$.

To evaluate these results, and evaluate the `best_model_{i}`, $i=0...4$ models on the test set use the following command:
```
./evaluation_wingbeats_fno.sh
``` 
It requires the files
`best_model_{i}.pt`, `inrun_results_{i}.csv`, `train_results_{i}.csv`, `valid_results_{i}.csv` for $i=0...4$ from the `results` directory and
creates the `results.dat` file which contains evaluation metrics achieved by the model with the highest validation accuracy. 
It also creates the confusion matrix corresponding the model with the highest validation accuracy,
and makes plots about the accuracies and the losses in `.svg` format among the 5 independent runs. 

## Best Models

The trained models are available at [Google Drive](---temporarily removed due to the double-blind review process---).

## About the datasets

The descriptions of how to generate the data files for the trainings and the tests can be found in the `create_datasets` directory.
 
