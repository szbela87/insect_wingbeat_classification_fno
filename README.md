# Applying Fourier Neural Operator to Insect Wingbeat Sound Classification: Introducing CF-ResNet-1D

This repository contains the code for the paper:
- [Applying Fourier Neural Operator to insect wingbeat sound classification: Introducing CF-ResNet-1D](https://www.sciencedirect.com/science/article/pii/S1574954125000640)

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#environment-installation">Environment Installation</a>
    </li>
    <li>
      <a href="#data-preparation">Data Preparation</a>
    </li>
    <li>
      <a href="#training-and-evaluation">Training and Evaluation</a>
    </li>
    <li>
      <a href="#selection-of-hyperparameters">Selection of Hyperparameters</a>
    </li>
    <li>
      <a href="#best-models">Best models</a>
    </li>
    <li><a href="#the-datasets">The datasets</a></li>
  </ol>
</details>

## Getting Started
### Environment Installation

First, please make sure you have installed Conda. Then, our environment can be installed by:
```
conda create -n insect_classification python=3.8
conda activate insect_classification
pip install -r requirements.txt
```

### Data Preparation
 
You can obtain all the four datasets from [Google Drive](https://drive.google.com/drive/folders/1Mt9Qpuc9PUXmThgX6iRlUFqE15nrvWTb?usp=sharing).

```
mkdir data
```
**Please put them in the `./data` directory.**

**Also copy the `evaluation.py`, `train_mgpu.py` and `functions.py` files in the corresponding directory.**

<p align="right">(<a href="#top">back to top</a>)</p>

### Training and Evaluation
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

The scripts for the vanilla FNO-ResNet9 models, where the convolutional kernel size is consistently set to 1, can be found in the `vanilla_scripts` directory.

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

<p align="right">(<a href="#top">back to top</a>)</p>

### Selection of Hyperparameters

The required files for the hyperparameter's selection procedure are in `ht_ft` directory.
To run the process use the following command:

```
./ht_ft.sh
``` 

A grid search approach was applied to optimize these hyperparameters, using 10%
of the training subset of the Wingbeats dataset. The selected small subset was further divided
into training and validation sets in a 75%-25% ratio. During the optimization process, we tuned the convolutional kernel size, the learning rate, and
the weight-decay hyperparameters, targeting the highest validation accuracy.

The small dataset was generated using the `create_data_ht.py` script. The link to the resulting dataset is available in the Google Drive directory shared in the `./create_datasets/Readme.md` file.

<p align="right">(<a href="#top">back to top</a>)</p>

## Best Models

The trained models are available at [Google Drive](https://drive.google.com/drive/folders/1QB-XOD96d909x_L64pBX0gFTI3V_BS4I?usp=sharing).

<p align="right">(<a href="#top">back to top</a>)</p>

## The datasets

The descriptions of how to generate the data files for the trainings and the tests can be found in the `create_datasets` directory.

<p align="right">(<a href="#top">back to top</a>)</p>

## Citations
If you use this repository or the methods described here in your research, please cite the following paper:

```
@article{SZEKERES2025103055,
title = {Applying Fourier Neural Operator to insect wingbeat sound classification: Introducing CF-ResNet-1D},
journal = {Ecological Informatics},
volume = {86},
pages = {103055},
year = {2025},
issn = {1574-9541},
doi = {https://doi.org/10.1016/j.ecoinf.2025.103055},
url = {https://www.sciencedirect.com/science/article/pii/S1574954125000640},
author = {Béla J. Szekeres and Máté Natabara Gyöngyössy and János Botzheim},
keywords = {Audio classification, ResNet architecture, Deep learning, Mosquito wingbeat, Fourier Neural Operator},
abstract = {Mosquitoes and other insects are vectors of severe diseases, posing significant health risks to millions worldwide yearly. Effective classification of insect species, particularly through their wingbeat sounds, is crucial for disease prevention and control. Despite recent advancements in Deep Learning, Fourier Neural Operators (FNO), efficient for solving Partial Differential Equations due to their global spectral representations, have yet to be thoroughly explored for real-world time series classification or regression tasks. This study explores the application of FNOs in insect wingbeat sound classification, focusing on their potential for improving the accuracy and efficiency of such tasks, particularly in the fight against mosquito-borne diseases. We introduce CF-ResNet-1D, a novel ResNet-inspired model that integrates Convolutional Fourier Layers, combining the strengths of FNOs and 1D-Convolutional processing. The model is designed to analyze raw time-domain signals, leveraging the parallel spectral processing capabilities of FNOs. Our findings demonstrate that CF-ResNet-1D significantly outperforms traditional spectrogram-based models in classifying insect wingbeat sounds, achieving state-of-the-art accuracy.}
}
```
 
