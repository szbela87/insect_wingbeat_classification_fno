# Creating the data files

The resulted data files can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1Mt9Qpuc9PUXmThgX6iRlUFqE15nrvWTb?usp=sharing).
Below is a description of how to create these files.

## The *Wingbeats* and the *Abuzz* datasets

For these datasets we have investigated the same training/testing sets splitting as in the article [^fn1].
The github code repository of the mentioned work is available at https://github.com/xutong30/WbNet-ResNet-Attention.
The `create_data_abuzz.py` and the `create_data_wingbeats.py` scripts creates numpy files for the training and the test sets.

Therefore, the following csv files were used with the audio filenames:
- `trainData_Wingbeats.csv`
- `testData_Wingbeats.csv`
- `trainData_Abuzz.csv`
- `testData_Abuzz.csv`

The original *Wingbeats* dataset can be downloaded from [timeseriesclassification.com](http://www.timeseriesclassification.com/description.php?Dataset=MosquitoSound)
or from [kaggle.com](https://www.kaggle.com/datasets/potamitis/wingbeats).

The used *Abuzz* dataset is available at a Google Drive link on the page [WbNet-link](https://github.com/xutong30/WbNet-ResNet-Attention) .
This is a preprocessed version of the original dataset available at https://web.stanford.edu/group/prakash-lab/cgi-bin/mosquitofreq/the-science/data/.
The processed dataset consists of 10 secs long audio signals. This preprocessing was done in the paper [^fn1].

### Example

To create the `train_abuzz.npy` and the `test_abuzz.npy` files just type the following command:
```
./create_data_abuzz.py
```

## The *Fruitflies* dataset

The original *Fruitflies* dataset can be downloaded from [timeseriesclassification.com](http://www.timeseriesclassification.com/description.php?Dataset=FruitFlies).
The `create_data_fruitflies.py` creates `.csv` files for the training and the test sets.


## The *Insects* dataset

The original *Insects* dataset can be downloaded from [timeseriesclassification.com](http://www.timeseriesclassification.com/description.php?Dataset=InsectSound).
The `create_data_insects.py` creates `.csv` files for the training and the test sets.

[^fn1]: Wei, X., Hossain, M.Z. & Ahmed, K.A. A ResNet attention model for classifying mosquitoes from wing-beating sounds. Sci Rep 12, 10334 (2022). https://doi.org/10.1038/s41598-022-14372-x

 
