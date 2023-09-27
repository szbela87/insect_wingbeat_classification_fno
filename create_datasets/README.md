# Creating the data files

The resulted data files can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1kt94eoQ4LKunu0DCHxmZfUbXmmrlpdK2?usp=sharing).
Below is a description of how to create these files.

## The *Wingbeats* and the *Abuzz* datasets

For these datasets we have investigated the same training/testing sets splitting as in the article [^fn1].
The github code repository of the mentioned work is available at https://github.com/xutong30/WbNet-ResNet-Attention.
The `create_data_abuzz.py` and the `create_data_wingbeats.py` scripts creates numpy files for the training and the test sets.

Therefore, the following csv files were used with the audio filenames.
[Google Drive](https://drive.google.com/drive/folders/1uDbzjY38QrmuglkwKVyIlLYxNpAvMKZk?usp=sharing).

The original *Wingbeats* dataset can be downloaded from [timeseriesclassification.com](http://www.timeseriesclassification.com/description.php?Dataset=MosquitoSound)
or from [kaggle.com](https://www.kaggle.com/datasets/potamitis/wingbeats).

The used *Abuzz* dataset is available at [Google Drive](https://drive.google.com/file/d/1iEX6DTU1euZyLbGX19EQgMwOGEXLGTte/view)
or [Google Drive](https://drive.google.com/file/d/1qRiiPYCpaoAxv--o2EGcFoHeQPk3JFZM/view?usp=sharing).
This is a preprocessed version of the original dataset available at https://web.stanford.edu/group/prakash-lab/cgi-bin/mosquitofreq/the-science/data/.
The processed dataset consists of 10 secs long audio signals. This preprocessing was done in the paper [^fn1].

### Example

To create the `train_abuzz.npy` and the `test_abuzz.npy` files just type the following command:
```
./create_data_abuzz.py
```

## The *Fruitflies* dataset

The original *Fruitflies* dataset can be downloaded from [timeseriesclassification.com](http://www.timeseriesclassification.com/description.php?Dataset=FruitFlies).

## The *Insects* dataset

The original *Insects* dataset can be downloaded from [timeseriesclassification.com](http://www.timeseriesclassification.com/description.php?Dataset=InsectSound).
The `create_data_insects.py` creates `.csv` files for the training and the test sets.

[^fn1]: Wei, X., Hossain, M.Z. & Ahmed, K.A. A ResNet attention model for classifying mosquitoes from wing-beating sounds. Sci Rep 12, 10334 (2022). https://doi.org/10.1038/s41598-022-14372-x

 
