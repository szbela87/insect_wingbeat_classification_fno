#!/bin/bash

MODEL=large
TEST_SPLIT_RATIO=0.25
EPOCHS=25
TRAIN_BATCH_SIZE=32
VALID_BATCH_SIZE=32
POOL_SIZE=2
MODES=16
N_TRIALS=50
EVAL_FREQ=4

python ht.py --model $MODEL --test_split_ratio $TEST_SPLIT_RATIO --epochs $EPOCHS --train_batch_size $TRAIN_BATCH_SIZE --valid_batch_size $VALID_BATCH_SIZE --pool_size $POOL_SIZE --modes $MODES --n_trials $N_TRIALS --eval_freq $EVAL_FREQ
