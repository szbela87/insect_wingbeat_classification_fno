#!/bin/bash

DATASET=Abuzz
MODEL=fno_med
TEST_SPLIT_RATIO=0.25
BATCH_SIZE=4
KERNEL_SIZE=11
POOL_SIZE=2
MODES=16

python evaluation.py --dataset $DATASET --model $MODEL --test_split_ratio $TEST_SPLIT_RATIO --batch_size $BATCH_SIZE --kernel_size $KERNEL_SIZE --pool_size $POOL_SIZE --modes $MODES
