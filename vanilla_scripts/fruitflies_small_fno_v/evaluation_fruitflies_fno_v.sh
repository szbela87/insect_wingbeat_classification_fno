#!/bin/bash

DATASET=Fruitflies
MODEL=fno
TEST_SPLIT_RATIO=0.25
BATCH_SIZE=8
KERNEL_SIZE=1
POOL_SIZE=2
MODES=16

python evaluation.py --dataset $DATASET --model $MODEL --test_split_ratio $TEST_SPLIT_RATIO --batch_size $BATCH_SIZE --kernel_size $KERNEL_SIZE --pool_size $POOL_SIZE --modes $MODES
