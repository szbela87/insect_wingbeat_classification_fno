#!/bin/bash

DATASET=Insects
MODEL=fno_med
TEST_SPLIT_RATIO=0.25
EPOCHS=25
WEIGHT_DECAY=0.005
LR=0.0005
DIV_FACTOR=50
FINAL_DIV_FACTOR=5000
OUT_FEATURES=10
TRAIN_BATCH_SIZE=32
VALID_BATCH_SIZE=128
EVAL_FREQ=200
KERNEL_SIZE=11
POOL_SIZE=2
MODES=16

python train_mgpu.py --dataset $DATASET --model $MODEL --test_split_ratio $TEST_SPLIT_RATIO --epochs $EPOCHS --weight_decay $WEIGHT_DECAY --lr $LR --div_factor $DIV_FACTOR --final_div_factor $FINAL_DIV_FACTOR --out_features $OUT_FEATURES --train_batch_size $TRAIN_BATCH_SIZE --valid_batch_size $VALID_BATCH_SIZE --eval_freq $EVAL_FREQ --kernel_size $KERNEL_SIZE --pool_size $POOL_SIZE --modes $MODES
