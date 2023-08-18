#!/bin/bash

DATASET=Fruitflies
MODEL=fno
TEST_SPLIT_RATIO=0.25
EPOCHS=25
WEIGHT_DECAY=0.005 
LR=0.0001
DIV_FACTOR=10
FINAL_DIV_FACTOR=1000
OUT_FEATURES=3
TRAIN_BATCH_SIZE=32
VALID_BATCH_SIZE=512
EVAL_FREQ=100
KERNEL_SIZE=11
POOL_SIZE=2

python train_mgpu.py --dataset $DATASET --model $MODEL --test_split_ratio $TEST_SPLIT_RATIO --epochs $EPOCHS --weight_decay $WEIGHT_DECAY --lr $LR --div_factor $DIV_FACTOR --final_div_factor $FINAL_DIV_FACTOR --out_features $OUT_FEATURES --train_batch_size $TRAIN_BATCH_SIZE --valid_batch_size $VALID_BATCH_SIZE --eval_freq $EVAL_FREQ --kernel_size $KERNEL_SIZE --pool_size $POOL_SIZE
