#!/bin/bash

if [ -n "$epoch_time" ]; then
    EPOCH_TIME=$epoch_time
else
    EPOCH_TIME=150
fi

if [ -n "$out_dir" ]; then
    OUT_DIR=$out_dir
else
    OUT_DIR='./model_out_acdc'
fi

if [ -n "$cfg" ]; then
    CFG=$cfg
else
    CFG='configs/swin_tiny_patch4_window7_224_lite.yaml'
fi

if [ -n "$data_dir" ]; then
    DATA_DIR=$data_dir
else
    DATA_DIR='../data/ACDC'
fi

if [ -n "$learning_rate" ]; then
    LEARNING_RATE=$learning_rate
else
    LEARNING_RATE=0.05
fi

if [ -n "$img_size" ]; then
    IMG_SIZE=$img_size
else
    IMG_SIZE=224
fi

if [ -n "$batch_size" ]; then
    BATCH_SIZE=$batch_size
else
    BATCH_SIZE=24
fi


echo "start test model"
python test.py --dataset ACDC --cfg $CFG --is_saveni --volume_path $DATA_DIR --max_epochs $EPOCH_TIME --output_dir $OUT_DIR --img_size $IMG_SIZE --base_lr $LEARNING_RATE --batch_size $BATCH_SIZE

