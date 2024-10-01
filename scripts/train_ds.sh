#!/bin/bash

JSON_FOLDER="./data/annotations"
VIDEO_FOLDER="./data/video"
INFO_FILE="./data/embodiedscan_infos_full_llava-3d.json"

deepspeed train_ds.py \
    --version ./checkpoints/llava-3d-7b \
    --dataset_dir ${VIDEO_FOLDER} \
    --info_file ${INFO_FILE} \
    --annos_dir ${JSON_FOLDER} \
    --num_frames 16 \
    --num_sample_tokens 1152 \
    --exp_name reground3d-7b \
    --no_eval
