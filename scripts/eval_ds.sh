#!/bin/bash
JSON_FOLDER="./data/annotations"
VIDEO_FOLDER="./data/video"
INFO_FILE="./data/embodiedscan_infos_full_llava-3d.json"

python eval_ds.py \
    --version ReGround3D-7B \
    --dataset_dir ${VIDEO_FOLDER} \
    --info_file ${INFO_FILE} \
    --annos_dir ${JSON_FOLDER} \
    --num_frames 20 \