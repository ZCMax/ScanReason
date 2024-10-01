#!/bin/bash

# Uncomment and set the following variables correspondingly to run this script:
JSON_FOLDER="/mnt/petrelfs/zhuchenming/LLaVA/playground/data/annotations"
VIDEO_FOLDER="/mnt/petrelfs/zhuchenming/LLaVA/playground/data/LLaVA-3D-Pretrain"
INFO_FILE="/mnt/petrelfs/zhuchenming/LLaVA/embodiedscan_infos_full_llava3d_v2.json"

MASTER_ADDR=`scontrol show hostname $SLURM_JOB_NODELIST | head -n1`
MASTER_PORT=$((RANDOM % 101 + 20000))
echo $MASTER_PORT

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun --partition=mozi-S1 \
    --job-name=llava3D_train \
    --gres=gpu:1 \
    --quotatype=reserved \
    --ntasks=1 \
    --ntasks-per-node=1 \
    --cpus-per-task=16 \
    --kill-on-bad-exit=1 \
    python eval_ds.py \
    --version ReGround3D-7B \
    --dataset_dir ${VIDEO_FOLDER} \
    --info_file ${INFO_FILE} \
    --annos_dir ${JSON_FOLDER} \
    --num_frames 20 \