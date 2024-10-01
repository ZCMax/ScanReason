#!/bin/bash
python merge_lora_weights_and_save_hf_model.py \
    --weight="runs/reground3d-7b/ckpt_model/pytorch_model.bin" \
    --save_path="./ReGround3D-7B"
