srun -p mozi-S1 \
    --gres=gpu:1 \
    --quotatype=auto \
    --ntasks=1 \
    --ntasks-per-node=1 \
    --cpus-per-task=16 \
    --kill-on-bad-exit=1 \
    python merge_lora_weights_and_save_hf_model.py \
        --weight="runs/reground3d-7b/ckpt_model/pytorch_model.bin" \
        --save_path="./ReGround3D-7B"
