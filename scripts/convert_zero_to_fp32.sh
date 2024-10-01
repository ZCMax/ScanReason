srun -p mozi-S1 \
    --gres=gpu:1 \
    --quotatype=auto \
    --ntasks=1 \
    --ntasks-per-node=1 \
    --cpus-per-task=12 \
    --kill-on-bad-exit=1 \
    python runs/reground3d-7b/ckpt_model/zero_to_fp32.py ./runs/reground3d-7b/ckpt_model ./runs/reground3d-7b/ckpt_model/pytorch_model.bin --tag global_step1000