#!/bin/bash
python runs/reground3d-7b/ckpt_model/zero_to_fp32.py ./runs/reground3d-7b/ckpt_model ./runs/reground3d-7b/ckpt_model/pytorch_model.bin --tag global_step1000