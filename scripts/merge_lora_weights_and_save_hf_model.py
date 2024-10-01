import argparse
import glob
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer

from model.ReGround3D import ReGround3DForCausalLM
from utils.utils import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN


def parse_args(args):
    parser = argparse.ArgumentParser(
        description="merge lora weights and save model with hf format"
    )
    parser.add_argument(
        "--version", default="/mnt/hwfile/openmmlab/zhuchenming/llava3d_checkpoints/llava3d-v1.5-7b-task-lora-v3-voxelize-region-1-4-tuning-mmscan/merged/llava3d-v1.5-7b-task-v3"
    )
    parser.add_argument("--vis_save_path", default="./vis_output", type=str)
    parser.add_argument(
        "--precision",
        default="fp32",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--vision_pretrained", default="/mnt/hwfile/openmmlab/zhuchenming/checkpoints/point_backbone/butd_detr_scanrefer_backbone.pth", type=str)
    parser.add_argument("--video_info_file", default="/mnt/petrelfs/zhuchenming/LLaVA/embodiedscan_infos_full_llava3d_v2.json", type=str)
    parser.add_argument("--out_dim", default=256, type=int)
    parser.add_argument("--image_size", default=336, type=int, help="image size")
    parser.add_argument("--model_max_length", default=2048, type=int)
    parser.add_argument(
        "--vision-tower", default="openai/clip-vit-large-patch14-336", type=str
    )
    parser.add_argument(
        "--video-tower", default="SpatialAwareModule", type=str
    )
    parser.add_argument(
        "--num_frames", default=20, type=int, help="frame number"
    )
    parser.add_argument(
        "--num_sample_tokens", default=1152, type=int, help="sample token number"
    )
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument("--lora_alpha", default=16, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    parser.add_argument("--lora_target_modules", default="q_proj,v_proj", type=str)
    parser.add_argument("--local-rank", default=0, type=int, help="node rank")
    parser.add_argument("--train_box_decoder", action="store_true", default=True)
    parser.add_argument("--use_mm_start_end", action="store_true", default=False)
    parser.add_argument(
        "--conv_type",
        default="llava_v1",
        type=str,
        choices=["llava_v1", "llava_llama_2"],
    )
    parser.add_argument("--weight", default="", type=str, required=True)
    parser.add_argument("--save_path", default="./reground3d_model", type=str, required=True)
    return parser.parse_args(args)


def main(args):
    args = parse_args(args)
    os.makedirs(args.vis_save_path, exist_ok=True)

    # Create model
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.version,
        cache_dir=None,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    num_added_tokens = tokenizer.add_tokens("[LOC]")
    # args.loc_token_idx = tokenizer("[LOC]", add_special_tokens=False).input_ids[0]
    args.loc_token_idx = tokenizer.convert_tokens_to_ids("[LOC]")

    if args.use_mm_start_end:
        tokenizer.add_tokens(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
        )

    model_args = {
        "train_box_decoder": args.train_box_decoder,
        "out_dim": args.out_dim,
        "num_frames": args.num_frames,
        "num_sample_tokens": args.num_sample_tokens,
        "loc_token_idx": args.loc_token_idx,
        "vision_pretrained": args.vision_pretrained,
        "vision_tower": args.vision_tower,
        "video_tower": args.video_tower,
        "video_info_file": args.video_info_file,
        "use_mm_start_end": args.use_mm_start_end,
    }

    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half
    model = ReGround3DForCausalLM.from_pretrained(
        args.version, torch_dtype=torch_dtype, low_cpu_mem_usage=True, **model_args
    )
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype)
    video_tower = model.get_model().get_video_tower()
    video_tower.to(dtype=torch_dtype)
    model.get_model().initialize_reground3d_modules(model.get_model().config)

    lora_r = args.lora_r
    if lora_r > 0:

        def find_linear_layers(model, lora_target_modules):
            cls = torch.nn.Linear
            lora_module_names = set()
            for name, module in model.named_modules():
                if (
                    isinstance(module, cls)
                    and all(
                        [
                            x not in name
                            for x in [
                                "visual_model",
                                "vision_tower",
                                "video_tower",
                                "mm_projector",
                                "text_hidden_fcs",
                            ]
                        ]
                    )
                    and any([x in name for x in lora_target_modules])
                ):
                    lora_module_names.add(name)
            return sorted(list(lora_module_names))

        lora_alpha = args.lora_alpha
        lora_dropout = args.lora_dropout
        lora_target_modules = find_linear_layers(
            model, args.lora_target_modules.split(",")
        )
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    model.resize_token_embeddings(len(tokenizer))

    state_dict = torch.load(args.weight, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)

    model = model.merge_and_unload()
    state_dict = {}
    for k, v in model.state_dict().items():
        if "vision_tower" not in k:
            state_dict[k] = v
    model.save_pretrained(args.save_path, state_dict=state_dict)
    tokenizer.save_pretrained(args.save_path)


if __name__ == "__main__":
    main(sys.argv[1:])
