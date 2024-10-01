import argparse
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, BitsAndBytesConfig
from functools import partial
import tqdm

from model.ReGround3D import ReGround3DForCausalLM
from utils.visual_grounding_dataset import VGValDataset
from model.llava import conversation as conversation_lib
from utils.dataset import collate_fn
from model.llava.mm_utils import tokenizer_image_token
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX, VG_QUESTION_LIST, AverageMeter, ProgressMeter, Summary, dict_to_cuda)
from embodiedscan.eval.metrics import GroundingMetric

def parse_args(args):
    parser = argparse.ArgumentParser(description="ReGround3D chat")
    parser.add_argument("--version", default="ReGround3D-7B")
    parser.add_argument("--vis_save_path", default="./vis_output", type=str)
    parser.add_argument(
        "--precision",
        default="fp32",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--image_size", default=336, type=int, help="image size")
    parser.add_argument("--model_max_length", default=2048, type=int)
    parser.add_argument("--num_frames", default=20, type=int, help="frame number")
    parser.add_argument("--dataset_dir", default="./dataset", type=str)
    parser.add_argument("--info_file", default="./dataset", type=str)
    parser.add_argument("--annos_dir", default="./dataset", type=str)
    parser.add_argument("--n_points", default=100000, type=int, help="frame number")
    parser.add_argument("--val_batch_size", default=1, type=int)
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument(
        "--vision-tower", default="openai/clip-vit-large-patch14-336", type=str
    )
    parser.add_argument(
        "--video-tower", default="SpatialAwareModule", type=str
    )
    parser.add_argument("--video_info_file", default="/mnt/petrelfs/zhuchenming/LLaVA/embodiedscan_infos_full_llava3d_v2.json", type=str)
    parser.add_argument("--local-rank", default=0, type=int, help="node rank")
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--use_mm_start_end", action="store_true", default=False)
    parser.add_argument(
        "--conv_type",
        default="llava_v1",
        type=str,
        choices=["llava_v1", "llava_llama_2"],
    )
    return parser.parse_args(args)

def validate(val_loader, model, args, tokenizer, metric):

    model.eval()

    iter_num = 0
    val_predictions = []
    for input_dict in tqdm.tqdm(val_loader):
        torch.cuda.empty_cache()

        device = torch.device("cuda", args.local_rank)
        input_dict = dict_to_cuda(input_dict, device)

        if args.precision == "fp16":
            for key, value in input_dict.items():
                if key in ['images_clip', 'depths', 'poses', 'intrinsics']:
                    input_dict[key] = input_dict[key].half()
        elif args.precision == "bf16":
            for key, value in input_dict.items():
                if key in ['images_clip', 'depths', 'poses', 'intrinsics']:
                    input_dict[key] = input_dict[key].bfloat16()
        else:
            for key, value in input_dict.items():
                if key in ['images_clip', 'depths', 'poses', 'intrinsics']:
                    input_dict[key] = input_dict[key].float()

        with torch.no_grad():
            predictions = model(**input_dict)

        iter_num +=1
        val_predictions += predictions

    metric.compute_metrics(val_predictions)
    
    return

def main(args):
    args = parse_args(args)
    os.makedirs(args.vis_save_path, exist_ok=True)

    # Create model
    tokenizer = AutoTokenizer.from_pretrained(
        args.version,
        cache_dir=None,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    args.loc_token_idx = tokenizer.convert_tokens_to_ids("[LOC]")


    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half

    kwargs = {"torch_dtype": torch_dtype}
    if args.load_in_4bit:
        kwargs.update(
            {
                "torch_dtype": torch.half,
                "load_in_4bit": True,
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    llm_int8_skip_modules=["visual_model"],
                ),
            }
        )
    elif args.load_in_8bit:
        kwargs.update(
            {
                "torch_dtype": torch.half,
                "quantization_config": BitsAndBytesConfig(
                    llm_int8_skip_modules=["visual_model"],
                    load_in_8bit=True,
                ),
            }
        )

    model = ReGround3DForCausalLM.from_pretrained(
        args.version, low_cpu_mem_usage=True, vision_tower=args.vision_tower, loc_token_idx=args.loc_token_idx, **kwargs
    )

    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype)
    video_tower = model.get_model().get_video_tower()
    video_tower.to(dtype=torch_dtype)

    if args.precision == "bf16":
        model = model.bfloat16().cuda()
    elif (
        args.precision == "fp16" and (not args.load_in_4bit) and (not args.load_in_8bit)
    ):
        vision_tower = model.get_model().get_vision_tower()
        model.model.vision_tower = None
        import deepspeed

        model_engine = deepspeed.init_inference(
            model=model,
            dtype=torch.half,
            replace_with_kernel_inject=True,
            replace_method="auto",
        )
        model = model_engine.module
        model.model.vision_tower = vision_tower.half().cuda()
    elif args.precision == "fp32":
        model = model.float().cuda()

    conversation_lib.default_conversation = conversation_lib.conv_templates[
        args.conv_type
    ]

    val_dataset = VGValDataset(
        data="scanrefer",
        video_folder=args.dataset_dir,
        video_info_file=args.info_file,
        json_folder=args.annos_dir,
        vision_tower=args.vision_tower,
        tokenizer=tokenizer,
        image_size=args.image_size,
        num_frames=args.num_frames,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=False,
        collate_fn=partial(
            collate_fn,
            tokenizer=tokenizer,
            conv_type=args.conv_type,
            use_mm_start_end=args.use_mm_start_end,
            local_rank=args.local_rank,
        ),
    )

    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(device=args.local_rank)
    video_tower = model.get_model().get_video_tower()
    video_tower.to(device=args.local_rank)

    # video_processor = video_tower.video_processor
    metric = GroundingMetric()
    validate(val_loader, model, args, tokenizer, metric)

if __name__ == "__main__":
    main(sys.argv[1:])
