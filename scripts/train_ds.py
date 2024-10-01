import argparse
import os
import shutil
import sys
import time
from functools import partial

import deepspeed
import numpy as np
import torch
import tqdm
import transformers
from peft import LoraConfig, get_peft_model
from torch.utils.tensorboard import SummaryWriter
from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live
from model.ReGround3D import ReGround3DForCausalLM
from model.llava import conversation as conversation_lib
from utils.dataset import Hybrid3DDataset, collate_fn
from utils.visual_grounding_dataset import VGValDataset
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         AverageMeter, ProgressMeter, Summary, dict_to_cuda,
                         intersectionAndUnionGPU)
from embodiedscan.eval.metrics import GroundingMetric

def parse_args(args):
    parser = argparse.ArgumentParser(description="ReGround3D Model Training")
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")
    parser.add_argument(
        "--version", default="/mnt/petrelfs/zhuchenming/LLaVA/checkpoints/llava3d-v1.5-7b-task-lora-v3-voxelize-region-1-4-tuning-mmscan/merged/llava3d-v1.5-7b-task-v3"
    )
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
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument(
        "--vision-tower", default="openai/clip-vit-large-patch14-336", type=str
    )
    parser.add_argument(
        "--video-tower", default="SpatialAwareModule", type=str
    )
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)

    parser.add_argument(
        "--dataset", default="vqa3d||vg", type=str
    )
    parser.add_argument("--sample_rates", default="0,1", type=str)
    parser.add_argument(
        "--vg_data", default="scanrefer||sr3d||nr3d", type=str
    )
    parser.add_argument(
        "--num_frames", default=20, type=int, help="frame number"
    )
    parser.add_argument(
        "--num_sample_tokens", default=1152, type=int, help="sample token number"
    )
    parser.add_argument("--vqa3d_data", default="sqa3d||scanqa", type=str)
    parser.add_argument("--dataset_dir", default="./dataset", type=str)
    parser.add_argument("--info_file", default="./dataset", type=str)
    parser.add_argument("--annos_dir", default="./dataset", type=str)
    parser.add_argument("--log_base_dir", default="./runs", type=str)
    parser.add_argument("--exp_name", default="reground3d", type=str)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--steps_per_epoch", default=100, type=int)
    parser.add_argument(
        "--batch_size", default=2, type=int, help="batch size per device per step"
    )
    parser.add_argument(
        "--grad_accumulation_steps",
        default=20,
        type=int,
    )
    parser.add_argument("--val_batch_size", default=1, type=int)
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--lr", default=0.0003, type=float)
    parser.add_argument("--ce_loss_weight", default=1.0, type=float)
    parser.add_argument("--det_loss_weight", default=1.0, type=float)
    parser.add_argument("--lora_alpha", default=16, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    parser.add_argument("--lora_target_modules", default="q_proj,v_proj", type=str)
    parser.add_argument("--explanatory", default=0.1, type=float)
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.95, type=float)
    parser.add_argument("--num_classes_per_sample", default=3, type=int)
    parser.add_argument("--exclude_val", action="store_true", default=False)
    parser.add_argument("--no_eval", action="store_true", default=False)
    parser.add_argument("--eval_only", action="store_true", default=False)
    parser.add_argument("--vision_pretrained", default="/mnt/hwfile/openmmlab/zhuchenming/checkpoints/point_backbone/butd_detr_scanrefer_backbone.pth", type=str)
    parser.add_argument("--video_info_file", default="/mnt/petrelfs/zhuchenming/LLaVA/embodiedscan_infos_full_llava3d_v2.json", type=str)
    parser.add_argument("--out_dim", default=256, type=int)
    parser.add_argument("--resume", default="", type=str)
    parser.add_argument("--print_freq", default=1, type=int)
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--train_box_decoder", action="store_true", default=True)
    parser.add_argument("--use_mm_start_end", action="store_true", default=False)
    parser.add_argument("--auto_resume", action="store_true", default=True)
    parser.add_argument(
        "--conv_type",
        default="llava_v1",
        type=str,
        choices=["llava_v1", "llava_llama_2"],
    )
    return parser.parse_args(args)


def main(args):
    args = parse_args(args)
    args.log_dir = os.path.join(args.log_base_dir, args.exp_name)
    if args.local_rank == 0:
        os.makedirs(args.log_dir, exist_ok=True)
        writer = SummaryWriter(args.log_dir)
    else:
        writer = None

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
        "ce_loss_weight": args.ce_loss_weight,
        "det_loss_weight": args.det_loss_weight,
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

    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype, device=args.local_rank)
    video_tower = model.get_model().get_video_tower()
    video_tower.to(dtype=torch_dtype, device=args.local_rank)
    if not args.eval_only:
        model.get_model().initialize_reground3d_modules(model.get_model().config)

    for p in vision_tower.parameters():
        p.requires_grad = False
    for p in video_tower.parameters():
        p.requires_grad = False
    for p in model.get_model().mm_projector.parameters():
        p.requires_grad = False

    conversation_lib.default_conversation = conversation_lib.conv_templates[
        args.conv_type
    ]

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

    # if args.local_rank == 0:
    #     estimate_zero3_model_states_mem_needs_all_live(model, num_gpus_per_node=1, num_nodes=1)

    # make text_hidden_fcs, decoder, lm_head, embed_tokens trainable
    for n, p in model.named_parameters():
        if any(
            [
                x in n
                for x in ["lm_head", "embed_tokens", "decoder", "feat_map", "bbox_head", "text_hidden_fcs"]
            ]
        ):
            print("n: ", n, "p.shape: ", p.shape)
            p.requires_grad = True

    world_size = torch.cuda.device_count()
    args.distributed = world_size > 1
    train_dataset = Hybrid3DDataset(
        args.dataset_dir,
        args.info_file,
        args.annos_dir,
        args.vision_tower,
        tokenizer=tokenizer,
        samples_per_epoch=args.batch_size
        * args.grad_accumulation_steps
        * args.steps_per_epoch
        * world_size,
        precision=args.precision,
        image_size=args.image_size,
        num_classes_per_sample=args.num_classes_per_sample,
        exclude_val=args.exclude_val,
        num_frames=args.num_frames,
        dataset=args.dataset,
        sample_rate=[float(x) for x in args.sample_rates.split(",")],
        vqa3d_data=args.vqa3d_data,
        vg_data=args.vg_data,
        explanatory=args.explanatory,
    )

    if args.no_eval == False:
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
        print(
            f"Training with {len(train_dataset)} examples and validating with {len(val_dataset)} examples."
        )
    else:
        val_dataset = None
        print(f"Training with {len(train_dataset)} examples.")

    ds_config = {
        "train_micro_batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps": args.grad_accumulation_steps,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": args.lr,
                "weight_decay": 0.0,
                "betas": (args.beta1, args.beta2),
            },
        },
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "total_num_steps": args.epochs * args.steps_per_epoch,
                "warmup_min_lr": 0,
                "warmup_max_lr": args.lr,
                "warmup_num_steps": 100,
                "warmup_type": "linear",
            },
        },
        "fp16": {
            "enabled": args.precision == "fp16",
        },
        "bf16": {
            "enabled": args.precision == "bf16",
        },
        "gradient_clipping": 1.0,
        "zero_optimization": {
            "stage": 3,
            "offload_optimizer": {
            "device": "cpu",
            "pin_memory": True
            },
            "offload_param": {
            "device": "cpu",
            "pin_memory": True
            },
            "overlap_comm": True,
            "contiguous_gradients": True,
            "sub_group_size": 1e9,
            "reduce_bucket_size": "auto",
            "stage3_prefetch_bucket_size": "auto",
            "stage3_param_persistence_threshold": "auto",
            "stage3_max_live_parameters": 1e9,
            "stage3_max_reuse_distance": 1e9,
            "gather_16bit_weights_on_model_save": True
        }
    }
    # ds_config = {
    #     "train_micro_batch_size_per_gpu": args.batch_size,
    #     "gradient_accumulation_steps": args.grad_accumulation_steps,
    #     "optimizer": {
    #         "type": "AdamW",
    #         "params": {
    #             "lr": args.lr,
    #             "weight_decay": 0.0,
    #             "betas": (args.beta1, args.beta2),
    #         },
    #     },
    #     "scheduler": {
    #         "type": "WarmupDecayLR",
    #         "params": {
    #             "total_num_steps": args.epochs * args.steps_per_epoch,
    #             "warmup_min_lr": 0,
    #             "warmup_max_lr": args.lr,
    #             "warmup_num_steps": 100,
    #             "warmup_type": "linear",
    #         },
    #     },
    #     "fp16": {
    #         "enabled": args.precision == "fp16",
    #     },
    #     "bf16": {
    #         "enabled": args.precision == "bf16",
    #     },
    #     "gradient_clipping": 1.0,
    #     "zero_optimization": {
    #         "stage": 2,
    #         "contiguous_gradients": True,
    #         "overlap_comm": True,
    #         "reduce_scatter": True,
    #         "reduce_bucket_size": 5e8,
    #         "allgather_bucket_size": 5e8,
    #     },
    # }

    # if args.local_rank == 0:
    #     print(ds_config)

    model_engine, optimizer, train_loader, scheduler = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        training_data=train_dataset,
        collate_fn=partial(
            collate_fn,
            tokenizer=tokenizer,
            conv_type=args.conv_type,
            use_mm_start_end=args.use_mm_start_end,
            local_rank=args.local_rank,
        ),
        config=ds_config,
    )

    # resume deepspeed checkpoint
    if args.auto_resume and len(args.resume) == 0:
        resume = os.path.join(args.log_dir, "ckpt_model")
        if os.path.exists(resume):
            args.resume = resume

    if args.resume:
        load_path, client_state = model_engine.load_checkpoint(args.resume)
        with open(os.path.join(args.resume, "latest"), "r") as f:
            ckpt_dir = f.readlines()[0].strip()
        args.start_epoch = (
            int(ckpt_dir.replace("global_step", "")) // args.steps_per_epoch
        )
        print(
            "resume training from {}, start from epoch {}".format(
                args.resume, args.start_epoch
            )
        )

    # validation dataset
    if val_dataset is not None:
        assert args.val_batch_size == 1
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset, shuffle=False, drop_last=False
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.val_batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=False,
            sampler=val_sampler,
            collate_fn=partial(
                collate_fn,
                tokenizer=tokenizer,
                conv_type=args.conv_type,
                use_mm_start_end=args.use_mm_start_end,
                local_rank=args.local_rank,
            ),
        )

    train_iter = iter(train_loader)
    best_score, cur_ciou = 0.0, 0.0

    if args.eval_only:
        validate(val_loader, model_engine, 0, writer, args)
        exit()

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train_iter = train(
            train_loader,
            model_engine,
            epoch,
            scheduler,
            writer,
            train_iter,
            args,
        )

        if args.no_eval:
            save_dir = os.path.join(args.log_dir, "ckpt_model")
            if args.local_rank == 0:
                # torch.save({"epoch": epoch}, args.log_dir)
                if os.path.exists(save_dir):
                    shutil.rmtree(save_dir)
            torch.distributed.barrier()
            model_engine.save_checkpoint(save_dir)


def train(
    train_loader,
    model,
    epoch,
    scheduler,
    writer,
    train_iter,
    args,
):
    """Main training loop."""
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    ce_losses = AverageMeter("CeLoss", ":.4f")
    cls_losses = AverageMeter("ClsLoss", ":.4f")
    bbox_losses = AverageMeter("BBoxLoss", ":.4f")
    det_losses = AverageMeter("DetLoss", ":.4f")

    progress = ProgressMeter(
        args.steps_per_epoch,
        [
            batch_time,
            losses,
            ce_losses,
            det_losses,
            cls_losses,
            bbox_losses,
        ],
        prefix="Epoch: [{}]".format(epoch)
    )

    # switch to train mode
    model.train()
    end = time.time()
    for global_step in range(args.steps_per_epoch):
        for i in range(args.grad_accumulation_steps):
            try:
                input_dict = next(train_iter)
            except:
                train_iter = iter(train_loader)
                input_dict = next(train_iter)

            data_time.update(time.time() - end)
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

            output_dict = model(**input_dict)

            loss = output_dict["loss"]
            ce_loss = output_dict["ce_loss"]
            bbox_loss = output_dict["bbox_loss"]
            cls_loss = output_dict["cls_loss"]
            det_loss = output_dict["det_loss"]
            # print('loss:', loss)
            losses.update(loss.item(), input_dict["images_clip"].size(0))
            ce_losses.update(ce_loss.item(), input_dict["images_clip"].size(0))
            cls_losses.update(cls_loss.item(), input_dict["images_clip"].size(0))
            bbox_losses.update(bbox_loss.item(), input_dict["images_clip"].size(0))
            det_losses.update(det_loss.item(), input_dict["images_clip"].size(0))
            model.backward(loss)
            model.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if global_step % args.print_freq == 0:
            if args.distributed:
                batch_time.all_reduce()
                data_time.all_reduce()

                losses.all_reduce()
                ce_losses.all_reduce()
                cls_losses.all_reduce()
                bbox_losses.all_reduce()
                det_losses.all_reduce()

            if args.local_rank == 0:
                progress.display(global_step + 1)
                writer.add_scalar("train/loss", losses.avg, global_step)
                writer.add_scalar("train/ce_loss", ce_losses.avg, global_step)
                writer.add_scalar(
                    "train/cls_loss", cls_losses.avg, global_step
                )
                writer.add_scalar(
                    "train/bbox_loss", bbox_losses.avg, global_step
                )
                writer.add_scalar("train/det_loss", det_losses.avg, global_step)
                writer.add_scalar(
                    "metrics/total_secs_per_batch", batch_time.avg, global_step
                )
                writer.add_scalar(
                    "metrics/data_secs_per_batch", data_time.avg, global_step
                )

            batch_time.reset()
            data_time.reset()
            losses.reset()
            ce_losses.reset()
            cls_losses.reset()
            bbox_losses.reset()
            det_losses.reset()

        if global_step != 0:
            curr_lr = scheduler.get_last_lr()
            if args.local_rank == 0:
                writer.add_scalar("train/lr", curr_lr[0], global_step)

    return train_iter


def validate(val_loader, model, args, tokenizer, metric):

    model.eval()

    iter_num = 0
    val_predictions = []
    for input_dict in tqdm.tqdm(val_loader):
        torch.cuda.empty_cache()

        print('gt_bboxes_3d:', input_dict['data_samples'][0].gt_instances_3d.bboxes_3d)
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

if __name__ == "__main__":
    main(sys.argv[1:])
