import glob
import os
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools import mask
from transformers import CLIPImageProcessor

from model.llava import conversation as conversation_lib
from model.llava.constants import (DEFAULT_IMAGE_TOKEN, IGNORE_INDEX,
                                   IMAGE_TOKEN_INDEX)
from model.llava.mm_utils import tokenizer_image_token

from .conversation import get_default_conv_template
from .visual_grounding_dataset import VGDataset
from .utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                    DEFAULT_IMAGE_TOKEN)
from .vqa_3d_dataset import VQA3DDataset
from model.llava.model.multimodal_encoder.video_processor import RGBDVideoProcessor
from mmcv.transforms import Resize

def collate_fn(
    batch, tokenizer=None, conv_type="llava_v1", use_mm_start_end=True, local_rank=-1
):
    video_path_list = []
    images_list = []
    points_list = []
    images_clip_list = []
    depth_list = []
    pose_list = []
    intrinsic_list = []
    images_clip_list = []
    conversation_list = []
    data_sample_list = []
    questions_list = []
    offset_list = [0]
    # cnt = 0
    box_cnt = 0
    inferences = []
    for (
        video_path,
        video_dict,
        scene_dict,
        conversation,
        data_sample,
        question,
        inference,
    ) in batch:
        video_path_list.append(video_path)
        images_clip_list.append(video_dict['images'])  # (V, 3, 336, 336)
        depth_list.append(video_dict['depth_images']) # (V, 336, 336)
        pose_list.append(video_dict['poses']) # (V, 4, 4)
        intrinsic_list.append(video_dict['intrinsic']) # (V, 4, 4)
        conversation_list.append(conversation)
        if len(data_sample.gt_instances_3d) > 0:
            images_list.append(scene_dict['imgs'])
            points_list.append(scene_dict['points'])
            data_sample_list.append(data_sample)
        questions_list.append(question)
        # cnt += len(conversations)
        box_cnt += len(data_sample.gt_instances_3d)
        offset_list.append(box_cnt)
        inferences.append(inference)

    if use_mm_start_end:
        # replace <image> token  to <im_start><image><im_end>
        for i in range(len(conversation_list)):
            replace_token = DEFAULT_IMAGE_TOKEN
            replace_token = (
                DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            )
            conversation_list[i] = conversation_list[i].replace(
                DEFAULT_IMAGE_TOKEN, replace_token
            )
    input_ids = [
        tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
        for prompt in conversation_list
    ]
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )  # 填充到一个 batch 最长的 token length   (B, MAX, C)
    attention_mask = input_ids.ne(tokenizer.pad_token_id)  # ne = not equal

    conv = conversation_lib.default_conversation.copy()
    targets = input_ids.clone()

    if conv_type == "llava_v1":
        sep = conv.sep + conv.roles[1] + ": "   #  ASSISTANT:
    else:
        sep = "[/INST] "
    # 这里是给 input_ids 加 label
    for conversation, target in zip(conversation_list, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())  # total token lens

        rounds = conversation.split(conv.sep2)  # </s> 用 sep2 分开多论对话
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX  # 屏蔽 bos_token_id
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            # if len(parts) != 2:
            #     break
            assert len(parts) == 2, (len(parts), rou)
            parts[0] += sep

            if DEFAULT_IMAGE_TOKEN in conversation:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX
        # 对于 system message 和 question 的部分都是设为了 ignore_index, answer 部分保留了原本的 token id

        # if False:
        #     z = target.clone()
        #     z = torch.where(z == IGNORE_INDEX, tokenizer.unk_token_id, z)
        #     if local_rank == 0:
        #         print(
        #             "conversation: ",
        #             conversation,
        #             "tokenizer.decode(z): ",
        #             tokenizer.decode(z),
        #         )

        if cur_len < tokenizer.model_max_length:
            assert cur_len == total_len

    if inferences[0] == False:
        # 因为这里目前的长度不代表最终的token长度，1 个 image_token_idx 会被替换成 256 个 image token
        # 这里是为了防止最终长度超过 max_length
        truncate_len = tokenizer.model_max_length - 1151

        if input_ids.shape[1] > truncate_len:
            input_ids = input_ids[:, :truncate_len]
            targets = targets[:, :truncate_len]
            attention_mask = attention_mask[:, :truncate_len]

    # print('conversation_list:', conversation_list)
    return {
        "video_paths": video_path_list,
        "images": images_list,  # list of (V, 3, H, W)
        "points": points_list, # list of points in the batch
        "images_clip": torch.stack(images_clip_list, dim=0),  # (B, V, 3, 336, 336)
        'depths': torch.stack(depth_list, dim=0),
        'poses': torch.stack(pose_list, dim=0),
        'intrinsics': torch.stack(intrinsic_list, dim=0),
        "input_ids": input_ids,
        "labels": targets,
        "attention_mask": attention_mask,
        "data_samples": data_sample_list,
        "offset": torch.LongTensor(offset_list),
        "questions_list": questions_list,
        "inference": inferences[0],
        "conversation_list": conversation_list,
    }

# class HybridDataset(torch.utils.data.Dataset):
#     pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
#     pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
#     img_size = 1024
#     ignore_label = 255

#     def __init__(
#         self,
#         base_image_dir,
#         tokenizer,
#         vision_tower,
#         samples_per_epoch=500 * 8 * 2 * 10,
#         precision: str = "fp32",
#         image_size: int = 224,
#         num_classes_per_sample: int = 3,
#         exclude_val=False,
#         dataset="sem_seg||refer_seg||vqa||reason_seg",
#         sample_rate=[9, 3, 3, 1],
#         sem_seg_data="ade20k||cocostuff||partimagenet||pascal_part||paco_lvis||mapillary",
#         refer_seg_data="refclef||refcoco||refcoco+||refcocog",
#         vqa_data="llava_instruct_150k",
#         reason_seg_data="ReasonSeg|train",
#         explanatory=0.1,
#     ):
#         self.exclude_val = exclude_val
#         self.dataset = dataset
#         self.samples_per_epoch = samples_per_epoch
#         self.explanatory = explanatory
#         self.num_classes_per_sample = num_classes_per_sample
#         sample_rate = np.array(sample_rate)
#         self.sample_rate = sample_rate / sample_rate.sum()

#         self.base_image_dir = base_image_dir
#         self.image_size = image_size
#         self.tokenizer = tokenizer
#         self.precision = precision

#         self.datasets = dataset.split("||")

#         self.all_datasets = []
#         for dataset in self.datasets:
#             # if dataset == "sem_seg":
#             #     self.all_datasets.append(
#             #         SemSegDataset(
#             #             base_image_dir,
#             #             tokenizer,
#             #             vision_tower,
#             #             samples_per_epoch,
#             #             precision,
#             #             image_size,
#             #             num_classes_per_sample,
#             #             exclude_val,
#             #             sem_seg_data,
#             #         )
#             #     )
#             # elif dataset == "refer_seg":
#             #     self.all_datasets.append(
#             #         ReferSegDataset(
#             #             base_image_dir,
#             #             tokenizer,
#             #             vision_tower,
#             #             samples_per_epoch,
#             #             precision,
#             #             image_size,
#             #             num_classes_per_sample,
#             #             exclude_val,
#             #             refer_seg_data,
#             #         )
#             #     )
#             if dataset == "vqa":
#                 self.all_datasets.append(
#                     VQADataset(
#                         base_image_dir,
#                         tokenizer,
#                         vision_tower,
#                         samples_per_epoch,
#                         precision,
#                         image_size,
#                         num_classes_per_sample,
#                         exclude_val,
#                         vqa_data,
#                     )
#                 )
#             elif dataset == "vg":
#                 self.all_datasets.append(
#                     VGDataset(
#                         base_image_dir,
#                         tokenizer,
#                         vision_tower,
#                         samples_per_epoch,
#                         precision,
#                         image_size,
#                         num_classes_per_sample,
#                         exclude_val,
#                         reason_seg_data,
#                         explanatory,
#                     )
#                 )

#     def __len__(self):
#         return self.samples_per_epoch

#     def __getitem__(self, idx):
#         ind = np.random.choice(list(range(len(self.datasets))), p=self.sample_rate)
#         data = self.all_datasets[ind]
#         inference = False
#         return *data[0], inference
    
class Hybrid3DDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        video_folder,
        video_info_file,
        json_folder,
        vision_tower,
        tokenizer=None,
        samples_per_epoch=500 * 8 * 2 * 10,
        precision: str = "fp32",
        image_size: int = 336,
        num_classes_per_sample: int = 1,
        exclude_val=False,
        num_frames=20,
        dataset="vqa3d||vg",
        sample_rate=[0, 1],
        vqa3d_data="scanqa||sqa3d",
        vg_data="scanrefer||sr3d||nr3d",
        explanatory=0.1,
    ):
        self.exclude_val = exclude_val
        self.dataset = dataset
        self.samples_per_epoch = samples_per_epoch
        self.explanatory = explanatory
        self.num_classes_per_sample = num_classes_per_sample
        sample_rate = np.array(sample_rate)
        self.sample_rate = sample_rate / sample_rate.sum()

        self.video_folder = video_folder
        self.video_info_file = video_info_file
        self.json_folder = json_folder
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        self.video_processor = RGBDVideoProcessor(vision_tower, video_info_file, num_frames=num_frames)
        self.resize = Resize(scale=(480, 480), keep_ratio=False)

        self.datasets = dataset.split("||")

        self.all_datasets = []
        for dataset in self.datasets:
            if dataset == "vg":
                self.all_datasets.append(
                    VGDataset(
                        data=vg_data,
                        video_folder=video_folder,
                        json_folder=json_folder,
                        video_processor=self.video_processor,
                        resize=self.resize,
                        precision=precision,
                        image_size=image_size,
                        num_classes_per_sample=num_classes_per_sample,
                        exclude_val=exclude_val,
                        num_frames=num_frames
                    )
                )
            elif dataset == "vqa3d":
                self.all_datasets.append(
                    VQA3DDataset(
                        data=vqa3d_data,
                        video_folder=video_folder,
                        json_folder=json_folder,
                        video_processor=self.video_processor,
                        resize=self.resize,
                        samples_per_epoch=samples_per_epoch,
                        precision=precision,
                        image_size=image_size,
                        num_classes_per_sample=num_classes_per_sample,
                        exclude_val=exclude_val,
                        num_frames=num_frames
                    )
                )

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        ind = np.random.choice(list(range(len(self.datasets))), p=self.sample_rate)
        data = self.all_datasets[ind]
        inference = False
        return *data[0], inference
