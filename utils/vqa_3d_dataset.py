import json
import os
import random

import torch
import mmengine
import numpy as np

from model.llava import conversation as conversation_lib

from .utils import DEFAULT_IMAGE_TOKEN, DEFAULT_VIDEO_TOKEN
from .base_3d_dataset import Base3DDataset
from mmengine.structures import InstanceData
from embodiedscan.utils.typing_config import Det3DDataElement
from embodiedscan.structures import get_box_type


class VQA3DDataset(Base3DDataset):

    def __init__(self,
        data="scanqa||sqa3d",
        box_type_3d = 'Euler-Depth',
        **kwargs
    ):
        super().__init__(data=data, **kwargs)
        self.box_type_3d, self.box_mode_3d = get_box_type(box_type_3d)

    def process_data(self):
        self.vqa_ds_list = self.data.split("||")  # ['scanqa', 'sqa3d']
        self.vqa_data = {}
        for ds in self.vqa_ds_list:
            annos = mmengine.load(os.path.join(self.json_folder, 'llava3d_scannet_' + ds + '_train.json'))
            train_scenes = []
            vqa_ds = {}

            scene2refs = {}  # ref 里需要包含 box 和 question
            for anno in annos:
                scene_id =  anno['video']  # scannet/scene0000_00
                if scene_id not in train_scenes:
                    train_scenes.append(scene_id)
                scene2refs[scene_id] = scene2refs.get(scene_id, []) + [
                        anno,]

            print(
                "dataset {} (train split) has {} scenes and {} annotations.".format(
                    ds,
                    len(train_scenes),
                    len(annos),
                )
            )
                        
            vqa_ds["scenes"] = train_scenes
            vqa_ds["scene2refs"] = scene2refs
            self.vqa_data[ds] = vqa_ds

    def __getitem__(self, idx):
        ds = random.randint(0, len(self.vqa_ds_list) - 1)
        ds = self.vqa_ds_list[ds]
        vqa_ds = self.vqa_data[ds]
        scenes = vqa_ds["scenes"]
        # annotations = vqa_ds["annotations"]
        scene2refs = vqa_ds["scene2refs"]
        idx = random.randint(0, len(scenes) - 1)
        scene_id = scenes[idx]
        # scene_id = 'scannet/scene0000_00'
        # image_path = image_info["file_name"]
        # image_id = image_info["id"]
        refs = scene2refs[scene_id]
        if len(refs) == 0:
            return self.__getitem__(0)

        questions = []
        answers = []
        for ref in refs:
            questions.append(ref["conversations"][0]["value"])
            answers.append(ref["conversations"][1]["value"])
        if len(questions) >= self.num_classes_per_sample:
            sampled_inds = np.random.choice(
                list(range(len(questions))), size=self.num_classes_per_sample, replace=False
            )
        else:
            sampled_inds = list(range(len(questions)))

        sampled_questions = [questions[ind] for ind in sampled_inds]
        sampled_answers = [answers[ind] for ind in sampled_inds]

        # preprocess video for video encoder
        video_path = os.path.join(self.video_folder, scene_id)
        video_dict = self.video_processor.preprocess(video_path, return_tensors="pt")
        # image = self.transform.apply_image(image)  # preprocess image for sam
        scene_dict = self.preprocess_embodiedscan(scene_id)   # preprocess points and images for embodiedscan

        refined_questions = []  # a scene could have multiple questions
        answers = []
        for idx, question in enumerate(sampled_questions):
            refined_question = question.replace(DEFAULT_VIDEO_TOKEN, DEFAULT_IMAGE_TOKEN)
            refined_questions.append(refined_question)
            answers.append(sampled_answers[idx])

        conversations = []
        conv = conversation_lib.default_conversation.copy()

        i = 0
        while i < len(refined_questions):
            conv.messages = []
            conv.append_message(conv.roles[0], refined_questions[i])
            conv.append_message(conv.roles[1], answers[i])
            conversations.append(conv.get_prompt())
            i += 1

        # 这是 numpy 转 tensor 再进行 normalize 以及 padding square 操作，
        # 前者对应 Pack3DInputs ，后者是 data_preprocesor
            
        data_sample = Det3DDataElement()
        gt_instances_3d = InstanceData()
                
        data_metas = {}
        for key, value in scene_dict.items():
            if key not in ['imgs', 'points']:
                data_metas[key] = value
    
        data_sample.set_metainfo(data_metas)
        sampled_bboxes = np.zeros((0, 9))
        gt_instances_3d['bboxes_3d'] = self.box_type_3d(
            sampled_bboxes,
            box_dim=sampled_bboxes.shape[-1],
            with_yaw=True,
            origin=(0.5, 0.5, 0.5))
        gt_instances_3d['labels_3d'] = torch.ones((len(sampled_bboxes),), dtype=torch.int64)
        data_sample.gt_instances_3d = gt_instances_3d

        return (
            video_path,
            video_dict,  # for MLLM
            scene_dict,  # for embodiedscan
            conversations[0],  # list of str, 1<=len<=3
            data_sample,
            refined_questions[0]
        )
