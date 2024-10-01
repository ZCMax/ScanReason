import os
import random

import numpy as np
import torch

from model.llava import conversation as conversation_lib

from .utils import ANSWER_LIST, VG_QUESTION_LIST, DEFAULT_IMAGE_TOKEN, DEFAULT_VIDEO_TOKEN
import mmengine
from .base_3d_dataset import Base3DDataset
from .base_3d_val_dataset import Base3DValDataset
from mmengine.structures import InstanceData
from embodiedscan.utils.typing_config import Det3DDataElement
from embodiedscan.structures import get_box_type

class VGDataset(Base3DDataset):

    def __init__(self, 
                 data = "scanrefer||sr3d||nr3d",
                 box_type_3d = 'Euler-Depth',
                 **kwargs):
        super().__init__(data=data, **kwargs)
        self.vg_question_list = VG_QUESTION_LIST
        self.answer_list = ANSWER_LIST
        self.box_type_3d, self.box_mode_3d = get_box_type(box_type_3d)

    def process_data(self):
        self.vg_ds_list = self.data.split("||")  # ['scanreason', 'sr3d', 'nr3d']
        self.vg_data = {}
        for ds in self.vg_ds_list:
            annos = mmengine.load(os.path.join(self.json_folder, 'llava3d_scannet_' + ds + '_train.json'))
            train_scenes = []
            vg_ds = {}

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
                        
            vg_ds["scenes"] = train_scenes
            vg_ds["scene2refs"] = scene2refs
            self.vg_data[ds] = vg_ds

    def __getitem__(self, idx):
        ds = random.randint(0, len(self.vg_ds_list) - 1)
        ds = self.vg_ds_list[ds]
        vg_ds = self.vg_data[ds]
        scenes = vg_ds["scenes"]
        # annotations = vg_ds["annotations"]
        scene2refs = vg_ds["scene2refs"]
        idx = random.randint(0, len(scenes) - 1)
        scene_id = scenes[idx]
        # scene_id = 'scannet/scene0000_00'
        # image_path = image_info["file_name"]
        # image_id = image_info["id"]
        refs = scene2refs[scene_id]
        if len(refs) == 0:
            return self.__getitem__(0)

        questions = []
        bboxes = []
        for ref in refs:
            questions.append(ref["conversations"][0]["value"])
            bboxes.append(np.array(ref["target"]["boxes"]))  # (num_boxes, 7)
        if len(questions) >= self.num_classes_per_sample:
            sampled_inds = np.random.choice(
                list(range(len(questions))), size=self.num_classes_per_sample, replace=False
            )
        else:
            sampled_inds = list(range(len(questions)))
        # sampled_sents = np.vectorize(sents.__getitem__)(sampled_inds).tolist()
        sampled_questions = [questions[ind] for ind in sampled_inds]
        # sampled_ann_ids = np.vectorize(ann_ids.__getitem__)(sampled_inds).tolist()
        sampled_bboxes = [bboxes[ind] for ind in sampled_inds][0]

        # preprocess video for video encoder
        video_path = os.path.join(self.video_folder, scene_id)
        video_dict = self.video_processor.preprocess(video_path, return_tensors="pt")
        # image = self.transform.apply_image(image)  # preprocess image for sam
        scene_dict = self.preprocess_embodiedscan(scene_id)   # preprocess points and images for embodiedscan

        refined_questions = []  # a scene could have multiple questions
        answers = []
        for question in sampled_questions:
            refined_question = question.replace('Please provide its coordinates.', 'Please respond with the 3D box.')
            refined_question = refined_question.replace(DEFAULT_VIDEO_TOKEN, DEFAULT_IMAGE_TOKEN)
            refined_questions.append(refined_question)
            answers.append(random.choice(self.answer_list))

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
            
        # bboxes = np.concatenate(sampled_bboxes, axis=0)  # (num_bboxes, 7)
        # bboxes = torch.from_numpy(bboxes)

        data_sample = Det3DDataElement()
        gt_instances_3d = InstanceData()
                
        data_metas = {}
        for key, value in scene_dict.items():
            if key not in ['imgs', 'points']:
                data_metas[key] = value
    
        data_sample.set_metainfo(data_metas)
        # hack for 7dof to 9dof
        sampled_bboxes = np.concatenate([sampled_bboxes, np.zeros((sampled_bboxes.shape[0], 2))], axis=-1)
        gt_instances_3d['bboxes_3d'] = self.box_type_3d(
            sampled_bboxes,
            box_dim=9,
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


class VGValDataset(Base3DValDataset):

    def __init__(self, 
                 data = "scanrefer",
                 box_type_3d = 'Euler-Depth',
                 **kwargs):
        super().__init__(data=data, **kwargs)
        self.vg_question_list = VG_QUESTION_LIST
        self.answer_list = ANSWER_LIST
        self.box_type_3d, self.box_mode_3d = get_box_type(box_type_3d)

    def __len__(self):
        return len(self.val_refs)
        
    def process_data(self):
        self.vg_ds_list = self.data.split("||")  # ['scanreason', 'sr3d', 'nr3d']
        for ds in self.vg_ds_list:
            annos = mmengine.load(os.path.join(self.json_folder, 'llava3d_scannet_' + ds + '_val.json'))
            val_refs = []
            val_refs += annos
        self.val_refs = val_refs

    def __getitem__(self, idx):
        ref = self.val_refs[idx]
        scene_id = ref['video']
        question = ref["conversations"][0]["value"]
        bboxes = np.array(ref["target"]["boxes"])

        # preprocess video for video encoder
        video_path = os.path.join(self.video_folder, scene_id)
        video_dict = self.video_processor.preprocess(video_path, return_tensors="pt")
        scene_dict = self.preprocess_embodiedscan(scene_id)   # preprocess points and images for embodiedscan

        refined_question = question.replace('Please provide its coordinates.', 'Please respond with the 3D box.')
        refined_question = refined_question.replace(DEFAULT_VIDEO_TOKEN, DEFAULT_IMAGE_TOKEN)
        answer = random.choice(self.answer_list)

        conversations = []
        conv = conversation_lib.default_conversation.copy()
        conv.messages = []
        conv.append_message(conv.roles[0], refined_question)
        conv.append_message(conv.roles[1], answer)
        conversations.append(conv.get_prompt())

        data_sample = Det3DDataElement()
        gt_instances_3d = InstanceData()
                
        data_metas = {}
        for key, value in scene_dict.items():
            if key not in ['imgs', 'points']:
                data_metas[key] = value
    
        data_sample.set_metainfo(data_metas)
        # hack for 7dof to 9dof
        bboxes = np.concatenate([bboxes, np.zeros((bboxes.shape[0], 2))], axis=-1)
        gt_instances_3d['bboxes_3d'] = self.box_type_3d(
            bboxes,
            box_dim=9,
            with_yaw=True,
            origin=(0.5, 0.5, 0.5))
        gt_instances_3d['labels_3d'] = torch.ones((len(bboxes),), dtype=torch.int64)
        data_sample.gt_instances_3d = gt_instances_3d
        inference = True

        return (
            video_path,
            video_dict,  # for MLLM
            scene_dict,  # for embodiedscan
            conversations[0],  # list of str, 1<=len<=3
            data_sample,
            refined_question,
            inference
        )