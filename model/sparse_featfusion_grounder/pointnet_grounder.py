# Copyright (c) OpenRobotLab. All rights reserved.
# Adapted from https://github.com/SamsungLabs/fcaf3d/blob/master/mmdet3d/models/detectors/single_stage_sparse.py # noqa
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

from mmengine.model import BaseModel
from mmengine.structures import InstanceData

from .pointnet_backbone import Pointnet2Backbone
from embodiedscan.models.layers import SparseFeatureFusionTransformerDecoder
from embodiedscan.registry import MODELS
from embodiedscan.utils import ConfigType, OptConfigType
from embodiedscan.utils.typing_config import (ForwardResults, InstanceList,
                                              OptSampleList, SampleList)


class PointNet3DGrounder(BaseModel):
    """SparseFusionSingleStage3DDetector.

    Args:
        backbone (dict): Config dict of detector's backbone.
        backbone_3d (dict): Config dict of detector's 3D backbone.
        bbox_head (dict, optional): Config dict of box head. Defaults to None.
        neck (dict, optional): Config dict of neck. Defaults to None.
        neck_3d (dict, optional): Config dict of 3D neck. Defaults to None.
        decoder (dict, optional): Config dict of decoder. Defaults to None.
        voxel_size (float): Voxel size for spatial quantization.
            Defaults to 0.01.
        num_queries (float): Number of object queries. Defaults to 512.
        max_num_entities (int): The maximum number of entities.
            Defaults to 256.
        train_cfg (dict, optional): Config dict of training hyper-parameters.
            Defaults to None.
        test_cfg (dict, optional): Config dict of test hyper-parameters.
            Defaults to None.
        data_preprocessor (dict or ConfigDict, optional): The pre-process
            config of :class:`BaseDataPreprocessor`.  it usually includes,
                ``pad_size_divisor``, ``pad_value``, ``mean`` and ``std``.
        use_xyz_feat (bool): Whether to use xyz features.
            Defaults to False.
        init_cfg (dict or ConfigDict, optional): the config to control the
            initialization. Defaults to None.
    """
    _version = 2

    def __init__(self,
                 bbox_head: ConfigType,
                 decoder: ConfigType = None,
                 num_queries: int = 50,
                 coord_type: str = 'CAMERA',
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 use_xyz_feat: bool = False,
                 init_cfg: OptConfigType = None):
        super().__init__(data_preprocessor=data_preprocessor,
                         init_cfg=init_cfg)
        self.backbone_3d = Pointnet2Backbone(input_feature_dim=3)
        self.feat_map = nn.Linear(288, 256)
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = MODELS.build(bbox_head)
        self.decoder = decoder
        self.coord_type = coord_type
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.num_queries = num_queries
        self.use_xyz_feat = use_xyz_feat
        self._init_layers()

    def _init_layers(self) -> None:
        """Initialize layers except for backbone, neck and bbox_head."""
        # text modules
        # t_type = 'roberta-base'
        # self.tokenizer = RobertaTokenizerFast.from_pretrained(t_type)
        # self.text_encoder = RobertaModel.from_pretrained(t_type)

        self.decoder = SparseFeatureFusionTransformerDecoder(**self.decoder)
        # map the text feature to the target dimension number
        # self.embed_dims = self.decoder.embed_dims
        # self.text_feat_map = nn.Linear(self.text_encoder.config.hidden_size,
        #                                self.embed_dims,
        #                                bias=True)

    @property
    def with_neck(self):
        """Whether the detector has a 2D backbone."""
        return hasattr(self, 'neck') and self.neck is not None

    @property
    def with_neck_3d(self):
        """Whether the detector has a 3D neck."""
        return hasattr(self, 'neck_3d') and self.neck_3d is not None

    def extract_feat(
        self, batch_inputs_dict: Dict[str,
                                      Tensor], batch_data_samples: SampleList
    ) -> Union[Tuple[torch.Tensor], Dict[str, Tensor]]:
        """Directly extract features from the backbone+neck.

        Args:
            batch_inputs_dict (dict): The model input dict which includes
                'points' keys.

                    - points (list[torch.Tensor]): Point cloud of each sample.

        Returns:
            tuple[Tensor] | dict:  For outside 3D object detection, we
                typically obtain a tuple of features from the backbone + neck,
                and for inside 3D object detection, usually a dict containing
                features will be obtained.
        """
        points = batch_inputs_dict['points']
        stack_points = torch.stack(points)  # (B, N, 3)
        feats_dict = self.backbone_3d(stack_points)
        coords = feats_dict['fp2_xyz']  # (B, 1024, 3)
        feats = feats_dict['fp2_features'].transpose(1, 2)  # (B, 1024, 288)
        feats = self.feat_map(feats).to(torch.bfloat16)
        scores = None
        # channel mapper feature of different level to the fixed number

        return feats, scores, coords

    def forward_transformer(self,
                            point_feats: List[Tensor],
                            scores: List[Tensor],
                            point_xyz: List[Tensor],
                            pred_embeddings: Tensor,
                            batch_data_samples: OptSampleList = None) -> Dict:
        decoder_inputs_dict, head_inputs_dict = self.pre_decoder(
            point_feats, scores, point_xyz, pred_embeddings)
        decoder_outputs_dict = self.forward_decoder(**decoder_inputs_dict)
        head_inputs_dict.update(decoder_outputs_dict)
        return head_inputs_dict

    def pre_decoder(
        self,
        feats_list: List[Tensor],
        scores_list: List[Tensor],
        xyz_list: List[Tensor],
        pred_embeddings: Tensor,
        batch_data_samples: OptSampleList = None,
    ) -> Tuple[Dict]:

        feats_with_pos_list = [
            torch.cat((feats, pos), dim=-1)
            for feats, pos in zip(feats_list, xyz_list)
        ]
        # batch the list of tensor
        max_feats_length = max(feats.size(0) for feats in feats_with_pos_list)
        min_feats_length = min(feats.size(0) for feats in feats_with_pos_list)
        padding_length = [
            max_feats_length - feats.size(0) for feats in feats_with_pos_list
        ]

        padded_feats_list = []
        feats_mask_list = []
        for batch_id, feats in enumerate(feats_with_pos_list):
            # If padding is needed, create a padding tensor
            # of the corresponding size.
            if padding_length[batch_id] > 0:
                padding_feats = torch.zeros(padding_length[batch_id],
                                            feats.size(1)).to(feats.device)
                padded_feats = torch.cat([feats, padding_feats], dim=0)
            else:
                padded_feats = feats
            padded_feats_list.append(padded_feats)
            feats_mask = torch.zeros(max_feats_length,
                                     dtype=torch.bool).to(feats.device)
            feats_mask[:feats.size(0)] = 1
            feats_mask_list.append(feats_mask)

        feats_with_pos = torch.stack(
            padded_feats_list)  # (b, max_feats_length, C+3)
        feats_mask = torch.stack(
            feats_mask_list).bool()  # (b, max_feats_length)

        feats, coords = feats_with_pos[..., :-3], feats_with_pos[..., -3:]

        # (b, max_feats_length, max_text_length)
        enc_outputs_class = self.bbox_head.cls_branches[0](feats, pred_embeddings,
                                     feats_mask)

        # calculate the min visual token sizes in the batch
        topk = min(self.num_queries, min_feats_length)
        topk_indices = torch.topk(enc_outputs_class.max(-1)[0], k=topk,
                                  dim=1)[1]

        # bbox_preds = self.bbox_head.reg_branches[self.decoder.num_layers](
        #     feats)
        bbox_preds = self.bbox_head.reg_branches[0](
            feats)
        bbox_pred_bboxes = self.bbox_head._bbox_pred_to_bbox(
            coords, bbox_preds)

        topk_query_coords = torch.gather(
            coords, 1,
            topk_indices.unsqueeze(-1).repeat(1, 1, 3))
        topk_pred_bboxes = torch.gather(
            bbox_pred_bboxes, 1,
            topk_indices.unsqueeze(-1).repeat(1, 1, 9))
        topk_feats = torch.gather(
            feats, 1,
            topk_indices.unsqueeze(-1).repeat(1, 1, feats.size(-1)))

        decoder_inputs_dict = dict(
            query=topk_feats,
            feats=feats,
            feats_attention_mask=~feats_mask,
            query_coords=topk_query_coords,
            feats_coords=coords,
            pred_bboxes=topk_pred_bboxes.detach().clone(),
            pred_embeddings=pred_embeddings)

        head_inputs_dict = dict(pred_embeddings=pred_embeddings)
        return decoder_inputs_dict, head_inputs_dict

    def forward_decoder(self, query: Tensor, feats: Tensor,
                        feats_attention_mask: Tensor, query_coords: Tensor,
                        feats_coords: Tensor, pred_bboxes: Tensor,
                        pred_embeddings: Tensor) -> Dict:
        """Forward with Transformer decoder.

        The forward procedure of the transformer is defined as:
        'pre_transformer' -> 'encoder' -> 'pre_decoder' -> 'decoder'
        More details can be found at `TransformerDetector.forward_transformer`
        in `mmdet/detector/base_detr.py`.

        Args:
            query (Tensor): The queries of decoder inputs, has shape
                (bs, num_queries_total, dim), where `num_queries_total` is the
                sum of `num_denoising_queries` and `num_matching_queries` when
                `self.training` is `True`, else `num_matching_queries`.

        Returns:
            dict: The dictionary of decoder outputs, which includes the
            `hidden_states` of the decoder output and `references` including
            the initial and intermediate reference_points.
        """
        inter_states, pred_bboxes = self.decoder(
            query=query,
            key=feats,
            value=feats,
            key_padding_mask=feats_attention_mask,
            self_attn_mask=None,
            cross_attn_mask=None,
            query_coords=query_coords,
            key_coords=feats_coords,
            pred_bboxes=pred_bboxes,
            text_feats=pred_embeddings,
            text_attention_mask=None,
            bbox_head=self.bbox_head)

        decoder_outputs_dict = dict(hidden_states=inter_states,
                                    all_layers_pred_bboxes=pred_bboxes)
        return decoder_outputs_dict

    def loss(self, batch_inputs_dict: dict, batch_data_samples: SampleList,
             pred_embeddings: torch.Tensor, **kwargs) -> Union[dict, list]:
        """Calculate losses from a batch of inputs dict and data samples.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                'points', 'img' keys.

                    - points (list[torch.Tensor]): Point cloud of each sample.
                    - imgs (torch.Tensor, optional): Image of each sample.

            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`, `gt_panoptic_seg_3d` and `gt_sem_seg_3d`.
            pred_embeddings (tensor): (batch_size, 1, C), the text embedding from
                the MLLM.

        Returns:
            dict: A dictionary of loss components.
        """

        point_feats, scores, point_xyz = self.extract_feat(
            batch_inputs_dict, batch_data_samples)
        head_inputs_dict = self.forward_transformer(point_feats, scores,
                                                    point_xyz, pred_embeddings,
                                                    batch_data_samples)
        losses = self.bbox_head.loss(**head_inputs_dict,
                                     batch_data_samples=batch_data_samples)
        return losses

    def predict(self, batch_inputs_dict, batch_data_samples, pred_embeddings):

        point_feats, scores, point_xyz = self.extract_feat(
            batch_inputs_dict, batch_data_samples)
        head_inputs_dict = self.forward_transformer(point_feats, scores,
                                                    point_xyz, pred_embeddings,
                                                    batch_data_samples)
        results_list = self.bbox_head.predict(
            **head_inputs_dict, batch_data_samples=batch_data_samples)

        for data_sample, pred_instances_3d in zip(batch_data_samples,
                                                  results_list):
            data_sample.pred_instances_3d = pred_instances_3d
        return batch_data_samples

    def create_positive_map(self, tokenized, tokens_positive,
                            batch_idx) -> Tensor:
        """construct a map such that positive_map[i,j] = True
        if box i is associated to token j

        Args:
            tokenized: The tokenized input.
            tokens_positive (list): A list of token ranges
                associated with positive boxes.

        Returns:
            torch.Tensor: The positive map.

        Raises:
            Exception: If an error occurs during token-to-char mapping.
        """
        # max number of tokens
        positive_map = torch.zeros(
            (len(tokens_positive), self.max_num_entities), dtype=torch.float)

        for j, tok_list in enumerate(tokens_positive):
            for (beg, end) in tok_list:
                try:
                    beg_pos = tokenized.char_to_token(batch_idx, beg)
                    end_pos = tokenized.char_to_token(batch_idx, end - 1)
                except Exception as e:
                    print('beg:', beg, 'end:', end)
                    print('token_positive:', tokens_positive)
                    raise e
                if beg_pos is None:
                    try:
                        beg_pos = tokenized.char_to_token(batch_idx, beg + 1)
                        if beg_pos is None:
                            beg_pos = tokenized.char_to_token(
                                batch_idx, beg + 2)
                    except Exception:
                        beg_pos = None
                if end_pos is None:
                    try:
                        end_pos = tokenized.char_to_token(batch_idx, end - 2)
                        if end_pos is None:
                            end_pos = tokenized.char_to_token(
                                batch_idx, end - 3)
                    except Exception:
                        end_pos = None
                if beg_pos is None or end_pos is None:
                    continue

                assert beg_pos is not None and end_pos is not None
                positive_map[j, beg_pos:end_pos + 1].fill_(1)

        return positive_map / (positive_map.sum(-1)[:, None] + 1e-6)

    def get_positive_map(self, tokenized, tokens_positive):
        # each data sample can contain a single box or multiple grounding boxes
        # Unify the single and multi grounding box cases
        positive_maps = []
        for idx, tp in enumerate(tokens_positive):
            positive_map = self.create_positive_map(tokenized, tp, idx)
            positive_maps.append(positive_map)
        return positive_maps

    def forward(self,
                inputs: Union[dict, List[dict]],
                data_samples: Optional[List] = None,
                pred_embeddings: Tensor = None,
                mode: str = 'tensor',
                **kwargs) -> ForwardResults:
        """The unified entry for a forward process in both training and test.

        The method should accept three modes: "tensor", "predict" and "loss":

        - "tensor": Forward the whole network and return tensor or tuple of
        tensor without any post-processing, same as a common nn.Module.
        - "predict": Forward and return the predictions, which are fully
        processed to a list of :obj:`Det3DDataSample`.
        - "loss": Forward and return a dict of losses according to the given
        inputs and data samples.

        Note that this method doesn't handle neither back propagation nor
        optimizer updating, which are done in the :meth:`train_step`.

        Args:
            inputs  (dict | list[dict]): When it is a list[dict], the
                outer list indicate the test time augmentation. Each
                dict contains batch inputs
                which include 'points' and 'imgs' keys.

                - points (list[torch.Tensor]): Point cloud of each sample.
                - imgs (torch.Tensor): Image tensor has shape (B, C, H, W).
            data_samples (list[:obj:`Det3DDataSample`],
                list[list[:obj:`Det3DDataSample`]], optional): The
                annotation data of every samples. When it is a list[list], the
                outer list indicate the test time augmentation, and the
                inter list indicate the batch. Otherwise, the list simply
                indicate the batch. Defaults to None.
            mode (str): Return what kind of value. Defaults to 'tensor'.

        Returns:
            The return type depends on ``mode``.

            - If ``mode="tensor"``, return a tensor or a tuple of tensor.
            - If ``mode="predict"``, return a list of :obj:`Det3DDataSample`.
            - If ``mode="loss"``, return a dict of tensor.
        """
        if mode == 'loss':
            return self.loss(inputs, data_samples, pred_embeddings, **kwargs)
        elif mode == 'predict':
            return self.predict(inputs, data_samples, pred_embeddings, **kwargs)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports loss, predict and tensor mode')

    def _forward(self,
                 batch_inputs_dict: dict,
                 batch_data_samples: OptSampleList = None,
                 **kwargs) -> Tuple[List[torch.Tensor]]:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

         Args:
            batch_inputs_dict (dict): The model input dict which include
                'points', 'img' keys.

                    - points (list[torch.Tensor]): Point cloud of each sample.
                    - imgs (torch.Tensor, optional): Image of each sample.

            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`, `gt_panoptic_seg_3d` and `gt_sem_seg_3d`.

        Returns:
            tuple[list]: A tuple of features from ``bbox_head`` forward.
        """
        x = self.extract_feat(batch_inputs_dict, batch_data_samples)
        results = self.bbox_head.forward(x)
        return results

    def add_pred_to_datasample(
        self,
        data_samples: SampleList,
        data_instances_3d: Optional[InstanceList] = None,
        data_instances_2d: Optional[InstanceList] = None,
    ) -> SampleList:
        """Convert results list to `Det3DDataSample`.

        Subclasses could override it to be compatible for some multi-modality
        3D detectors.

        Args:
            data_samples (list[:obj:`Det3DDataSample`]): The input data.
            data_instances_3d (list[:obj:`InstanceData`], optional): 3D
                Detection results of each sample.
            data_instances_2d (list[:obj:`InstanceData`], optional): 2D
                Detection results of each sample.

        Returns:
            list[:obj:`Det3DDataSample`]: Detection results of the
            input. Each Det3DDataSample usually contains
            'pred_instances_3d'. And the ``pred_instances_3d`` normally
            contains following keys.

            - scores_3d (Tensor): Classification scores, has a shape
              (num_instance, )
            - labels_3d (Tensor): Labels of 3D bboxes, has a shape
              (num_instances, ).
            - bboxes_3d (Tensor): Contains a tensor with shape
              (num_instances, C) where C >=7.

            When there are image prediction in some models, it should
            contains  `pred_instances`, And the ``pred_instances`` normally
            contains following keys.

            - scores (Tensor): Classification scores of image, has a shape
              (num_instance, )
            - labels (Tensor): Predict Labels of 2D bboxes, has a shape
              (num_instances, ).
            - bboxes (Tensor): Contains a tensor with shape
              (num_instances, 4).
        """

        assert (data_instances_2d is not None) or \
               (data_instances_3d is not None),\
               'please pass at least one type of data_samples'

        if data_instances_2d is None:
            data_instances_2d = [
                InstanceData() for _ in range(len(data_instances_3d))
            ]
        if data_instances_3d is None:
            data_instances_3d = [
                InstanceData() for _ in range(len(data_instances_2d))
            ]

        for i, data_sample in enumerate(data_samples):
            data_sample.pred_instances_3d = data_instances_3d[i]
            data_sample.pred_instances = data_instances_2d[i]
        return data_samples