from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BitsAndBytesConfig, CLIPVisionModel

from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_PATCH_TOKEN)

from .llava.model.language_model.llava_llama import (LlavaLlamaForCausalLM,
                                                     LlavaLlamaModel)
from embodiedscan.models.data_preprocessors.data_preprocessor import Det3DDataPreprocessor
from .sparse_featfusion_grounder.build_sparse_featfusion_grounder import build_sparse_featfusion_grounder, build_pointnet_grounder


def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
    scale=1000,  # 100000.0,
    eps=1e-6,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1, 2)
    targets = targets.flatten(1, 2)
    numerator = 2 * (inputs / scale * targets).sum(-1)
    denominator = (inputs / scale).sum(-1) + (targets / scale).sum(-1)
    loss = 1 - (numerator + eps) / (denominator + eps)
    loss = loss.sum() / (num_masks + 1e-8)
    return loss


def sigmoid_ce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = loss.flatten(1, 2).mean(1).sum() / (num_masks + 1e-8)
    return loss


class ReGround3DMetaModel:
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(ReGround3DMetaModel, self).__init__(config)

        self.config = config
        if not hasattr(self.config, "train_box_decoder"):
            self.config.train_box_decoder = kwargs["train_box_decoder"]
            self.config.out_dim = kwargs["out_dim"]
            self.vision_pretrained = kwargs.get("vision_pretrained", None)
        else:
            self.vision_pretrained = kwargs.get("vision_pretrained", None)
            self.initialize_reground3d_modules(self.config)

    def initialize_reground3d_modules(self, config):
        # ReGround3D
        self.visual_model = build_pointnet_grounder(self.vision_pretrained)
        # self.visual_model = build_sparse_featfusion_grounder(self.vision_pretrained)
        for param in self.visual_model.parameters():
            param.requires_grad = False
        if config.train_box_decoder:
            self.visual_model.train()
            for param in self.visual_model.parameters():
                param.requires_grad = True
            # self.visual_model.decoder.train()
            # self.visual_model.feat_map.train()
            # self.visual_model.bbox_head.train()
            # for param in self.visual_model.decoder.parameters():
            #     param.requires_grad = True
            # for param in self.visual_model.feat_map.parameters():
            #     param.requires_grad = True
            # for param in self.visual_model.bbox_head.parameters():
            #     param.requires_grad = True

        # Projection layer
        in_dim = config.hidden_size
        out_dim = config.out_dim
        text_fc = [
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),
            nn.Dropout(0.0),
        ]
        self.text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_fc)])
        self.text_hidden_fcs.train()
        for param in self.text_hidden_fcs.parameters():
            param.requires_grad = True


class ReGround3DModel(ReGround3DMetaModel, LlavaLlamaModel):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(ReGround3DModel, self).__init__(config, **kwargs)

        self.config.use_cache = False
        self.config.vision_tower = self.config.mm_vision_tower
        self.config.video_tower = self.config.mm_video_tower
        self.config.mm_vision_select_feature = "patch"
        self.config.image_aspect_ratio = "square"
        self.config.image_grid_pinpoints = None
        self.config.tune_mm_mlp_adapter = False
        self.config.freeze_mm_mlp_adapter = True
        self.config.pretrain_mm_mlp_adapter = None
        self.config.mm_use_im_patch_token = False


class ReGround3DForCausalLM(LlavaLlamaForCausalLM):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        if not hasattr(config, "train_box_decoder"):
            config.mm_use_im_start_end = kwargs.pop("use_mm_start_end", True)
            config.mm_vision_tower = kwargs.get(
                "vision_tower", "openai/clip-vit-large-patch14-336"
            )
            config.mm_video_tower = kwargs.get(
                "video_tower", "SpatialAwareModule"
            )
            config.video_info_file = kwargs.pop("video_info_file", None)
            self.ce_loss_weight = kwargs.pop("ce_loss_weight", None)
            self.det_loss_weight = kwargs.pop("det_loss_weight", None)
        else:
            config.mm_vision_tower = config.vision_tower
            config.mm_video_tower = config.video_tower
            
        self.loc_token_idx = kwargs.pop("loc_token_idx")

        super().__init__(config)

        self.model = ReGround3DModel(config, **kwargs)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

        self.data_preprocessor = Det3DDataPreprocessor(
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            pad_mask=False,
        )

    def forward(self, **kwargs):
        if "past_key_values" in kwargs:
            return super().forward(**kwargs)
        return self.model_forward(**kwargs)
    
    def model_forward(
        self,
        points: List[torch.FloatTensor],
        images: List[torch.FloatTensor],
        images_clip: torch.FloatTensor,
        depths: torch.FloatTensor,
        poses: torch.FloatTensor,
        intrinsics: torch.FloatTensor,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor,
        attention_mask: torch.LongTensor,
        offset: torch.LongTensor,
        data_samples,
        inference: bool = False,
        **kwargs,
    ):
        # image_embeddings = self.get_visual_embs(images)
        # batch_size = image_embeddings.shape[0]
        batch_size = images_clip.shape[0]
        assert batch_size == len(offset) - 1

        grounder_targets= dict()

        # print('clip shape:', len(images_clip), 'points image shape:', len(images))
        grounder_targets['inputs'] = dict(img=images, points=points)  # could be empty list
        grounder_targets['data_samples'] = data_samples
        # print('len datasamples:', len(data_samples))

        # print('img_type:', images[0].dtype)
        # print('points_type:', points[0].dtype)

        # print(input_ids)
        # print('loc_token_idx:', self.loc_token_idx)
        loc_token_mask = input_ids[:, 1:] == self.loc_token_idx
        loc_token_mask = torch.cat(
            [
                loc_token_mask,
                torch.zeros((loc_token_mask.shape[0], 1)).bool().cuda(),
            ],
            dim=1,
        )  # (B, S)
        # hack for IMAGE_TOKEN_INDEX (we suppose that there is only one image, and it is in the front)
        loc_token_mask = torch.cat(
            [torch.zeros((loc_token_mask.shape[0], 1151)).bool().cuda(), loc_token_mask],
            dim=1,
        )

        if inference:
            assert images_clip.shape[0] == 1
            output = super().forward(
                images=images_clip,
                depths=depths,
                poses=poses,
                intrinsics=intrinsics,
                attention_mask=attention_mask,
                input_ids=input_ids,
                output_hidden_states=True,
            )
            output_hidden_states = output.hidden_states
        else:
            # image features is list of feature tensor
            output = super().forward(
                images=images_clip,
                depths=depths,
                poses=poses,
                intrinsics=intrinsics,
                attention_mask=attention_mask,
                input_ids=input_ids,
                labels=labels,
                output_hidden_states=True,
            )
            output_hidden_states = output.hidden_states   

        # print([feats.shape for feats in image_features])
        # assert len(image_features) == len(images_clip)

        hidden_states = []

        assert len(self.model.text_hidden_fcs) == 1
        # hidden_states.append(self.model.text_hidden_fcs[0](output_hidden_states[-1].to(torch.float32)))
        hidden_states.append(self.model.text_hidden_fcs[0](output_hidden_states[-1]))   # (bs, S, C')
        last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)  # (bs, S, C)

        # print('last_hidden_state:', last_hidden_state.shape)
        pred_embeddings = last_hidden_state[loc_token_mask]  # (num_loc_token, S)
        # pred_embeddings = pred_embeddings.to(torch.bfloat16)
        # print('pred_embddings:', pred_embeddings.shape)

        # print('loc_token_mask shape:', loc_token_mask.shape)
        # print([torch.sum(mask).item() for mask in loc_token_mask])
        # actual_loc_token_mask = []
        # for idx, feat in enumerate(image_features):
        #     mask = torch.cat([torch.zeros((len(feat)-1)).bool().cuda(), loc_token_mask[idx]])
        #     mask = torch.cat([mask, torch.zeros((last_hidden_state.shape[1]-mask.shape[0])).bool().cuda()])
        #     actual_loc_token_mask.append(mask)

        # print('actual_loc_token_mask:', [mask.shape for mask in actual_loc_token_mask])
        # pred_embeddings = []
        # for idx, hidden_state in enumerate(last_hidden_state):
        #     pred_embedding = hidden_state[actual_loc_token_mask[idx]]
        # pred_embeddings.append(pred_embedding)
        # pred_embeddings = torch.cat(pred_embeddings, dim=0)

        assert len(pred_embeddings) == len(images)  # 每个定为 conv 只有一个 [loc] token

        pred_embeddings = pred_embeddings.unsqueeze(1)  # (num_loc_token, 1, 256)

        # print(grounder_inputs)
        
        # print(f"pred_embeddings: {pred_embeddings.shape} on device: {pred_embeddings.device}")
        if not inference:
            model_output = output
            output = model_output.logits
            ce_loss = model_output.loss
            ce_loss = ce_loss * self.ce_loss_weight
            cls_loss = torch.tensor(0).to(dtype=ce_loss.dtype, device=ce_loss.device)
            bbox_loss = torch.tensor(0).to(dtype=ce_loss.dtype, device=ce_loss.device)

        if len(grounder_targets['inputs']['img']) > 0:
            self.data_preprocessor.to(self.model.device)
            grounder_inputs = self.data_preprocessor(grounder_targets)
            # self.model.visual_model.to(torch.float32)
            if not inference:
                grounder_output = self.model.visual_model.loss(grounder_inputs['inputs'], grounder_inputs['data_samples'], pred_embeddings)
                for key, value in grounder_output.items():
                    if 'loss_bbox' in key:
                        bbox_loss += value.squeeze(0)
                    elif 'loss_cls' in key:
                        cls_loss += value.squeeze(0)
            else:
                grounder_output = self.model.visual_model.predict(grounder_inputs['inputs'], grounder_inputs['data_samples'], pred_embeddings)
                predictions = []
                for i, pred in enumerate(grounder_output):
                    det_anno = dict()
                    gt_anno = dict()
                    det_anno["bboxes_3d"] = pred.pred_instances_3d.bboxes_3d
                    det_anno["target_scores_3d"] = pred.pred_instances_3d.scores_3d
                    gt_anno["gt_bboxes_3d"] = data_samples[i].gt_instances_3d.bboxes_3d
                    predictions.append((gt_anno, det_anno))
                # pred_bboxes_3d = grounder_output[0].pred_instances_3d.bboxes_3d
                # pred_scores_3d = grounder_output[0].pred_instances_3d.scores_3d
                return predictions

        cls_loss = cls_loss / 5
        bbox_loss = bbox_loss / 5
        det_loss = cls_loss + bbox_loss
        # print('cls_loss dtype:', cls_loss.dtype, 'bbox_loss:', bbox_loss.dtype)
        det_loss = det_loss * self.det_loss_weight

        loss = ce_loss + det_loss

        # print('losss:', loss, loss.device)
        # print('ce_loss dtype:', ce_loss.dtype)
        # print('loss value:', loss, 'loss type:', loss.dtype)
        return {
            "loss": loss,
            "ce_loss": ce_loss,
            "cls_loss": cls_loss,
            "bbox_loss": bbox_loss,
            "det_loss": det_loss,
        }

    def evaluate(
        self,
        points: List[torch.FloatTensor],
        images: List[torch.FloatTensor],
        images_clip: torch.FloatTensor,
        depths: torch.FloatTensor,
        poses: torch.FloatTensor,
        intrinsics: torch.FloatTensor,
        input_ids: torch.LongTensor,
        data_samples,
        max_new_tokens=32,
        tokenizer=None,
    ):
        with torch.no_grad():
            outputs = self.generate(
                input_ids,
                images=images_clip,
                depths=depths,
                poses=poses,
                intrinsics=intrinsics,
                max_new_tokens=max_new_tokens,
                num_beams=1,
                output_hidden_states=True,
                return_dict_in_generate=True
            )
            output_hidden_states = outputs.hidden_states[-1]
            output_ids = outputs.sequences

            loc_token_mask = output_ids[:, 1:] == self.loc_token_idx  # 找出模型的输出中预测 id 为 loc_token_id  # (B, max_tokens)
            # hack for IMAGE_TOKEN_INDEX (we suppose that there is only one image, and it is in the front)
            loc_token_mask = torch.cat(
                [
                    torch.zeros((loc_token_mask.shape[0], 1151)).bool().cuda(),
                    loc_token_mask,
                ],
                dim=1,
            )

            hidden_states = []

            assert len(self.model.text_hidden_fcs) == 1
            hidden_states.append(self.model.text_hidden_fcs[0](output_hidden_states))

            last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
            pred_embeddings = last_hidden_state[loc_token_mask]
            pred_embeddings = pred_embeddings.unsqueeze(1)  # (num_loc_token, 1, 256)

            grounder_targets= dict()

            grounder_targets['inputs'] = dict(img=images, points=points)  # could be empty list
            grounder_targets['data_samples'] = data_samples
            self.data_preprocessor.to(self.model.device)
            grounder_inputs = self.data_preprocessor(grounder_targets)
            grounder_output = self.model.visual_model.predict(grounder_inputs['inputs'], grounder_inputs['data_samples'], pred_embeddings)

        return output_ids, grounder_output
