from .sparse_featfusion_grounder import SparseFeatureFusion3DGrounder
from .pointnet_grounder import PointNet3DGrounder
from .grounding_head import ReGround3DGroundingHead
from .match_cost import ReGround3DBinaryFocalLossCost
import torch

def build_sparse_featfusion_grounder(checkpoint=None):
    model = SparseFeatureFusion3DGrounder(
        num_queries=100,
        voxel_size=0.01,
        backbone=dict(
            type='mmdet.ResNet',
            depth=50,
            base_channels=16,  # to make it consistent with mink resnet
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            frozen_stages=1,
            norm_cfg=dict(type='BN', requires_grad=False),
            norm_eval=True,
            init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
            style='pytorch'),
        backbone_3d=dict(type='MinkResNet', in_channels=3, depth=34),
        use_xyz_feat=True,
        # change due to no img feature fusion
        neck_3d=dict(type='MinkNeck',
                    num_classes=1,
                    in_channels=[128, 256, 512, 1024],
                    out_channels=256,
                    voxel_size=0.01,
                    pts_prune_threshold=1000),
        decoder=dict(
            num_layers=6,
            return_intermediate=True,
            layer_cfg=dict(
                # query self attention layer
                self_attn_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
                # cross attention layer query to text
                cross_attn_text_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
                # cross attention layer query to image
                cross_attn_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
                ffn_cfg=dict(embed_dims=256,
                            feedforward_channels=2048,
                            ffn_drop=0.0)),
            post_norm_cfg=None),
        bbox_head=dict(type='ReGround3DGroundingHead',
                    num_classes=1,
                    sync_cls_avg_factor=True,
                    decouple_bbox_loss=True,
                    decouple_groups=4,
                    share_pred_layer=True,
                    decouple_weights=[0.2, 0.2, 0.2, 0.4],
                    contrastive_cfg=dict(log_scale='auto', bias=True),
                    loss_cls=dict(type='mmdet.FocalLoss',
                                    use_sigmoid=True,
                                    gamma=2.0,
                                    alpha=0.25,
                                    loss_weight=1.0),
                    loss_bbox=dict(type='BBoxCDLoss',
                                    mode='l1',
                                    loss_weight=1.0,
                                    group='g8')),
        coord_type='DEPTH',
        # training and testing settings
        train_cfg=dict(assigner=dict(type='HungarianAssigner3D',
                                    match_costs=[
                                        dict(type='ReGround3DBinaryFocalLossCost',
                                            weight=1.0),
                                        dict(type='BBox3DL1Cost', weight=2.0),
                                        dict(type='IoU3DCost', weight=2.0)
                                    ]), ),
        test_cfg=None)
    model.eval()
    # for name, param in model.state_dict().items():
    #     print(f"{name}: {param.size()}")
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)['state_dict']
        model.load_state_dict(state_dict, strict=False)
        print('Successfully load the grounder checkpoints!')
    return model

def build_pointnet_grounder(checkpoint=None):
    model = PointNet3DGrounder(
        num_queries=50,
        use_xyz_feat=True,
        decoder=dict(
            num_layers=4,
            return_intermediate=True,
            layer_cfg=dict(
                # query self attention layer
                self_attn_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
                # cross attention layer query to text
                cross_attn_text_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
                # cross attention layer query to image
                cross_attn_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
                ffn_cfg=dict(embed_dims=256,
                            feedforward_channels=2048,
                            ffn_drop=0.0)),
            post_norm_cfg=None),
        bbox_head=dict(type='ReGround3DGroundingHead',
                    num_classes=1,
                    sync_cls_avg_factor=True,
                    decouple_bbox_loss=True,
                    decouple_groups=4,
                    share_pred_layer=True,
                    decouple_weights=[0.2, 0.2, 0.2, 0.4],
                    contrastive_cfg=dict(log_scale='auto', bias=True),
                    loss_cls=dict(type='mmdet.FocalLoss',
                                    use_sigmoid=True,
                                    gamma=2.0,
                                    alpha=0.25,
                                    loss_weight=1.0),
                    loss_bbox=dict(type='BBoxCDLoss',
                                    mode='l1',
                                    loss_weight=1.0,
                                    group='g8')),
        coord_type='DEPTH',
        # training and testing settings
        train_cfg=dict(assigner=dict(type='HungarianAssigner3D',
                                    match_costs=[
                                        dict(type='ReGround3DBinaryFocalLossCost',
                                            weight=1.0),
                                        dict(type='BBox3DL1Cost', weight=2.0),
                                        dict(type='IoU3DCost', weight=2.0)
                                    ]), ),
        test_cfg=None)
    model.eval()
    # for name, param in model.state_dict().items():
    #     print(f"{name}: {param.size()}")
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)['state_dict']
        model.load_state_dict(state_dict, strict=False)
        print('Successfully load the grounder checkpoints!')
    return model