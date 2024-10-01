import os

import numpy as np
import torch
from typing import Union
from embodiedscan.structures.bbox_3d import points_img2cam, points_cam2img
from mmcv.transforms import Resize
from mmengine.structures import InstanceData
from embodiedscan.utils.typing_config import Det3DDataElement
from embodiedscan.structures.bbox_3d.euler_depth_box3d import EulerDepthInstance3DBoxes

from PIL import Image
from pathlib import Path

MEAN_RGB = np.array([109.8, 97.2, 83.8])

def process_videos(videos, video_processor, mode='random', device=None, text=None):
    if isinstance(videos, str):
        videos = [videos]
    new_videos = []
    for video in videos:
        video = video_processor.preprocess(video, return_tensors='pt', mode=mode, device=device, text=text)
        new_videos.append(video)

    new_images = [video['images'] for video in new_videos]
    new_depths = [video['depth_images'] for video in new_videos]
    new_poses = [video['poses'] for video in new_videos]
    new_intrinsics = [video['intrinsic'] for video in new_videos]
    
    videos_dict = dict()
    videos_dict['images'] = torch.stack(new_images, dim=0)
    videos_dict['depths'] = torch.stack(new_depths, dim=0)
    videos_dict['poses'] = torch.stack(new_poses, dim=0)
    videos_dict['intrinsics'] = torch.stack(new_intrinsics, dim=0)
    return videos_dict

def load_grounder_input(video_path, video_processor, num_frame, n_points):
    scene_dict = preprocess_embodiedscan(video_path, video_processor, num_frame, n_points)
    data_sample = Det3DDataElement()
    gt_instances_3d = InstanceData()
                
    data_metas = {}
    for key, value in scene_dict.items():
        if key not in ['imgs', 'points']:
            data_metas[key] = value
    data_sample.set_metainfo(data_metas)
    # hack for 7dof to 9dof
    # sampled_bboxes = np.concatenate([sampled_bboxes, np.zeros((sampled_bboxes.shape[0], 2))], axis=-1)
    sampled_bboxes = np.zeros((0, 9))
    gt_instances_3d['bboxes_3d'] = EulerDepthInstance3DBoxes(
        sampled_bboxes,
        box_dim=9,
        with_yaw=True,
        origin=(0.5, 0.5, 0.5))
    gt_instances_3d['labels_3d'] = torch.ones((len(sampled_bboxes),), dtype=torch.int64)
    data_sample.gt_instances_3d = gt_instances_3d

    return scene_dict, data_sample


def preprocess_embodiedscan(video, video_processor, num_frames, n_points) -> dict:
    resize = Resize(scale=(480, 480), keep_ratio=False)
    video_path = Path(video)
    video_name = str(Path(*video_path.parts[-2:]))
    video_folder = str(Path(*video_path.parts[:-2]))
    video_info = video_processor.scene[video_name]
    dataset = video.split('/')[-2]
    video_frames = [str(key) for key in video_info.keys() if key.startswith(dataset)]  # remove other paramters
    
    ids = np.arange(len(video_frames))
    replace = True if num_frames > len(ids) else False
    step = (len(ids) - 1) // (num_frames - 1)  # TODO: BUG, fix from branch fbocc
    if step > 0:
        ids = ids[::step]
        # sometimes can not get the accurate num_frames in this way
        # then take the first num_frames one
        ids = ids[:num_frames]
    else:  # the number of images < pre-set num_frames
        # randomly select num_frames ids to enable batch-wise inference
        # In practice, can directly use the original ids to avoid
        # redundant computation
        ids = np.random.choice(ids, num_frames, replace=replace)

    axis_align_matrix = np.array(video_info['axis_align_matrix'])  # 4x4 array

    if dataset == 'matterport3d':
        depth_shift  = 4000.0
    else:
        depth_shift = 1000.0

    if 'depth_intrinsic' in video_info:
        depth_intrinsic = np.array(video_info['depth_intrinsic'])  # 4x4 array

    if 'intrinsic' in video_info:
        intrinsic = np.array(video_info['intrinsic'])  # 4x4 array

    imgs = []
    points = []
    intrinsics = []
    extrinsics = []
    results = dict()

    for i in ids.tolist():
        frame = video_frames[i]
        img_path = os.path.join(video_folder, frame)
        img = Image.open(img_path).convert('RGB')
        img = np.array(img)  # convert from PIL to array
        depth_img_path =  os.path.join(video_folder, video_info[frame]['depth'])
        depth_img = Image.open(depth_img_path)
        depth_img = np.array(depth_img) / depth_shift  # obtain the true depth
        pose = np.array(video_info[frame]['pose']) # 4x4 array
        extrinsic = np.linalg.inv(axis_align_matrix @ pose).astype(np.float32)
        cam2img = video_info[frame].get('intrinsc', intrinsic)
        if depth_intrinsic is not None:
            depth_cam2img = depth_intrinsic
        else:
            depth_cam2img = cam2img

        frame_points = convert_rgbd_to_points(depth_img, depth_cam2img, img, cam2img)
        frame_points = points_random_sampling(frame_points, n_points//10)
        result = dict()
        result['img'] = img
        result = resize(result)
        imgs.append(result['img'])
        points.append(frame_points)
        extrinsics.append(extrinsic)
        intrinsics.append(cam2img)
    
    for key in result.keys():
        if key != 'img':
            results[key] = result[key]

    # stack and then convert to tensor
    imgs = np.stack(imgs, axis=0)
    imgs = torch.from_numpy(imgs).permute(0, 3, 1, 2).contiguous()
    results['imgs'] = imgs   # (V, 3, H, W)

    global_points = []
    for idx in range(len(points)):
        point = torch.from_numpy(points[idx]).to(torch.float32)
        point_xyz = point[:, :3]
        point_xyz = torch.cat([point_xyz, point_xyz.new_ones(point_xyz.shape[0], 1)],
                        dim=1)
        global2ego = torch.from_numpy(extrinsics[idx]).to(point_xyz.device)
        global_point_xyz = (torch.linalg.solve(global2ego, point_xyz.transpose(
            0, 1))).transpose(0, 1)
        point[:, :3] = global_point_xyz[:, :3]
        global_points.append(point)
    global_points = torch.cat(global_points)
    global_points = points_random_sampling(global_points, n_points)

    results['points'] = global_points  # (N, 3)

    results['depth2img'] = dict()
    results['depth2img']['intrinsic'] = intrinsics
    results['depth2img']['extrinsic'] = extrinsics

    # 'img_shape', 'scale', 'scale_factor', 'keep_ratio'
    return results


def points_random_sampling(
    points: np.array,
    num_samples: Union[int, float],
    replace: bool = False,
    return_choices: bool = False
):
    """Points random sampling.

    Sample points to a certain number.

    Args:
        points (:obj:`np.array`): 3D Points.
        num_samples (int, float): Number of samples to be sampled. If
            float, we sample random fraction of points from num_points
            to 100%.
        replace (bool): Sampling with or without replacement.
            Defaults to False.
        return_choices (bool): Whether return choice. Defaults to False.

    Returns:
        tuple[:obj:`BasePoints`, np.ndarray] | :obj:`BasePoints`:

            - points (:obj:`BasePoints`): 3D Points.
            - choices (np.ndarray, optional): The generated random samples.
    """
    if isinstance(num_samples, float):
        assert num_samples < 1
        num_samples = int(
            np.random.uniform(len(points), 1.) *
            points.shape[0])  # TODO: confusion

    if not replace:
        replace = (points.shape[0] < num_samples)
    point_range = range(len(points))
    choices = np.random.choice(point_range, num_samples, replace=replace)
    if return_choices:
        return points[choices], choices
    else:
        return points[choices]
    
def convert_rgbd_to_points(depth_img, depth_cam2img, img, cam2img, use_color=True, normalize_color=True):
    # obtain the camera coordinates points 
    ws = np.arange(depth_img.shape[1])
    hs = np.arange(depth_img.shape[0])
    us, vs = np.meshgrid(ws, hs)
    grid = np.stack(
        [us.astype(np.float32),
            vs.astype(np.float32), depth_img], axis=-1).reshape(-1, 3)
    nonzero_indices = depth_img.reshape(-1).nonzero()[0]
    grid3d = points_img2cam(grid, depth_cam2img)
    points = grid3d[nonzero_indices]

    if use_color:
        h, w = img.shape[0], img.shape[1]
        points2d = np.round(points_cam2img(points,
                                            cam2img)).astype(np.int32)
        us = np.clip(points2d[:, 0], a_min=0, a_max=w - 1)
        vs = np.clip(points2d[:, 1], a_min=0, a_max=h - 1)
        rgb_points = img[vs, us]
        if normalize_color:
            rgb_points = rgb_points - MEAN_RGB
            rgb_points = rgb_points / 255.0
        points = np.concatenate([points, rgb_points], axis=-1)

    return points
