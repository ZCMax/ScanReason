import os

import numpy as np
import torch
from abc import abstractmethod
from typing import Union
from embodiedscan.structures.bbox_3d import points_img2cam, points_cam2img

from PIL import Image

class Base3DDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        video_folder,
        json_folder,
        video_processor,
        resize,
        samples_per_epoch=500 * 8 * 2 * 10,
        precision: str = "fp32",
        image_size: int = 336,
        num_classes_per_sample: int = 1,
        exclude_val=False,
        num_frames=26,
        n_points=100000,
        use_color=True,
        normalize_color=True,
        data=None,
    ):
        self.exclude_val = exclude_val
        self.samples_per_epoch = samples_per_epoch
        self.num_classes_per_sample = num_classes_per_sample

        self.video_folder = video_folder
        self.json_folder = json_folder
        self.image_size = image_size
        # self.tokenizer = tokenizer
        self.precision = precision
        self.video_processor = video_processor
        self.resize = resize

        self.num_frames = num_frames
        self.n_points = n_points
        self.use_color = use_color
        self.normalize_color = normalize_color
        self.mean_rgb = np.array([109.8, 97.2, 83.8])

        self.data = data
        self.process_data()

    def __len__(self):
        return self.samples_per_epoch

    @abstractmethod
    def process_data(self):
        pass
        
    def preprocess_embodiedscan(self, video) -> dict:
        video_info = self.video_processor.scene[video]
        dataset = video.split('/')[-2]
        video_frames = [str(key) for key in video_info.keys() if key.startswith(dataset)]  # remove other paramters
        
        ids = np.arange(len(video_frames))
        replace = True if self.num_frames > len(ids) else False
        step = (len(ids) - 1) // (self.num_frames - 1
                                    )  # TODO: BUG, fix from branch fbocc
        if step > 0:
            ids = ids[::step]
            # sometimes can not get the accurate num_frames in this way
            # then take the first num_frames one
            ids = ids[:self.num_frames]
        else:  # the number of images < pre-set num_frames
            # randomly select num_frames ids to enable batch-wise inference
            # In practice, can directly use the original ids to avoid
            # redundant computation
            ids = np.random.choice(ids, self.num_frames, replace=replace)

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
            img_path = os.path.join(self.video_folder, frame)
            img = Image.open(img_path).convert('RGB')
            img = np.array(img)  # convert from PIL to array
            depth_img_path =  os.path.join(self.video_folder, video_info[frame]['depth'])
            depth_img = Image.open(depth_img_path)
            depth_img = np.array(depth_img) / depth_shift  # obtain the true depth
            pose = np.array(video_info[frame]['pose']) # 4x4 array
            extrinsic = np.linalg.inv(axis_align_matrix @ pose).astype(np.float32)
            cam2img = video_info[frame].get('intrinsc', intrinsic)
            if depth_intrinsic is not None:
                depth_cam2img = depth_intrinsic
            else:
                depth_cam2img = cam2img

            frame_points = self.convert_rgbd_to_points(depth_img, depth_cam2img, img, cam2img)
            frame_points = self.points_random_sampling(frame_points, self.n_points//10)
            result = dict()
            result['img'] = img
            result = self.resize(result)
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
        global_points = self.points_random_sampling(global_points, self.n_points)

        results['points'] = global_points  # (N, 3)

        results['depth2img'] = dict()
        results['depth2img']['intrinsic'] = intrinsics
        results['depth2img']['extrinsic'] = extrinsics

        # 'img_shape', 'scale', 'scale_factor', 'keep_ratio'
        return results


    def points_random_sampling(
        self,
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
                np.random.uniform(self.num_points, 1.) *
                points.shape[0])  # TODO: confusion

        if not replace:
            replace = (points.shape[0] < num_samples)
        point_range = range(len(points))
        choices = np.random.choice(point_range, num_samples, replace=replace)
        if return_choices:
            return points[choices], choices
        else:
            return points[choices]
        
    def convert_rgbd_to_points(self, depth_img, depth_cam2img, img, cam2img):
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

        if self.use_color:
            h, w = img.shape[0], img.shape[1]
            points2d = np.round(points_cam2img(points,
                                               cam2img)).astype(np.int32)
            us = np.clip(points2d[:, 0], a_min=0, a_max=w - 1)
            vs = np.clip(points2d[:, 1], a_min=0, a_max=h - 1)
            rgb_points = img[vs, us]
            if self.normalize_color:
                rgb_points = rgb_points - self.mean_rgb
                rgb_points = rgb_points / 255.0
            points = np.concatenate([points, rgb_points], axis=-1)

        return points
