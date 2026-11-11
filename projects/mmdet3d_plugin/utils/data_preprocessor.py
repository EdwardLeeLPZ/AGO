# ------------------------------------------------------------------------
# AGO
# Copyright (c) 2025 Peizheng Li. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------

from typing import List
import torch
import numpy as np
from torch import Tensor
import torchvision.transforms as T
from mmdet.models.utils.misc import samplelist_boxtype2tensor
from mmdet3d.registry import MODELS
from mmdet3d.structures.det3d_data_sample import SampleList
from mmdet3d.models import Det3DDataPreprocessor
from mmdet3d.models.data_preprocessors.voxelize import dynamic_scatter_3d

from featup.util import norm


@MODELS.register_module()
class GroundingDataPreprocessor(Det3DDataPreprocessor):

    def __init__(self, num_classes=18, max_num_pts=40000, mask_clip_img_resize=None, **kwargs):
            super(GroundingDataPreprocessor, self).__init__(**kwargs)
            self.num_classes = num_classes
            self.max_num_pts = max_num_pts
            assert mask_clip_img_resize is not None, "mask_clip_img_resize must be set"
            self.mask_clip_img_resize = tuple(mask_clip_img_resize)
            
            if 'grid_shape' in kwargs['voxel_layer']:
                self.grid_shape = torch.tensor(kwargs['voxel_layer']['grid_shape']).int()
            elif 'voxel_size' in kwargs['voxel_layer']:
                self.grid_shape = torch.tensor(self.voxel_layer.grid_shape).int()

    def simple_process(self, data: dict, training: bool = False) -> dict:
        """Collate the batch, preprocess the raw images for MaskCLIP and
        voxelize the points if ``voxel`` is True.

        Args:
            data (dict): Data sampled from dataloader.
            training (bool): Whether to apply ``batch_augments``.
                Defaults to False.

        Returns:
            dict: Data in the same format as the model input.
        """
        if 'img' in data['inputs']:
            batch_pad_shape = self._get_pad_shape(data)

        data = self.collate_data(data)
        inputs, data_samples = data['inputs'], data['data_samples']
        batch_inputs = inputs

        # preprocess the raw images for MaskCLIP
        if 'raw_img' in inputs:
            transform = T.Compose([
                T.Resize(self.mask_clip_img_resize),
                T.ToTensor(),
                norm])

            # inputs['raw_img'][cam][sample] after collation
            batch_inputs['raw_imgs'] = torch.stack([
                torch.stack([transform(cam_imgs[i]) for cam_imgs in inputs['raw_img']])
                for i in range(len(inputs['raw_img'][0]))]).float() # B, N, C, H, W


        if 'imgs' in inputs:
            imgs = inputs['imgs']

            if data_samples is not None:
                batch_input_shape = tuple(imgs[0].size()[-2:])
                for data_sample, pad_shape in zip(data_samples,
                                                  batch_pad_shape):
                    data_sample.set_metainfo({
                        'batch_input_shape': batch_input_shape,
                        'pad_shape': pad_shape
                    })

                if self.boxtype2tensor:
                    samplelist_boxtype2tensor(data_samples)
                if self.pad_mask:
                    self.pad_gt_masks(data_samples)
                if self.pad_seg:
                    self.pad_gt_sem_seg(data_samples)

        if 'points' in inputs:
            batch_inputs['points'] = inputs['points']

            if self.voxel:
                voxel_dict = self.voxelize(inputs, data_samples)
                batch_inputs['voxels'] = voxel_dict

            if training and self.batch_augments is not None:
                for batch_aug in self.batch_augments:
                    imgs, data_samples = batch_aug(imgs, data_samples)
            batch_inputs['imgs'] = imgs

        return {'inputs': batch_inputs, 'data_samples': data_samples}

    @torch.no_grad()
    def voxelize(self, inputs: List[Tensor],
                 data_samples: SampleList) -> List[Tensor]:
        """Voxelize the points in the ego frame and store the point, voxel and
        camera coordinates in ``data_samples``.

        Args:
            inputs (dict): The input data, including points and images.
            data_samples (List[:obj:`Det3DDataSample`]): The annotation data
                of every sample.
        """
        for data_sample, point in zip(data_samples, inputs['points']):
            point_coors = point[:, :3]

            if hasattr(data_sample, 'lidar2ego'):
                # change the coordinate system from lidar to ego
                lidar2ego = np.asarray(data_sample.lidar2ego)
                lidar2ego = point_coors.new_tensor(lidar2ego)  # (4, 4)

                point_coors_h = torch.cat(
                    (point_coors, torch.ones_like(point_coors[..., :1])), -1).unsqueeze(-1) # (N, 4, 1)
                lidar2ego = lidar2ego.view(1, 4, 4).repeat(point_coors_h.shape[0], 1, 1) # (N, 4, 4)
                point_coors = torch.matmul(
                    lidar2ego.to(torch.float32),
                    point_coors_h.to(torch.float32)).squeeze(-1)[:, :3] # (N, 3)

            min_bound = point_coors.new_tensor(self.voxel_layer.point_cloud_range[:3])
            coors = (point_coors - min_bound) / point_coors.new_tensor(self.voxel_layer.voxel_size)
            
            valid = (torch.floor(coors).long() >= 0).all(dim=-1) & (torch.floor(coors).long() < self.grid_shape.to(coors.device)).all(dim=-1)
            point_coors, coors = point_coors[valid], coors[valid]
                    
            data_sample.point_coors_float = coors.clone()
            coors = torch.floor(coors).long()
            data_sample.point_coors = coors

            self.get_camera_coords(point[valid, :3], data_sample, inputs)
            self.get_voxel_seg(coors, data_sample)

            data_sample.grid_shape = point_coors.new_tensor(self.grid_shape)
            data_sample.voxel_size = point_coors.new_tensor(self.voxel_layer.voxel_size)
            data_sample.point_cloud_range = point_coors.new_tensor(self.voxel_layer.point_cloud_range)
            data_sample.voxel_coords_center = torch.round((0 - min_bound) / point_coors.new_tensor(self.voxel_layer.voxel_size)).long()

    def get_voxel_seg(self, res_coors: Tensor, data_sample: SampleList):
        """Store the occupied voxel coordinates and the point-to-voxel and
        voxel-to-point maps in ``data_sample``.

        Args:
            res_coors (Tensor): The voxel coordinates of points, Nx3.
            data_sample (:obj:`Det3DDataSample`): The annotation data of the
                sample.
        """
        pts_semantic_mask = torch.ones([res_coors.shape[0], 1], device=res_coors.device).float()
        voxel_semantic_mask, voxel_coors, point2voxel_map = \
            dynamic_scatter_3d(pts_semantic_mask, res_coors, 'mean', True)
        voxel_semantic_mask = torch.argmax(voxel_semantic_mask, dim=-1)
        data_sample.gt_pts_seg.voxel_semantic_mask = voxel_semantic_mask
        point2voxel_map = point2voxel_map.long()
        data_sample.point2voxel_map = point2voxel_map

        point_indices = torch.arange(point2voxel_map.shape[0], device=point2voxel_map.device, dtype=point2voxel_map.dtype)
        voxel2point_map = point2voxel_map.new_zeros(voxel_coors.shape[0])
        voxel2point_map.scatter_(0, point2voxel_map, point_indices)
        data_sample.voxel2point_map = voxel2point_map

        data_sample.voxel_coors = voxel_coors
    
    def get_camera_coords(self, point_coors, data_sample, inputs):
        lidar2img = np.asarray(data_sample.lidar2img)
        lidar2img = point_coors.new_tensor(lidar2img)  # (num_cam, 4, 4)
        num_cam = lidar2img.shape[0]
        num_points = point_coors.shape[0]

        point_coors_h = torch.cat(
            (point_coors, torch.ones_like(point_coors[..., :1])), -1) # (N, 4)
        point_coors_h = point_coors_h.view(-1, 1, 4).repeat(1, num_cam, 1).unsqueeze(-1) # (N, num_cam, 4, 1)
        lidar2img = lidar2img.view(1, num_cam, 4, 4).repeat(num_points, 1, 1, 1) # (N, num_cam, 4, 4)

        point_coors_cam = torch.matmul(
            lidar2img.to(torch.float32),
            point_coors_h.to(torch.float32)).squeeze(-1) # (N, num_cam, 4)
        point_depth_cam = point_coors_cam[..., 2:3]
        point_coors_cam = point_coors_cam[..., 0:2] / point_coors_cam[..., 2:3] # (N, num_cam, 2)

        image_shape = data_sample.batch_input_shape
        point_cam_mask = (
            (point_depth_cam > 0)
            & (point_coors_cam[..., 0:1] > 0)
            & (point_coors_cam[..., 0:1] < image_shape[1])
            & (point_coors_cam[..., 1:2] > 0)
            & (point_coors_cam[..., 1:2] < image_shape[0])
        ).squeeze() # (N, num_cam)
        data_sample.point_coors_cam = point_coors_cam
        data_sample.point_cam_mask = point_cam_mask
