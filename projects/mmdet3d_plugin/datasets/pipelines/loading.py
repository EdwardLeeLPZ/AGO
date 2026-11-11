# ------------------------------------------------------------------------
# AGO
# Copyright (c) 2025 Peizheng Li. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------

import os
import copy
import numpy as np
from PIL import Image
from typing import Optional

import mmcv
from mmcv import BaseTransform
from mmengine.fileio import get
from mmdet3d.registry import TRANSFORMS
from mmdet3d.datasets.transforms.formating import to_tensor
from mmdet3d.datasets.transforms import LoadMultiViewImageFromFiles

from .nuscenes_token2name import token2name

@TRANSFORMS.register_module()
class OccLoadMultiViewImageFromFiles(LoadMultiViewImageFromFiles):
    """Load the multi-view images and camera parameters of the current frame.

    Takes the arguments of ``LoadMultiViewImageFromFiles`` and adds
    ``img_path``, ``lidar2img`` and ``raw_img`` (RGB PIL images for MaskCLIP)
    to its output keys.
    """

    def transform(self, results: dict) -> Optional[dict]:
        filename, cam2img, lidar2cam, lidar2img = [], [], [], []
        for _, cam_item in results['images'].items():
            filename.append(cam_item['img_path'])
            lidar2cam.append(cam_item['lidar2cam'])

            lidar2cam_array = np.array(cam_item['lidar2cam'])
            cam2img_array = np.eye(4).astype(np.float64)
            cam2img_array[:3, :3] = np.array(cam_item['cam2img'])
            cam2img.append(cam2img_array)
            lidar2img.append(cam2img_array @ lidar2cam_array)

        results['img_path'] = filename
        results['cam2img'] = np.stack(cam2img, axis=0)
        results['lidar2cam'] = np.stack(lidar2cam, axis=0)
        results['lidar2img'] = np.stack(lidar2img, axis=0)

        results['ori_cam2img'] = copy.deepcopy(results['cam2img'])

        imgs = [mmcv.imfrombytes(get(name, backend_args=self.backend_args), flag=self.color_type)
                for name in filename]
        img_shapes = np.stack([img.shape for img in imgs], axis=0)
        img_shape_max = np.max(img_shapes, axis=0)
        img_shape_min = np.min(img_shapes, axis=0)
        assert img_shape_min[-1] == img_shape_max[-1]
        if not np.all(img_shape_max == img_shape_min):
            pad_shape = img_shape_max[:2]
        else:
            pad_shape = None
        if pad_shape is not None:
            imgs = [
                mmcv.impad(img, shape=pad_shape, pad_val=0) for img in imgs
            ]

        if self.to_float32:
            imgs = [img.astype(np.float32) for img in imgs]

        results['filename'] = filename
        results['img'] = imgs
        results['img_shape'] = imgs[0].shape[:2]
        results['ori_shape'] = imgs[0].shape[:2]
        results['pad_shape'] = imgs[0].shape[:2]
        if self.set_default_scale:
            results['scale_factor'] = 1.0
        num_channels = 1 if len(imgs[0].shape) < 3 else imgs[0].shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        results['num_views'] = self.num_views
        results['num_ref_frames'] = self.num_ref_frames

        # keep the raw images for MaskCLIP
        results['raw_img'] = [Image.fromarray(img[..., [2, 1, 0]]) for img in results['img']]

        return results


@TRANSFORMS.register_module()
class LoadOccGTFromFile(BaseTransform):
    """Load Occ3D ground truth and, if available, grounded pseudo labels.

    Args:
        data_root (str): Root directory containing ``occ3d/gts`` and
            ``occ3d/pseudo_labels``.
        use_lidar_mask (bool): Mask voxels by the Occ3D LiDAR visibility mask
            for training and evaluation. Defaults to False.
        use_image_mask (bool): Mask voxels by the Occ3D camera visibility mask
            for training and evaluation. Defaults to True.
        load_pseudo_label (bool): Load pseudo labels from
            ``occ3d/pseudo_labels``. Defaults to True.
    """

    def __init__(
            self,
            data_root,
            use_lidar_mask=False,
            use_image_mask=True,
            load_pseudo_label=True,
        ):
        self.gt_data_root = os.path.join(data_root, 'occ3d', 'gts')
        self.pl_data_root = os.path.join(data_root, 'occ3d', 'pseudo_labels')
        self.use_lidar_mask = use_lidar_mask
        self.use_image_mask = use_image_mask
        self.load_pseudo_label = load_pseudo_label

    def transform(self, results: dict) -> Optional[dict]:
        sample_token = results['token']
        scene_name = token2name[sample_token]
        occ_gt_path = os.path.join(self.gt_data_root, scene_name, sample_token, 'labels.npz')

        occ_labels = np.load(occ_gt_path)
        semantics = occ_labels['semantics']
        mask_lidar = occ_labels['mask_lidar'].astype(bool)
        mask_camera = occ_labels['mask_camera'].astype(bool)

        if self.use_lidar_mask and not self.use_image_mask:
            mask = mask_lidar
        elif not self.use_lidar_mask and self.use_image_mask:
            mask = mask_camera
        elif self.use_lidar_mask and self.use_image_mask:
            mask = np.logical_and(mask_lidar, mask_camera)
        else:
            mask = np.ones_like(semantics).astype(bool)
        
        results['voxel_semantics'] = to_tensor(semantics)
        results['voxel_mask'] = to_tensor(mask)

        if 'eval_ann_info' not in results:
            results['eval_ann_info'] = {}
        results['eval_ann_info']['pts_semantic_mask'] = semantics[mask]

        if self.load_pseudo_label:
            occ_pl_path = os.path.join(self.pl_data_root, scene_name, sample_token, 'labels.npz')
            pseudo_labels = np.load(occ_pl_path)['semantics'][0]
            results['pseudo_voxel_semantics'] = to_tensor(pseudo_labels)

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        return "{} (data_root={})".format(
            self.__class__.__name__, self.gt_data_root)
