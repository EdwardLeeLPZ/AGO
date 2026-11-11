# ------------------------------------------------------------------------
# AGO
# Copyright (c) 2025 Peizheng Li. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------

from mmdet3d.registry import TRANSFORMS
from mmdet3d.datasets.transforms.formating import Pack3DDetInputs

@TRANSFORMS.register_module()
class GroundingPack3DDetInputs(Pack3DDetInputs):
    INPUTS_KEYS = [
        'raw_img', 'img', 'points', 
        'voxel_semantics', 'voxel_mask', 'pseudo_voxel_semantics',
    ]
    INSTANCEDATA_3D_KEYS = [
        'gt_bboxes_3d', 'gt_labels_3d', 'attr_labels', 'depths', 'centers_2d'
    ]
    INSTANCEDATA_2D_KEYS = [
        'gt_bboxes', 'gt_bboxes_labels',
    ]
    SEG_KEYS = [
        'gt_seg_map', 'pts_instance_mask', 'pts_semantic_mask', 'gt_semantic_seg'
    ]