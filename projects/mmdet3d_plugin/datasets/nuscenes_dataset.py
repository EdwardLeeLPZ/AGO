# ------------------------------------------------------------------------
# AGO
# Copyright (c) 2025 Peizheng Li. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------

import json
import random
import numpy as np
import os.path as osp
from collections import defaultdict
from typing import Callable, List, Union
from mmengine.dataset import BaseDataset, force_full_init
from mmdet3d.registry import DATASETS

from .pipelines.open_world_token2classes import token2classes
from .label_space import OPENWORLD2ID, OPENWORLD_FINE2COARSE

@DATASETS.register_module()
class Occ3DNuScenesDataset(BaseDataset):
    r"""Occ3D-nuScenes dataset.

    Args:
        data_root (str): Path of dataset root.
        ann_file (str): Path of annotation file.
        pipeline (list[dict]): Pipeline used for data processing.
            Defaults to [].
        test_mode (bool): Store `True` when building test or val dataset.
        serialize_data (bool): Whether to serialize the data list.
            Defaults to False.
        version (str): Suffix of the few-shot sample file
            ``nuscenes_selected_samples_{version}.json``. Defaults to 'trainval'.
        open_world_stage (str): One of 'all', 'pretraining', 'finetuning'.
        num_shot (int): Number of samples per class for few-shot finetuning.
            Defaults to 100.
    """
    METAINFO = {
        'classes':
        ('others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
         'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
         'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
         'vegetation', 'free'),
        'ignore_index':
        18,
        'label_mapping':
        dict([(1, 0), (5, 0), (7, 0), (8, 0), (10, 0), (11, 0), (13, 0),
              (19, 0), (20, 0), (0, 0), (29, 0), (31, 0), (9, 1), (14, 2),
              (15, 3), (16, 3), (17, 4), (18, 5), (21, 6), (2, 7), (3, 7),
              (4, 7), (6, 7), (12, 8), (22, 9), (23, 10), (24, 11), (25, 12),
              (26, 13), (27, 14), (28, 15), (30, 16)]),
        'palette': [
            [0, 0, 0],  # others
            [255, 120, 50],  # barrier
            [255, 192, 203],  # bicycle
            [255, 255, 0],  # bus
            [0, 150, 245],  # car
            [0, 255, 255],  # construction_vehicle
            [255, 127, 0],  # motorcycle
            [255, 0, 0],  # pedestrian
            [255, 240, 150],  # traffic_cone
            [135, 60, 0],  # trailer
            [160, 32, 240],  # truck
            [255, 0, 255],  # driveable_surface
            [139, 137, 137],  # other_flat
            [75, 0, 75],  # sidewalk
            [150, 240, 80],  # terrain
            [230, 230, 250],  # manmade
            [0, 175, 0],  # vegetation
        ]
    }

    def __init__(self,
                 data_root: str,
                 ann_file: str,
                 pipeline: List[Union[dict, Callable]] = [],
                 test_mode: bool = False,
                 serialize_data: bool = False,
                 version: str = 'trainval',
                 open_world_stage=None,
                 num_shot=100,
                 **kwargs) -> None:
        metainfo = dict(label2cat={
            i: cat_name
            for i, cat_name in enumerate(self.METAINFO['classes'])
        })
        super().__init__(
            ann_file=ann_file,
            data_root=data_root,
            metainfo=metainfo,
            pipeline=pipeline,
            test_mode=test_mode,
            serialize_data=serialize_data,
            **kwargs)
        
        self.open_world_stage = open_world_stage
        self.num_shot = num_shot
        if not self.test_mode and self.open_world_stage == 'finetuning':
            self.select_sweep(version)
    
    def select_sweep(self, version):
        few_shot_tokens = self.few_shot_select({sample['token'] for sample in self.data_list}, version)
        self.data_list = [d for d in self.data_list if d['token'] in few_shot_tokens]

    def few_shot_select(self, sample_tokens, version='trainval'):
        sample_file = osp.join(osp.dirname(osp.abspath(__file__)), f'nuscenes_selected_samples_{version}.json')
        if osp.exists(sample_file):
            with open(sample_file, 'r') as f:
                return set(json.load(f))

        few_shot_classes = list(OPENWORLD_FINE2COARSE)
        token2class_ids = {token: [int(c) for c in classes.split(',')]
                           for token, classes in token2classes.items() if token in sample_tokens}
        class2tokens = defaultdict(list)
        for token, class_ids in token2class_ids.items():
            for class_id in class_ids:
                class2tokens[class_id].append(token)

        selected_tokens = set()
        for class_id in few_shot_classes:
            selected_tokens.update(random.sample(class2tokens[class_id], min(self.num_shot, len(class2tokens[class_id]))))
        few_shot_counter = np.zeros(max(OPENWORLD2ID.values()) + 1)
        for token in selected_tokens:
            few_shot_counter[token2class_ids[token]] += 1

        # drop samples whose removal keeps every class in at least num_shot samples
        for token in random.sample(sorted(selected_tokens), len(selected_tokens)):
            few_shot_counter[token2class_ids[token]] -= 1
            if np.all(few_shot_counter[few_shot_classes] >= self.num_shot):
                selected_tokens.remove(token)
            else:
                few_shot_counter[token2class_ids[token]] += 1

        return selected_tokens

    @force_full_init
    def __len__(self) -> int:
        """Get the length of filtered dataset and automatically call
        ``full_init`` if the dataset has not been fully init.

        Returns:
            int: The length of filtered dataset.
        """
        if self.serialize_data:
            return len(self.data_address)
        else:
            return len(self.data_list)

    def parse_data_info(self, info: dict) -> Union[List[dict], dict]:
        """Convert the relative paths of a raw data info to absolute paths.

        Args:
            info (dict): Raw info dict.

        Returns:
            List[dict] or dict: The processed data info.
        """

        data_list = []
        info['lidar_points']['lidar_path'] = \
            osp.join(
                self.data_prefix.get('pts', ''),
                info['lidar_points']['lidar_path'])

        for cam_id, img_info in info['images'].items():
            if 'img_path' in img_info:
                if cam_id in self.data_prefix:
                    cam_prefix = self.data_prefix[cam_id]
                else:
                    cam_prefix = self.data_prefix.get('img', '')
                img_info['img_path'] = osp.join(cam_prefix,
                                                img_info['img_path'])

        if 'pts_semantic_mask_path' in info:
            info['pts_semantic_mask_path'] = \
                osp.join(self.data_prefix.get('pts_semantic_mask', ''),
                         info['pts_semantic_mask_path'])

        info['seg_label_mapping'] = self.metainfo['label_mapping']

        if self.test_mode:
            info['eval_ann_info'] = dict()

        data_list.append(info)
        return data_list
