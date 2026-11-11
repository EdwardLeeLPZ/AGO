# ------------------------------------------------------------------------
# AGO
# Copyright (c) 2025 Peizheng Li. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------

import copy
import numpy as np
from typing import Dict, Optional, Sequence
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger
from mmdet3d.registry import METRICS


from .metrics_util import Metric_mIoU


@METRICS.register_module()
class OccSegMetric(BaseMetric):
    """3D semantic occupancy evaluation metric.

    Args:
        collect_device (str, optional): Device name used for collecting
            results from different ranks during distributed training.
            Must be 'cpu' or 'gpu'. Defaults to 'cpu'.
        prefix (str): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Default: None.
        open_world_stage (str): One of 'all', 'pretraining', 'finetuning'.
    """

    def __init__(self,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 open_world_stage=None,
                 **kwargs):
        self.open_world_stage = open_world_stage

        super(OccSegMetric, self).__init__(
            prefix=prefix, collect_device=collect_device)

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions.

        The processed results should be stored in ``self.results``,
        which will be used to compute the metrics when all batches
        have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from
                the model.
        """
        for data_sample in data_samples:
            pred_3d = data_sample['pred_pts_seg']
            eval_ann_info = data_sample['eval_ann_info']
            cpu_pred_3d = dict()
            for k, v in pred_3d.items():
                if hasattr(v, 'to'):
                    cpu_pred_3d[k] = v.to('cpu').numpy()
                else:
                    cpu_pred_3d[k] = v
            self.results.append((eval_ann_info, cpu_pred_3d))

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()

        self.mIoU_eval_metrics = Metric_mIoU(name='', open_world_stage=self.open_world_stage)
        self.lidar_visible_mIoU_eval_metrics = Metric_mIoU(name='lidar visible ', open_world_stage=self.open_world_stage)

        if self.open_world_stage == 'pretraining':
            self.mIoU_eval_metrics_zs = Metric_mIoU(name='', open_world_stage='finetuning')
            self.lidar_visible_mIoU_eval_metrics_zs = Metric_mIoU(name='lidar visible ', open_world_stage='finetuning')

        for eval_ann, sinlge_pred_results in results:
            gt_semantic_mask = copy.deepcopy(eval_ann['pts_semantic_mask'].astype(np.int32))
            pred_semantic_mask = copy.deepcopy(sinlge_pred_results['pts_semantic_mask'].astype(np.int32))

            lidar_visible_mask = copy.deepcopy(eval_ann['lidar_visible_mask'])
            lidar_visible_gt_semantic_mask = gt_semantic_mask[lidar_visible_mask]
            lidar_visible_pred_semantic_mask = pred_semantic_mask[lidar_visible_mask]

            self.mIoU_eval_metrics.add_batch(pred_semantic_mask, gt_semantic_mask)
            self.lidar_visible_mIoU_eval_metrics.add_batch(lidar_visible_pred_semantic_mask, lidar_visible_gt_semantic_mask)

            if self.open_world_stage == 'pretraining':
                gt_semantic_mask = copy.deepcopy(eval_ann['pts_semantic_mask'].astype(np.int32))
                pred_semantic_mask = copy.deepcopy(sinlge_pred_results['pts_semantic_mask'].astype(np.int32))
                lidar_visible_mask = copy.deepcopy(eval_ann['lidar_visible_mask'])
                lidar_visible_gt_semantic_mask = gt_semantic_mask[lidar_visible_mask]
                lidar_visible_pred_semantic_mask = pred_semantic_mask[lidar_visible_mask]
                
                self.mIoU_eval_metrics_zs.add_batch(pred_semantic_mask, gt_semantic_mask)
                self.lidar_visible_mIoU_eval_metrics_zs.add_batch(lidar_visible_pred_semantic_mask, lidar_visible_gt_semantic_mask)

        ret_dict = self.mIoU_eval_metrics.count_miou(logger)
        for k, v in self.lidar_visible_mIoU_eval_metrics.count_miou(logger).items():
            ret_dict[f'lidar_visible_{k}'] = v

        if self.open_world_stage == 'pretraining':
            for k, v in self.mIoU_eval_metrics_zs.count_miou(logger).items():
                ret_dict[f'{k}_zs'] = v
            for k, v in self.lidar_visible_mIoU_eval_metrics_zs.count_miou(logger).items():
                ret_dict[f'lidar_visible_{k}_zs'] = v

        ret_dict.update({f'{self.open_world_stage}/{k}': v for k, v in ret_dict.items()})

        return ret_dict
