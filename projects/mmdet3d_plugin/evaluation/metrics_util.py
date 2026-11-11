# ------------------------------------------------------------------------
# AGO
# Copyright (c) 2025 Peizheng Li. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from CVPR2023-3D-Occupancy-Prediction (https://github.com/CVPR2023-3D-Occupancy-Prediction/CVPR2023-3D-Occupancy-Prediction)
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
# ------------------------------------------------------------------------

import copy
import numpy as np
from terminaltables import AsciiTable

from mmengine.logging import print_log

from ..datasets.label_space import NUSCENES2ID, OPENWORLD2ID, NUSCENES2OPENWORLD, OPENWORLD_FINE2COARSE

np.seterr(divide='ignore', invalid='ignore')

class Metric_mIoU():
    def __init__(self,
                 name="mIoU",
                 save_dir='.',
                 open_world_stage=None):
        self.open_world_stage = open_world_stage
        if self.open_world_stage in ['pretraining', 'finetuning']:
            self.class_names = ['pedestrian', 'driveable surface', 'sidewalk', 'vehicle', 'cycle', 'car', 'bus', 'truck', 'construction vehicle',
                                'trailer', 'bicycle', 'motorcycle', 'barrier', 'traffic cone', 'terrain', 'manmade', 'vegetation', 'free']
        else:
            self.class_names = ['others','barrier', 'bicycle', 'bus', 'car', 'construction vehicle',
                            'motorcycle', 'pedestrian', 'traffic cone', 'trailer', 'truck',
                            'driveable surface', 'other flat', 'sidewalk',
                            'terrain', 'manmade', 'vegetation', 'free']
            
        self.name = name
        self.save_dir = save_dir
        self.num_classes = len(self.class_names)

        self.open_world_stage = open_world_stage
        if self.open_world_stage == 'all':
            self.known_class_names = ['others','barrier', 'bicycle', 'bus', 'car', 'construction vehicle',
                            'motorcycle', 'pedestrian', 'traffic cone', 'trailer', 'truck',
                            'driveable surface', 'other flat', 'sidewalk',
                            'terrain', 'manmade', 'vegetation']
            self.unknown_class_names = []
            self.known_class_ids = [NUSCENES2ID[name] for name in self.known_class_names]
            self.unknown_class_ids = [NUSCENES2ID[name] for name in self.unknown_class_names]
        elif self.open_world_stage == 'pretraining':
            self.known_class_names = ['pedestrian', 'driveable surface', 'sidewalk', 'vehicle', 'cycle']
            self.unknown_class_names = ['car', 'bus', 'construction vehicle', 'truck', 'trailer', 
                                'bicycle', 'motorcycle', 'barrier', 'traffic cone', 'terrain', 'manmade', 'vegetation']           
            self.known_class_ids = [OPENWORLD2ID[name] for name in self.known_class_names]
            self.unknown_class_ids = [OPENWORLD2ID[name] for name in self.unknown_class_names]
        elif self.open_world_stage == 'finetuning':
            self.known_class_names = ['pedestrian', 'driveable surface', 'sidewalk']
            self.unknown_class_names = ['car', 'bus', 'construction vehicle', 'truck', 'trailer', 
                                'bicycle', 'motorcycle', 'barrier', 'traffic cone', 'terrain', 'manmade', 'vegetation']
            self.known_class_ids = [OPENWORLD2ID[name] for name in self.known_class_names]
            self.unknown_class_ids = [OPENWORLD2ID[name] for name in self.unknown_class_names]
        else:
            raise ValueError(f'No match open world stage for {self.open_world_stage}')

        self.class_names = [name.replace(' ', '_') for name in self.class_names]

        self.point_cloud_range = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
        self.occupancy_size = [0.4, 0.4, 0.4]
        self.voxel_size = 0.4
        self.occ_xdim = int((self.point_cloud_range[3] - self.point_cloud_range[0]) / self.occupancy_size[0])
        self.occ_ydim = int((self.point_cloud_range[4] - self.point_cloud_range[1]) / self.occupancy_size[1])
        self.occ_zdim = int((self.point_cloud_range[5] - self.point_cloud_range[2]) / self.occupancy_size[2])
        self.voxel_num = self.occ_xdim * self.occ_ydim * self.occ_zdim
        self.hist = np.zeros((self.num_classes, self.num_classes))
        self.cnt = 0

    def hist_info(self, n_cl, pred, gt):
        """Build the confusion matrix.

        Args:
            n_cl (int): Number of classes.
            pred (np.ndarray): Predicted labels, 1-D.
            gt (np.ndarray): Ground truth labels, 1-D. Labels outside
                ``[0, n_cl)`` are ignored.

        Returns:
            tuple: Confusion matrix, number of correct predictions and number
            of labeled voxels.
        """
        assert pred.shape == gt.shape
        k = (gt >= 0) & (gt < n_cl)
        labeled = np.sum(k)
        correct = np.sum((pred[k] == gt[k]))

        return (
            np.bincount(
                n_cl * gt[k].astype(int) + pred[k].astype(int), minlength=n_cl ** 2
            ).reshape(n_cl, n_cl),
            correct,
            labeled,
        )

    def per_class_iu(self, hist):

        return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

    def per_class_recall(self, hist):

        return np.diag(hist) / hist.sum(1)

    def compute_mIoU(self, pred, label, n_classes):
        hist = np.zeros((n_classes, n_classes))
        new_hist, correct, labeled = self.hist_info(n_classes, pred.flatten(), label.flatten())
        hist += new_hist
        mIoUs = self.per_class_iu(hist)
        return round(np.nanmean(mIoUs) * 100, 2), hist

    def add_batch(self, semantics_pred, semantics_gt):
        if self.open_world_stage in ['pretraining', 'finetuning']:
            semantics_ori_pred = copy.deepcopy(semantics_pred)
            semantics_ori_gt = copy.deepcopy(semantics_gt)
            for nusc_id, ow_id in NUSCENES2OPENWORLD.items():
                semantics_pred[semantics_ori_pred == nusc_id] = ow_id
                semantics_gt[semantics_ori_gt == nusc_id] = ow_id
            
            if self.open_world_stage == 'pretraining':
                for fine_id, coarse_id in OPENWORLD_FINE2COARSE.items():
                    if coarse_id == -1:
                        continue
                    semantics_pred[semantics_pred == fine_id] = coarse_id
                    semantics_gt[semantics_gt == fine_id] = coarse_id
            
            semantics_pred = semantics_pred[semantics_gt != -1]
            semantics_gt = semantics_gt[semantics_gt != -1]

            semantics_pred[semantics_pred == -1] = self.num_classes - 1
            
            assert semantics_pred.shape == semantics_gt.shape, f'semantics_pred shape {semantics_pred.shape} != semantics_gt shape {semantics_gt.shape}'
        self.cnt += 1
        _, _hist = self.compute_mIoU(semantics_pred, semantics_gt, self.num_classes)
        self.hist += _hist
    
    def count_miou(self, logger=None):
        ret_dict = dict()
        print_log(f'===> pred hist: \n{self.hist.sum(0).astype(np.int64)}',  logger=logger)
        print_log(f'===> gt hist: \n{self.hist.sum(1).astype(np.int64)}',  logger=logger)

        mIoU = self.per_class_iu(self.hist)
        header, table_columns = ['classes'], [['results']]
        for ind_class in range(self.num_classes):
            ret_dict[self.class_names[ind_class]] = round(mIoU[ind_class] * 100, 2)
            header.append(self.class_names[ind_class])
            table_columns.append([round(mIoU[ind_class] * 100, 2)])
        ret_dict['mIoU'] = round(np.nanmean(mIoU[:self.num_classes-1]) * 100, 2)
        header.append('mIoU')
        table_columns.append([round(np.nanmean(mIoU[:self.num_classes-1]) * 100, 2)])

        if self.open_world_stage is not None:
            ret_dict['known_mIoU'] = round(np.nanmean(mIoU[self.known_class_ids]) * 100, 2) if len(self.known_class_ids) > 0 else 0
            header.append('known_mIoU')
            table_columns.append([round(np.nanmean(mIoU[self.known_class_ids]) * 100, 2) if len(self.known_class_ids) > 0 else 0])

            ret_dict['unknown_mIoU'] = round(np.nanmean(mIoU[self.unknown_class_ids]) * 100, 2) if len(self.unknown_class_ids) > 0 else 0
            header.append('unknown_mIoU')
            table_columns.append([round(np.nanmean(mIoU[self.unknown_class_ids]) * 100, 2) if len(self.unknown_class_ids) > 0 else 0])

            recall = self.per_class_recall(self.hist)
            ret_dict['unknown_recall'] = round(np.nanmean(recall[self.unknown_class_ids]) * 100, 2) if len(self.unknown_class_ids) > 0 else 0
            header.append('unknown_recall')
            table_columns.append([round(np.nanmean(recall[self.unknown_class_ids]) * 100, 2) if len(self.unknown_class_ids) > 0 else 0])

        table_data = [header]
        table_rows = list(zip(*table_columns))
        table_data += table_rows
        table = AsciiTable(table_data)
        table.inner_footing_row_border = True
        print_log(f'{self.name}semantic occupancy results of {self.open_world_stage}: \n' + table.table, logger=logger)

        if self.open_world_stage is not None and len(self.known_class_ids) > 0:
            self.occ_hist = np.zeros((3, 3))
            self.occ_hist[0, 0] = self.hist[self.known_class_ids][:, self.known_class_ids].sum()
            self.occ_hist[0, 1] = self.hist[self.known_class_ids][:, self.unknown_class_ids].sum()
            self.occ_hist[0, 2] = self.hist[self.known_class_ids][:, -1].sum()
            self.occ_hist[1, 0] = self.hist[self.unknown_class_ids][:, self.known_class_ids].sum()
            self.occ_hist[1, 1] = self.hist[self.unknown_class_ids][:, self.unknown_class_ids].sum()
            self.occ_hist[1, 2] = self.hist[self.unknown_class_ids][:, -1].sum()
            self.occ_hist[2, 0] = self.hist[-1][self.known_class_ids].sum()
            self.occ_hist[2, 1] = self.hist[-1][self.unknown_class_ids].sum()
            self.occ_hist[2, 2] = self.hist[-1][-1]

            IoU = self.per_class_iu(self.occ_hist)
            header, table_columns = ['classes'], [['results']]
            for i, name in enumerate(['known_IoU', 'unknown_IoU']):
                ret_dict[name] = round(IoU[i] * 100, 2)
                header.append(name)
                table_columns.append([round(IoU[i] * 100, 2)])
            
            self.occ_hist = np.zeros((2, 2))
            self.occ_hist[0, 0] = self.hist[np.arange(self.num_classes-1)][:, np.arange(self.num_classes-1)].sum()
            self.occ_hist[0, 1] = self.hist[np.arange(self.num_classes-1)][:, -1].sum()
            self.occ_hist[1, 0] = self.hist[-1][np.arange(self.num_classes-1)].sum()
            self.occ_hist[1, 1] = self.hist[-1][-1]

            IoU = self.per_class_iu(self.occ_hist)
            ret_dict['IoU'] = round(IoU[0] * 100, 2)
            header.append('IoU')
            table_columns.append([round(IoU[0] * 100, 2)])
            
            table_data = [header]
            table_rows = list(zip(*table_columns))
            table_data += table_rows
            table = AsciiTable(table_data)
            table.inner_footing_row_border = True
            print_log(f'{self.name}occupancy results of {self.open_world_stage}: \n' + table.table, logger=logger)
        else:
            self.occ_hist = np.zeros((2, 2))
            self.occ_hist[0, 0] = self.hist[np.arange(self.num_classes-1), np.arange(self.num_classes-1)].sum()
            self.occ_hist[0, 1] = self.hist[np.arange(self.num_classes-1), -1].sum()
            self.occ_hist[1, 0] = self.hist[-1, np.arange(self.num_classes-1)].sum()
            self.occ_hist[1, 1] = self.hist[-1, -1]

            IoU = self.per_class_iu(self.occ_hist)
            header, table_columns = ['classes'], [['results']]
            ret_dict['IoU'] = round(IoU[0] * 100, 2)
            header.append('IoU')
            table_columns.append([round(IoU[0] * 100, 2)])

            table_data = [header]
            table_rows = list(zip(*table_columns))
            table_data += table_rows
            table = AsciiTable(table_data)
            table.inner_footing_row_border = True
            print_log(f'{self.name}occupancy results of {self.open_world_stage}: \n' + table.table, logger=logger)

        return ret_dict
