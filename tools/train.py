# ------------------------------------------------------------------------
# AGO
# Copyright (c) 2025 Peizheng Li. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------

import argparse
import logging
import os
import os.path as osp
import time
import copy

from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.registry import RUNNERS
from mmengine.runner import Runner
from mmengine.dist import init_dist, get_dist_info

from mmdet3d.utils import replace_ceph_backend


def parse_args():
    parser = argparse.ArgumentParser(description='Train AGO')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--tag', default='', help='the tag of this experiment')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--sync_bn',
        choices=['none', 'torch', 'mmcv'],
        default='none',
        help='convert all BatchNorm layers in the model to SyncBatchNorm '
        '(SyncBN) or mmcv.ops.sync_bn.SyncBatchNorm (MMSyncBN) layers.')
    parser.add_argument(
        '--auto-scale-lr',
        action='store_true',
        help='enable automatically scaling LR.')
    parser.add_argument(
        '--resume',
        nargs='?',
        type=str,
        const='auto',
        help='If specify checkpoint path, resume from it, while if not '
        'specify, try to auto resume from the latest checkpoint '
        'in the work directory.')
    parser.add_argument(
        '--ceph', action='store_true', help='Use ceph as data storage backend')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    if args.ceph:
        cfg = replace_ceph_backend(cfg)

    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    
    # the logger depends on the dist info
    if args.launcher != 'none':
        init_dist(args.launcher)
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = './work_dirs'
    
    tag = args.tag if args.tag != '' else time.strftime('%Y%m%d_%H%M%S', time.localtime())
    cfg.work_dir = osp.join(cfg.work_dir, osp.basename(args.config).split('.')[0], tag)

    open_world_stage = cfg.open_world_stage
    ori_work_dir = copy.deepcopy(cfg.work_dir)
    for stage in open_world_stage:
        print_log('Start training open world stage: {}'.format(stage), logger='current', level=logging.INFO)
        cfg.val_evaluator.open_world_stage = stage
        if hasattr(cfg.model, 'decode_head'):
            cfg.model.decode_head.open_world_stage = stage
        elif hasattr(cfg.model, 'pts_bbox_head'):
            cfg.model.pts_bbox_head.open_world_stage = stage
        else:
            raise ValueError("No available decode_head or pts_bbox_head.")
        cfg.train_dataloader.dataset.open_world_stage = stage
        cfg.val_dataloader.dataset.open_world_stage = stage

        if cfg.work_dir != ori_work_dir:
            # a later stage starts from the last checkpoint of the previous one
            max_epochs = cfg.train_cfg.max_epochs
            cfg.load_from = osp.join(copy.deepcopy(cfg.work_dir), f'epoch_{max_epochs}.pth')
        cfg.work_dir = osp.join(ori_work_dir, stage)
        if args.amp is True:
            optim_wrapper = cfg.optim_wrapper.type
            if optim_wrapper == 'AmpOptimWrapper':
                print_log(
                    'AMP training is already enabled in your config.',
                    logger='current',
                    level=logging.WARNING)
            else:
                assert optim_wrapper == 'OptimWrapper', (
                    '`--amp` is only supported when the optimizer wrapper type is '
                    f'`OptimWrapper` but got {optim_wrapper}.')
                cfg.optim_wrapper.type = 'AmpOptimWrapper'
                cfg.optim_wrapper.loss_scale = 'dynamic'
        if args.sync_bn != 'none':
            cfg.sync_bn = args.sync_bn
        if args.auto_scale_lr:
            if 'auto_scale_lr' in cfg and \
                    'enable' in cfg.auto_scale_lr and \
                    'base_batch_size' in cfg.auto_scale_lr:
                cfg.auto_scale_lr.enable = True
            else:
                raise RuntimeError('Can not find "auto_scale_lr" or '
                                '"auto_scale_lr.enable" or '
                                '"auto_scale_lr.base_batch_size" in your'
                                ' configuration file.')

        # resume is determined in this priority: resume from > auto_resume
        if args.resume == 'auto':
            cfg.resume = True
            cfg.load_from = None
        elif args.resume is not None:
            cfg.resume = True
            cfg.load_from = args.resume
        if 'runner_type' not in cfg:
            runner = Runner.from_cfg(cfg)
        else:
            runner = RUNNERS.build(cfg)
        runner.train()


if __name__ == '__main__':
    main()
