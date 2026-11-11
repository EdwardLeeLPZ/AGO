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
from mmengine.registry import RUNNERS
from mmengine.runner import Runner
from mmengine.dist import init_dist, get_dist_info
from mmengine.logging import print_log

from mmdet3d.utils import replace_ceph_backend


def parse_args():
    parser = argparse.ArgumentParser(
        description='Test AGO')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--tag', default='', help='the tag of this experiment')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
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
    # will pass the `--local-rank` parameter to `tools/test.py` instead
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

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = './work_dirs'

    tag = args.tag if args.tag != '' else time.strftime('%Y%m%d_%H%M%S', time.localtime())
    cfg.work_dir = osp.join(cfg.work_dir, osp.basename(args.config).split('.')[0], tag)

    # the logger depends on the dist info
    if args.launcher != 'none':
        init_dist(args.launcher)
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    cfg.load_from = args.checkpoint

    open_world_stage = cfg.open_world_stage
    ori_work_dir = copy.deepcopy(cfg.work_dir)
    for stage in open_world_stage:
        print_log('Start testing  open world stage: {}'.format(stage), logger='current', level=logging.INFO)
        cfg.test_evaluator.open_world_stage = stage
        if hasattr(cfg.model, 'decode_head'):
            cfg.model.decode_head.open_world_stage = stage
        elif hasattr(cfg.model, 'pts_bbox_head'):
            cfg.model.pts_bbox_head.open_world_stage = stage
        else:
            raise ValueError("No available decode_head or pts_bbox_head.")
        cfg.test_dataloader.dataset.open_world_stage = stage

        cfg.work_dir = osp.join(ori_work_dir, stage)
        if 'runner_type' not in cfg:
            runner = Runner.from_cfg(cfg)
        else:
            runner = RUNNERS.build(cfg)
        runner.test()


if __name__ == '__main__':
    main()
