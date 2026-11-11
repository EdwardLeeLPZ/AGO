# ------------------------------------------------------------------------
# AGO
# Copyright (c) 2025 Peizheng Li. All Rights Reserved.
# ------------------------------------------------------------------------

_base_ = ['./_base_/ago_nuscenes.py']

open_world_stage = ['finetuning']
load_from = 'checkpoints/ago_nuscenes_open_world_pretraining.pth'

num_epoch = 12
param_scheduler = [
    dict(type='LinearLR', start_factor=1e-2, by_epoch=False, begin=0, end=500),
    dict(
        type='CosineAnnealingLR',
        begin=0,
        T_max=num_epoch,
        by_epoch=True,
        eta_min=1e-6,
        convert_to_iter_based=True)
]
train_cfg = dict(max_epochs=num_epoch)
