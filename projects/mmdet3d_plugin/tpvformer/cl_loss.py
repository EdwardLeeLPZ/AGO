# ------------------------------------------------------------------------
# AGO
# Copyright (c) 2025 Peizheng Li. All Rights Reserved.
# ------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet3d.registry import MODELS

@MODELS.register_module()
class CosineSimLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(CosineSimLoss, self).__init__()
        self.reduction = reduction
        self.criterion = nn.CosineSimilarity()

    def forward(self, pred, target):
        if pred.dim() == target.dim():
            # pred and target: (N, C)
            cos_sim = self.criterion(F.normalize(pred, p=2, dim=-1), F.normalize(target, p=2, dim=-1)) # (N,)
            loss = 1.0 - cos_sim
        elif pred.dim() + 1 == target.dim():
            # pred: (N, C), target: (N, K, C)
            cos_sim = [self.criterion(F.normalize(pred, p=2, dim=-1), F.normalize(target[:, i, :], p=2, dim=-1)) for i in range(target.shape[1])] # K of (N,)
            loss = 1.0 - torch.stack(cos_sim, dim=-1).mean(dim=-1) # (N,)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
