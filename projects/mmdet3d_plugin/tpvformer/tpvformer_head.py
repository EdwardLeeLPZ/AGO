# ------------------------------------------------------------------------
# AGO
# Copyright (c) 2025 Peizheng Li. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------

import copy
import torch
import torch.nn.functional as F
from torch import nn
from mmengine.model import BaseModule
from mmdet3d.registry import MODELS

from ..datasets.label_space import LABEL_PROMPT, OPEN_WORLD_CLASS, OCC3D_NUSCENES_MAPPING, NUSCENES2ID, NUSCENES2OPENWORLD, OPENWORLD2NUSCENES, OPENWORLD_COARSE2FINE, OPENWORLD_FINE2COARSE

class TPVFormerDecoder(BaseModule):

    def __init__(self,
                 tpv_h,
                 tpv_w,
                 tpv_z,
                 num_classes=20,
                 in_dims=64,
                 hidden_dims=128,
                 out_dims=None,
                 scale_h=2,
                 scale_w=2,
                 scale_z=2,
                 ignore_index=0,
                 lovasz_input='points',
                 ce_input='voxel'):
        super().__init__()
        self.tpv_h = tpv_h
        self.tpv_w = tpv_w
        self.tpv_z = tpv_z
        self.scale_h = scale_h
        self.scale_w = scale_w
        self.scale_z = scale_z

        out_dims = in_dims if out_dims is None else out_dims
        self.in_dims = in_dims
        self.decoder = nn.Sequential(
            nn.Linear(in_dims, hidden_dims), nn.Softplus(),
            nn.Linear(hidden_dims, out_dims))

        self.classifier = nn.Linear(out_dims, num_classes)
        self.ignore_index = ignore_index
        self.lovasz_input = lovasz_input
        self.ce_input = ce_input


@MODELS.register_module()
class GroundingTPVFormerDecoder(TPVFormerDecoder):

    def __init__(self,
                 tpv_h,
                 tpv_w,
                 tpv_z,
                 num_classes=18,
                 in_dims=64,
                 hidden_dims=128,
                 out_dims=None,
                 scale_h=2,
                 scale_w=2,
                 scale_z=2,
                 ignore_index=0,
                 loss_lovasz=None,
                 loss_ce=None,
                 loss_occ_ce=None,
                 loss_sim=None,
                 open_world_stage='all', # one of 'all', 'pretraining', 'finetuning'
                 training_type='grounding', # 'grounding' or 'image_align'
                 auxiliary_decoder=True): 
        super().__init__(
            tpv_h=tpv_h,
            tpv_w=tpv_w,
            tpv_z=tpv_z,
            num_classes=num_classes,
            in_dims=in_dims,
            hidden_dims=hidden_dims,
            out_dims=out_dims,
            scale_h=scale_h,
            scale_w=scale_w,
            scale_z=scale_z,
            ignore_index=ignore_index,)
        self.out_dims = out_dims
        self.num_classes = num_classes
        self.training_type = training_type

        self.open_world_stage = open_world_stage
        self.caption = None
        self.caption_list = []

        self.class_list_known = []
        self.class_list_unknown = []

        self.phrase2classid = {phrase:NUSCENES2ID[OCC3D_NUSCENES_MAPPING[phrase]] \
                               for phrase in OCC3D_NUSCENES_MAPPING.keys()}

        self.decoder = nn.Sequential(
            nn.Linear(in_dims, hidden_dims), nn.Softplus(),
            nn.Linear(hidden_dims, out_dims))
        
        self.free_embed = nn.Parameter(torch.ones(1, self.out_dims), requires_grad=True)

        if training_type == 'grounding':            
            self.loss_lovasz = MODELS.build(loss_lovasz) if loss_lovasz is not None else None
            self.loss_ce = MODELS.build(loss_ce) if loss_ce is not None else None
            self.loss_occ_ce = MODELS.build(loss_occ_ce) if loss_occ_ce is not None else None
            self.loss_sim = MODELS.build(loss_sim) if loss_sim is not None else None
            self.auxiliary_decoder = nn.Sequential(
                nn.Linear(out_dims, out_dims), nn.Softplus(),
                nn.Linear(out_dims, out_dims)) if auxiliary_decoder else None
        elif training_type == 'image_align':
            self.loss_lovasz = None
            self.loss_ce = None
            self.loss_sim = MODELS.build(loss_sim) if loss_sim is not None else None
            self.loss_occ_ce = MODELS.build(loss_occ_ce) if loss_occ_ce is not None else None
            self.auxiliary_decoder = nn.Sequential(
                nn.Linear(in_dims, out_dims), nn.Softplus(),
                nn.Linear(out_dims, out_dims)) if auxiliary_decoder else None
            self.occ_classifier = nn.Linear(out_dims, 2)
        else:
            raise ValueError(f'No match training type for {self.training_type}')

    def get_caption_infos(self):
        self.caption = LABEL_PROMPT
        self.caption_list = [phrase.strip() for phrase in self.caption.split(".") if phrase.strip()]

        if self.open_world_stage == 'all':
            self.class_list_known = [cls.strip() for cls in OPEN_WORLD_CLASS['all'].split(".") if cls.strip()]
        elif self.open_world_stage == 'pretraining':
            self.class_list_known = [cls.strip() for cls in OPEN_WORLD_CLASS['pretraining'].split(".") if cls.strip()]
            self.class_list_unknown = list(set([cls.strip() for cls in OPEN_WORLD_CLASS['finetuning'].split(".") if cls.strip()]) - set(self.class_list_known))
        elif self.open_world_stage == 'finetuning':
            self.class_list_known = [cls.strip() for cls in OPEN_WORLD_CLASS['finetuning'].split(".") if cls.strip()]
        else:
            raise ValueError(f'No match open world stage for {self.open_world_stage}')

        return self.caption, self.caption_list

    def get_target_embeddings(self, inputs, data_samples):
        batch_maskclip_image_embed, batch_visible = [], []
        for i, data_sample in enumerate(data_samples):
            maskclip_image_embed = inputs['image_embeds'][i] # num_cam, d_maskclip, h, w

            voxel2point_map = data_sample.voxel2point_map # num_voxels, num_points
            point_coors_cam = data_sample.point_coors_cam # num_points, num_cam, 2
            point_cam_mask = data_sample.point_cam_mask # num_points, num_cam
            point_coors_cam[~point_cam_mask] = 0
            point_coors_cam, point_cam_mask = point_coors_cam[voxel2point_map], point_cam_mask[voxel2point_map]

            image_shape = data_sample.batch_input_shape
            cam_x = point_coors_cam[..., 1] / image_shape[-2] * maskclip_image_embed.shape[-2]
            cam_y = point_coors_cam[..., 0] / image_shape[-1] * maskclip_image_embed.shape[-1]
            num_voxels, num_cam = point_cam_mask.shape
            cam_indices = torch.arange(num_cam, device=point_coors_cam.device).unsqueeze(0).expand(num_voxels, -1)

            maskclip_image_embed = maskclip_image_embed[cam_indices, :, cam_x.long(), cam_y.long()] # num_voxels, num_cam, d_maskclip
            # average over the cameras that observe each voxel
            maskclip_image_embed[~point_cam_mask] = 0
            maskclip_image_embed = maskclip_image_embed.sum(dim=1) / (point_cam_mask.sum(dim=1, keepdim=True) + 1e-6)

            batch_maskclip_image_embed.append(maskclip_image_embed)
            batch_visible.append(point_cam_mask.any(dim=1))

        return torch.cat(batch_maskclip_image_embed, dim=0), torch.cat(batch_visible, dim=0) # bs*num_voxels, d_maskclip; bs*num_voxels
    
    def loss(self, tpv_list, batch_inputs, batch_data_samples, text_embeds_dict):
        ################# Embedding Generation #################

        caption_embeds = text_embeds_dict['caption_embeds']
        position_id_list = text_embeds_dict['position_id_list']
        noise_caption_embeds = text_embeds_dict['noise_caption_embeds']

        ################# TPVFormer Decoder #################
        tpv_hw, tpv_zh, tpv_wz = tpv_list
        bs, _, c = tpv_hw.shape
        tpv_hw = tpv_hw.permute(0, 2, 1).reshape(bs, c, self.tpv_h, self.tpv_w)
        tpv_zh = tpv_zh.permute(0, 2, 1).reshape(bs, c, self.tpv_z, self.tpv_h)
        tpv_wz = tpv_wz.permute(0, 2, 1).reshape(bs, c, self.tpv_w, self.tpv_z)

        if self.scale_h != 1 or self.scale_w != 1:
            tpv_hw = F.interpolate(
                tpv_hw,
                size=(self.tpv_h * self.scale_h, self.tpv_w * self.scale_w),
                mode='bilinear')
        if self.scale_z != 1 or self.scale_h != 1:
            tpv_zh = F.interpolate(
                tpv_zh,
                size=(self.tpv_z * self.scale_z, self.tpv_h * self.scale_h),
                mode='bilinear')
        if self.scale_w != 1 or self.scale_z != 1:
            tpv_wz = F.interpolate(
                tpv_wz,
                size=(self.tpv_w * self.scale_w, self.tpv_z * self.scale_z),
                mode='bilinear')
            
        tpv_hw_vox = tpv_hw.unsqueeze(-1).permute(0, 1, 3, 2, 4).expand(
            -1, -1, -1, -1, self.scale_z * self.tpv_z)
        tpv_zh_vox = tpv_zh.unsqueeze(-1).permute(0, 1, 4, 3, 2).expand(
            -1, -1, self.scale_w * self.tpv_w, -1, -1)
        tpv_wz_vox = tpv_wz.unsqueeze(-1).permute(0, 1, 2, 4, 3).expand(
            -1, -1, -1, self.scale_h * self.tpv_h, -1)
        fused_vox = tpv_hw_vox + tpv_zh_vox + tpv_wz_vox # bs, in_dims, h, w, z
        vox_logits = self.decoder(fused_vox.permute(0, 2, 3, 4, 1)) # bs, h, w, z, out_dims
        if self.auxiliary_decoder is not None:
            if self.training_type == 'grounding':
                auxiliary_vox_logits = self.auxiliary_decoder(vox_logits) # bs, h, w, z, out_dims
            elif self.training_type == 'image_align':
                auxiliary_vox_logits = self.auxiliary_decoder(fused_vox.permute(0, 2, 3, 4, 1)) # bs, h, w, z, out_dims
        else:
            auxiliary_vox_logits = vox_logits

        ################# Label Generation #################        
        sem_gt_labels = torch.stack([batch_inputs['voxel_semantics'][i].long() for i in range(bs)], dim=0) # bs, h, w, z
        sem_masks = torch.stack([batch_inputs['voxel_mask'][i].bool() for i in range(bs)], dim=0) # bs, h, w, z
        subclass_pseudo_labels = torch.stack([batch_inputs['pseudo_voxel_semantics'][i].long() for i in range(bs)], dim=0) # bs, h, w, z

        # map prompt-level pseudo labels to class labels
        sem_pseudo_labels = subclass_pseudo_labels.new_full(subclass_pseudo_labels.shape, self.ignore_index)
        for phraseid, phrase in enumerate(self.caption_list):
            label = torch.tensor(self.phrase2classid[phrase], device=subclass_pseudo_labels.device)
            sem_pseudo_labels[subclass_pseudo_labels == phraseid] = label
        sem_pseudo_labels[subclass_pseudo_labels == len(self.caption_list)] = self.num_classes - 1
        
        target_point_embeds, target_point_visible = self.get_target_embeddings(batch_inputs, batch_data_samples)
        voxel_coords = torch.cat([
            torch.cat([coords.new_full((len(coords), 1), i), coords], dim=1)
            for i, coords in enumerate(data_sample.voxel_coors.long() for data_sample in batch_data_samples)]) # bs*num_voxels, 4

        target_vox_embeds = target_point_embeds.new_zeros([*sem_gt_labels.shape, target_point_embeds.shape[-1]]) # bs, h, w, z, embedding_dims
        target_vox_embeds[voxel_coords.unbind(dim=1)] = target_point_embeds
        target_vox_visible = torch.zeros_like(sem_masks)
        target_vox_visible[voxel_coords.unbind(dim=1)] = target_point_visible

        if self.loss_occ_ce is not None:
            occ_labels = sem_pseudo_labels.new_full(sem_pseudo_labels.shape, self.ignore_index)
            occ_labels[sem_pseudo_labels == self.num_classes-1] = 0 # free
            occ_labels[(sem_pseudo_labels != (self.num_classes - 1)) & (sem_pseudo_labels != self.ignore_index)] = 1  # occupied

            sem_pseudo_labels[sem_pseudo_labels == self.num_classes-1] = self.ignore_index

        if self.open_world_stage == 'all':
            sem_labels = sem_pseudo_labels
        elif self.open_world_stage in ['pretraining', 'finetuning']:
            sem_labels = copy.deepcopy(sem_pseudo_labels)
            for nusc_id, ow_id in NUSCENES2OPENWORLD.items():
                sem_labels[sem_pseudo_labels == nusc_id] = ow_id
            
            if self.open_world_stage == 'pretraining':
                for fine_id, coarse_id in OPENWORLD_FINE2COARSE.items():
                    sem_labels[sem_labels == fine_id] = coarse_id
                if self.loss_occ_ce is not None:
                    occ_labels[(occ_labels == 1) & (sem_labels == self.ignore_index)] = self.ignore_index
        else:
            raise ValueError(f'No match open world stage for {self.open_world_stage}')

        ################# Grounding Layer #################
        if self.training_type == 'grounding':
            phrase_logits = {self.phrase2classid[phrase]: [] for phrase in self.caption_list}
            for i, phrase in enumerate(self.caption_list):
                position_ids = position_id_list[i]
                phrase_logits[self.phrase2classid[phrase]].append(
                    (vox_logits @ caption_embeds[position_ids].float().T).mean(dim=-1).unsqueeze(-1)) # bs, h, w, z, 1
            
            phrase_logits[self.num_classes-1] = [vox_logits @ self.free_embed.T.to(vox_logits.device)] # bs, h, w, z, 1

            sem_logits = vox_logits.new_zeros(vox_logits.shape[:-1] + (self.num_classes,)) # bs, h, w, z, num_classes
            for phrase_id, phrase_logit in phrase_logits.items():
                sem_logits[..., phrase_id] = torch.cat(phrase_logit, dim=-1).max(dim=-1)[0] # bs, h, w, z
            
            if self.open_world_stage in ['pretraining', 'finetuning']:
                reorder_n2o = [OPENWORLD2NUSCENES[ow_id] for ow_id in range(len(OPENWORLD2NUSCENES))] + [self.num_classes - 1]
                sem_logits = sem_logits[..., reorder_n2o]

                # pretraining: the coarse known classes only; finetuning: all classes except the coarse ones
                if self.open_world_stage == 'pretraining':
                    sem_logits[..., 3] = torch.max(sem_logits[..., [5, 6, 7, 8, 9]], dim=-1)[0]
                    sem_logits[..., 4] = torch.max(sem_logits[..., [10, 11]], dim=-1)[0]
                    excluded_ids = list(OPENWORLD_FINE2COARSE)
                else:
                    excluded_ids = list(OPENWORLD_COARSE2FINE)
                sem_logits[..., excluded_ids] = float('-inf')

            sem_logits = torch.cat([sem_logits, vox_logits @ noise_caption_embeds.float().T], dim=-1)

            occ_logits = torch.cat([
                sem_logits[..., self.num_classes-1].unsqueeze(-1), # free
                sem_logits[..., :self.num_classes-1].max(dim=-1)[0].unsqueeze(-1), # occupied                
            ], dim=-1) # bs, h, w, z, 2
        elif self.training_type == 'image_align':
            occ_logits = self.occ_classifier(auxiliary_vox_logits) # bs, h, w, z, 2
        else:              
            raise ValueError(f'No match training type for {self.training_type}')
        
        ################# Loss Calculation #################
        loss = dict()
        sem_labels[~sem_masks] = self.ignore_index

        if self.loss_sim is not None:
            align_mask = (sem_labels != self.ignore_index) & (sem_labels != self.num_classes - 1) & target_vox_visible
            if self.training_type == 'grounding':
                loss['loss_sim'] = self.loss_sim(auxiliary_vox_logits[align_mask], target_vox_embeds[align_mask])
            elif self.training_type == 'image_align':
                loss['loss_sim'] = self.loss_sim(vox_logits[align_mask], target_vox_embeds[align_mask])
            else:
                raise ValueError(f'No match training type for {self.training_type}')

        if self.loss_ce is not None:
            loss['loss_ce'] = self.loss_ce(
                sem_logits[sem_labels != self.ignore_index].reshape(-1, sem_logits.shape[-1]), 
                sem_labels[sem_labels != self.ignore_index].reshape(-1),
            )
        
        if self.loss_lovasz is not None:
            loss['loss_lovasz'] = self.loss_lovasz(
                sem_logits[sem_labels != self.ignore_index].reshape(-1, sem_logits.shape[-1]),
                sem_labels[sem_labels != self.ignore_index].reshape(-1)
            )

        if self.loss_occ_ce is not None:
            occ_labels[~sem_masks] = self.ignore_index
            loss['loss_ce_occ'] = self.loss_occ_ce(
                occ_logits[occ_labels != self.ignore_index].reshape(-1, occ_logits.shape[-1]), 
                occ_labels[occ_labels != self.ignore_index].reshape(-1)
            )

        return loss
    
    def predict(self, tpv_list, batch_inputs, batch_data_samples, text_embeds_dict):
        ################# Embedding Generation #################

        caption_embeds = text_embeds_dict['caption_embeds']
        position_id_list = text_embeds_dict['position_id_list']

        ################# TPVFormer Decoder #################
        tpv_hw, tpv_zh, tpv_wz = tpv_list
        bs, _, c = tpv_hw.shape
        tpv_hw = tpv_hw.permute(0, 2, 1).reshape(bs, c, self.tpv_h, self.tpv_w)
        tpv_zh = tpv_zh.permute(0, 2, 1).reshape(bs, c, self.tpv_z, self.tpv_h)
        tpv_wz = tpv_wz.permute(0, 2, 1).reshape(bs, c, self.tpv_w, self.tpv_z)

        if self.scale_h != 1 or self.scale_w != 1:
            tpv_hw = F.interpolate(
                tpv_hw,
                size=(self.tpv_h * self.scale_h, self.tpv_w * self.scale_w),
                mode='bilinear')
        if self.scale_z != 1 or self.scale_h != 1:
            tpv_zh = F.interpolate(
                tpv_zh,
                size=(self.tpv_z * self.scale_z, self.tpv_h * self.scale_h),
                mode='bilinear')
        if self.scale_w != 1 or self.scale_z != 1:
            tpv_wz = F.interpolate(
                tpv_wz,
                size=(self.tpv_w * self.scale_w, self.tpv_z * self.scale_z),
                mode='bilinear')
            
        tpv_hw_vox = tpv_hw.unsqueeze(-1).permute(0, 1, 3, 2, 4).expand(
            -1, -1, -1, -1, self.scale_z * self.tpv_z)
        tpv_zh_vox = tpv_zh.unsqueeze(-1).permute(0, 1, 4, 3, 2).expand(
            -1, -1, self.scale_w * self.tpv_w, -1, -1)
        tpv_wz_vox = tpv_wz.unsqueeze(-1).permute(0, 1, 2, 4, 3).expand(
            -1, -1, -1, self.scale_h * self.tpv_h, -1)
        fused_vox = tpv_hw_vox + tpv_zh_vox + tpv_wz_vox # bs, in_dims, h, w, z
        vox_logits = self.decoder(fused_vox.permute(0, 2, 3, 4, 1)) # bs, h, w, z, out_dims
        if self.auxiliary_decoder is not None:
            if self.training_type == 'grounding':
                auxiliary_vox_logits = self.auxiliary_decoder(vox_logits) # bs, h, w, z, out_dims
            elif self.training_type == 'image_align':
                auxiliary_vox_logits = self.auxiliary_decoder(fused_vox.permute(0, 2, 3, 4, 1)) # bs, h, w, z, out_dims
        else:
            auxiliary_vox_logits = vox_logits

        ################# Grounding Layer #################
        if self.training_type in ['grounding', 'image_align']:
            phrase_logits = {self.phrase2classid[phrase]: [] for phrase in self.caption_list}
            for i, phrase in enumerate(self.caption_list):
                position_ids = position_id_list[i]
                phrase_logits[self.phrase2classid[phrase]].append(
                    (vox_logits @ caption_embeds[position_ids].float().T).mean(dim=-1).unsqueeze(-1))
            
            caption_logits = vox_logits.new_zeros(*vox_logits.shape[:-1], self.num_classes-1) # bs, h, w, z, num_classes - 1
            for phrase_id, phrase_logit in phrase_logits.items():
                if self.open_world_stage == 'all' or phrase_id not in [0, 12]:
                    caption_logits[..., phrase_id] = torch.cat(phrase_logit, dim=-1).max(dim=-1)[0]

            if self.training_type == 'grounding' and self.open_world_stage == 'pretraining' and self.auxiliary_decoder is not None:
                aux_phrase_logits = {self.phrase2classid[phrase]: [] for phrase in self.caption_list}
                for i, phrase in enumerate(self.caption_list):
                    position_ids = position_id_list[i]
                    aux_phrase_logits[self.phrase2classid[phrase]].append(
                        (auxiliary_vox_logits @ caption_embeds[position_ids].float().T).mean(dim=-1).unsqueeze(-1))
                
                aux_caption_logits = auxiliary_vox_logits.new_zeros(*vox_logits.shape[:-1], self.num_classes-1)  # bs, h, w, z, num_classes - 1
                for phrase_id, phrase_logit in aux_phrase_logits.items():
                    if phrase_id not in [0, 12]:
                        aux_caption_logits[..., phrase_id] = torch.cat(phrase_logit, dim=-1).max(dim=-1)[0]

                # open-world identifier: keep the output with lower entropy
                caption_prob = nn.Softmax(dim=-1)(caption_logits)
                aux_caption_prob = nn.Softmax(dim=-1)(aux_caption_logits)
                caption_entropy = -torch.sum(caption_prob * torch.log(caption_prob + 1e-8), dim=-1)
                aux_caption_entropy = -torch.sum(aux_caption_prob * torch.log(aux_caption_prob + 1e-8), dim=-1)
                caption_logits[aux_caption_entropy <= caption_entropy] = aux_caption_logits[aux_caption_entropy <= caption_entropy]

            if self.training_type == 'grounding':
                logits = torch.cat([caption_logits, vox_logits @ self.free_embed.T.to(vox_logits.device)], dim=-1)
            elif self.training_type != 'grounding':
                logits = torch.cat([caption_logits, caption_logits.new_zeros(*caption_logits.shape[:-1], 1)], dim=-1)

                logit = torch.tensor(self.num_classes-1, device=logits.device)
                logit = F.one_hot(logit, num_classes=self.num_classes).float()
                occ_logits = self.occ_classifier(auxiliary_vox_logits)
                occ_preds = occ_logits.argmax(dim=-1) # bs, h, w, z
                logits[occ_preds == 0] = logit
        else:
            raise ValueError(f'No match training type for {self.training_type}')

        return logits
