# ------------------------------------------------------------------------
# AGO
# Copyright (c) 2025 Peizheng Li. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------

import nltk
from nltk.corpus import words
import random
import torch
import torch.distributed as dist
from torch import nn, Tensor
from typing import List, Optional, Union
from mmdet3d.models import Base3DSegmentor
from mmdet3d.registry import MODELS
from mmdet3d.structures import PointData
from mmdet3d.structures.det3d_data_sample import SampleList

from featup.featurizers.maskclip.clip import tokenize

from ..utils.ray_sample import ray_sample

FEATUP_HUB = 'mhamilton723/FeatUp:6b5a6c0e91f75e69194807128dcbc39c3084a30d'

class TPVFormer(Base3DSegmentor):

    def __init__(self,
                 data_preprocessor: Optional[Union[dict, nn.Module]] = None,
                 backbone=None,
                 neck=None,
                 encoder=None,
                 decode_head=None):

        super().__init__(data_preprocessor=data_preprocessor)

        self.backbone = MODELS.build(backbone)
        if neck is not None:
            self.neck = MODELS.build(neck)
        self.encoder = MODELS.build(encoder)
        self.decode_head = MODELS.build(decode_head)

    def extract_feat(self, img):
        """Extract features of images."""
        B, N, C, H, W = img.size()
        img = img.view(B * N, C, H, W)
        img_feats = self.backbone(img)

        if hasattr(self, 'neck'):
            img_feats = self.neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            _, C, H, W = img_feat.size()
            img_feats_reshaped.append(img_feat.view(B, N, C, H, W))
        return img_feats_reshaped

    def _forward(self, batch_inputs, batch_data_samples):
        raise NotImplementedError

    def loss(self, batch_inputs: dict,
             batch_data_samples: SampleList) -> SampleList:
        img_feats = self.extract_feat(batch_inputs['imgs'])
        queries = self.encoder(img_feats, batch_data_samples)
        losses = self.decode_head.loss(queries, batch_data_samples)
        return losses

    def predict(self, batch_inputs: dict,
                batch_data_samples: SampleList) -> SampleList:
        """Forward predict function."""
        img_feats = self.extract_feat(batch_inputs['imgs'])
        tpv_queries = self.encoder(img_feats, batch_data_samples)
        seg_logits = self.decode_head.predict(tpv_queries, batch_data_samples)
        seg_preds = [seg_logit.argmax(dim=1) for seg_logit in seg_logits]

        return self.postprocess_result(seg_preds, batch_data_samples)

    def aug_test(self, batch_inputs, batch_data_samples):
        pass

    def encode_decode(self, batch_inputs: dict,
                      batch_data_samples: SampleList) -> SampleList:
        pass


@MODELS.register_module()
class GroundingTPVFormer(TPVFormer):

    def __init__(self,
                 data_preprocessor: Optional[Union[dict, nn.Module]] = None,
                 backbone=None,
                 neck=None,
                 encoder=None,
                 decode_head=None,
                 num_noise_prompts=100,
                 ):
        super().__init__(data_preprocessor=data_preprocessor,
                         backbone=backbone,
                         neck=neck,
                         encoder=encoder,
                         decode_head=decode_head)


        self.caption, self.caption_list = self.decode_head.get_caption_infos()
        
        self._load_vision_encoders()
        
        # noise prompts sampled from the English dictionary
        if not dist.is_initialized() or dist.get_rank() == 0:
            nltk.download('words')
        if dist.is_initialized():
            dist.barrier()
        self.num_noise_prompts = num_noise_prompts
        self.word_list = words.words()
        self.noise_caption_list = []
    
    def _load_vision_encoders(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            torch.hub.load(FEATUP_HUB, 'maskclip', use_norm=False, trust_repo=True, skip_validation=True)
        if dist.is_initialized():
            dist.barrier()
        maskclip_module = torch.hub.load(FEATUP_HUB, 'maskclip', use_norm=False, trust_repo=True, skip_validation=True)
        maskclip_module.cuda().eval()
        self.vision_encoders = maskclip_module
    
    def generate_text_embeds(self, device='cuda', with_noise=True):
        position_id_list = [[i] for i in range(len(self.caption_list))]
        text = tokenize(self.caption_list).to(device)
        with torch.no_grad():
            caption_embeds = self.vision_encoders.model.model.encode_text(text)
        if not with_noise:
            return caption_embeds, position_id_list, None

        self.noise_caption_list = random.sample(self.word_list, self.num_noise_prompts)
        noise_text = tokenize(self.noise_caption_list).to(device)
        with torch.no_grad():
            noise_caption_embeds = self.vision_encoders.model.model.encode_text(noise_text)

        return caption_embeds, position_id_list, noise_caption_embeds
    
    def generate_image_embeds(self, images):
        img_shape = images.shape
        images = images.view(-1, *img_shape[-3:])

        maskclip_features = self.vision_encoders(images)
        image_embeds = maskclip_features.view(*img_shape[:-3], -1, *maskclip_features.shape[-2:]) # B, N, C, H, W
        
        return image_embeds
    
    def loss(self, batch_inputs: dict,
             batch_data_samples: SampleList) -> SampleList:
        img_feats = self.extract_feat(batch_inputs['imgs'])

        caption_embeds, position_id_list, noise_caption_embeds = self.generate_text_embeds(img_feats[0].device)
        text_embeds_dict = {'caption_embeds': caption_embeds,
                            'position_id_list': position_id_list,
                            'noise_caption_embeds': noise_caption_embeds}

        with torch.no_grad():
            image_embeds = self.generate_image_embeds(batch_inputs['raw_imgs'])
            batch_inputs['image_embeds'] = image_embeds

        queries = self.encoder(img_feats, batch_data_samples)
        losses = self.decode_head.loss(queries, batch_inputs, batch_data_samples, text_embeds_dict)
        return losses
    
    def predict(self, batch_inputs: dict,
                batch_data_samples: SampleList) -> SampleList:
        """Forward predict function."""
        img_feats = self.extract_feat(batch_inputs['imgs'])

        caption_embeds, position_id_list, _ = self.generate_text_embeds(img_feats[0].device, with_noise=False)
        text_embeds_dict = {'caption_embeds': caption_embeds, 'position_id_list': position_id_list}

        tpv_queries = self.encoder(img_feats, batch_data_samples)
        seg_logits = self.decode_head.predict(tpv_queries, batch_inputs, batch_data_samples, text_embeds_dict)
        return self.postprocess_result(seg_logits, batch_inputs, batch_data_samples)
    
    def postprocess_result(self, seg_logits_list: List[Tensor],
                           batch_inputs: dict, batch_data_samples: SampleList) -> SampleList:

        for i in range(len(seg_logits_list)):
            seg_logits = seg_logits_list[i]
            seg_pred = seg_logits.argmax(dim=-1)

            # get voxel coordinates that are visible to lidar
            voxel_coors = batch_data_samples[i].voxel_coors.long()
            voxel_coords_free = ray_sample(batch_data_samples[i].voxel_coors.long(), batch_data_samples[i].voxel_coords_center.long(), batch_data_samples[i].grid_shape)
            lidar_visible_voxel_coors = torch.unique(torch.cat([voxel_coors, voxel_coords_free], dim=0), dim=0).cpu().numpy()

            mask = batch_inputs['voxel_mask'][i].bool()

            lidar_visible_mask = torch.zeros_like(mask, dtype=torch.bool)
            lidar_visible_mask[lidar_visible_voxel_coors[:, 0], lidar_visible_voxel_coors[:, 1], lidar_visible_voxel_coors[:, 2]] = True

            if batch_data_samples[i].eval_ann_info:
                batch_data_samples[i].eval_ann_info['lidar_visible_mask'] = lidar_visible_mask[mask].cpu().numpy()
            else:
                batch_data_samples[i].eval_ann_info = {'lidar_visible_mask': lidar_visible_mask[mask].cpu().numpy()}
            
            batch_data_samples[i].set_data({
                'pts_seg_logits':
                PointData(**{'pts_seg_logits': seg_logits[mask, :]}),
                'pred_pts_seg':
                PointData(**{'pts_semantic_mask': seg_pred[mask]})
            })
        return batch_data_samples
