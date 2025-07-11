import torch
import torch.nn as nn
import numpy as np
from forecast.models.model_template import ModelTemplate
from einops import rearrange
from forecast.utils.loss_utils import lovasz_softmax

import torch.nn.functional as F

class OccFmVAE(ModelTemplate):
    def __init__(self, model_cfg, loss_cfg, **kwargs):
        super().__init__(model_cfg)
        self.input_height = self.model_cfg.EMBEDDING.HEIGHT_NUM
        self.cate = self.model_cfg.EMBEDDING.FEAT_DIM
        self.loss_weight = loss_cfg

        self.module_list = self.build_compressor()


    def forward(self, batch_dict, **kwargs):

        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        decoded_map = batch_dict['decoded_map']
        template = self.embedding.class_embeds.weight.T.unsqueeze(0).detach()
        decoded_map = rearrange(decoded_map, 'b (d c) h w -> b h w d c', d=self.input_height, c=self.cate)
        similarity = torch.matmul(decoded_map, template)
        batch_dict['similarity'] = similarity

        loss, tb_dict, disp_dict = self.get_training_loss(batch_dict)

        return loss, tb_dict, disp_dict

    def get_training_loss(self, batch_dict):

        pred_occ, gt_occ = batch_dict['similarity'], batch_dict['semantic_occ']
        gt_occ = gt_occ.long()
        disp_dict = {}

        rec_loss = F.cross_entropy(pred_occ.permute(0, 4, 1, 2, 3), gt_occ, ignore_index=-100)
        weighted_rec_loss = self.loss_weight['RECON_LOSS_WEIGHT'] * rec_loss
        disp_dict['weighted_rec_loss'] = weighted_rec_loss

        # miou
        pred_occ = pred_occ.permute(0, 4, 1, 2, 3).softmax(dim=1)
        loss = lovasz_softmax(pred_occ, gt_occ)
        weighted_lova_loss = self.loss_weight['LOVASZ_LOSS_WEIGHT'] * loss
        disp_dict['weighted_lova_loss'] = weighted_lova_loss

        # KL divergence
        kl_loss = self.quantization.get_loss()
        weighted_kl_loss = self.loss_weight['KL_DIVERGENCE_WEIGHT'] * kl_loss
        disp_dict['weighted_kl_loss'] = weighted_kl_loss

        loss = weighted_rec_loss + weighted_lova_loss + weighted_kl_loss
        disp_dict['loss'] = loss

        return disp_dict, disp_dict, disp_dict