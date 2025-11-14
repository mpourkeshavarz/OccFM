import torch
import torch.nn as nn
import time
from forecast.models.model_template import ModelTemplate
from einops import rearrange

from forecast.utils.common_utils import cuda_timer
from forecast.utils.loss_utils import lovasz_softmax

import torch.nn.functional as F

class OccFmVAE(ModelTemplate):
    def __init__(self, model_cfg, loss_cfg, **kwargs):
        super().__init__(model_cfg.COMPRESSOR_CONFIG)
        self.input_height = self.model_cfg.EMBEDDING.HEIGHT_NUM
        self.cate = self.model_cfg.EMBEDDING.FEAT_DIM
        self.loss_weight = loss_cfg

        self.module_list = self.build_model(self.compressor_topology, skip_list=[])

    @cuda_timer
    def nn_forward(self, batch_dict):

        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        decoded_map = batch_dict['decoded_map']
        template = self.embedding.class_embeds.weight.T.unsqueeze(0).detach()
        decoded_map = rearrange(decoded_map, 'b (d c) h w -> b h w d c', d=self.input_height, c=self.cate)
        similarity = torch.matmul(decoded_map, template)
        batch_dict['similarity'] = similarity

        return batch_dict

    def forward(self, batch_dict, **kwargs):
        eval_fps = batch_dict.get('eval_fps', False)

        batch_dict, forward_time = self.nn_forward(batch_dict)
        loss, tb_dict, disp_dict = self.get_training_loss(batch_dict)

        if eval_fps and forward_time is not None:
            disp_dict['time'] = forward_time

        return loss, tb_dict, disp_dict

    def get_training_loss(self, batch_dict):

        pred_occ, gt_occ = batch_dict['similarity'], batch_dict['semantic_occ']

        assert gt_occ.shape[1] == 1, "video input not supported now"
        gt_occ = gt_occ.long()[:, 0] if len(gt_occ.shape) == 5 else gt_occ.long()

        # Safety check: clamp ground truth labels to valid range [0, num_classes-1]
        # pred_occ has shape [B, H, W, D, NUM_CATE], so num_classes = pred_occ.shape[-1]
        num_classes = pred_occ.shape[-1]
        gt_occ = torch.clamp(gt_occ, 0, num_classes - 1)

        tb_dict, disp_dict = {}, {}

        # Permute pred_occ to [B, C, H, W, D] for cross-entropy
        pred_occ_permuted = pred_occ.permute(0, 4, 1, 2, 3)
        
        # Check for NaN or Inf in predictions
        if torch.isnan(pred_occ_permuted).any() or torch.isinf(pred_occ_permuted).any():
            print(f"Warning: NaN or Inf detected in predictions! NaN: {torch.isnan(pred_occ_permuted).sum()}, Inf: {torch.isinf(pred_occ_permuted).sum()}")
            pred_occ_permuted = torch.nan_to_num(pred_occ_permuted, nan=0.0, posinf=1.0, neginf=-1.0)

        rec_loss = F.cross_entropy(pred_occ_permuted, gt_occ, ignore_index=-100)
        
        # Check for NaN in loss
        if torch.isnan(rec_loss):
            print(f"Warning: NaN in reconstruction loss! pred_occ shape: {pred_occ.shape}, gt_occ shape: {gt_occ.shape}, gt_occ range: [{gt_occ.min()}, {gt_occ.max()}]")
            rec_loss = torch.tensor(0.0, device=rec_loss.device, requires_grad=True)
        
        weighted_rec_loss = self.loss_weight['RECON_LOSS_WEIGHT'] * rec_loss
        tb_dict['weighted_rec_loss'] = weighted_rec_loss

        # miou - use permuted version for consistency
        pred_occ_softmax = pred_occ_permuted.softmax(dim=1)
        loss = lovasz_softmax(pred_occ_softmax, gt_occ)
        
        # Check for NaN in lovasz loss
        if torch.isnan(loss):
            print(f"Warning: NaN in Lovasz loss!")
            loss = torch.tensor(0.0, device=loss.device, requires_grad=True)
        
        weighted_lova_loss = self.loss_weight['LOVASZ_LOSS_WEIGHT'] * loss
        tb_dict['weighted_lova_loss'] = weighted_lova_loss

        # KL divergence
        kl_loss = self.quantization.get_loss()
        
        # Check for NaN in KL loss
        if torch.isnan(kl_loss):
            print(f"Warning: NaN in KL loss!")
            kl_loss = torch.tensor(0.0, device=kl_loss.device, requires_grad=True)
        
        weighted_kl_loss = self.loss_weight['KL_DIVERGENCE_WEIGHT'] * kl_loss
        tb_dict['weighted_kl_loss'] = weighted_kl_loss

        loss = weighted_rec_loss + weighted_lova_loss + weighted_kl_loss
        
        # Final check for NaN in total loss
        if torch.isnan(loss):
            print(f"Warning: NaN in total loss! rec: {weighted_rec_loss.item()}, lovasz: {weighted_lova_loss.item()}, kl: {weighted_kl_loss.item()}")
            loss = torch.tensor(0.0, device=loss.device, requires_grad=True)
        
        tb_dict['loss'] = loss

        disp_dict['pred_occ'] = torch.argmax(pred_occ_softmax, 1).unsqueeze(1)
        disp_dict['gt_occ'] = gt_occ.unsqueeze(1)
        disp_dict['gt_path'] = batch_dict['paths']
        if 'trajectory' in batch_dict:
            disp_dict['trajectory'] = batch_dict['trajectory']

        if batch_dict['mu'] is not None:
            disp_dict['mu'] = batch_dict['mu']
            disp_dict['sigma'] = batch_dict['sigma']

        if self.quantization.latent_cache:
            disp_dict['x_sampled'] = batch_dict['sampled_features']

        return tb_dict, tb_dict, disp_dict