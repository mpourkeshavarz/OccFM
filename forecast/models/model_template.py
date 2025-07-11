import os

import torch
import torch.nn as nn
import numpy as np

from .modules import embedding, encoders, quantization, decoders

from forecast.utils import common_utils

class ModelTemplate(nn.Module):
    def __init__(self, model_cfg, dataset=None):
        super().__init__()
        self.model_cfg = model_cfg.COMPRESSOR_CONFIG.MODEL
        self.dataset = dataset
        self.world_model_topology = ['embedding', 'encoder', 'quantization', 'transition_model', 'decoder', 'planner']
        self.compressor_topology = ['embedding', 'encoder', 'quantization', 'decoder']
        self.global_step = 0

    def update_global_step(self):
        self.global_step += 1

    def build_world_model(self):
        print()

    def build_compressor(self):
        model_info_dict = {
            'module_list': [],
            'component_configs': self.model_cfg
        }
        for module_name in self.compressor_topology:
            module, model_info_dict = getattr(self, 'build_%s' % module_name)(
                model_info_dict=model_info_dict
            )
            self.add_module(module_name, module)
        return model_info_dict['module_list']

    def build_embedding(self, model_info_dict):

        if model_info_dict['component_configs'].get('EMBEDDING', None) is None:
            return None, model_info_dict

        embed_config = model_info_dict['component_configs']['EMBEDDING']
        embed_module = embedding.__all__[embed_config.NAME](
            model_cfg = embed_config
        )
        model_info_dict['module_list'].append(embed_module)
        return embed_module, model_info_dict

    def build_encoder(self, model_info_dict):

        if model_info_dict['component_configs'].get('ENCODER', None) is None:
            return None, model_info_dict

        encoder_config = model_info_dict['component_configs']['ENCODER']
        encoder_module = encoders.__all__[encoder_config.NAME](
            **common_utils.lowercase_keys(encoder_config)
        )
        model_info_dict['module_list'].append(encoder_module)
        return encoder_module, model_info_dict

    def build_quantization(self, model_info_dict):

        if model_info_dict['component_configs'].get('QUANTIZATION', None) is None:
            return None, model_info_dict
        quant_config = model_info_dict['component_configs']['QUANTIZATION']
        quant_module = quantization.__all__[quant_config.NAME](
            **common_utils.lowercase_keys(quant_config)
        )
        model_info_dict['module_list'].append(quant_module)
        return quant_module, model_info_dict

    def build_transition_model(self, model_info_dict):
        if model_info_dict['component_configs'].get('TRANSITION', None) is None:
            return None, model_info_dict

    def build_decoder(self, model_info_dict):
        if model_info_dict['component_configs'].get('DECODER', None) is None:
            return None, model_info_dict

        decoder_config = model_info_dict['component_configs']['DECODER']
        decoder_module = decoders.__all__[decoder_config.NAME](
            **common_utils.lowercase_keys(decoder_config)
        )
        model_info_dict['module_list'].append(decoder_module)
        return decoder_module, model_info_dict

    def build_planner(self, model_info_dict):
        print()

    def forward(self, **kwargs):
        raise NotImplementedError

