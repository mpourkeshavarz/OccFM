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
        self.count_fps = False

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

    # @staticmethod
    def recover_training(self, weight_path):

        pl_sd = torch.load(weight_path, map_location="cpu", weights_only=True)
        self.load_state_dict(pl_sd['state_dict'])

        """
        old_name, weight = list(pl_sd['state_dict'].keys()), list(pl_sd['state_dict'].values())
        current_name = [x[0] for x in self.state_dict().items()]
        remap_list = []
        for name in current_name:
            if name.startswith('embedding'):
                new_name = name[10:]
            else:
                new_name = name

            if new_name in old_name:
                idx = old_name.index(new_name)
                remap_list.append(idx)
            else:
                raise ValueError('%s is not in remap_list' % name)

        new_state_dict = {}
        weights = [weight[x] for x in remap_list]
        for name, weights in zip(current_name, weights):
            new_state_dict[name] = weights

        pl_sd['state_dict'] = new_state_dict
        
        self.load_state_dict(new_state_dict)
        """
        return pl_sd


