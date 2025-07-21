import os
from collections import OrderedDict

import torch
import torch.nn as nn
import numpy as np

from .modules import embedding, encoders, quantization, decoders, transition_models, planners

from forecast.utils import common_utils

class ModelTemplate(nn.Module):
    def __init__(self, model_cfg, dataset=None, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg.MODEL
        self.dataset = dataset
        self.world_model_topology = ['embedding', 'encoder', 'quantization', 'transition_model', 'decoder', 'planner']
        self.compressor_topology = ['embedding', 'encoder', 'quantization', 'decoder']
        self.global_step = 0

    def update_global_step(self):
        self.global_step += 1

    def build_model(self, topology, skip_list=None):
        model_info_dict = {
            'module_list': [],
            'component_configs': self.model_cfg,
        }
        for module_name in topology:
            module, model_info_dict = getattr(self, 'build_%s' % module_name)(
                model_info_dict=model_info_dict, skip= module_name in skip_list
            )
            self.add_module(module_name, module)
        return model_info_dict['module_list']

    def build_embedding(self, model_info_dict, skip=False):

        if model_info_dict['component_configs'].get('EMBEDDING', None) is None:
            return None, model_info_dict

        embed_config = model_info_dict['component_configs']['EMBEDDING']
        embed_config['skip'] = skip
        embed_module = embedding.__all__[embed_config.NAME](
            model_cfg = embed_config
        )
        model_info_dict['module_list'].append(embed_module)
        return embed_module, model_info_dict

    def build_encoder(self, model_info_dict, skip=False):

        if model_info_dict['component_configs'].get('ENCODER', None) is None:
            return None, model_info_dict

        encoder_config = model_info_dict['component_configs']['ENCODER']
        encoder_config['skip'] = skip
        encoder_module = encoders.__all__[encoder_config.NAME](
            **common_utils.lowercase_keys(encoder_config)
        )
        model_info_dict['module_list'].append(encoder_module)
        return encoder_module, model_info_dict

    def build_quantization(self, model_info_dict, skip=False):

        if model_info_dict['component_configs'].get('QUANTIZATION', None) is None:
            return None, model_info_dict
        quant_config = model_info_dict['component_configs']['QUANTIZATION']
        quant_config['skip'] = skip
        quant_module = quantization.__all__[quant_config.NAME](
            **common_utils.lowercase_keys(quant_config)
        )
        model_info_dict['module_list'].append(quant_module)
        return quant_module, model_info_dict

    def build_transition_model(self, model_info_dict, skip=False):
        if model_info_dict['component_configs'].get('TRANSITION_MODEL', None) is None:
            return None, model_info_dict

        transition_config = model_info_dict['component_configs']['TRANSITION_MODEL']
        transition_config['skip'] = skip
        transition_module = transition_models.__all__[transition_config.NAME](
            **common_utils.lowercase_keys(transition_config)
        )
        model_info_dict['module_list'].append(transition_module)
        return transition_module, model_info_dict

    def build_decoder(self, model_info_dict, skip=False):
        if model_info_dict['component_configs'].get('DECODER', None) is None:
            return None, model_info_dict

        decoder_config = model_info_dict['component_configs']['DECODER']
        decoder_config['skip'] = skip
        decoder_module = decoders.__all__[decoder_config.NAME](
            **common_utils.lowercase_keys(decoder_config)
        )
        model_info_dict['module_list'].append(decoder_module)
        return decoder_module, model_info_dict

    def build_planner(self, model_info_dict, skip=False):

        if model_info_dict['component_configs'].get('PLANNER_CONFIG', None) is None:
            return None, model_info_dict

        planner_config = model_info_dict['component_configs']['PLANNER_CONFIG']
        planner_config['skip'] = skip
        planner_module = planners.__all__[planner_config.NAME](
            **common_utils.lowercase_keys(planner_config)
        )
        model_info_dict['module_list'].append(planner_module)
        return planner_module, model_info_dict

    def forward(self, **kwargs):
        raise NotImplementedError

    # @staticmethod
    def recover_training(self, weight_path):

        pl_sd = torch.load(weight_path, map_location="cpu", weights_only=True)

        """
        all_parts =  [name for name in self.state_dict()]
        #mod = self.get_submodule('transition_model.downs.0.4')
        #print()
        
        transition_parameters = {}
        params = pl_sd['state_dict']
        for name, param in params.items():
            if name.split('.')[0] == 'transition_model':
                transition_parameters[name] = param

        vae_weight = torch.load('./logs/occfm_vae/100ep_3docc_sem_voxel/ckpt/epoch=000100.ckpt', map_location="cpu", weights_only=True)
        vae_weight = vae_weight['state_dict']

        ordered_dict = OrderedDict()
        for name in all_parts:
            if name in list(transition_parameters.keys()):
                ordered_dict[name] = transition_parameters[name]
            elif name in list(vae_weight.keys()):
                ordered_dict[name] = vae_weight[name]
            else:
                print(name)

        pl_sd['state_dict'] = ordered_dict
        torch.save(pl_sd, './epoch=000197.pt')
        """

        key = list(pl_sd['state_dict'].keys())[0]
        new_param_dict = common_utils.remove_module_prefix_from_ddp(pl_sd['state_dict']) \
            if key.startswith('module') else pl_sd['state_dict']
        self.load_state_dict(new_param_dict)

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

    def recover_compressor(self, weight_path):
        compressor_weight = torch.load(weight_path, map_location="cpu", weights_only=True)
        key = list(compressor_weight['state_dict'].keys())[0]
        new_param_dict = common_utils.remove_module_prefix_from_ddp(compressor_weight['state_dict']) \
            if key.startswith('module') else compressor_weight['state_dict']
        self.load_state_dict(new_param_dict, strict=False)
        return compressor_weight


