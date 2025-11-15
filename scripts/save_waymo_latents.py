#!/usr/bin/env python3
"""
Script to load VAE and save latents for Waymo dataset.
Saves latents in the same format as trajectories (pickle file with gt_path, gt_trajs, x_sampled).

Usage:
    python scripts/save_waymo_latents.py \
        --cfg_file tools/cfgs/occfm_vae_occ_only_waymo.yaml \
        --vae_ckpt logs/occfm_vae_occ_only_waymo/train_new/ckpt/epoch=000002.ckpt \
        --output_dir ./data/waymo_latent_vae/x16 \
        --batch_size 16 \
        --num_workers 4

The script will create two pickle files:
    - waymo_latent_train.pkl (for training set)
    - waymo_latent_val.pkl (for validation set)

Each pickle file contains a list of dictionaries with:
    - 'x_sampled': VAE latents (numpy array)
    - 'gt_path': Path to the .npz file
    - 'gt_trajs': Trajectory (numpy array)

To use these latents in OccFM training, add to your config:
    PICKLE_PATH:
        train: './data/waymo_latent_vae/x16/waymo_latent_train.pkl'
        test: './data/waymo_latent_vae/x16/waymo_latent_val.pkl'
"""

import argparse
import os
import pickle
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from easydict import EasyDict

from forecast.config import cfg_from_yaml_file
from forecast.datasets import build_dataloader
from forecast.models import build_network
from forecast.utils import common_utils


def parse_config():
    parser = argparse.ArgumentParser(description='Save VAE latents for Waymo dataset')
    parser.add_argument('--cfg_file', type=str, required=True, help='VAE config file (e.g., tools/cfgs/occfm_vae_occ_only_waymo.yaml)')
    parser.add_argument('--vae_ckpt', type=str, required=True, help='Path to VAE checkpoint file')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory to save pickle files')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for processing')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for dataloader')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
    
    args = parser.parse_args()
    cfg = EasyDict()
    cfg.ROOT_DIR = (Path(__file__).resolve().parent / '../').resolve()
    
    return args, cfg


def encode_semantic_occ(model, semantic_occ_tensor, device):
    """
    Encode semantic occupancy through VAE to get latents.
    
    Args:
        model: VAE model (OccFmVAE)
        semantic_occ_tensor: Tensor of shape [B, H, W, D] or [B, T, H, W, D]
        device: Device to run on
    
    Returns:
        sampled_features: Latents of shape [B, C, H', W'] or [B, T, C, H', W']
    """
    model.eval()
    
    # Handle both single frame and sequence inputs
    if len(semantic_occ_tensor.shape) == 4:
        # Single frame: [B, H, W, D]
        semantic_occ_tensor = semantic_occ_tensor.unsqueeze(1)  # [B, 1, H, W, D]
        squeeze_output = True
    else:
        squeeze_output = False
    
    B, T = semantic_occ_tensor.shape[0], semantic_occ_tensor.shape[1]
    
    # Reshape to process each frame: [B*T, H, W, D]
    semantic_occ_flat = semantic_occ_tensor.view(B * T, *semantic_occ_tensor.shape[2:])
    semantic_occ_flat = semantic_occ_flat.to(device)
    
    # Process in chunks to reduce memory usage
    chunk_size = 10
    sampled_features_list = []
    
    with torch.no_grad():
        for chunk_start in range(0, B * T, chunk_size):
            chunk_end = min(chunk_start + chunk_size, B * T)
            semantic_occ_chunk = semantic_occ_flat[chunk_start:chunk_end].contiguous()
            
            # Encode through VAE: embedding -> encoder -> quantization
            temp_dict = {'semantic_occ': semantic_occ_chunk}
            
            # Embedding
            temp_dict = model.embedding(temp_dict)
            
            # Encoder
            temp_dict = model.encoder(temp_dict)
            
            # Quantization
            temp_dict = model.quantization(temp_dict)
            
            # Get sampled_features (latents)
            sampled_chunk = temp_dict['sampled_features'].detach().cpu()  # [chunk_size, C, H', W']
            sampled_features_list.append(sampled_chunk)
            
            del temp_dict, semantic_occ_chunk
    
    del semantic_occ_flat
    
    # Concatenate all chunks
    sampled_features = torch.cat(sampled_features_list, dim=0)  # [B*T, C, H', W']
    del sampled_features_list
    
    # Reshape back to [B, T, C, H', W']
    sampled_features = sampled_features.view(B, T, *sampled_features.shape[1:])
    
    if squeeze_output:
        sampled_features = sampled_features.squeeze(1)  # [B, C, H', W']
    
    return sampled_features


def process_split(model, dataset, dataloader, split_name, device, output_dir):
    """
    Process a dataset split (train or validation) and save latents.
    
    Args:
        model: VAE model
        dataset: Dataset instance
        dataloader: DataLoader instance
        split_name: 'train' or 'val'
        device: Device to use
        output_dir: Output directory
    """
    print(f"\nProcessing {split_name} split...")
    
    all_cached_data = []
    
    # Iterate through dataset
    for batch_idx, batch_dict in enumerate(tqdm(dataloader, desc=f"Encoding {split_name}")):
        # Get paths and trajectories
        paths = batch_dict['paths']  # List of paths for this batch
        trajectories = batch_dict['trajectory']  # [B, T*trajectory_length, 2]
        
        # Get semantic_occ from batch_dict
        semantic_occ_list = batch_dict['semantic_occ']  # List of numpy arrays
        
        # Convert to tensor
        semantic_occ_tensors = []
        for occ in semantic_occ_list:
            if isinstance(occ, np.ndarray):
                occ_tensor = torch.from_numpy(occ).long()
            else:
                occ_tensor = occ.long()
            semantic_occ_tensors.append(occ_tensor)
        
        # Stack into tensor: [B, H, W, D] (assuming sequence_length=1 for VAE training)
        semantic_occ_tensor = torch.stack(semantic_occ_tensors, dim=0)  # [B, H, W, D]
        
        # Encode to get latents
        x_sampled = encode_semantic_occ(model, semantic_occ_tensor, device)  # [B, C, H', W']
        
        # Convert to numpy
        x_sampled_np = x_sampled.numpy()
        trajectories_np = trajectories.numpy() if isinstance(trajectories, torch.Tensor) else trajectories
        
        # Save each sample in the batch
        batch_size = len(paths)
        for i in range(batch_size):
            # Handle path format (can be list or string)
            gt_path = paths[i] if isinstance(paths[i], str) else paths[i][0]
            
            # For single frame, x_sampled is [C, H', W']
            x_sampled_i = x_sampled_np[i]  # [C, H', W']
            
            # Trajectory: handle different shapes
            # For sequence_length=1, trajectory from dataset is [trajectory_length, 2]
            # For batch, trajectories_np might be [B, trajectory_length, 2] or [B*trajectory_length, 2]
            if isinstance(trajectories_np, np.ndarray):
                if len(trajectories_np.shape) == 2:
                    # Shape [B*trajectory_length, 2] - need to split by batch
                    traj_length = trajectories_np.shape[0] // batch_size
                    traj_i = trajectories_np[i * traj_length:(i + 1) * traj_length]
                elif len(trajectories_np.shape) == 3:
                    # Shape [B, trajectory_length, 2]
                    traj_i = trajectories_np[i]
                else:
                    traj_i = trajectories_np[i] if trajectories_np.shape[0] == batch_size else trajectories_np
            else:
                # If it's already a list or single array
                traj_i = trajectories_np[i] if isinstance(trajectories_np, (list, tuple)) else trajectories_np
            
            all_cached_data.append({
                "x_sampled": x_sampled_i,
                "gt_path": gt_path,
                "gt_trajs": traj_i
            })
    
    # Save to pickle file
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'waymo_latent_{split_name}.pkl')
    
    print(f"Saving {len(all_cached_data)} samples to {output_file}...")
    with open(output_file, 'wb') as f:
        pickle.dump(all_cached_data, f)
    
    print(f"✓ Saved {len(all_cached_data)} samples to {output_file}")
    
    return output_file


def main():
    args, cfg = parse_config()
    
    # Load config
    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"Using device: {device}")
    
    # Load checkpoint first to determine NUM_CATE from embedding weights
    print(f"Loading VAE checkpoint from {args.vae_ckpt}...")
    checkpoint = torch.load(args.vae_ckpt, map_location=device, weights_only=True)
    
    # Handle DDP prefix if present
    state_dict = checkpoint['state_dict']
    key = list(state_dict.keys())[0]
    if key.startswith('module.'):
        state_dict = common_utils.remove_module_prefix_from_ddp(state_dict)
    
    # Determine NUM_CATE from checkpoint embedding weights
    embedding_key = 'embedding.class_embeds.weight'
    if embedding_key not in state_dict:
        # Try with module prefix
        embedding_key = 'module.embedding.class_embeds.weight'
        if embedding_key in state_dict:
            temp_state_dict = common_utils.remove_module_prefix_from_ddp(state_dict)
            embedding_key = 'embedding.class_embeds.weight'
            state_dict = temp_state_dict
    
    if embedding_key in state_dict:
        checkpoint_num_cate = state_dict[embedding_key].shape[0]
        print(f"Detected NUM_CATE from checkpoint: {checkpoint_num_cate}")
        
        # After cfg_from_yaml_file, COMPRESSOR_CONFIG is moved to MODEL.COMPRESSOR_CONFIG
        # The structure is: MODEL.COMPRESSOR_CONFIG.MODEL.EMBEDDING.NUM_CATE
        # But build_network expects model_cfg.COMPRESSOR_CONFIG, so we need to check both structures
        if hasattr(cfg, 'MODEL') and 'COMPRESSOR_CONFIG' in cfg.MODEL:
            # After config loading, structure is MODEL.COMPRESSOR_CONFIG
            if not hasattr(cfg.MODEL.COMPRESSOR_CONFIG, 'MODEL'):
                cfg.MODEL.COMPRESSOR_CONFIG.MODEL = EasyDict()
            if not hasattr(cfg.MODEL.COMPRESSOR_CONFIG.MODEL, 'EMBEDDING'):
                cfg.MODEL.COMPRESSOR_CONFIG.MODEL.EMBEDDING = EasyDict()
            cfg.MODEL.COMPRESSOR_CONFIG.MODEL.EMBEDDING.NUM_CATE = checkpoint_num_cate
        elif hasattr(cfg, 'COMPRESSOR_CONFIG'):
            # Before config loading or if structure is different
            if not hasattr(cfg.COMPRESSOR_CONFIG, 'MODEL'):
                cfg.COMPRESSOR_CONFIG.MODEL = EasyDict()
            if not hasattr(cfg.COMPRESSOR_CONFIG.MODEL, 'EMBEDDING'):
                cfg.COMPRESSOR_CONFIG.MODEL.EMBEDDING = EasyDict()
            cfg.COMPRESSOR_CONFIG.MODEL.EMBEDDING.NUM_CATE = checkpoint_num_cate
        
        print(f"Set NUM_CATE to {checkpoint_num_cate} in config")
    else:
        print("Warning: Could not find embedding weights in checkpoint, using config NUM_CATE")
    
    # Build VAE model
    # After cfg_from_yaml_file, COMPRESSOR_CONFIG is moved to MODEL.COMPRESSOR_CONFIG
    # But build_network expects model_cfg with COMPRESSOR_CONFIG at top level
    # So we need to restore it or pass the right structure
    print("Building VAE model...")
    
    # build_network checks for TRANSITION_MODEL_CONFIG to decide between world model and compressor
    # For VAE, we need to pass a config with COMPRESSOR_CONFIG at top level
    # But after cfg_from_yaml_file, it's in MODEL.COMPRESSOR_CONFIG
    # So we create a temporary config structure that build_network expects
    if hasattr(cfg, 'MODEL') and 'COMPRESSOR_CONFIG' in cfg.MODEL:
        # Create a model_cfg with COMPRESSOR_CONFIG at top level for build_network
        model_cfg = EasyDict()
        model_cfg.NAME = cfg.NAME
        model_cfg.COMPRESSOR_CONFIG = cfg.MODEL.COMPRESSOR_CONFIG
        model = build_network(model_cfg=model_cfg, loss_cfg=cfg.LOSS).to(device)
    else:
        # Fallback: pass cfg directly (shouldn't happen after cfg_from_yaml_file)
        model = build_network(model_cfg=cfg, loss_cfg=cfg.LOSS).to(device)
    
    model.load_state_dict(state_dict)
    model.eval()
    print("✓ VAE model loaded successfully")
    
    # Build dataloaders for train and validation
    print("\nBuilding dataloaders...")
    
    # Training set
    train_set, train_loader = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        gen_training=False,  # VAE training mode
        training=True,
        rank=0,
        world_size=1
    )
    
    # Validation set
    val_set, val_loader = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        gen_training=False,  # VAE training mode
        training=False,
        rank=0,
        world_size=1
    )
    
    print(f"✓ Training samples: {len(train_set)}")
    print(f"✓ Validation samples: {len(val_set)}")
    
    # Process training set
    train_output = process_split(model, train_set, train_loader, 'train', device, args.output_dir)
    
    # Process validation set
    val_output = process_split(model, val_set, val_loader, 'val', device, args.output_dir)
    
    print("\n" + "="*60)
    print("✓ All latents saved successfully!")
    print(f"  Training: {train_output}")
    print(f"  Validation: {val_output}")
    print("\nTo use these latents in OccFM training, add to your config:")
    print(f"  PICKLE_PATH:")
    print(f"    train: '{train_output}'")
    print(f"    test: '{val_output}'")
    print("="*60)


if __name__ == '__main__':
    main()

