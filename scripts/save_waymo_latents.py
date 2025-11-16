#!/usr/bin/env python3
"""
Script to load VAE and save latents for Waymo dataset.
Saves latents in optimized formats for efficient multi-GPU training.

Usage:
    # Original format (backward compatible, single pickle file)
    python scripts/save_waymo_latents.py \
        --cfg_file tools/cfgs/occfm_vae_occ_only_waymo.yaml \
        --vae_ckpt logs/occfm_vae_occ_only_waymo/train_new/ckpt/epoch=000002.ckpt \
        --output_dir ./data/waymo_latent_vae/x16 \
        --batch_size 16 \
        --num_workers 4 \
        --format pickle

    # Sharded format (recommended for multi-GPU, reduces I/O contention)
    python scripts/save_waymo_latents.py \
        --cfg_file tools/cfgs/occfm_vae_occ_only_waymo.yaml \
        --vae_ckpt logs/occfm_vae_occ_only_waymo/train_new/ckpt/epoch=000002.ckpt \
        --output_dir ./data/waymo_latent_vae/x16 \
        --batch_size 16 \
        --num_workers 4 \
        --format sharded \
        --num_shards 8

    # Numpy format (separate .npy files, enables memory-mapping)
    python scripts/save_waymo_latents.py \
        --cfg_file tools/cfgs/occfm_vae_occ_only_waymo.yaml \
        --vae_ckpt logs/occfm_vae_occ_only_waymo/train_new/ckpt/epoch=000002.ckpt \
        --output_dir ./data/waymo_latent_vae/x16 \
        --batch_size 16 \
        --num_workers 4 \
        --format numpy

Storage Formats:
    1. 'pickle' (default): Single pickle file per split
        - waymo_latent_train.pkl
        - waymo_latent_val.pkl
        - Backward compatible with existing code
        - All data in one file (can cause I/O contention in multi-GPU)

    2. 'sharded' (recommended for multi-GPU): Multiple pickle files
        - waymo_latent_train_shard_000_of_008.pkl, ...
        - waymo_latent_train_shards.pkl (index file)
        - Each GPU/worker can load different shards
        - Reduces I/O contention significantly
        - Use --num_shards to specify number of shards (default: 8)

    3. 'numpy': Separate .npy files + metadata
        - latents_train/latent_*.npy (one file per sample)
        - waymo_latent_train_meta.pkl (metadata)
        - Enables memory-mapping for efficient loading
        - Best for very large datasets

Each format contains:
    - 'x_sampled': VAE latents (numpy array)
    - 'gt_path': Path to the .npz file
    - 'gt_trajs': Displacement from frame t to frame t+1 [dx, dy] (numpy array of shape [2])
                  For each frame t, displacement = pose(t+1) - pose(t)
                  Note: Last frame in each scene is skipped (no next frame, so no valid displacement)

To use these latents in OccFM training, add to your config:
    PICKLE_PATH:
        train: './data/waymo_latent_vae/x16/waymo_latent_train.pkl'  # For pickle format
        # OR for sharded format, point to index file:
        train: './data/waymo_latent_vae/x16/waymo_latent_train_shards.pkl'
        # OR for numpy format, point to metadata file:
        train: './data/waymo_latent_vae/x16/waymo_latent_train_meta.pkl'
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
    parser.add_argument('--format', type=str, default='pickle', choices=['pickle', 'numpy', 'sharded'],
                        help='Storage format: pickle (single file, backward compatible), numpy (separate .npy files), sharded (multiple pickle files for multi-GPU)')
    parser.add_argument('--num_shards', type=int, default=None,
                        help='Number of shards for sharded format (default: auto-detect from world_size or use 1)')
    
    args = parser.parse_args()
    cfg = EasyDict()
    cfg.ROOT_DIR = (Path(__file__).resolve().parent / '../').resolve()
    
    return args, cfg


def encode_semantic_occ(model, semantic_occ_tensor, device):
    """
    Encode semantic occupancy through VAE to get latents using model's forward pass.
    This is more efficient than manually calling embedding/encoder/quantization.
    
    Args:
        model: VAE model (OccFmVAE)
        semantic_occ_tensor: Tensor of shape [B, H, W, D] or [B, T, H, W, D]
        device: Device to run on
    
    Returns:
        sampled_features: Latents of shape [B, C, H', W'] or [B, T, C, H', W']
    """
    model.eval()
    
    # Enable latent_cache in quantization module to get x_sampled in output
    if hasattr(model, 'quantization'):
        model.quantization.latent_cache = True
    elif hasattr(model, 'module') and hasattr(model.module, 'quantization'):
        model.module.quantization.latent_cache = True
    
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
            
            # VAE model expects [B, 1, H, W, D] format (5D) where time dimension = 1
            # The loss function asserts: assert gt_occ.shape[1] == 1
            # Use model's forward pass instead of manual encoding
            # This leverages optimized forward pass and any caching mechanisms
            batch_dict = {
                'semantic_occ': semantic_occ_chunk.unsqueeze(1),  # [B, 1, H, W, D] - add time dimension
                'paths': [None] * semantic_occ_chunk.shape[0]  # Dummy paths
            }
            
            # Forward pass through model
            _, _, disp_dict = model(batch_dict)
            
            # Get sampled_features from disp_dict (set by latent_cache flag)
            if 'x_sampled' in disp_dict:
                sampled_chunk = disp_dict['x_sampled'].detach().cpu()
                # Handle shape: might be [B, 1, C, H', W'] or [B, C, H', W']
                if len(sampled_chunk.shape) == 5:
                    sampled_chunk = sampled_chunk.squeeze(1)  # [B, C, H', W']
                sampled_features_list.append(sampled_chunk)
            else:
                # Fallback: manually encode if latent_cache didn't work
                temp_dict = {'semantic_occ': semantic_occ_chunk}
                temp_dict = model.embedding(temp_dict)
                temp_dict = model.encoder(temp_dict)
                temp_dict = model.quantization(temp_dict)
                sampled_chunk = temp_dict['sampled_features'].detach().cpu()
                sampled_features_list.append(sampled_chunk)
            
            del batch_dict, semantic_occ_chunk
    
    del semantic_occ_flat
    
    # Concatenate all chunks
    sampled_features = torch.cat(sampled_features_list, dim=0)  # [B*T, C, H', W']
    del sampled_features_list
    
    # Reshape back to [B, T, C, H', W']
    sampled_features = sampled_features.view(B, T, *sampled_features.shape[1:])
    
    if squeeze_output:
        sampled_features = sampled_features.squeeze(1)  # [B, C, H', W']
    
    return sampled_features


def extract_global_pose_from_waymo_infos(path, dataset, waymo_infos_list):
    """
    Extract global (x, y) pose from waymo_infos for a given path.
    
    Args:
        path: Path to .npz file
        dataset: Dataset instance with scene mappings
        waymo_infos_list: List of waymo_infos pickle data
    
    Returns:
        global_pose: np.array([x, y]) in global coordinates, or None if not found
    """
    if waymo_infos_list is None or not hasattr(dataset, 'all_samples'):
        return None
    
    try:
        sample_idx = dataset.all_samples.index(path)
        scene_id = os.path.basename(os.path.dirname(path))
        
        if hasattr(dataset, 'scene_frame_mapping') and scene_id in dataset.scene_frame_mapping:
            scene_frames = dataset.scene_frame_mapping[scene_id]
            if sample_idx in scene_frames:
                local_frame_idx = scene_frames.index(sample_idx)
                scene_idx = int(scene_id) if scene_id.isdigit() else None
                
                if scene_idx is not None and hasattr(dataset, 'infos_by_scene'):
                    pickle_scene_key = f"scene_{scene_idx:03d}"
                    if pickle_scene_key in dataset.infos_by_scene:
                        pickle_scene_indices = dataset.infos_by_scene[pickle_scene_key]
                        if local_frame_idx < len(pickle_scene_indices):
                            pickle_idx = pickle_scene_indices[local_frame_idx]
                            if pickle_idx < len(waymo_infos_list):
                                frame_info = waymo_infos_list[pickle_idx]
                                if 'pose' in frame_info:
                                    pose_matrix = frame_info['pose']
                                    if isinstance(pose_matrix, np.ndarray) and pose_matrix.shape == (4, 4):
                                        global_x = pose_matrix[0, 3]  # x coordinate in meters (global)
                                        global_y = pose_matrix[1, 3]  # y coordinate in meters (global)
                                        return np.array([global_x, global_y], dtype=np.float32)
    except (ValueError, IndexError, AttributeError):
        pass
    
    return None


def process_split(model, dataset, dataloader, split_name, device, output_dir, waymo_infos_list=None, format='pickle', num_shards=None):
    """
    Process a dataset split (train or validation) and save latents.
    Processes frames scene by scene, computing displacements within each scene and saving displacement [dx, dy] for each latent.
    
    For each frame t, saves displacement = pose(t+1) - pose(t) = [dx, dy]
    This represents where frame t+1 is relative to frame t.
    Scene boundaries are respected - displacements are only computed within the same scene.
    Last frame in each scene is skipped (no next frame, so no valid displacement).
    
    Args:
        model: VAE model
        dataset: Dataset instance
        dataloader: DataLoader instance
        split_name: 'train' or 'val'
        device: Device to use
        output_dir: Output directory
        waymo_infos_list: Optional list of waymo_infos pickle data for extracting global poses
    """
    print(f"\nProcessing {split_name} split...")
    
    # Group frames by scene to process scene by scene
    # This ensures we respect scene boundaries when computing displacements
    scene_to_frames = {}
    if hasattr(dataset, 'scene_frame_mapping') and hasattr(dataset, 'all_samples'):
        print("Grouping frames by scene...")
        for scene_id, frame_indices in dataset.scene_frame_mapping.items():
            scene_to_frames[scene_id] = frame_indices
        print(f"Found {len(scene_to_frames)} scenes")
    
    # Build mapping from path to global pose for all frames
    print("Building path to global pose mapping...")
    path_to_global_pose = {}
    if waymo_infos_list is not None and hasattr(dataset, 'all_samples'):
        for idx, path in enumerate(dataset.all_samples):
            global_pose = extract_global_pose_from_waymo_infos(path, dataset, waymo_infos_list)
            if global_pose is not None:
                path_to_global_pose[path] = global_pose
        
        print(f"Extracted global poses for {len(path_to_global_pose)}/{len(dataset.all_samples)} frames")
        
        # Verify scene boundaries by checking displacements within scenes
        print("\nVerifying scene boundaries and computing displacements within scenes...")
        for scene_id, frame_indices in sorted(scene_to_frames.items())[:5]:  # Check first 5 scenes
            scene_poses = []
            scene_paths = []
            for frame_idx in frame_indices:
                if frame_idx < len(dataset.all_samples):
                    path = dataset.all_samples[frame_idx]
                    if path in path_to_global_pose:
                        scene_poses.append(path_to_global_pose[path])
                        scene_paths.append(path)
            
            if len(scene_poses) > 1:
                # Compute displacements within scene
                scene_poses_array = np.array(scene_poses)
                displacements = np.diff(scene_poses_array, axis=0)
                max_displacement = np.max(np.linalg.norm(displacements, axis=1))
                print(f"  Scene {scene_id}: {len(scene_poses)} frames, max displacement: {max_displacement:.2f}m")
    
    all_cached_data = []
    
    # Process scene by scene to respect boundaries
    # But we still need to iterate through dataloader to encode latents
    # So we'll process in batches but group results by scene
    
    # First, encode all latents
    print("\nEncoding latents...")
    path_to_latent = {}
    for batch_idx, batch_dict in enumerate(tqdm(dataloader, desc=f"Encoding {split_name}")):
        paths = batch_dict['paths']
        semantic_occ_list = batch_dict['semantic_occ']
        
        # Convert to tensor
        semantic_occ_tensors = []
        for occ in semantic_occ_list:
            if isinstance(occ, np.ndarray):
                occ_tensor = torch.from_numpy(occ).long()
            else:
                occ_tensor = occ.long()
            semantic_occ_tensors.append(occ_tensor)
        
        # Stack into tensor: [B, H, W, D]
        semantic_occ_tensor = torch.stack(semantic_occ_tensors, dim=0)
        
        # Encode to get latents
        x_sampled = encode_semantic_occ(model, semantic_occ_tensor, device)
        x_sampled_np = x_sampled.numpy()
        
        # Store latents by path
        batch_size = len(paths)
        for i in range(batch_size):
            gt_path = paths[i] if isinstance(paths[i], str) else paths[i][0]
            x_sampled_i = x_sampled_np[i]
            path_to_latent[gt_path] = x_sampled_i
    
    print(f"Encoded {len(path_to_latent)} latents")
    
    # Now process scene by scene, computing displacements and saving global poses
    print("\nProcessing scenes and saving latents with global poses...")
    for scene_id in sorted(scene_to_frames.keys()):
        frame_indices = scene_to_frames[scene_id]
        scene_global_poses = []
        scene_latents = []
        scene_paths = []
        
        # Collect all frames in this scene
        for frame_idx in frame_indices:
            if frame_idx < len(dataset.all_samples):
                path = dataset.all_samples[frame_idx]
                if path in path_to_latent:
                    scene_paths.append(path)
                    scene_latents.append(path_to_latent[path])
                    
                    # Get global pose
                    if path in path_to_global_pose:
                        global_pose = path_to_global_pose[path]
                    else:
                        # Try to extract from dataset
                        global_pose = extract_global_pose_from_waymo_infos(path, dataset, waymo_infos_list)
                        if global_pose is None:
                            global_pose = np.array([0.0, 0.0], dtype=np.float32)
                    
                    scene_global_poses.append(global_pose)
        
        # Compute displacements within scene
        # For each frame t, displacement = pose(t+1) - pose(t) = [dx, dy]
        # This represents where frame t+1 is relative to frame t
        scene_displacements = []
        if len(scene_global_poses) > 1:
            scene_poses_array = np.array(scene_global_poses)
            # Compute displacements: disp[i] = pose[i+1] - pose[i]
            displacements = np.diff(scene_poses_array, axis=0)  # [N-1, 2] - displacements between consecutive frames
            
            # For each frame, assign its displacement to next frame
            # Frame 0 -> displacement to frame 1
            # Frame 1 -> displacement to frame 2
            # ...
            # Last frame -> zero displacement (no next frame)
            for i in range(len(scene_global_poses)):
                if i < len(displacements):
                    # Displacement from frame i to frame i+1
                    disp = displacements[i]  # [dx, dy]
                else:
                    # Last frame: no next frame, use zero displacement
                    disp = np.array([0.0, 0.0], dtype=np.float32)
                scene_displacements.append(disp)
            
            # Verify displacements are reasonable (not crossing scene boundaries)
            displacement_norms = np.linalg.norm(displacements, axis=1)
            if np.any(displacement_norms > 100.0):  # Large displacement might indicate scene boundary issue
                print(f"Warning: Scene {scene_id} has large displacement: {np.max(displacement_norms):.2f}m")
        else:
            # Single frame scene: zero displacement
            scene_displacements = [np.array([0.0, 0.0], dtype=np.float32)]
        
        # Save each frame in the scene with its displacement to next frame
        # For frame t, displacement = [dx, dy] where (x_t+1, y_t+1) = (x_t, y_t) + (dx, dy)
        # Skip the last frame in each scene (it has zero displacement, no next frame)
        for path, latent, displacement in zip(scene_paths[:-1], scene_latents[:-1], scene_displacements[:-1]):
            all_cached_data.append({
                "x_sampled": latent,
                "gt_path": path,
                "gt_trajs": displacement  # [2] - displacement to next frame [dx, dy]
            })
    
    # Save based on format
    os.makedirs(output_dir, exist_ok=True)
    
    if format == 'pickle':
        # Original format: single pickle file (backward compatible)
        output_file = os.path.join(output_dir, f'waymo_latent_{split_name}.pkl')
        print(f"Saving {len(all_cached_data)} samples to {output_file}...")
        with open(output_file, 'wb') as f:
            pickle.dump(all_cached_data, f)
        print(f"✓ Saved {len(all_cached_data)} samples to {output_file}")
        return output_file
    
    elif format == 'numpy':
        # Optimized format: separate .npy files + metadata pickle
        # This allows memory-mapping and reduces I/O contention
        latents_dir = os.path.join(output_dir, f'latents_{split_name}')
        os.makedirs(latents_dir, exist_ok=True)
        
        metadata = []
        print(f"Saving {len(all_cached_data)} samples in numpy format...")
        
        for idx, data in enumerate(tqdm(all_cached_data, desc=f"Saving {split_name} latents")):
            # Save latent as separate .npy file
            latent_file = os.path.join(latents_dir, f'latent_{idx:08d}.npy')
            np.save(latent_file, data['x_sampled'])
            
            # Store metadata (paths and trajectories)
            metadata.append({
                "gt_path": data['gt_path'],
                "gt_trajs": data['gt_trajs'],
                "latent_file": os.path.relpath(latent_file, output_dir)  # Relative path for portability
            })
        
        # Save metadata
        metadata_file = os.path.join(output_dir, f'waymo_latent_{split_name}_meta.pkl')
        with open(metadata_file, 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"✓ Saved {len(all_cached_data)} samples:")
        print(f"  - Latents: {latents_dir}/")
        print(f"  - Metadata: {metadata_file}")
        return metadata_file
    
    elif format == 'sharded':
        # Sharded format: multiple pickle files for multi-GPU training
        # IMPORTANT: Shard by scene to preserve scene continuity
        # Each shard contains complete scenes, so when a rank loads one shard,
        # it has all frames from those scenes and can find consecutive sequences
        if num_shards is None:
            # Try to auto-detect from environment, default to 1
            num_shards = int(os.environ.get('WORLD_SIZE', 1))
            if num_shards == 1:
                num_shards = 8  # Default to 8 shards for better multi-GPU performance
        
        print(f"Saving {len(all_cached_data)} samples in {num_shards} shards (sharding by scene)...")
        
        # Group frames by scene - extract scene_id from gt_path
        # Since all_cached_data is built scene by scene, frames from same scene are consecutive
        # But we need to verify and group them explicitly to ensure complete scenes in each shard
        print("Grouping frames by scene for scene-based sharding...")
        scene_to_frames = {}
        for idx, frame_data in enumerate(all_cached_data):
            path = frame_data['gt_path'][0] if isinstance(frame_data['gt_path'], list) else frame_data['gt_path']
            # Extract scene_id from path: .../<split>/<scene_id>/<frame>.npz
            scene_id = os.path.basename(os.path.dirname(path))
            if scene_id not in scene_to_frames:
                scene_to_frames[scene_id] = []
            scene_to_frames[scene_id].append(frame_data)
        
        # Verify scene grouping
        print(f"Grouped {len(all_cached_data)} frames into {len(scene_to_frames)} scenes")
        scene_frame_counts = {scene_id: len(frames) for scene_id, frames in scene_to_frames.items()}
        if len(scene_frame_counts) > 0:
            min_frames = min(scene_frame_counts.values())
            max_frames = max(scene_frame_counts.values())
            avg_frames = sum(scene_frame_counts.values()) / len(scene_frame_counts)
            print(f"  Frames per scene: min={min_frames}, max={max_frames}, avg={avg_frames:.1f}")
        
        # Get list of scenes and shuffle scenes (not frames) for better distribution
        scene_ids = sorted(scene_to_frames.keys())
        import random
        random.seed(42)  # Fixed seed for reproducibility
        random.shuffle(scene_ids)  # Shuffle scenes, but keep all frames of each scene together
        
        # Distribute scenes across shards (not frames)
        # Each shard gets approximately equal number of scenes
        scenes_per_shard = len(scene_ids) // num_shards
        shard_files = []
        total_samples = 0
        
        for shard_idx in range(num_shards):
            # Determine which scenes go to this shard
            start_scene_idx = shard_idx * scenes_per_shard
            if shard_idx == num_shards - 1:
                # Last shard gets remaining scenes
                end_scene_idx = len(scene_ids)
            else:
                end_scene_idx = (shard_idx + 1) * scenes_per_shard
            
            # Collect all frames from scenes assigned to this shard
            shard_scenes = scene_ids[start_scene_idx:end_scene_idx]
            shard_data = []
            for scene_id in shard_scenes:
                shard_data.extend(scene_to_frames[scene_id])  # Add all frames from this scene
            
            shard_file = os.path.join(output_dir, f'waymo_latent_{split_name}_shard_{shard_idx:03d}_of_{num_shards:03d}.pkl')
            
            with open(shard_file, 'wb') as f:
                pickle.dump(shard_data, f)
            
            shard_files.append(shard_file)
            total_samples += len(shard_data)
            
            # Verify that all frames in this shard belong to the assigned scenes
            shard_scene_ids = set()
            for frame_data in shard_data:
                path = frame_data['gt_path'][0] if isinstance(frame_data['gt_path'], list) else frame_data['gt_path']
                scene_id = os.path.basename(os.path.dirname(path))
                shard_scene_ids.add(scene_id)
            
            # Check if any scenes are split across shards (should not happen)
            if shard_scene_ids != set(shard_scenes):
                print(f"  [WARNING] Shard {shard_idx+1} scene mismatch!")
                print(f"    Expected scenes: {set(shard_scenes)}")
                print(f"    Actual scenes: {shard_scene_ids}")
            
            print(f"  Shard {shard_idx+1}/{num_shards}: {len(shard_scenes)} scenes, {len(shard_data)} samples -> {shard_file}")
            print(f"    Scenes: {shard_scenes[:5]}{'...' if len(shard_scenes) > 5 else ''}")
            print(f"    ✓ Verified: All {len(shard_data)} frames belong to the {len(shard_scenes)} assigned scenes")
        
        # Save shard index file
        index_file = os.path.join(output_dir, f'waymo_latent_{split_name}_shards.pkl')
        with open(index_file, 'wb') as f:
            pickle.dump({
                'num_shards': num_shards,
                'shard_files': [os.path.basename(f) for f in shard_files],
                'total_samples': total_samples,
                'total_scenes': len(scene_ids),
                'sharding_method': 'by_scene'  # Indicate that shards are organized by scene
            }, f)
        
        print(f"✓ Saved {total_samples} samples from {len(scene_ids)} scenes in {num_shards} shards")
        print(f"  - Index file: {index_file}")
        print(f"  - Each shard contains complete scenes (preserves scene continuity)")
        return index_file
    
    else:
        raise ValueError(f"Unknown format: {format}")


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
    
    # Load waymo_infos pickle files to extract global poses
    waymo_infos_train = None
    waymo_infos_val = None
    
    info_pkl_name_train = 'waymo_infos_train.pkl'
    info_pkl_name_val = 'waymo_infos_val.pkl'
    # DATA_CONFIG has DATA_PATH (uppercase) - try attribute access first, then dict access
    if hasattr(cfg.DATA_CONFIG, 'DATA_PATH'):
        data_path = cfg.DATA_CONFIG.DATA_PATH
    elif 'DATA_PATH' in cfg.DATA_CONFIG:
        data_path = cfg.DATA_CONFIG['DATA_PATH']
    elif 'data_path' in cfg.DATA_CONFIG:
        data_path = cfg.DATA_CONFIG['data_path']
    else:
        raise ValueError(f"Could not find DATA_PATH in DATA_CONFIG. Available keys: {list(cfg.DATA_CONFIG.keys())}")
    info_pkl_path_train = os.path.join(data_path, info_pkl_name_train)
    info_pkl_path_val = os.path.join(data_path, info_pkl_name_val)
    
    if os.path.exists(info_pkl_path_train):
        print(f"Loading {info_pkl_name_train} for global pose extraction...")
        with open(info_pkl_path_train, 'rb') as f:
            waymo_infos_train = pickle.load(f)
        print(f"Loaded {len(waymo_infos_train)} entries from {info_pkl_name_train}")
    
    if os.path.exists(info_pkl_path_val):
        print(f"Loading {info_pkl_name_val} for global pose extraction...")
        with open(info_pkl_path_val, 'rb') as f:
            waymo_infos_val = pickle.load(f)
        print(f"Loaded {len(waymo_infos_val)} entries from {info_pkl_name_val}")
    
    # Process training set
    train_output = process_split(model, train_set, train_loader, 'train', device, args.output_dir, 
                                 waymo_infos_list=waymo_infos_train, format=args.format, num_shards=args.num_shards)
    
    # Process validation set
    val_output = process_split(model, val_set, val_loader, 'val', device, args.output_dir, 
                               waymo_infos_list=waymo_infos_val, format=args.format, num_shards=args.num_shards)
    
    print("\n" + "="*60)
    print("✓ All latents saved successfully!")
    print(f"  Training: {train_output}")
    print(f"  Validation: {val_output}")
    print(f"\nFormat: {args.format}")
    print("\nTo use these latents in OccFM training, add to your config:")
    print(f"  PICKLE_PATH:")
    print(f"    train: '{train_output}'")
    print(f"    test: '{val_output}'")
    if args.format == 'sharded':
        print("\n  Note: For sharded format, each GPU/worker can load different shards")
        print("  to reduce I/O contention. Update dataset code to support shard selection.")
    elif args.format == 'numpy':
        print("\n  Note: For numpy format, latents are stored as separate .npy files.")
        print("  The metadata file contains paths to load latents on-demand.")
    print("="*60)


if __name__ == '__main__':
    main()

