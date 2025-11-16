#!/usr/bin/env python3
"""
Script to inspect waymo_infos pickle file structure and metadata for pose conversion.
"""

import pickle
import numpy as np
import os
import sys

def inspect_waymo_pkl(pkl_path, num_samples=3):
    """Inspect waymo_infos pickle file structure."""
    if not os.path.exists(pkl_path):
        print(f"Error: File not found: {pkl_path}")
        return
    
    print(f"\n{'='*80}")
    print(f"Inspecting: {pkl_path}")
    print(f"{'='*80}\n")
    
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Data type: {type(data).__name__}")
    
    if isinstance(data, list):
        print(f"Number of items: {len(data)}")
        print(f"\nInspecting first {num_samples} items:\n")
        
        for i in range(min(num_samples, len(data))):
            item = data[i]
            print(f"{'─'*80}")
            print(f"Item {i}:")
            print(f"{'─'*80}")
            
            if isinstance(item, dict):
                print(f"  Keys: {list(item.keys())}")
                print()
                
                # Inspect each key
                for key in item.keys():
                    val = item[key]
                    val_type = type(val).__name__
                    
                    if key == 'pose':
                        if isinstance(val, np.ndarray):
                            print(f"  {key}:")
                            print(f"    Type: {val_type}")
                            print(f"    Shape: {val.shape}")
                            print(f"    Dtype: {val.dtype}")
                            print(f"    Translation (x, y, z): [{val[0, 3]:.2f}, {val[1, 3]:.2f}, {val[2, 3]:.2f}]")
                            print(f"    Rotation matrix (3x3):")
                            print(f"      {val[0, :3]}")
                            print(f"      {val[1, :3]}")
                            print(f"      {val[2, :3]}")
                            # Extract rotation angle (yaw) from rotation matrix
                            yaw = np.arctan2(val[1, 0], val[0, 0])
                            print(f"    Yaw angle (rad): {yaw:.4f}, (deg): {np.degrees(yaw):.2f}")
                    
                    elif key == 'timestamp':
                        print(f"  {key}: {val} (type: {val_type})")
                    
                    elif key == 'image':
                        if isinstance(val, dict):
                            print(f"  {key}:")
                            print(f"    Keys: {list(val.keys())}")
                            for img_key, img_val in val.items():
                                if img_key == 'image_path':
                                    print(f"      {img_key}: {img_val}")
                                elif img_key == 'image_idx':
                                    print(f"      {img_key}: {img_val}")
                                elif img_key == 'image_shape':
                                    if isinstance(img_val, np.ndarray):
                                        print(f"      {img_key}: shape={img_val.shape}")
                                    else:
                                        print(f"      {img_key}: {img_val}")
                    
                    elif key == 'calib':
                        if isinstance(val, dict):
                            print(f"  {key}:")
                            print(f"    Keys: {list(val.keys())}")
                            for calib_key in ['P0', 'P1', 'P2', 'P3', 'P4', 'R0_rect', 'Tr_velo_to_cam']:
                                if calib_key in val:
                                    calib_val = val[calib_key]
                                    if isinstance(calib_val, np.ndarray):
                                        print(f"      {calib_key}: shape={calib_val.shape}, dtype={calib_val.dtype}")
                    
                    elif key == 'point_cloud':
                        if isinstance(val, dict):
                            print(f"  {key}:")
                            print(f"    Keys: {list(val.keys())}")
                            for pc_key, pc_val in val.items():
                                print(f"      {pc_key}: {pc_val}")
                    
                    elif key == 'annos':
                        if isinstance(val, dict):
                            print(f"  {key}:")
                            print(f"    Keys: {list(val.keys())[:10]}...")  # Show first 10 keys
                    
                    elif key == 'sweeps':
                        print(f"  {key}: {val_type}, length: {len(val) if hasattr(val, '__len__') else 'N/A'}")
                    
                    else:
                        if isinstance(val, np.ndarray):
                            print(f"  {key}: shape={val.shape}, dtype={val.dtype}")
                        elif isinstance(val, (list, tuple)):
                            print(f"  {key}: {val_type}, length={len(val)}")
                        else:
                            val_str = str(val)
                            if len(val_str) > 100:
                                val_str = val_str[:100] + "..."
                            print(f"  {key}: {val_str}")
            
            print()
    
    elif isinstance(data, dict):
        print(f"Dictionary with keys: {list(data.keys())}")
        if 'infos' in data:
            print(f"\n'infos' contains: {type(data['infos']).__name__}")
            if isinstance(data['infos'], dict):
                print(f"  Number of scenes: {len(data['infos'])}")
                first_scene_key = list(data['infos'].keys())[0]
                first_scene = data['infos'][first_scene_key]
                print(f"  First scene key: {first_scene_key}")
                print(f"  First scene type: {type(first_scene).__name__}")
                if isinstance(first_scene, list):
                    print(f"  Frames in first scene: {len(first_scene)}")
                    if len(first_scene) > 0:
                        print(f"  First frame keys: {list(first_scene[0].keys()) if isinstance(first_scene[0], dict) else 'N/A'}")
    
    print(f"\n{'='*80}")
    print("Summary of pose conversion metadata:")
    print(f"{'='*80}")
    print("""
Available in 'pose' field (4x4 transformation matrix):
  - Translation (global coordinates): pose[0, 3] = x, pose[1, 3] = y, pose[2, 3] = z
  - Rotation matrix: pose[:3, :3] = 3x3 rotation matrix
  - Can extract yaw angle: yaw = arctan2(pose[1, 0], pose[0, 0])
  
Additional metadata:
  - 'timestamp': Microsecond timestamp for temporal ordering
  - 'image': Contains image_path, image_idx, image_shape
  - 'calib': Camera calibration matrices (P0-P4, R0_rect, Tr_velo_to_cam)
  - 'point_cloud': LiDAR data paths
    """)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python inspect_waymo_pkl.py <path_to_waymo_infos.pkl> [num_samples]")
        sys.exit(1)
    
    pkl_path = sys.argv[1]
    num_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    
    inspect_waymo_pkl(pkl_path, num_samples)



