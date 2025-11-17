#!/usr/bin/env python3
"""Detailed exploration of Waymo pickle structure"""
import pickle
import numpy as np
import os

def explore_detailed(pkl_path):
    print(f"\n{'='*80}")
    print(f"Detailed exploration: {os.path.basename(pkl_path)}")
    print(f"{'='*80}\n")
    
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Type: {type(data)}")
    print(f"Length: {len(data)}")
    
    # Show first few items
    print(f"\nFirst 3 items structure:")
    for i in range(min(3, len(data))):
        item = data[i]
        print(f"\n  Item {i}:")
        print(f"    Keys: {list(item.keys())}")
        print(f"    timestamp: {item.get('timestamp', 'N/A')}")
        
        if 'pose' in item:
            pose = item['pose']
            print(f"    pose shape: {pose.shape if hasattr(pose, 'shape') else 'N/A'}")
            print(f"    pose type: {type(pose)}")
            if isinstance(pose, np.ndarray) and pose.shape == (4, 4):
                # Extract translation (x, y, z)
                translation = pose[:3, 3]
                print(f"    pose translation (x, y, z): {translation}")
                # Extract rotation (yaw from rotation matrix)
                yaw = np.arctan2(pose[1, 0], pose[0, 0])
                print(f"    pose yaw (approx): {np.degrees(yaw):.2f}°")
        
        if 'image' in item:
            img_info = item['image']
            print(f"    image_path: {img_info.get('image_path', 'N/A')}")
        
        if 'point_cloud' in item:
            pc_info = item['point_cloud']
            print(f"    velodyne_path: {pc_info.get('velodyne_path', 'N/A')}")
    
    # Check if there's scene grouping
    print(f"\n{'='*80}")
    print("Checking for scene organization...")
    print(f"{'='*80}\n")
    
    # Try to group by image path prefix (scene ID)
    scene_groups = {}
    for i, item in enumerate(data[:1000]):  # Sample first 1000
        if 'image' in item and 'image_path' in item['image']:
            img_path = item['image']['image_path']
            # Extract scene ID from path (e.g., "training/image_0/0000000.png" -> "image_0")
            parts = img_path.split('/')
            if len(parts) >= 2:
                scene_id = parts[-2]  # e.g., "image_0"
                if scene_id not in scene_groups:
                    scene_groups[scene_id] = []
                scene_groups[scene_id].append(i)
    
    print(f"Found {len(scene_groups)} scene groups in first 1000 items:")
    for scene_id, indices in list(scene_groups.items())[:5]:
        print(f"  {scene_id}: {len(indices)} frames (indices {indices[:5]}...{indices[-5:] if len(indices) > 5 else ''})")
    
    # Show trajectory construction example
    print(f"\n{'='*80}")
    print("Example trajectory construction from poses...")
    print(f"{'='*80}\n")
    
    if len(data) > 5:
        # Get first scene's frames
        first_scene_id = list(scene_groups.keys())[0]
        scene_indices = scene_groups[first_scene_id][:5]
        
        print(f"Scene: {first_scene_id}")
        print(f"Frames: {scene_indices}")
        print(f"\nPose translations (x, y) for first 5 frames:")
        
        poses_xy = []
        for idx in scene_indices:
            item = data[idx]
            if 'pose' in item:
                pose = item['pose']
                if isinstance(pose, np.ndarray) and pose.shape == (4, 4):
                    x, y = pose[0, 3], pose[1, 3]
                    poses_xy.append([x, y])
                    print(f"  Frame {idx}: x={x:.2f}, y={y:.2f}")
        
        if len(poses_xy) > 1:
            print(f"\nRelative trajectories (from first frame):")
            base_pose = np.array(poses_xy[0])
            for i, (x, y) in enumerate(poses_xy[1:], 1):
                rel_pos = np.array([x, y]) - base_pose
                print(f"  Frame {i} relative: dx={rel_pos[0]:.2f}, dy={rel_pos[1]:.2f}")
    
    # Check if there are any pre-computed trajectory fields
    print(f"\n{'='*80}")
    print("Checking for trajectory-related fields in all items...")
    print(f"{'='*80}\n")
    
    traj_keys_found = set()
    for i, item in enumerate(data[:100]):  # Check first 100
        if isinstance(item, dict):
            for key in item.keys():
                key_lower = key.lower()
                if any(term in key_lower for term in ['traj', 'motion', 'future', 'pred']):
                    traj_keys_found.add(key)
    
    if traj_keys_found:
        print(f"Found trajectory-related keys: {traj_keys_found}")
    else:
        print("No pre-computed trajectory fields found. Trajectories must be constructed from poses.")
    
    # Check how to match with .npz files
    print(f"\n{'='*80}")
    print("Matching strategy with .npz files...")
    print(f"{'='*80}\n")
    
    print("NPZ file structure: <split>/<scene_id>/<frame>_*.npz")
    print("Pickle structure: flat list with image_path like 'training/image_0/0000000.png'")
    print("\nMatching strategy:")
    print("  1. Extract scene_id from .npz path: <scene_id>/<frame>.npz")
    print("  2. Extract frame number from .npz filename: <frame>_04.npz -> <frame>")
    print("  3. Find matching item in pickle by:")
    print("     - Matching scene_id with image_path directory")
    print("     - Matching frame number with image_idx or timestamp")

if __name__ == "__main__":
    base_path = '/data/dataset/waymo_perception_v1_4_0/Occupancy3D-Waymo'
    val_pkl = os.path.join(base_path, 'waymo_infos_val.pkl')
    
    if os.path.exists(val_pkl):
        explore_detailed(val_pkl)









