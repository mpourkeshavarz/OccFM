#!/usr/bin/env python3
"""Check how pickle items match .npz files"""
import pickle
import numpy as np
import os

base_path = '/data/dataset/waymo_perception_v1_4_0/Occupancy3D-Waymo'

# Load pickle
val_pkl = os.path.join(base_path, 'waymo_infos_val.pkl')
with open(val_pkl, 'rb') as f:
    data = pickle.load(f)

print(f"Pickle has {len(data)} items\n")

# Check first few items
print("First 5 pickle items:")
for i in range(min(5, len(data))):
    item = data[i]
    img_path = item['image']['image_path']
    img_idx = item['image']['image_idx']
    timestamp = item['timestamp']
    pose = item['pose']
    x, y = pose[0, 3], pose[1, 3]
    
    print(f"  Item {i}:")
    print(f"    image_path: {img_path}")
    print(f"    image_idx: {img_idx}")
    print(f"    timestamp: {timestamp}")
    print(f"    pose (x, y): ({x:.2f}, {y:.2f})")
    print()

# Check .npz file structure
val_dir = os.path.join(base_path, 'validation')
if os.path.exists(val_dir):
    scene_dirs = sorted([d for d in os.listdir(val_dir) if os.path.isdir(os.path.join(val_dir, d))])
    print(f"Validation directory has {len(scene_dirs)} scene directories")
    print(f"First 5 scenes: {scene_dirs[:5]}\n")
    
    # Check first scene
    first_scene = scene_dirs[0]
    scene_path = os.path.join(val_dir, first_scene)
    npz_files = sorted([f for f in os.listdir(scene_path) if f.endswith('_04.npz')])
    print(f"Scene {first_scene} has {len(npz_files)} .npz files")
    print(f"First 5 .npz files: {npz_files[:5]}\n")
    
    # Try to match
    print("Matching strategy:")
    print("  Pickle item image_path: training/image_0/1000000.png")
    print("  NPZ file: validation/000/000_04.npz")
    print("\n  Need to find mapping between:")
    print("    - image_idx (1000000) <-> frame number (000)")
    print("    - image directory (image_0) <-> scene directory (000)")
    
    # Check if there's a pattern
    print("\nChecking for patterns...")
    
    # Group pickle items by image directory
    from collections import defaultdict
    scene_groups = defaultdict(list)
    for i, item in enumerate(data[:1000]):
        img_path = item['image']['image_path']
        parts = img_path.split('/')
        if len(parts) >= 2:
            scene_key = parts[-2]  # e.g., "image_0"
            scene_groups[scene_key].append((i, item))
    
    print(f"Found {len(scene_groups)} unique image directories in first 1000 items")
    for scene_key, items in list(scene_groups.items())[:3]:
        print(f"  {scene_key}: {len(items)} items")
        print(f"    First item idx: {items[0][0]}, image_idx: {items[0][1]['image']['image_idx']}")
        print(f"    Last item idx: {items[-1][0]}, image_idx: {items[-1][1]['image']['image_idx']}")







