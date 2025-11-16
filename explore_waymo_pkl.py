#!/usr/bin/env python3
"""Script to explore Waymo info pickle file structure"""
import pickle
import sys
import os

def explore_pickle(pkl_path, max_depth=3, max_items=5):
    """Recursively explore pickle file structure"""
    print(f"\n{'='*80}")
    print(f"Exploring: {os.path.basename(pkl_path)}")
    print(f"{'='*80}\n")
    
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"Error loading pickle: {e}")
        import traceback
        traceback.print_exc()
        return
    
    def explore_obj(obj, path="root", depth=0, max_depth=3, max_items=5):
        indent = "  " * depth
        obj_type = type(obj).__name__
        
        if depth > max_depth:
            print(f"{indent}{path}: {obj_type} (max depth reached)")
            return
        
        if isinstance(obj, dict):
            keys = list(obj.keys())
            print(f"{indent}{path}: dict with {len(keys)} keys")
            if len(keys) > 0:
                print(f"{indent}  Keys: {keys[:max_items]}{'...' if len(keys) > max_items else ''}")
                
                # Explore first few items
                for key in keys[:max_items]:
                    try:
                        val = obj[key]
                        if isinstance(val, (dict, list)):
                            explore_obj(val, f"{path}.{key}", depth+1, max_depth, max_items)
                        else:
                            val_str = str(val)
                            if len(val_str) > 100:
                                val_str = val_str[:100] + "..."
                            print(f"{indent}  {key}: {type(val).__name__} = {val_str}")
                    except Exception as e:
                        print(f"{indent}  {key}: Error accessing - {e}")
        
        elif isinstance(obj, list):
            print(f"{indent}{path}: list with {len(obj)} items")
            if len(obj) > 0:
                print(f"{indent}  First item type: {type(obj[0]).__name__}")
                if isinstance(obj[0], (dict, list)):
                    explore_obj(obj[0], f"{path}[0]", depth+1, max_depth, max_items)
                else:
                    val_str = str(obj[0])
                    if len(val_str) > 100:
                        val_str = val_str[:100] + "..."
                    print(f"{indent}  First item: {val_str}")
        
        elif hasattr(obj, '__dict__'):
            attrs = dir(obj)
            print(f"{indent}{path}: {obj_type} with attributes")
            print(f"{indent}  Attributes: {attrs[:max_items]}{'...' if len(attrs) > max_items else ''}")
        
        else:
            val_str = str(obj)
            if len(val_str) > 200:
                val_str = val_str[:200] + "..."
            print(f"{indent}{path}: {obj_type} = {val_str}")
    
    explore_obj(data, max_depth=max_depth, max_items=max_items)
    
    # Try to find trajectory-related information
    print(f"\n{'='*80}")
    print("Searching for trajectory/ego pose related fields...")
    print(f"{'='*80}\n")
    
    def find_trajectory_fields(obj, path="root", depth=0, max_depth=4):
        if depth > max_depth:
            return
        
        if isinstance(obj, dict):
            for key, val in obj.items():
                key_lower = key.lower()
                if any(term in key_lower for term in ['traj', 'ego', 'pose', 'translation', 'rotation', 'position', 'motion']):
                    val_type = type(val).__name__
                    val_str = str(val)
                    if len(val_str) > 150:
                        val_str = val_str[:150] + "..."
                    print(f"  Found: {path}.{key} = {val_type}: {val_str}")
                
                if isinstance(val, (dict, list)):
                    find_trajectory_fields(val, f"{path}.{key}", depth+1, max_depth)
        
        elif isinstance(obj, list) and len(obj) > 0:
            find_trajectory_fields(obj[0], f"{path}[0]", depth+1, max_depth)
    
    find_trajectory_fields(data)
    
    # Show sample structure for first scene if it's a dict of scenes
    print(f"\n{'='*80}")
    print("Sample scene structure (if applicable)...")
    print(f"{'='*80}\n")
    
    if isinstance(data, dict):
        if 'infos' in data:
            infos = data['infos']
            if isinstance(infos, dict) and len(infos) > 0:
                first_scene_key = list(infos.keys())[0]
                print(f"First scene key: {first_scene_key}")
                first_scene = infos[first_scene_key]
                print(f"First scene type: {type(first_scene).__name__}")
                
                if isinstance(first_scene, list) and len(first_scene) > 0:
                    print(f"Number of frames: {len(first_scene)}")
                    print(f"\nFirst frame structure:")
                    first_frame = first_scene[0]
                    if isinstance(first_frame, dict):
                        print(f"  Keys: {list(first_frame.keys())}")
                        for key, val in list(first_frame.items())[:15]:
                            val_type = type(val).__name__
                            val_str = str(val)
                            if hasattr(val, 'shape'):
                                val_str = f"shape={val.shape}, dtype={getattr(val, 'dtype', 'unknown')}"
                            elif len(val_str) > 80:
                                val_str = val_str[:80] + "..."
                            print(f"    {key}: {val_type} = {val_str}")
                    
                    if len(first_scene) > 1:
                        print(f"\nSecond frame keys (for comparison):")
                        second_frame = first_scene[1]
                        if isinstance(second_frame, dict):
                            print(f"  Keys: {list(second_frame.keys())}")

if __name__ == "__main__":
    base_path = '/data/dataset/waymo_perception_v1_4_0/Occupancy3D-Waymo'
    
    # Explore validation pickle
    val_pkl = os.path.join(base_path, 'waymo_infos_val.pkl')
    if os.path.exists(val_pkl):
        explore_pickle(val_pkl, max_depth=4, max_items=10)
    
    print("\n\n")
    
    # Explore training pickle (smaller sample)
    train_pkl = os.path.join(base_path, 'waymo_infos_train.pkl')
    if os.path.exists(train_pkl):
        print("Note: Training pickle is large, showing structure only...")
        explore_pickle(train_pkl, max_depth=3, max_items=5)







