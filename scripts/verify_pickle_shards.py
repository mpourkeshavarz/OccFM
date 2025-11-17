#!/usr/bin/env python3
"""
Script to verify pickle shard files are complete and not corrupted.
"""

import pickle
import os
import sys

def verify_pickle_file(pkl_path):
    """Try to load a pickle file and report if it's corrupted."""
    if not os.path.exists(pkl_path):
        print(f"❌ File not found: {pkl_path}")
        return False
    
    file_size = os.path.getsize(pkl_path)
    print(f"Checking: {os.path.basename(pkl_path)} ({file_size / (1024**3):.2f} GB)")
    
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        if isinstance(data, list):
            print(f"  ✓ Successfully loaded {len(data)} samples")
        elif isinstance(data, dict):
            print(f"  ✓ Successfully loaded dictionary with keys: {list(data.keys())}")
        else:
            print(f"  ✓ Successfully loaded {type(data).__name__}")
        
        return True
    except (pickle.UnpicklingError, EOFError) as e:
        print(f"  ❌ CORRUPTED: {type(e).__name__}: {str(e)}")
        return False
    except Exception as e:
        print(f"  ❌ ERROR: {type(e).__name__}: {str(e)}")
        return False

def main():
    if len(sys.argv) < 2:
        print("Usage: python verify_pickle_shards.py <shard_index_file.pkl>")
        print("\nExample:")
        print("  python verify_pickle_shards.py data/waymo_latent_vae_4shards/x16/waymo_latent_val_shards.pkl")
        sys.exit(1)
    
    index_file = sys.argv[1]
    
    if not os.path.exists(index_file):
        print(f"Error: Index file not found: {index_file}")
        sys.exit(1)
    
    print(f"\n{'='*80}")
    print(f"Verifying pickle shards from: {index_file}")
    print(f"{'='*80}\n")
    
    # Load index file
    try:
        with open(index_file, 'rb') as f:
            shard_info = pickle.load(f)
    except Exception as e:
        print(f"❌ Failed to load index file: {e}")
        sys.exit(1)
    
    if not isinstance(shard_info, dict) or 'num_shards' not in shard_info:
        print(f"❌ Invalid index file format. Expected dict with 'num_shards' key.")
        sys.exit(1)
    
    num_shards = shard_info['num_shards']
    shard_files = shard_info['shard_files']
    base_dir = os.path.dirname(index_file)
    
    print(f"Found {num_shards} shards\n")
    
    all_valid = True
    for shard_idx, shard_file in enumerate(shard_files):
        shard_path = os.path.join(base_dir, shard_file)
        is_valid = verify_pickle_file(shard_path)
        if not is_valid:
            all_valid = False
        print()
    
    print(f"{'='*80}")
    if all_valid:
        print("✓ All shard files are valid!")
    else:
        print("❌ Some shard files are corrupted!")
        print("\nTo fix this, regenerate the shards by running:")
        print("  python scripts/save_waymo_latents.py --format sharded --num_shards 4")
    print(f"{'='*80}\n")

if __name__ == '__main__':
    main()

