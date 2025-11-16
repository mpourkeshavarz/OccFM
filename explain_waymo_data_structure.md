# Waymo Cached Latents Data Structure

## Data Format

The cached latents are stored as a **flat list of dictionaries**. Each dictionary represents one frame:

```python
cached_files = [
    {
        'x_sampled': np.array(...),  # VAE latent for this frame, shape depends on VAE config
        'gt_path': 'data/waymo_perception_v1_4_0/Occupancy3D-Waymo/training/000/000_04.npz',
        'gt_trajs': np.array([dx, dy])  # [2] displacement from this frame to next frame
    },
    {
        'x_sampled': np.array(...),
        'gt_path': 'data/waymo_perception_v1_4_0/Occupancy3D-Waymo/training/000/001_04.npz',
        'gt_trajs': np.array([dx, dy])
    },
    # ... more frames
]
```

## How Consecutive Frames Are Found

### Step 1: Extract Paths
```python
gt_path = [x['gt_path'][0] if isinstance(x['gt_path'], list) else x['gt_path'] 
           for x in cached_files]
self.all_samples = gt_path.copy()  # List of paths: ['.../training/000/000_04.npz', ...]
```

### Step 2: Extract Scene IDs from Paths
The scene ID is extracted from the directory name in the path:
```python
# Path format: .../training/<scene_id>/<frame_id>_04.npz
# Example: .../training/000/000_04.npz -> scene_id = "000"
scenes_list = [os.path.basename(os.path.dirname(p)) for p in self.all_samples]
# Result: ['000', '000', '000', ..., '001', '001', ...]
```

### Step 3: Find Valid Sequences
`select_valid` looks for sequences where `safe_length` (90) consecutive frames have the same scene ID:

```python
safe_length = hist_length + forecast_length  # 10 + 80 = 90

for idx, scene in enumerate(scenes_list):
    # Check if next 90 frames all have the same scene ID
    sub_seq = scenes_list[idx: idx + safe_length]
    if len(set(sub_seq)) == 1 and len(sub_seq) == safe_length:
        # All 90 frames are from the same scene - valid!
        self.valid_idx.append(idx)
```

### Example:
```
scenes_list = ['000', '000', '000', ..., '000', '001', '001', ...]
                ↑                                    ↑
            idx=0 (valid)                      idx=180 (invalid - scene changes)
            
For idx=0: sub_seq = ['000'] * 90  → all same scene → VALID
For idx=180: sub_seq = ['000', '001', ...] → different scenes → INVALID
```

## Why Sorting is Important

When data is sharded, frames from the same scene might be split across shards. If we only load one shard:
- Shard 0 might have: scene 000 frames 0-50, scene 005 frames 0-30, scene 010 frames 0-20
- This doesn't have 90 consecutive frames from the same scene → no valid indices!

By loading ALL shards and sorting by scene:
- All scene 000 frames come together: 000/000, 000/001, ..., 000/199
- Now we can find 90 consecutive frames from scene 000 → valid indices found!

## Data Flow

1. **Load pickle files** → `cached_files` (flat list of dicts)
2. **Extract paths** → `self.all_samples` (list of path strings)
3. **Extract scene IDs** → `scenes_list` (list of scene ID strings)
4. **Find valid sequences** → `self.valid_idx` (list of starting indices)
5. **Dataset.__getitem__(idx)** → Returns sequence starting at `self.valid_idx[idx]`

