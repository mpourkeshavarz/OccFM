# Waymo Pickle File Structure and Trajectory Construction

## Pickle File Structure

The `waymo_infos_train.pkl` and `waymo_infos_val.pkl` files have a **flat list structure**, not organized by scenes like NuScenes.

### Structure:
```python
data = [
    {
        'timestamp': int64,           # e.g., 1522688014970187
        'image': {
            'image_idx': int,         # e.g., 1000000
            'image_path': str,        # e.g., 'training/image_0/1000000.png'
            'image_shape': ndarray    # e.g., [1280, 1920]
        },
        'point_cloud': {
            'num_features': int,      # e.g., 6
            'velodyne_path': str      # e.g., 'training/velodyne/1000000.bin'
        },
        'calib': {
            'P0', 'P1', 'P2', 'P3', 'P4': ndarray,  # Camera projection matrices
            'R0_rect': ndarray,                      # Rectification matrix
            'Tr_velo_to_cam': ndarray               # LiDAR to camera transform
        },
        'pose': ndarray,              # 4x4 transformation matrix (EGO POSE!)
        'annos': {
            'name', 'bbox', 'dimensions', 'location', 'rotation_y', ...
        },
        'sweeps': []                  # Usually empty
    },
    ...
]
```

### Key Fields:

1. **`pose`**: 4x4 transformation matrix representing ego vehicle pose in global coordinates
   - `pose[:3, 3]` = translation vector `[x, y, z]` in meters
   - `pose[:3, :3]` = rotation matrix (3x3)
   - Example: `pose[0, 3] = 7213.27`, `pose[1, 3] = -1571.12` (x, y coordinates)

2. **`image['image_path']`**: Path like `'training/image_0/1000000.png'`
   - Directory part (`image_0`) might correspond to scene ID
   - Filename (`1000000`) is the image index

3. **`timestamp`**: Microsecond timestamp for temporal ordering

## Dataset Directory Structure

The Occ3D-Waymo dataset is organized as:
```
Occ3D-Waymo/
├── waymo_infos_train.pkl
├── waymo_infos_val.pkl
├── cam_infos.pkl
├── cam_infos_vali.pkl
├── training/
│   ├── 000/
│   │   ├── 000_04.npz
│   │   ├── 001_04.npz
│   │   ├── 002_04.npz
│   │   └── ...
│   ├── 001/
│   │   ├── 000_04.npz
│   │   └── ...
│   ├── ...
│   └── 797/
│       └── ...
└── validation/
    ├── 000/
    │   ├── 000_04.npz
    │   └── ...
    ├── ...
    └── 201/
        └── ...
```

### .npz File Contents

Each `*.npz` file contains:
- **`voxel_label`**: Semantic ground truth (occupancy labels 0-15)
- **`origin_voxel_state`**: LiDAR mask (indicates voxels observed by LiDAR)
- **`final_voxel_state`**: Camera mask (indicates voxels observed in current camera view)
- **`infov`**: Field of view mask (indicates voxels within camera FOV, since Waymo has 5 cameras, not 360°)

Note: `*_04.npz` files use 0.4m voxel size (200x200x16), while files without `_04` suffix use 0.1m voxel size (1600x1600x64).

## Matching with .npz Files

**Matching Strategy:**
- Scene ID from .npz path: `validation/000/000_04.npz` → scene_id = `"000"`
- Frame number from .npz filename: `000_04.npz` → frame_name = `"000"`
- Need to match with pickle item by:
  1. Finding items with matching scene (from `image_path` directory like `"image_0"`)
  2. Finding items with matching frame index (from `image_idx` or frame number)
  
The current implementation matches by:
- Scene index: `.npz` scene `"000"` → pickle scene at index 0 (e.g., `"image_0"`)
- Frame index: Frame `"000"` → local frame index 0 within the scene

## Trajectory Construction

**Important:** The pickle file does NOT contain pre-computed trajectories. We must construct them from consecutive poses.

### Method:

1. **Extract ego pose (x, y) from each frame:**
   ```python
   pose_matrix = frame_info['pose']  # 4x4 matrix
   x, y = pose_matrix[0, 3], pose_matrix[1, 3]  # Extract translation
   ```

2. **For each frame, get future frames' poses:**
   - Find consecutive frames in the same scene
   - Extract their poses
   - Compute relative positions

3. **Construct trajectory:**
   ```python
   current_pose = np.array([x_current, y_current])
   trajectory = []
   for future_frame in future_frames:
       future_pose = np.array([x_future, y_future])
       rel_pos = future_pose - current_pose  # Relative position
       trajectory.append(rel_pos)
   ```

4. **Format:** Trajectory should be `(trajectory_length, 2)` array for (x, y) coordinates
   - Each row is relative position from current frame
   - Example: `[[dx1, dy1], [dx2, dy2], [dx3, dy3], [dx4, dy4]]`

### Example:

From the exploration:
- Frame 0: x=7213.27, y=-1571.12
- Frame 1: x=7213.04, y=-1569.33
- Frame 2: x=7212.79, y=-1567.53

Relative trajectory from Frame 0:
- Frame 1: dx=-0.23, dy=1.80
- Frame 2: dx=-0.47, dy=3.59

## Implementation

The code now correctly handles the flat list structure:

1. **Loads pickle file** as a flat list of frames
2. **Groups by scene** using `image_path` directory (e.g., `"image_0"`, `"image_1"`)
3. **Matches with .npz files** by:
   - Mapping scene index: `.npz` scene `"000"` → pickle scene at index 0
   - Mapping frame index: Frame `"000"` → local frame index 0 within scene
4. **Extracts ego pose** from `pose` matrix (4x4 transformation matrix)
5. **Constructs trajectories** by computing relative positions from consecutive frames

### Trajectory Format

- Shape: `(trajectory_length, 2)` where default `trajectory_length = 4`
- Each row: `[dx, dy]` - relative position from current frame to future frame
- Coordinates: (x, y) in meters, relative to current ego position
- Example: `[[-0.23, 1.80], [-0.47, 3.59], [-0.71, 5.39], [-0.95, 7.19]]`

This format matches what the model expects (same as NuScenes dataset).


