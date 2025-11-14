import os
import pickle
import numpy as np
from .dataset import DatasetTemplate

try:
    from waymo_open_dataset import dataset_pb2
    from waymo_open_dataset.utils import frame_utils
    WAYMO_AVAILABLE = True
except ImportError:
    WAYMO_AVAILABLE = False


class WaymoDatasetOccOnly(DatasetTemplate):
    def __init__(self, dataset_cfg, batch_size, training, gen_training):
        """
        Occ-only dataset for Waymo Occupancy3D:
        - Scans split directories to build ordered per-frame .npz label paths
        - Constructs ego vehicle trajectory from Waymo dataset or frame sequences
        - Returns 'paths', 'semantic_occ', and 'trajectory' per item
        """
        super().__init__(dataset_cfg, training, gen_training)

        # self.cache_mode = cache_mode
        self.sequence_length = dataset_cfg.sequence_length
        self.sem_mode = dataset_cfg.get('sem_mode', True)
        self.label_name = dataset_cfg.get('label_name', None)
        self.win_size = dataset_cfg.get('win_size', 1)
        self.npz_label_key = 'voxel_label'
        self.use_voxel_04 = dataset_cfg.get('use_voxel_04', True)
        # Waymo Occ3D classes: 0..14 are semantic, 15 is free
        self.num_classes = 16
        self.free_label_id = 15

        # Trajectory construction options
        self.waymo_tfrecord_path = dataset_cfg.get('waymo_tfrecord_path', None)
        self.trajectory_mode = dataset_cfg.get('trajectory_mode', 'pkl')  # 'pkl', 'waymo', 'frame_based', 'zero'
        self.trajectory_length = dataset_cfg.get('trajectory_length', 4)  # Number of future frames for trajectory
        
        # Load Waymo info pickle file if available
        # Note: Waymo pickle files are flat lists, not organized by scenes like NuScenes
        self.infos_list = None  # Flat list of all frames
        self.infos_by_scene = None  # Dict: scene_id -> list of frame indices in infos_list
        info_pkl_name = 'waymo_infos_train.pkl' if training else 'waymo_infos_val.pkl'
        info_pkl_path = os.path.join(dataset_cfg['data_path'], info_pkl_name)
        if os.path.exists(info_pkl_path):
            try:
                with open(info_pkl_path, 'rb') as f:
                    pkl_data = pickle.load(f)
                    
                    # Handle different formats
                    if isinstance(pkl_data, list):
                        # Flat list format (Waymo format)
                        self.infos_list = pkl_data
                        
                        # Group pickle frames by matching .npz scene structure
                        # The .npz files define the ground truth scene boundaries
                        # We match pickle frames to .npz scenes by frame count
                        self.infos_by_scene = {}
                        
                        # Get actual .npz scene structure
                        split_dirname = 'training' if training else 'validation'
                        split_path = os.path.join(dataset_cfg['data_path'], split_dirname)
                        
                        if os.path.isdir(split_path):
                            # Get scene directories and count frames per scene
                            scene_dirs = sorted([d for d in os.listdir(split_path) 
                                                if os.path.isdir(os.path.join(split_path, d))])
                            
                            # Verify pickle frames are ordered (by image_idx, which is monotonically increasing)
                            if len(self.infos_list) > 1:
                                image_indices = []
                                for item in self.infos_list[:100]:  # Check first 100
                                    if isinstance(item, dict) and 'image' in item:
                                        img_idx = item['image'].get('image_idx', None)
                                        if img_idx is not None:
                                            image_indices.append(img_idx)
                                
                                if len(image_indices) > 1:
                                    is_ordered = all(image_indices[i] <= image_indices[i+1] 
                                                    for i in range(len(image_indices)-1))
                                    if not is_ordered:
                                        print(f"Warning: Pickle frames may not be ordered by image_idx!")
                            
                            pickle_idx = 0
                            total_mapped_frames = 0
                            for scene_id in scene_dirs:
                                scene_path = os.path.join(split_path, scene_id)
                                # Count .npz files in this scene
                                npz_files = sorted([f for f in os.listdir(scene_path) 
                                                   if f.endswith('_04.npz') or (f.endswith('.npz') and not f.endswith('_04.npz'))])
                                num_frames = len(npz_files)
                                
                                if num_frames > 0 and pickle_idx < len(self.infos_list):
                                    # Map this .npz scene to pickle frames
                                    scene_key = f"scene_{int(scene_id):03d}"  # e.g., "scene_000"
                                    # Take the next num_frames from pickle file
                                    scene_frames = list(range(pickle_idx, min(pickle_idx + num_frames, len(self.infos_list))))
                                    self.infos_by_scene[scene_key] = scene_frames
                                    
                                    # Verify timestamps are continuous within scene (no large gaps)
                                    if len(scene_frames) > 1:
                                        scene_timestamps = [self.infos_list[i]['timestamp'] 
                                                          for i in scene_frames if 'timestamp' in self.infos_list[i]]
                                        if len(scene_timestamps) > 1:
                                            max_gap = max(scene_timestamps[i+1] - scene_timestamps[i] 
                                                         for i in range(len(scene_timestamps)-1))
                                            # Large gap (>1s) indicates potential mismatch
                                            if max_gap > 1e6:
                                                print(f"Warning: Scene {scene_id} has timestamp gap of {max_gap/1e6:.2f}s (may indicate mapping issue)")
                                    
                                    pickle_idx += num_frames
                                    total_mapped_frames += len(scene_frames)
                            
                            # Verify all frames are mapped
                            if total_mapped_frames != len(self.infos_list):
                                print(f"Warning: Mapped {total_mapped_frames} frames but pickle has {len(self.infos_list)} frames")
                            
                            print(f"Loaded Waymo info pickle: {info_pkl_name} ({len(self.infos_list)} frames, {len(self.infos_by_scene)} scenes matched to .npz structure)")
                        else:
                            # Fallback: use timestamp gaps if .npz structure not available
                            scene_idx = 0
                            current_scene_frames = []
                            scene_boundary_threshold = 0.5e6  # 0.5 seconds
                            
                            for idx, frame_info in enumerate(self.infos_list):
                                if not isinstance(frame_info, dict) or 'timestamp' not in frame_info:
                                    continue
                                
                                timestamp = frame_info['timestamp']
                                
                                if len(current_scene_frames) > 0:
                                    prev_idx = current_scene_frames[-1]
                                    prev_timestamp = self.infos_list[prev_idx]['timestamp']
                                    time_gap = timestamp - prev_timestamp
                                    
                                    if time_gap > scene_boundary_threshold:
                                        scene_key = f"scene_{scene_idx:03d}"
                                        self.infos_by_scene[scene_key] = current_scene_frames.copy()
                                        scene_idx += 1
                                        current_scene_frames = []
                                
                                current_scene_frames.append(idx)
                            
                            if len(current_scene_frames) > 0:
                                scene_key = f"scene_{scene_idx:03d}"
                                self.infos_by_scene[scene_key] = current_scene_frames
                            
                            print(f"Loaded Waymo info pickle: {info_pkl_name} ({len(self.infos_list)} frames, {len(self.infos_by_scene)} scenes detected from timestamp gaps)")
                    elif isinstance(pkl_data, dict) and 'infos' in pkl_data:
                        # NuScenes-like format
                        self.infos = pkl_data['infos']
                        print(f"Loaded Waymo info pickle (NuScenes format): {info_pkl_name} ({len(self.infos)} scenes)")
                    elif isinstance(pkl_data, dict):
                        # Direct dict format
                        self.infos = pkl_data
                        print(f"Loaded Waymo info pickle (dict format): {info_pkl_name}")
            except Exception as e:
                print(f"Warning: Could not load Waymo info pickle {info_pkl_path}: {e}")
                import traceback
                traceback.print_exc()
                self.infos_list = None
                self.infos_by_scene = None

        # Handle gen_training mode (for OccFM training)
        # Can use cached VAE latents (pickle_path) or encode on-the-fly (no pickle_path)
        if gen_training:
            # If pickle_path is provided, load cached latents; otherwise encode on-the-fly in model
            if dataset_cfg.get('pickle_path', None) is not None:
                path = dataset_cfg['pickle_path']['train' if training else 'test']
                with open(path, 'rb') as f:
                    cached_files = pickle.load(f)
                
                # Extract paths, trajectories, and x_sampled from cached files
                gt_path = [x['gt_path'][0] if isinstance(x['gt_path'], list) else x['gt_path'] for x in cached_files]
                
                # Match cached files to .npz structure by path
                # Waymo paths are like: .../training/scene_000/frame_000_04.npz
                # We need to match them to our all_samples structure
                split_dirname = 'training' if training else 'validation'
                base_path = dataset_cfg['data_path']
                split_path = os.path.join(base_path, split_dirname)
                assert os.path.isdir(split_path), f"Split path not found: {split_path}"
                
                # Build mapping from path to index
                scene_ids = sorted([d for d in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, d))])
                path_to_idx = {}
                for scene_id in scene_ids:
                    scene_dir = os.path.join(split_path, scene_id)
                    if self.use_voxel_04:
                        frame_files = sorted([f for f in os.listdir(scene_dir) if f.endswith('_04.npz')])
                    else:
                        frame_files = sorted([f for f in os.listdir(scene_dir) if f.endswith('.npz') and not f.endswith('_04.npz')])
                    
                    for f in frame_files:
                        full_path = os.path.join(scene_dir, f)
                        path_to_idx[full_path] = len(self.all_samples)
                        self.all_samples.append(full_path)
                
                # Match cached files to all_samples by path
                self.traj = []
                self.x_sampled = []
                matched_indices = []
                
                for cached_item in cached_files:
                    cached_path = cached_item['gt_path'][0] if isinstance(cached_item['gt_path'], list) else cached_item['gt_path']
                    # Normalize path for matching
                    if cached_path in path_to_idx:
                        matched_indices.append(path_to_idx[cached_path])
                        self.traj.append(cached_item['gt_trajs'])
                        self.x_sampled.append(cached_item['x_sampled'])
                    else:
                        # Try to match by filename
                        cached_filename = os.path.basename(cached_path)
                        matched = False
                        for full_path, idx in path_to_idx.items():
                            if os.path.basename(full_path) == cached_filename:
                                matched_indices.append(idx)
                                self.traj.append(cached_item['gt_trajs'])
                                self.x_sampled.append(cached_item['x_sampled'])
                                matched = True
                                break
                        if not matched:
                            print(f"Warning: Could not match cached path: {cached_path}")
                
                # Reorder to match all_samples order
                if len(matched_indices) == len(self.all_samples):
                    # Reorder traj and x_sampled to match all_samples order
                    reordered_traj = [None] * len(self.all_samples)
                    reordered_x_sampled = [None] * len(self.all_samples)
                    for orig_idx, new_idx in enumerate(matched_indices):
                        reordered_traj[new_idx] = self.traj[orig_idx]
                        reordered_x_sampled[new_idx] = self.x_sampled[orig_idx]
                    self.traj = reordered_traj
                    self.x_sampled = reordered_x_sampled
                
                # Select valid indices
                self.select_valid(training, gen_test=getattr(dataset_cfg, 'eval_mode', False), vae_training=False)
            else:
                # No pickle_path: will encode semantic_occ on-the-fly in OccFM model
                # Load semantic_occ from .npz files (same as VAE training mode)
                split_dirname = 'training' if training else 'validation'
                base_path = dataset_cfg['data_path']
                split_path = os.path.join(base_path, split_dirname)
                assert os.path.isdir(split_path), f"Split path not found: {split_path}"

                scene_ids = sorted([d for d in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, d))])
                self.scene_frame_mapping = {}
                
                for scene_id in scene_ids:
                    scene_dir = os.path.join(split_path, scene_id)
                    if self.use_voxel_04:
                        frame_files = sorted([f for f in os.listdir(scene_dir) if f.endswith('_04.npz')])
                    else:
                        frame_files = sorted([f for f in os.listdir(scene_dir) if f.endswith('.npz') and not f.endswith('_04.npz')])
                    
                    start_idx = len(self.all_samples)
                    self.all_samples.extend([os.path.join(scene_dir, f) for f in frame_files])
                    end_idx = len(self.all_samples)
                    self.scene_frame_mapping[scene_id] = list(range(start_idx, end_idx))

                self._infer_label_spec_from_sample()
                self._construct_trajectories()
                # Initialize x_sampled to None to indicate we'll encode on-the-fly
                self.x_sampled = None
                self.select_valid(training, gen_test=getattr(dataset_cfg, 'eval_mode', False), vae_training=False)
        else:
            # VAE training mode: load from .npz files
            # Determine split directory
            split_dirname = 'training' if training else 'validation'
            base_path = dataset_cfg['data_path']
            split_path = os.path.join(base_path, split_dirname)
            assert os.path.isdir(split_path), f"Split path not found: {split_path}"

            # Build list of label file paths per frame by scanning scenes
            # Expected structure: <base_path>/<split>/<scene_id>/<frame>_*.npz
            scene_ids = sorted([d for d in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, d))])
            self.scene_frame_mapping = {}  # Map scene_id -> list of frame indices in all_samples
            
            for scene_id in scene_ids:
                scene_dir = os.path.join(split_path, scene_id)
                if self.use_voxel_04:
                    # only use 0.4m resolution files
                    frame_files = sorted([f for f in os.listdir(scene_dir) if f.endswith('_04.npz')])
                else:
                    # only use default high-res files without _04 suffix
                    frame_files = sorted([f for f in os.listdir(scene_dir) if f.endswith('.npz') and not f.endswith('_04.npz')])
                
                start_idx = len(self.all_samples)
                # Absolute paths
                self.all_samples.extend([os.path.join(scene_dir, f) for f in frame_files])
                end_idx = len(self.all_samples)
                self.scene_frame_mapping[scene_id] = list(range(start_idx, end_idx))

            # Infer label specification (num_classes and free_label_id) from a sample if available
            self._infer_label_spec_from_sample()

            # Construct ego trajectories
            self._construct_trajectories()

            # Select valid starting indices for fixed-length sequences (stay within same scene)
            self.select_valid(training, gen_test=False, vae_training=True)

    def select_valid(self, training, gen_test=False, vae_training=False):
        self.valid_idx = []
        
        if gen_test:
            self.safe_length = self.roll_out_length + self.hist_length
        else:
            self.safe_length = self.forecast_length + self.hist_length if self.gen_training else getattr(self, 'roll_out_length', self.sequence_length)

        # scenes_list parsed from path: .../<split>/<scene>/<frame>.npz
        scenes_list = [os.path.basename(os.path.dirname(p)) for p in self.all_samples]
        for idx, scene in enumerate(scenes_list):
            sub_seq = scenes_list[idx: idx + self.safe_length]
            if len(set(sub_seq)) == 1 and len(sub_seq) == self.safe_length:
                self.valid_idx.append(idx)
        self.valid_idx = self.valid_idx[::self.win_size] if training else self.valid_idx

    def _construct_trajectories(self):
        """
        Construct ego vehicle trajectories for all frames.
        Supports four modes:
        1. 'pkl': Load from Waymo info pickle files (default, requires waymo_infos_*.pkl)
        2. 'waymo': Load from original Waymo tfrecord files (requires waymo_tfrecord_path)
        3. 'frame_based': Construct from frame sequence (assumes forward motion)
        4. 'zero': Return zero trajectories (placeholder)
        """
        self.traj = []
        
        # Try loading from pickle file first (preferred method)
        if self.trajectory_mode == 'pkl' and (self.infos_list is not None or self.infos is not None):
            self._load_trajectories_from_pkl()
            return
        
        # Fallback to tfrecord files
        if self.trajectory_mode == 'waymo' and self.waymo_tfrecord_path:
            if not WAYMO_AVAILABLE:
                print("Warning: waymo_open_dataset not available. Falling back to frame_based trajectory.")
                self.trajectory_mode = 'frame_based'
            else:
                self._load_trajectories_from_waymo()
                return
        
        # Default: construct frame-based trajectories
        for i in range(len(self.all_samples)):
            traj = self._get_trajectory_for_frame(i)
            self.traj.append(traj)

    def _load_trajectories_from_pkl(self):
        """
        Load ego trajectories from Waymo info pickle file.
        
        Waymo pickle files have a flat list structure:
        - Each item is a dict with 'pose' (4x4 matrix), 'image', 'timestamp', etc.
        - Need to group by scene and match with .npz files
        - Extract ego pose (x, y) from pose matrix and construct trajectories
        """
        # Handle flat list format (Waymo format)
        if self.infos_list is not None:
            self._load_trajectories_from_flat_list()
            return
        
        # Handle NuScenes-like format (dict of scenes)
        if self.infos is not None:
            self._load_trajectories_from_scene_dict()
            return
        
        # Fallback to frame-based
        print("Warning: Could not parse pickle file structure, using frame-based trajectories")
        for i in range(len(self.all_samples)):
            self.traj.append(self._get_trajectory_for_frame(i))
    
    def _load_trajectories_from_flat_list(self):
        """
        Load trajectories from flat list format (Waymo format).
        Each item has 'pose' (4x4 matrix) with ego pose in global coordinates.
        """
        # Create mapping from (scene_id, frame_idx) to pickle list index
        # scene_id from .npz path, frame_idx from .npz filename
        
        # Build scene mapping: map .npz scene_id to pickle scene keys
        # .npz scenes: "000", "001", etc.
        # Pickle scenes: "image_0", "image_1", etc. (from image_path)
        scene_id_mapping = {}  # .npz scene_id -> list of pickle indices
        
        # Group pickle items by scene (from image_path)
        for pickle_idx, frame_info in enumerate(self.infos_list):
            if not isinstance(frame_info, dict) or 'image' not in frame_info:
                continue
            
            img_path = frame_info['image'].get('image_path', '')
            parts = img_path.split('/')
            if len(parts) >= 2:
                pickle_scene_key = parts[-2]  # e.g., "image_0"
                
                # Try to map to .npz scene_id
                # Strategy: match by order or by trying to find pattern
                # For now, we'll match by index in the scene list
                if pickle_scene_key not in scene_id_mapping:
                    scene_id_mapping[pickle_scene_key] = []
                scene_id_mapping[pickle_scene_key].append(pickle_idx)
        
        # Now match .npz files to pickle items and construct trajectories
        for i, path in enumerate(self.all_samples):
            scene_id = os.path.basename(os.path.dirname(path))  # e.g., "000"
            frame_name = os.path.basename(path).replace('_04.npz', '').replace('.npz', '')  # e.g., "000"
            
            # Try to find matching pickle item
            # Strategy 1: Match by scene order and frame index
            pickle_idx = None
            scene_frames = self.scene_frame_mapping.get(scene_id, [])
            local_frame_idx = scene_frames.index(i) if i in scene_frames else -1
            
            # Find corresponding pickle scene (match by scene index)
            # .npz scene "000" -> pickle scene "scene_000" (index 0)
            scene_idx = int(scene_id) if scene_id.isdigit() else None
            if scene_idx is not None:
                # Match .npz scene index to pickle scene
                pickle_scene_key = f"scene_{scene_idx:03d}"  # e.g., "scene_000"
                if pickle_scene_key in self.infos_by_scene:
                    pickle_scene_indices = self.infos_by_scene[pickle_scene_key]
                
                if local_frame_idx >= 0 and local_frame_idx < len(pickle_scene_indices):
                    pickle_idx = pickle_scene_indices[local_frame_idx]
            
            # Extract pose and construct trajectory
            if pickle_idx is not None and pickle_idx < len(self.infos_list):
                frame_info = self.infos_list[pickle_idx]
                
                # Extract ego pose from 4x4 transformation matrix
                if 'pose' in frame_info:
                    pose_matrix = frame_info['pose']
                    if isinstance(pose_matrix, np.ndarray) and pose_matrix.shape == (4, 4):
                        current_x, current_y = pose_matrix[0, 3], pose_matrix[1, 3]
                        current_pose = np.array([current_x, current_y])
                        
                        # Get future frames' poses from the same scene
                        traj = []
                        for j in range(self.trajectory_length):
                            future_local_idx = local_frame_idx + j + 1
                            if future_local_idx < len(pickle_scene_indices):
                                future_pickle_idx = pickle_scene_indices[future_local_idx]
                                if future_pickle_idx < len(self.infos_list):
                                    future_frame_info = self.infos_list[future_pickle_idx]
                                    if 'pose' in future_frame_info:
                                        future_pose_matrix = future_frame_info['pose']
                                        if isinstance(future_pose_matrix, np.ndarray) and future_pose_matrix.shape == (4, 4):
                                            future_x, future_y = future_pose_matrix[0, 3], future_pose_matrix[1, 3]
                                            future_pose = np.array([future_x, future_y])
                                            # Relative position from current frame
                                            rel_pos = future_pose - current_pose
                                            traj.append(rel_pos)
                                            continue
                            
                            # Pad if future frame not available
                            if len(traj) > 0:
                                traj.append(traj[-1])  # Repeat last relative position
                            else:
                                traj.append(np.array([0.0, 0.0]))
                        
                        if len(traj) > 0:
                            self.traj.append(np.array(traj, dtype=np.float32))
                            continue
            
            # Fallback: use frame-based construction
            traj = self._get_trajectory_for_frame(i)
            self.traj.append(traj)
    
    def _load_trajectories_from_scene_dict(self):
        """
        Load trajectories from NuScenes-like format (dict of scenes).
        This is a fallback for if the pickle file has that structure.
        """
        # Similar to original implementation but handle pose matrix extraction
        frame_info_map = {}
        for scene_name, frame_list in self.infos.items():
            if not isinstance(frame_list, list):
                continue
            for frame_info in frame_list:
                if not isinstance(frame_info, dict):
                    continue
                frame_id = None
                for key in ['token', 'frame_id', 'frame_name', 'frame_token']:
                    if key in frame_info:
                        frame_id = frame_info[key]
                        break
                if frame_id is not None:
                    frame_info_map[(scene_name, str(frame_id))] = frame_info
        
        for i, path in enumerate(self.all_samples):
            scene_id = os.path.basename(os.path.dirname(path))
            frame_name = os.path.basename(path).replace('_04.npz', '').replace('.npz', '')
            
            frame_info = None
            if (scene_id, frame_name) in frame_info_map:
                frame_info = frame_info_map[(scene_id, frame_name)]
            elif scene_id in self.infos:
                scene_frames = self.infos[scene_id]
                if isinstance(scene_frames, list):
                    for f_info in scene_frames:
                        if isinstance(f_info, dict):
                            for key in ['token', 'frame_id', 'frame_name', 'frame_token']:
                                if key in f_info and str(f_info[key]) == frame_name:
                                    frame_info = f_info
                                    break
                            if frame_info:
                                break
            
            # Extract trajectory or construct from pose
            if frame_info is not None:
                # Check for pre-computed trajectory
                traj = None
                for key in ['gt_ego_fut_trajs', 'ego_fut_trajs', 'trajectory', 'ego_trajectory', 'gt_trajs']:
                    if key in frame_info:
                        traj_data = frame_info[key]
                        if isinstance(traj_data, list) and len(traj_data) > 0:
                            traj = np.array(traj_data[0] if isinstance(traj_data[0], (list, np.ndarray)) else traj_data)
                        elif isinstance(traj_data, np.ndarray):
                            traj = traj_data
                        elif isinstance(traj_data, (list, tuple)):
                            traj = np.array(traj_data)
                        
                        if traj is not None:
                            if len(traj.shape) == 1 and traj.shape[0] >= 2:
                                traj = traj[:self.trajectory_length * 2].reshape(-1, 2)
                            elif len(traj.shape) == 2:
                                traj = traj[:self.trajectory_length, :2]
                            break
                
                if traj is not None and traj.shape[0] > 0:
                    if traj.shape[0] < self.trajectory_length:
                        last_pos = traj[-1] if traj.shape[0] > 0 else np.array([0.0, 0.0])
                        padding = np.tile(last_pos, (self.trajectory_length - traj.shape[0], 1))
                        traj = np.concatenate([traj, padding], axis=0)
                    else:
                        traj = traj[:self.trajectory_length]
                    self.traj.append(traj.astype(np.float32))
                    continue
                
                # Construct from pose if available
                ego_pose = None
                if 'pose' in frame_info and isinstance(frame_info['pose'], np.ndarray) and frame_info['pose'].shape == (4, 4):
                    pose_matrix = frame_info['pose']
                    ego_pose = np.array([pose_matrix[0, 3], pose_matrix[1, 3]])
                else:
                    for key in ['ego2global_translation', 'ego_translation', 'ego_pose']:
                        if key in frame_info:
                            pose_data = frame_info[key]
                            if isinstance(pose_data, (list, tuple, np.ndarray)) and len(pose_data) >= 2:
                                ego_pose = np.array(pose_data[:2])
                                break
                
                if ego_pose is not None:
                    scene_frames = self.scene_frame_mapping.get(scene_id, [])
                    local_idx = scene_frames.index(i) if i in scene_frames else -1
                    
                    if local_idx >= 0 and scene_id in self.infos:
                        scene_info_list = self.infos[scene_id]
                        if isinstance(scene_info_list, list) and local_idx < len(scene_info_list):
                            traj = []
                            current_pose = ego_pose
                            
                            for j in range(self.trajectory_length):
                                future_idx = local_idx + j + 1
                                if future_idx < len(scene_info_list):
                                    future_frame_info = scene_info_list[future_idx]
                                    future_pose = None
                                    
                                    if 'pose' in future_frame_info and isinstance(future_frame_info['pose'], np.ndarray) and future_frame_info['pose'].shape == (4, 4):
                                        pose_matrix = future_frame_info['pose']
                                        future_pose = np.array([pose_matrix[0, 3], pose_matrix[1, 3]])
                                    else:
                                        for key in ['ego2global_translation', 'ego_translation', 'ego_pose']:
                                            if key in future_frame_info:
                                                pose_data = future_frame_info[key]
                                                if isinstance(pose_data, (list, tuple, np.ndarray)) and len(pose_data) >= 2:
                                                    future_pose = np.array(pose_data[:2])
                                                    break
                                    
                                    if future_pose is not None:
                                        rel_pos = future_pose - current_pose
                                        traj.append(rel_pos)
                                    else:
                                        if len(traj) > 0:
                                            traj.append(traj[-1])
                                        else:
                                            traj.append(np.array([0.0, 0.0]))
                                else:
                                    if len(traj) > 0:
                                        traj.append(traj[-1])
                                    else:
                                        traj.append(np.array([0.0, 0.0]))
                            
                            if len(traj) > 0:
                                self.traj.append(np.array(traj, dtype=np.float32))
                                continue
            
            # Fallback
            traj = self._get_trajectory_for_frame(i)
            self.traj.append(traj)

    def _load_trajectories_from_waymo(self):
        """
        Load ego poses from Waymo tfrecord files and construct trajectories.
        This requires the original Waymo Open Dataset tfrecord files.
        """
        import tensorflow as tf
        
        if not os.path.exists(self.waymo_tfrecord_path):
            print(f"Warning: Waymo tfrecord path not found: {self.waymo_tfrecord_path}")
            print("Falling back to frame_based trajectory.")
            self.trajectory_mode = 'frame_based'
            for i in range(len(self.all_samples)):
                self.traj.append(self._get_trajectory_for_frame(i))
            return
        
        # Map scene_id and frame_id to ego poses
        # This is a simplified version - you may need to adjust based on your Waymo data structure
        ego_poses = {}  # (scene_id, frame_id) -> (x, y, heading)
        
        # Load tfrecord files and extract ego poses
        # Note: This is a placeholder - actual implementation depends on your Waymo data structure
        tfrecord_files = [f for f in os.listdir(self.waymo_tfrecord_path) if f.endswith('.tfrecord')]
        
        for tfrecord_file in tfrecord_files:
            dataset = tf.data.TFRecordDataset(os.path.join(self.waymo_tfrecord_path, tfrecord_file))
            for data in dataset:
                frame = dataset_pb2.Frame()
                frame.ParseFromString(bytearray(data.numpy()))
                
                # Extract ego pose (x, y, heading)
                pose = frame.pose
                x = pose.transform.translation.x
                y = pose.transform.translation.y
                heading = np.arctan2(pose.transform.rotation.y, pose.transform.rotation.x) * 2
                
                # Map to scene and frame (you'll need to match this with your Occ3D structure)
                # This is a placeholder - adjust based on your actual data mapping
                scene_id = str(frame.context.name).split('_')[-1] if hasattr(frame.context, 'name') else 'unknown'
                frame_id = frame.timestamp_micros
                ego_poses[(scene_id, frame_id)] = np.array([x, y])
        
        # Construct trajectories from ego poses
        for i in range(len(self.all_samples)):
            path = self.all_samples[i]
            scene_id = os.path.basename(os.path.dirname(path))
            frame_name = os.path.basename(path).replace('_04.npz', '').replace('.npz', '')
            
            # Try to find matching ego pose
            # This is simplified - you may need to adjust the matching logic
            traj = self._get_trajectory_for_frame(i, ego_poses)
            self.traj.append(traj)

    def _get_trajectory_for_frame(self, frame_idx, ego_poses=None):
        """
        Get trajectory for a specific frame.
        Returns trajectory of shape (trajectory_length, 2) for (x, y) coordinates.
        
        Args:
            frame_idx: Index of the frame in all_samples
            ego_poses: Optional dict mapping (scene_id, frame_id) -> (x, y) pose
        """
        if self.trajectory_mode == 'zero':
            # Return zero trajectory
            return np.zeros((self.trajectory_length, 2), dtype=np.float32)
        
        # Extract scene and frame info
        path = self.all_samples[frame_idx]
        scene_id = os.path.basename(os.path.dirname(path))
        frame_name = os.path.basename(path).replace('_04.npz', '').replace('.npz', '')
        
        # Get frame index within scene
        scene_frames = self.scene_frame_mapping.get(scene_id, [])
        if frame_idx not in scene_frames:
            # Fallback to zero trajectory if frame not found in scene
            return np.zeros((self.trajectory_length, 2), dtype=np.float32)
        
        local_frame_idx = scene_frames.index(frame_idx)
        num_frames_in_scene = len(scene_frames)
        
        # If ego_poses are available, use them to construct trajectory
        if ego_poses is not None:
            # Try to find current frame's pose
            current_pose_key = (scene_id, frame_name)
            if current_pose_key in ego_poses:
                current_pose = ego_poses[current_pose_key]
                traj = []
                for i in range(self.trajectory_length):
                    future_local_idx = local_frame_idx + i + 1
                    if future_local_idx < num_frames_in_scene:
                        future_global_idx = scene_frames[future_local_idx]
                        future_path = self.all_samples[future_global_idx]
                        future_frame_name = os.path.basename(future_path).replace('_04.npz', '').replace('.npz', '')
                        future_pose_key = (scene_id, future_frame_name)
                        
                        if future_pose_key in ego_poses:
                            future_pose = ego_poses[future_pose_key]
                            # Relative position from current frame
                            rel_pos = future_pose - current_pose
                            traj.append(rel_pos)
                        else:
                            # Extrapolate if future pose not found
                            if len(traj) > 0:
                                # Use last known velocity
                                last_rel = traj[-1]
                                traj.append(last_rel)
                            else:
                                traj.append(np.array([0.0, 0.0]))
                    else:
                        # Pad with last known relative position
                        if len(traj) > 0:
                            traj.append(traj[-1])
                        else:
                            traj.append(np.array([0.0, 0.0]))
                return np.array(traj, dtype=np.float32)
        
        # Frame-based trajectory: construct from frame sequence (fallback)
        # Use frame index as proxy for position (assuming forward motion)
        traj = []
        for i in range(self.trajectory_length):
            future_idx = local_frame_idx + i + 1
            if future_idx < num_frames_in_scene:
                # Assume ~0.1m per frame forward motion (adjust based on your data)
                # Waymo typically samples at 10Hz, so ~1m/s average speed = 0.1m/frame
                x = float(i + 1) * 0.1  # Forward motion
                y = 0.0  # Assume straight motion (no lateral movement)
            else:
                # Pad with last known position if beyond scene
                x = float(num_frames_in_scene - local_frame_idx - 1) * 0.1 if num_frames_in_scene > local_frame_idx else 0.0
                y = 0.0
            traj.append([x, y])
        
        return np.array(traj, dtype=np.float32)

    def __getitem__(self, idx):
        sample_idx = self.valid_idx[idx]
        paths = self.all_samples[sample_idx: sample_idx + self.safe_length]
        data_dict = {
            'paths': paths
        }
        
        # Add trajectory: concatenate trajectories for all frames in the sequence
        trajectories = [self.traj[i] for i in range(sample_idx, sample_idx + self.safe_length)]
        data_dict['trajectory'] = np.concatenate(trajectories, axis=0)
        
        if self.gen_training:
            # For OccFM training: check if we have cached x_sampled or need to encode on-the-fly
            if hasattr(self, 'x_sampled') and self.x_sampled is not None:
                # Return cached x_sampled (VAE latents)
                data_dict['x_sampled'] = np.concatenate(self.x_sampled[sample_idx: sample_idx + self.safe_length])
            else:
                # No cached latents: return semantic_occ for on-the-fly encoding in model
                semantic_occ_list = []
                for p in paths:
                    labels = np.load(p)[self.npz_label_key].copy()
                    # Remap free label from 23 (ground truth format) to 15 (model format)
                    if self.free_label_id == 15:
                        labels[labels == 23] = 15
                    semantic_occ_list.append(labels)
                data_dict['semantic_occ'] = semantic_occ_list
        else:
            # For VAE training: load semantic occupancy labels from .npz files
            # Load semantic occupancy labels for each frame in the sequence (Waymo key: 'voxel_label')
            # Always return a list to match NuScenes dataset format, even for single frames
            semantic_occ_list = []
            for p in paths:
                labels = np.load(p)[self.npz_label_key].copy()
                # Remap free label from 23 (ground truth format) to 15 (model format)
                # Waymo ground truth uses label 23 for free, but model expects 15
                if self.free_label_id == 15:
                    labels[labels == 23] = 15
                semantic_occ_list.append(labels)
            data_dict['semantic_occ'] = semantic_occ_list

        return data_dict

    def _infer_label_spec_from_sample(self):
        """
        Infer label specification from sample data.
        Note: For Waymo, ground truth uses label 23 for free, but we remap it to 15.
        So we should NOT infer from raw data - use the configured values instead.
        """
        if not self.all_samples:
            return
        
        # For Waymo dataset, we know the correct specification:
        # - 15 semantic classes (0-14) + 1 free class (15) = 16 total
        # - Ground truth files use label 23 for free, but we remap to 15
        # So we should use the configured values, not infer from raw data
        if self.num_classes == 16 and self.free_label_id == 15:
            # Already correctly configured, don't override
            return
        
        # Only infer if not already set correctly
        sample_path = self.all_samples[0]
        try:
            arr = np.load(sample_path)[self.npz_label_key]
            max_id = int(arr.max())
            
            # Check if max_id is 23 (Waymo free label) - if so, use configured values
            if max_id == 23 and self.num_classes == 16 and self.free_label_id == 15:
                # This is Waymo format - use configured values, don't infer
                return
            
            # Otherwise, infer from data (for other datasets)
            inferred_num_classes = max_id + 1
            inferred_free_id = max_id
            self.num_classes = inferred_num_classes
            self.free_label_id = inferred_free_id
            # If provided label_name length mismatches, generate a generic list
            if isinstance(self.label_name, list):
                if len(self.label_name) != self.num_classes:
                    # Keep last as 'free'
                    gen_names = [f'class_{i}' for i in range(self.num_classes - 1)] + ['free']
                    self.label_name = gen_names
            else:
                gen_names = [f'class_{i}' for i in range(self.num_classes - 1)] + ['free']
                self.label_name = gen_names
        except Exception:
            # Fallback to defaults if any issue occurs
            pass


