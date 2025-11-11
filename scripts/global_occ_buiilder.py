import os, pickle
import numpy as np
import argparse
from pyquaternion import Quaternion
from rich.progress import track

def vox_idx_to_ego(idx):
    """
    idx: (N, 3) int, 假设 occ.shape = (X, Y, Z)，顺序 (x_idx, y_idx, z_idx)
    返回: (N, 3) ego 坐标 (x,y,z)
    """
    x_idx, y_idx, z_idx = idx[:, 0], idx[:, 1], idx[:, 2]

    x = x_min + (x_idx + 0.5) * vx
    y = y_min + (y_idx + 0.5) * vy
    z = z_min + (z_idx + 0.5) * vz
    return np.stack([x, y, z], axis=-1)

def ego_to_global(pts_ego, T_ego2global):
    N = pts_ego.shape[0]
    pts_h = np.concatenate([pts_ego, np.ones((N, 1), dtype=pts_ego.dtype)], axis=-1)
    pts_g = (T_ego2global @ pts_h.T).T
    return pts_g[:, :3]

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--dataset', type=str, default=None, help='specify the dataset for fg remove')
    parser.add_argument('--start_dir', type=str, default=None, help='a path for dataset need to be preprocessed')
    parser.add_argument('--write_dir', type=str, default=None, help='a path for dataset processed')
    return parser.parse_args()

if __name__ == "__main__":

    args = parse_config()

    if args.dataset == "nuscenes":
        root_data_path = '/data/dataset/nuscenes'
        pkl_files_train = os.path.join(root_data_path, 'nuscenes_infos_train_temporal_v3_scene.pkl')
        pkl_files_val = os.path.join(root_data_path, 'nuscenes_infos_val_temporal_v3_scene.pkl')

        POINT_CLOUD_RANGE = np.array([-40, -40, -1, 40, 40, 5.4])
        VOXEL_SIZE = np.array([0.4, 0.4, 0.4])  # [vx, vy, vz]
        x_voxel_size, y_voxel_size, z_voxel_size = 200, 200, 16
        x_min, y_min, z_min, x_max, y_max, z_max = POINT_CLOUD_RANGE
        vx, vy, vz = VOXEL_SIZE
    else:
        raise NotImplementedError

    start_path = args.start_dir
    write_path = args.write_dir

    with open(pkl_files_train, 'rb') as f:
        nuscenes_infos_train = pickle.load(f)
    with open(pkl_files_val, 'rb') as f:
        nuscenes_infos_val = pickle.load(f)
    nuscenes_infos_val['infos'].update(nuscenes_infos_train['infos'])
    all_infos = nuscenes_infos_val['infos']

    for seq_idx, (seq_name, seq_info) in enumerate(track(all_infos.items())):

        scope_xyz = []
        all_scenes = []
        ego2global = []
        frame_path_save = []

        for frame in seq_info:
            frame_path = os.path.join(start_path, seq_name, frame['token'], 'labels.npz')
            frame_path_save.append(os.path.join(seq_name, frame['token']))

            occ = np.load(frame_path)['semantics']

            t = np.array(frame['ego2global_translation'])  # [tx, ty, tz]
            q = Quaternion(frame['ego2global_rotation'])  # nuScenes: (w, x, y, z)
            R = q.rotation_matrix

            T_ego2global = np.eye(4)
            T_ego2global[:3, :3] = R
            T_ego2global[:3, 3] = t

            idx = np.argwhere(occ != 9)
            semantic_label = occ[idx[:, 0], idx[:, 1], idx[:, 2]]

            pts_ego = vox_idx_to_ego(idx)  # (N,3)
            pts_g = ego_to_global(pts_ego, T_ego2global).astype(np.float32)  # (N,3)
            ego2global.append(T_ego2global)

            frame_x_min, frame_y_min, frame_z_min, frame_x_max, frame_y_max, frame_z_max \
                = np.min(pts_g[:, 0]), np.min(pts_g[:, 1]), np.min(pts_g[:, 2]), np.max(pts_g[:, 0]), np.max(
                pts_g[:, 1]), np.max(pts_g[:, 2])
            scope_xyz.append(np.asarray([frame_x_min, frame_y_min, frame_z_min, frame_x_max, frame_y_max, frame_z_max]))

            scene_preserve = np.concatenate((pts_g, semantic_label.reshape(-1, 1)), axis=-1)
            all_scenes.append(scene_preserve)

        # stack every frame to form a sequence
        scope_xyz = np.array(scope_xyz)
        seq_x_min, seq_y_min, seq_z_min, seq_x_max, seq_y_max, seq_z_max \
            = np.min(scope_xyz[:, 0]), np.min(scope_xyz[:, 1]), np.min(scope_xyz[:, 2]), np.max(
            scope_xyz[:, 3]), np.max(scope_xyz[:, 4]), np.max(scope_xyz[:, 5])

        origin = np.array([
            np.floor(seq_x_min / vx) * vx,
            np.floor(seq_y_min / vy) * vy,
            np.floor(seq_z_min / vz) * vz,
        ], dtype=np.float32)  # 这里还是可以是负数

        extent = np.array([seq_x_max, seq_y_max, seq_z_max], dtype=np.float32) - origin
        size = np.ceil(extent / VOXEL_SIZE).astype(int)
        Nx, Ny, Nz = size.tolist()
        global_counts = np.ones((Nx, Ny, Nz), dtype=np.uint8) * 9  # [Nx, Ny, Nz]

        all_scenes = np.concatenate(all_scenes)
        rel = all_scenes[:, :-1] - origin[None, :]
        ix = np.floor(rel[:, 0] / vx).astype(int)
        iy = np.floor(rel[:, 1] / vy).astype(int)
        iz = np.floor(rel[:, 2] / vz).astype(int)
        global_counts[ix, iy, iz] = all_scenes[:, -1].astype(int)

        gx, gy, gz = np.meshgrid(np.arange(x_voxel_size), np.arange(y_voxel_size), np.arange(z_voxel_size), indexing='ij')
        local_idx = np.stack([gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)], axis=-1)  # (N_vox, 3)
        pts_ego_local = vox_idx_to_ego(local_idx).astype(np.float32)
        fused_seq = []

        for frame_wise_e2g in ego2global:
            T_ego2global = frame_wise_e2g.astype(np.float32)

            # ego -> world
            pts_g = ego_to_global(pts_ego_local, T_ego2global)  # (N_vox, 3)

            # world -> global voxel index
            rel = pts_g - origin[None, :]  # (N_vox, 3)
            ix_g = np.floor(rel[:, 0] / vx).astype(int)
            iy_g = np.floor(rel[:, 1] / vy).astype(int)
            iz_g = np.floor(rel[:, 2] / vz).astype(int)

            ix_g = np.clip(ix_g, 0, Nx - 1)
            iy_g = np.clip(iy_g, 0, Ny - 1)
            iz_g = np.clip(iz_g, 0, Nz - 1)

            frame_occ = global_counts[ix_g, iy_g, iz_g].reshape(x_voxel_size, y_voxel_size, z_voxel_size)
            fused_seq.append(frame_occ)

        for path, fused_frame in zip(frame_path_save, fused_seq):
            write_path_full = os.path.join(write_path, path)  # .../scene-0003/<token>
            writedback_dir = os.path.join(write_path_full, 'labels.npz')  # .../scene-0003/<token>/labels.npz
            os.makedirs(os.path.dirname(writedback_dir), exist_ok=True)  # 建到 token 这一层
            np.savez(writedback_dir, semantics=fused_frame)
