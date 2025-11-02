import numpy as np
import yaml
import os
import argparse, pickle
from scipy import ndimage
from nuscenes.nuscenes import NuScenes
from rich.progress import track


class FilterFg(object):
    def __init__(self, preprocessor_config, steps_use, **kwargs):

        self.steps_use = steps_use

        self.cluster_tolerance = preprocessor_config['cluster_tolerance']
        self.driveable_area = preprocessor_config['drive_area_index']
        self.ori_cate_num = preprocessor_config['ori_cate_num']
        self.non_vehicle_index = preprocessor_config['fg_non_vehicle_index']

        # background voxels include empty
        self.dynamic_object_idx = preprocessor_config['dynamic_objects']
        self.background_object_idx = [x for x in range(self.ori_cate_num+1) if x not in self.dynamic_object_idx]


    @staticmethod
    def mask_dynamic_voxels(sem_arr, dynamic_indices, mask_idx):
        mask = np.isin(sem_arr, dynamic_indices)
        out = sem_arr.copy()
        out[mask] = mask_idx
        return out, mask

    @staticmethod
    def find_clusters_with_tolerance(foreground_mask_3d: np.ndarray, tolerance_voxels: int = 0):
        """
        Args:
            foreground_mask_3d (np.ndarray): a bool mask, True for foreground。
            tolerance_voxels (int): 2 clusters combined if distance between 2 class below this number。

        Returns:
            list[np.ndarray]: a list of different clusters' coordinates。
        """

        dilated_mask = ndimage.binary_dilation(foreground_mask_3d, iterations=tolerance_voxels)
        labeled_dilated, num_clusters = ndimage.label(dilated_mask)

        if num_clusters == 0:
            return []

        clusters = []
        for cluster_id in range(1, num_clusters + 1):
            dilated_cluster_mask = (labeled_dilated == cluster_id)
            original_voxels_in_cluster = foreground_mask_3d & dilated_cluster_mask
            coords = np.argwhere(original_voxels_in_cluster)
            if coords.shape[0] > 0:
                clusters.append(coords)

        return clusters

    @staticmethod
    def query_sem(data, cluster, empty_index):
        non_empty_idx = np.where(data[cluster[:, 0], cluster[:, 1], :] != empty_index)
        non_empty_3d_idx = np.concatenate([cluster[non_empty_idx[0], :], non_empty_idx[1].reshape(-1, 1)], axis=1)
        sem_cate = data[non_empty_3d_idx[:, 0], non_empty_3d_idx[:, 1], non_empty_3d_idx[:, 2]]
        return sem_cate

    @staticmethod
    def get_external_neighbors(coords: np.ndarray, scene_shape=(200, 200)):
        """
        使用Numpy广播计算一组2D坐标点的所有在边界内的外部邻居。

        Args:
            coords (np.ndarray):
                一个形状为 (N, 2) 的Numpy数组，包含N个2D坐标 (y, x)。
            scene_shape (tuple[int, int], optional):
                场景的形状 (height, width)，用于限制坐标范围。
                生成的邻居坐标 (y, x) 必须满足 0 <= y < height 和 0 <= x < width。
                默认为 (200, 200)。

        Returns:
            np.ndarray: 一个形状为 (M, 2) 的Numpy数组，包含M个唯一的、在边界内的外部邻居坐标。
        """
        if coords.shape[0] == 0:
            return np.array([], dtype=coords.dtype).reshape(0, 2)

        height, width = scene_shape

        # all possible neighbours
        offsets = np.array([
            [-1, -1], [-1, 0], [-1, 1],
            [0, -1], [0, 1],
            [1, -1], [1, 0], [1, 1]
        ])
        all_neighbors_flat = (coords[:, np.newaxis, :] + offsets).reshape(-1, 2)

        # boundary condition
        valid_mask = ((all_neighbors_flat[:, 0] >= 0) & (all_neighbors_flat[:, 0] < height) &
                      (all_neighbors_flat[:, 1] >= 0) & (all_neighbors_flat[:, 1] < width))

        # 使用掩码筛选出在边界内的邻居点
        bounded_neighbors = all_neighbors_flat[valid_mask]

        # --- 步骤3：去重并移除原始输入点 ---

        # 对在边界内的邻居点进行去重
        unique_potential_neighbors = np.unique(bounded_neighbors, axis=0)

        # 使用视图技巧，高效地从邻居中移除原始输入点
        row_dtype = np.dtype((np.void, coords.dtype.itemsize * coords.shape[1]))
        coords_view = np.ascontiguousarray(coords).view(row_dtype)
        neighbors_view = np.ascontiguousarray(unique_potential_neighbors).view(row_dtype)

        external_neighbors_view = np.setdiff1d(neighbors_view, coords_view, assume_unique=True)
        external_neighbors = external_neighbors_view.view(coords.dtype).reshape(-1, coords.shape[1])

        return external_neighbors

    def filter_fg(self, data_dict):

        semantic_occ = data_dict['semantic_occ']
        removed_data, fg_mask = self.mask_dynamic_voxels(semantic_occ, self.dynamic_object_idx, self.ori_cate_num)
        fg_mask_2d = np.any(fg_mask, axis=2)

        cluster_voxels = self.find_clusters_with_tolerance(fg_mask_2d, self.cluster_tolerance)
        if len(cluster_voxels) == 0:
            return data_dict

        combined_voxels = np.concatenate(cluster_voxels, axis=0)
        pillar = (semantic_occ[combined_voxels[:, 0], combined_voxels[:, 1]] != self.ori_cate_num)
        first_indices = np.argmax(pillar, axis=1)

        split_indices = np.cumsum([x.shape[0] for x in cluster_voxels])[:-1]
        segments = np.split(first_indices, split_indices)
        cluster_ground_heights = [np.argmax(np.bincount(s)) for s in segments]

        # The semantic category
        ground_sem_cate_cluster = []
        for cluster, cluster_height in zip(cluster_voxels, segments):

            sem_cate = self.query_sem(semantic_occ, cluster, self.ori_cate_num)
            cluster_sem_list = np.bincount(sem_cate)
            cluster_sem = np.argmax(cluster_sem_list)

            while cluster_sem in self.background_object_idx:
                cluster_sem_list[cluster_sem] = 0
                cluster_sem = np.argmax(cluster_sem_list)

            assert cluster_sem in self.dynamic_object_idx, f"wrong category {cluster_sem}"
            if cluster_sem in self.non_vehicle_index:
                around_cluster = self.get_external_neighbors(cluster, scene_shape=semantic_occ.shape[:-1])
                sem_ground_around_cate = self.query_sem(semantic_occ, around_cluster, self.ori_cate_num)

                if len(sem_ground_around_cate) == 0:
                    ground_sem_cate_cluster.append(self.driveable_area)
                else:
                    ground_sem_cate_cluster.append(np.argmax(np.bincount(sem_ground_around_cate), axis=-1))
            else:
                ground_sem_cate_cluster.append(self.driveable_area)  # all vehicles should in drivable area

        loc_need_repair = np.concatenate([
            np.hstack([coords, np.full((coords.shape[0], 1), height, dtype=coords.dtype),
                       np.full((coords.shape[0], 1), sem, dtype=coords.dtype)])
            for coords, height, sem in zip(cluster_voxels, cluster_ground_heights, ground_sem_cate_cluster)
        ])

        removed_data[loc_need_repair[:, 0], loc_need_repair[:, 1], loc_need_repair[:, 2]] = loc_need_repair[:, 3]
        data_dict['semantic_occ'] = removed_data
        return data_dict

    def index_reorder(self, data_dict):

        semantic_occ = data_dict['semantic_occ']
        for new_idx, old_idx in enumerate(self.background_object_idx):
            semantic_occ[semantic_occ==old_idx] = new_idx
        data_dict['semantic_occ'] = semantic_occ[None,]
        return data_dict

    def __call__(self, data_dict):

        if len(self.steps_use) == 0:
            return data_dict

        for step in self.steps_use:
            data_dict = getattr(self, step)(data_dict)

        return data_dict

def to_lowercase_keys(d):
    if isinstance(d, dict):
        new_d = {}
        for k, v in d.items():
            new_key = k.lower() if isinstance(k, str) else k
            new_d[new_key] = to_lowercase_keys(v)
        return new_d
    elif isinstance(d, list):
        return [to_lowercase_keys(item) for item in d]
    elif isinstance(d, tuple):
        return tuple(to_lowercase_keys(item) for item in d)
    else:
        return d

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--dataset', type=str, default=None, help='specify the dataset for fg remove')
    parser.add_argument('--write_dir', type=str, default=None, help='path for new dataset')
    return parser.parse_args()

if __name__ == "__main__":


    args = parse_config()
    writeback_dir = args.write_dir
    preprocessor = {}

    if args.dataset == "nuscenes":

        cfg_file = './tools/cfgs/dataset_configs/nuscenes_dataset.yaml'

        with open(cfg_file, 'r') as f:
            dataset_cfg = to_lowercase_keys(yaml.safe_load(f))

        preprocessor['dynamic_objects'] = [dataset_cfg['label_name'].index(x) for x in dataset_cfg['dynamic_classes']]
        preprocessor['drive_area_index'] = 11
        preprocessor['fg_non_vehicle_index'] = [2, 6, 7]
        preprocessor['ori_cate_num'] = len(dataset_cfg['label_name'])
        preprocessor['cluster_tolerance'] = dataset_cfg['preprocessor']['cluster_tolerance']

        filter = FilterFg(preprocessor, steps_use=['filter_fg', 'index_reorder'])
        nuSc_context_manager = NuScenes(version='v1.0-trainval', dataroot='/data/dataset/nuscenes')
        pickle_paths = ['nuscenes_infos_train_temporal_v3_scene.pkl', 'nuscenes_infos_val_temporal_v3_scene.pkl']

        for pickle_path in pickle_paths:
            with open('/data/dataset/nuscenes/' + pickle_path, 'rb') as f:
                infos = pickle.load(f)['infos']

            using_scenes = list(infos.keys())
            all_samples = []
            for scene in nuSc_context_manager.scene:
                if scene["name"] not in using_scenes:
                    continue
                all_token_with_order = [infos[scene["name"]][i]['token'] for i in
                                        range(len(infos[scene["name"]]))]
                path = [f'/gts/{scene["name"]}/' + x + '/labels.npz' for x in
                        all_token_with_order]
                all_samples.extend(path)

            for occ_path in track(all_samples):
                readable_path = '/data/dataset/nuscenes' + occ_path
                np_file = np.load(readable_path)

                input_dict = {'semantic_occ': np_file['semantics']}
                bg_occ = filter(input_dict)['semantic_occ'][0]
                writedback_dir = writeback_dir + occ_path
                save_path = writedback_dir[:-10]

                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                np.savez(writedback_dir, semantics=bg_occ)





