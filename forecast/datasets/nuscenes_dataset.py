import os
import pickle
import numpy as np

from .dataset import DatasetTemplate
from nuscenes.nuscenes import NuScenes

class NuScenesDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, batch_size, training, cache_mode):
        super().__init__(dataset_cfg, training)

        self.cache_mode = cache_mode
        self.sem_mode = dataset_cfg.sem_mode
        self.label_name = dataset_cfg.label_name
        self.sequence_length = dataset_cfg.sequence_length
        self.x_sampled = None

        pickle_path = dataset_cfg['info_path']['train' if training else 'test'][0]
        pickle_path = os.path.join(dataset_cfg['data_path'], pickle_path)
        with open(pickle_path, 'rb') as f:
            self.infos = pickle.load(f)['infos']

        if not cache_mode: # used to control dataset in VAE or CFM
            self.nuSc_context_manager = NuScenes(version='v1.0-trainval', dataroot=dataset_cfg['data_path'])

            using_scenes = list(self.infos.keys())
            init_pos = 0
            for scene in self.nuSc_context_manager.scene:
                if scene["name"] not in using_scenes:
                    continue

                all_token_with_order = [self.infos[scene["name"]][i]['token'] for i in range(len(self.infos[scene["name"]]))]
                path = [dataset_cfg['data_path'] + f'/gts/{scene["name"]}/' + x + '/labels.npz' for x in all_token_with_order]
                self.valid_idx.extend(range(init_pos + 1, init_pos + len(path)))
                self.all_samples.extend(path)
                init_pos += len(path)

                info_seq = self.infos[scene["name"]]
                for token, sample in zip(all_token_with_order, info_seq):
                    self.traj.append(sample['gt_ego_fut_trajs'][0])
        else:
            assert dataset_cfg.get('pickle_path', None) is not None, "Should provide cached pickles"

            path = dataset_cfg['pickle_path']['train' if training else 'test']
            with open(path, 'rb') as f:
                cached_files = pickle.load(f)
            gt_path = [x['gt_path'][0] for x in cached_files]
            token_seq = [x.split('/')[4] for x in gt_path]
            token_ori_sort = []
            for scene_idx, value in self.infos.items():
                for frame in value:
                    token_ori_sort.append(frame['token'])
            indices = [token_seq.index(token) for token in token_ori_sort if token in token_seq]
            sorted_cache_file = [cached_files[x] for x in indices]

            self.traj = [x['gt_trajs'] for x in sorted_cache_file]
            self.all_samples = [x['gt_path'][0] for x in sorted_cache_file]
            self.x_sampled = [x['x_sampled'] for x in sorted_cache_file]
            self.valid_idx = []
            scenes_list = [x.split('/')[3] for x in self.all_samples]
            for idx, scene in enumerate(scenes_list):
                sub_seq = scenes_list[idx: idx + self.sequence_length * 2]
                if len(set(sub_seq)) == 1 and len(sub_seq) == self.sequence_length * 2:
                    self.valid_idx.append(idx)

    def __getitem__(self, idx):

        sample_idx = self.valid_idx[idx]
        if self.cache_mode:
            data_dict = {
                'paths': self.all_samples[sample_idx: sample_idx + self.sequence_length * 2],
                'trajectory': np.concat(self.traj[sample_idx: sample_idx + self.sequence_length * 2]),
                'x_sampled': np.concat(self.x_sampled[sample_idx: sample_idx + self.sequence_length * 2]),
            }

        else:
            paths = self.all_samples[sample_idx - self.sequence_length: sample_idx]
            data_dict = {
                'paths': self.all_samples[sample_idx - self.sequence_length: sample_idx],
                'trajectory': self.traj[sample_idx - self.sequence_length: sample_idx],
                'semantic_occ': [np.load(path)['semantics'] for path in paths] if len(paths) > 1 else np.load(paths[0])['semantics']
            }
        return data_dict
