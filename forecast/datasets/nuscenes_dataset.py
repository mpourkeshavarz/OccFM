import os
import pickle
import numpy as np

from .dataset import DatasetTemplate
from nuscenes.nuscenes import NuScenes

class NuScenesDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, batch_size, training, cache_mode):
        super().__init__(dataset_cfg, training)

        if not cache_mode:
            self.nuSc_context_manager = NuScenes(version='v1.0-trainval', dataroot=dataset_cfg['data_path'])

            pickle_path = dataset_cfg['info_path']['train' if training else 'test'][0]
            pickle_path = os.path.join(dataset_cfg['data_path'], pickle_path)
            with open(pickle_path, 'rb') as f:
                self.infos = pickle.load(f)['infos']

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


    def __getitem__(self, idx):
        sample_idx = self.valid_idx[idx]
        paths = self.all_samples[sample_idx - self.sequence_length: sample_idx]

        data_dict = {
            'paths': self.all_samples[sample_idx - self.sequence_length: sample_idx],
            'trajectory': self.traj[sample_idx - self.sequence_length: sample_idx],
            'semantic_occ': [np.load(path)['semantics'] for path in paths] if len(paths) > 1 else np.load(paths[0])['semantics']
        }
        return data_dict
