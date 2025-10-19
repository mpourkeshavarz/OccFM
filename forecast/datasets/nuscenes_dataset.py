import os
import pickle
import numpy as np

from .dataset import DatasetTemplate
from nuscenes.nuscenes import NuScenes
from .preprocessor import Preprocessor

class NuScenesDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, batch_size, training, gen_training):
        super().__init__(dataset_cfg, training, gen_training)

        self.sem_mode = dataset_cfg.sem_mode
        self.label_name = dataset_cfg.label_name
        self.iou_eval_length = dataset_cfg.iou_eval_length
        self.roll_out_step = dataset_cfg.roll_out_step

        dataset_cfg.preprocessor['dynamic_objects'] = [dataset_cfg['label_name'].index(x) for x in dataset_cfg['dynamic_classes']]
        dataset_cfg.preprocessor['drive_area_index'] = 11
        dataset_cfg.preprocessor['fg_non_vehicle_index'] = [2, 6, 7]
        dataset_cfg.preprocessor['cate_num'] = len(self.label_name)

        preprocess_step = getattr(dataset_cfg, 'preprocess_step', [])
        self.preprocessor = Preprocessor(dataset_cfg.preprocessor, preprocess_step)

        pickle_path = dataset_cfg['info_path']['train' if training else 'test'][0]
        pickle_path = os.path.join(dataset_cfg['data_path'], pickle_path)

        with open(pickle_path, 'rb') as f:
            self.infos = pickle.load(f)['infos']

        if gen_training: # used to control dataset in VAE or CFM

            assert dataset_cfg.get('pickle_path', None) is not None, "Should provide cached pickles"

            path = dataset_cfg['pickle_path']['train' if training else 'test']
            with open(path, 'rb') as f:
                cached_files = pickle.load(f)
            gt_path = [x['gt_path'][0] for x in cached_files]
            token_seq = [x[0].split('/')[4] if isinstance(x, list) else x.split('/')[4] for x in gt_path]
            token_ori_sort = []
            for scene_idx, value in self.infos.items():
                for frame in value:
                    token_ori_sort.append(frame['token'])
            indices = [token_seq.index(token) for token in token_ori_sort if token in token_seq]
            sorted_cache_file = [cached_files[x] for x in indices]

            self.traj = [x['gt_trajs'] for x in sorted_cache_file]
            self.all_samples = [x['gt_path'][0] for x in sorted_cache_file]
            self.x_sampled = [x['x_sampled'] for x in sorted_cache_file]
            self.select_valid(training) # cache mode only during cfm training
            self.only_bg = 'fg_only' in path

        else:
            self.nuSc_context_manager = NuScenes(version='v1.0-trainval', dataroot=dataset_cfg['data_path'])

            using_scenes = list(self.infos.keys())
            init_pos = 0
            for scene in self.nuSc_context_manager.scene:
                if scene["name"] not in using_scenes:
                    continue

                all_token_with_order = [self.infos[scene["name"]][i]['token'] for i in
                                        range(len(self.infos[scene["name"]]))]
                path = [dataset_cfg['data_path'] + f'/gts/{scene["name"]}/' + x + '/labels.npz' for x in
                        all_token_with_order]
                self.all_samples.extend(path)
                init_pos += len(path)

                info_seq = self.infos[scene["name"]]
                for token, sample in zip(all_token_with_order, info_seq):
                    self.traj.append(sample['gt_ego_fut_trajs'][0])
            self.select_valid(training)

            self.only_bg = 'filter_fg' in preprocess_step  # VAE should have this step

    def __getitem__(self, idx):

        sample_idx = self.valid_idx[idx]
        data_dict = {
            'paths': self.all_samples[sample_idx: sample_idx + self.safe_length],
            'trajectory': np.concat(self.traj[sample_idx: sample_idx + self.safe_length])
        }
        if self.gen_training:
            data_dict['x_sampled'] = np.concat(self.x_sampled[sample_idx: sample_idx + self.safe_length])
        else:
            paths = data_dict['paths']
            data_dict['semantic_occ'] = [np.load(path)['semantics'] for path in paths] if len(paths) > 1 else np.load(paths[0])['semantics']

        data_dict = self.preprocessor(data_dict)
        return data_dict