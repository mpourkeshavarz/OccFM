import numpy as np
import torch.utils.data as torch_data

class DatasetTemplate(torch_data.Dataset):
    def __init__(self, dataset_cfg, training):
        super().__init__()
        self.training = training
        self.sequence_length = dataset_cfg.sequence_length
        self.win_size = dataset_cfg.get('win_size', 1)
        self.valid_idx, self.all_samples, self.traj = [], [], []

    def __len__(self):
        return len(self.valid_idx[:20])

    def select_valid(self, training, vae_training=False):
        self.valid_idx = []
        self.safe_length = self.sequence_length + getattr(self, 'hist_last', self.sequence_length) if not vae_training else self.sequence_length
        scenes_list = [x[0].split('/')[3] if isinstance(x, list) else x.split('/')[3] for x in self.all_samples]
        for idx, scene in enumerate(scenes_list):
            sub_seq = scenes_list[idx: idx + self.safe_length]
            if len(set(sub_seq)) == 1 and len(sub_seq) == self.safe_length:
                self.valid_idx.append(idx)
        self.valid_idx = self.valid_idx[::self.win_size] if training else self.valid_idx

    @staticmethod
    def collate_batch(data_list, _unused=False):
        batch_dict = {}
        keys = data_list[0].keys()
        for key in keys:
            batch_dict[key] = []

        for data in data_list:
            for k, v in data.items():
                batch_dict[k].append(v)

        for key, value in batch_dict.items():
            if key in ['paths']:
                batch_dict[key] = [x for x in batch_dict[key]]
            elif key in ['trajectory']:

                traj = batch_dict[key]
                all_same_shape = all(t.shape == traj[0].shape for t in traj)
                if not all_same_shape:
                    print()

                batch_dict[key] = np.stack([x for x in batch_dict[key]])
            elif key in ['semantic_occ', 'x_sampled']:
                batch_dict[key] = np.stack([x for x in batch_dict[key]])
            else:
                raise KeyError(key)

        return batch_dict