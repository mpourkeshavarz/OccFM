import numpy as np
import torch.utils.data as torch_data

class DatasetTemplate(torch_data.Dataset):
    def __init__(self, dataset_cfg, training, gen_training):
        super().__init__()
        self.training = training
        self.gen_training = gen_training

        # VAE only support 1 frame, used for training
        self.forecast_length = dataset_cfg.forecast_length if self.gen_training else 1
        self.hist_length = dataset_cfg.hist_length if self.gen_training else 0

        self.iou_eval_length = dataset_cfg.iou_eval_length
        self.roll_out_step = dataset_cfg.roll_out_step
        self.roll_out_length = dataset_cfg.roll_out_length

        self.only_bg = dataset_cfg.only_bg

        self.win_size = dataset_cfg.get('win_size', 1)
        self.valid_idx, self.all_samples, self.traj = [], [], []

    def __len__(self):
        return len(self.valid_idx)

    def select_valid(self, training, gen_test=False):
        self.valid_idx = []

        if gen_test:
            self.safe_length = self.iou_eval_length + self.hist_length
        else:
            self.safe_length = self.forecast_length + self.hist_length if self.gen_training else getattr(self, 'roll_out_length', 1)

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