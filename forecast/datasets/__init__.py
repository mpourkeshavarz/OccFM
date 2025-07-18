from .dataset import DatasetTemplate
from .nuscenes_dataset import NuScenesDataset
from functools import partial

from forecast.utils import common_utils

from torch.utils.data import DataLoader

__all__ = {
    'DatasetTemplate': DatasetTemplate,
    'NuScenesDataset': NuScenesDataset,
}


def build_dataloader(dataset_cfg, batch_size, num_workers, cache_mode, training=True, seed=None, rank=None):
    dataset = __all__[dataset_cfg.DATASET](
        dataset_cfg=common_utils.lowercase_keys(dataset_cfg), batch_size=batch_size, training=training, cache_mode=cache_mode
    )

    dataloader = DataLoader(
        dataset, batch_size=batch_size, pin_memory=True, num_workers=num_workers,
        shuffle=training, collate_fn=dataset.collate_batch,
        drop_last=False, timeout=0, worker_init_fn=partial(common_utils.worker_init_fn, seed=seed)
    )
    return dataset, dataloader

def reset_batch_size(data_loader, new_batch_size):
    return DataLoader(
        dataset=data_loader.dataset,
        batch_size=new_batch_size,
        shuffle=False,  # 注意：通常 cache 不需要 shuffle
        num_workers=data_loader.num_workers,
        pin_memory=data_loader.pin_memory,
        drop_last=False,
        collate_fn=data_loader.collate_fn,
    )