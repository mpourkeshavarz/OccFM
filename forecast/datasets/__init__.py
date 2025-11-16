from .dataset import DatasetTemplate
from .nuscenes_dataset import NuScenesDataset
from .waymo_dataset_occ_only import WaymoDatasetOccOnly
from functools import partial

from forecast.utils import common_utils

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch
import torch.distributed as dist

__all__ = {
    'DatasetTemplate': DatasetTemplate,
    'NuScenesDataset': NuScenesDataset,
    'WaymoDatasetOccOnly': WaymoDatasetOccOnly,


}


def check_and_balance_dataset_sizes(dataset, rank, world_size, training=True):
    """
    Check dataset sizes across all ranks and balance them to the minimum size.
    This ensures perfect synchronization and prevents NCCL timeouts.
    
    Args:
        dataset: Dataset instance
        rank: Current rank
        world_size: Total number of ranks
        training: Whether this is training dataset
    
    Returns:
        dataset: Dataset with balanced size (truncated to minimum across all ranks)
    """
    if not dist.is_initialized() or world_size <= 1:
        return dataset
    
    # Get dataset size (number of valid sequences)
    dataset_size = len(dataset)
    
    # Gather sizes from all ranks
    # NCCL backend requires CUDA tensors, so we need to use the GPU
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
    size_tensor = torch.tensor([dataset_size], dtype=torch.long, device=device)
    size_list = [torch.zeros_like(size_tensor) for _ in range(world_size)]
    dist.all_gather(size_list, size_tensor)
    
    # Convert to Python list
    sizes = [int(s.item()) for s in size_list]
    min_size = min(sizes)
    max_size = max(sizes)
    avg_size = sum(sizes) / len(sizes)
    imbalance_ratio = (max_size - min_size) / max(min_size, 1)
    
    if rank == 0:
        split_name = "Training" if training else "Validation"
        print(f"\n[{split_name}] Dataset size check across {world_size} ranks:")
        print(f"  Original sizes per rank: {sizes}")
        print(f"  Min: {min_size}, Max: {max_size}, Avg: {avg_size:.1f}")
        if imbalance_ratio > 0:
            print(f"  Imbalance ratio: {imbalance_ratio:.2%}")
            print(f"  [INFO] Balancing all ranks to minimum size: {min_size} (for perfect synchronization)")
        else:
            print(f"  [OK] All ranks already have the same size: {min_size}")
    
    # Always balance by truncating to minimum size to ensure perfect synchronization
    if dataset_size > min_size:
        # Truncate valid_idx to minimum size
        if hasattr(dataset, 'valid_idx') and len(dataset.valid_idx) > min_size:
            original_size = len(dataset.valid_idx)
            dataset.valid_idx = dataset.valid_idx[:min_size]
            if rank == 0:
                print(f"  [INFO] All ranks balanced to {min_size} sequences (from {original_size} on rank with max)")
    
    # Synchronize all ranks after balancing
    dist.barrier()
    
    return dataset


def build_dataloader(dataset_cfg, batch_size, num_workers, gen_training, training=True, seed=None, rank=None, world_size=None):
    dataset = __all__[dataset_cfg.DATASET](
        dataset_cfg=common_utils.lowercase_keys(dataset_cfg), batch_size=batch_size, training=training, gen_training=gen_training
    )
    
    # Check and balance dataset sizes across ranks
    if dist.is_initialized() and world_size > 1:
        dataset = check_and_balance_dataset_sizes(dataset, rank, world_size, training=training)
    
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=training, drop_last=True)

    dataloader = DataLoader(
        dataset, batch_size=batch_size, pin_memory=True, num_workers=num_workers, collate_fn=dataset.collate_batch,
        sampler=sampler, drop_last=True, timeout=0, worker_init_fn=partial(common_utils.worker_init_fn, seed=seed)
    )
    return dataset, dataloader

def reset_batch_size(data_loader, new_batch_size, rank, world_size, training=False):
    sampler = DistributedSampler(data_loader.dataset, num_replicas=world_size,
                                 rank=rank, shuffle=training)
    return DataLoader(
        dataset=data_loader.dataset,
        batch_size=new_batch_size,
        shuffle=False, sampler=sampler,
        num_workers=data_loader.num_workers,
        pin_memory=data_loader.pin_memory,
        drop_last=False,
        collate_fn=data_loader.collate_fn,
    )