import torch
import numpy as np
import os
from rich.console import Group
import torch.distributed as dist

from tools.common_utils.display_utils import format_disp_dict
from tools.common_utils.common_utils import accumulate_disp_dict
from forecast.utils.eval_utils import multi_step_MeanIou, DistributedDictMeanCounter


def setup_occ_comparsion(label_name, frame, rank):

    unique_labels = np.asarray([x for x in range(len(label_name))])
    unique_label_str = [label_name[l] for l in unique_labels]
    IoU_counter = multi_step_MeanIou([1], -100, ['occupied'], 'vox', times=frame, rank=rank)
    IoU_counter.reset()
    mIoU_counter = multi_step_MeanIou(unique_labels, -100, unique_label_str, 'sem', times=frame, rank=rank)
    mIoU_counter.reset()
    return IoU_counter, mIoU_counter


def val_model(model, val_loader, model_func, progress, console_live, use_amp=False, eval_iou=False,
              eval_fps=False, is_main_process=None, rank=None, test_cfm=False, teach_forcing=True,
              fid_eval_path=None):

    label_name = val_loader.dataset.label_name
    cond_length = val_loader.dataset.hist_length
    roll_out_step = val_loader.dataset.roll_out_step

    if test_cfm:
        iter_num = val_loader.dataset.roll_out_length // roll_out_step # should be number rollout now
    else:
        iter_num = val_loader.dataset.forecast_length // roll_out_step

    if val_loader.dataset.roll_out_length > val_loader.dataset.iou_eval_length:
        assert fid_eval_path is not None, "fid eval path should be given if roll_out_length > iou_eval_length"
        os.makedirs(os.path.dirname(fid_eval_path), exist_ok=True)

    model.eval()

    if is_main_process:
        val_task = progress.add_task(description="Eval samples", total=len(val_loader))

    iou_eval_length = val_loader.dataset.iou_eval_length
    IoU_counter, mIoU_counter = setup_occ_comparsion(label_name, iou_eval_length, rank)
    metrics_mean_counter = DistributedDictMeanCounter(rank)

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            with torch.amp.autocast('cuda', enabled=use_amp, dtype=torch.bfloat16):
                batch['eval_fps'] = eval_fps
                batch['cfm_eval'] = test_cfm
                batch['cond_length'] = cond_length

                all_trajectorys = batch['trajectory']
                all_paths = batch['paths']

                input_name = 'x_sampled' if 'x_sampled' in batch.keys() else 'semantic_occ'
                all_x_samples_gt = batch[input_name]

                batch[input_name] = all_x_samples_gt[:, :cond_length + roll_out_step]

                pred_occs_near = []
                pred_occs_all = []
                for iter_idx in range(iter_num):

                    start_idx = iter_idx * roll_out_step
                    end_idx = start_idx + cond_length + roll_out_step

                    # always ground truth trajectory and gt path
                    batch['trajectory'] = all_trajectorys[:, start_idx:end_idx]
                    batch['paths'] = [x[start_idx:end_idx] for x in all_paths]

                    val_loss, tb_dict, val_disp_dict = model_func(model, batch)

                    if eval_fps:
                        tb_dict["time"] = val_disp_dict["time"]

                    metrics_mean_counter.update(tb_dict)

                    if eval_iou:
                        pred_occs_all.append(val_disp_dict['pred_occ'].detach().cpu())
                        if iter_idx < iou_eval_length:
                            pred_occs_near.append(val_disp_dict['pred_occ'].detach().cpu())

                    # for vae model, here the ground truth will be replaced by forecasted results
                    # it's ok so far since there is only 1 time inference for vae training
                    if val_loader.dataset.gen_training and iter_idx < iter_num - 1:
                        if teach_forcing and not test_cfm: # no teach force during test
                            next_start = (iter_idx + 1) * roll_out_step
                            next_end = next_start + cond_length + roll_out_step
                            batch[input_name] = all_x_samples_gt[:, next_start:next_end]
                        else:
                            # replace last frame of condition with forecasted one, padding with 0.
                            # last frame will be treated as noise
                            ori_cond = batch[input_name][:, roll_out_step:roll_out_step + cond_length - 1].detach()
                            forecasted_frame = val_disp_dict['future_seq']
                            batch[input_name] = torch.cat((ori_cond, forecasted_frame,
                                                           torch.zeros_like(forecasted_frame)), dim=1)
                # generation end here, save & compare
                if eval_iou:
                    if 'gt_occ' not in val_disp_dict:
                        all_seq_gtocc_path = all_paths[0][cond_length:cond_length + iou_eval_length]
                        # Load ground truth - use same key as dataset (voxel_label) or fallback to semantics
                        npz_label_key = getattr(val_loader.dataset, 'npz_label_key', 'voxel_label')
                        gt_occ_list = []
                        missing_files = []
                        for x in all_seq_gtocc_path:
                            file_path = x[0] if isinstance(x, list) else x
                            # Resolve relative paths - check if file exists, if not try resolving relative to data_path
                            if not os.path.exists(file_path):
                                # Try resolving relative to dataset's data_path
                                dataset_cfg = getattr(val_loader.dataset, 'dataset_cfg', None)
                                if dataset_cfg and 'data_path' in dataset_cfg:
                                    data_path = dataset_cfg['data_path']
                                    # Extract relative part from path (e.g., 'validation/000/010_04.npz')
                                    if 'validation' in file_path or 'training' in file_path:
                                        parts = file_path.split(os.sep)
                                        try:
                                            split_idx = next(i for i, p in enumerate(parts) if p in ['validation', 'training'])
                                            relative_path = os.sep.join(parts[split_idx:])
                                            resolved_path = os.path.join(data_path, relative_path)
                                            if os.path.exists(resolved_path):
                                                file_path = resolved_path
                                        except (StopIteration, ValueError):
                                            pass
                            
                            # If file still doesn't exist, skip IoU evaluation for this batch
                            if not os.path.exists(file_path):
                                missing_files.append(file_path)
                                continue
                            
                            try:
                                labels = np.load(file_path)[npz_label_key].copy()
                                # Remap free label from 23 (ground truth format) to 15 (model format)
                                free_label_id = getattr(val_loader.dataset, 'free_label_id', 15)
                                if free_label_id == 15:
                                    labels[labels == 23] = 15
                                gt_occ_list.append(labels)
                            except Exception as e:
                                if is_main_process and batch_idx == 0:
                                    print(f"[WARNING] Failed to load {file_path}: {e}. Skipping IoU evaluation for this batch.")
                                missing_files.append(file_path)
                                continue
                        
                        # If we couldn't load any ground truth files, skip IoU evaluation
                        if len(gt_occ_list) == 0:
                            if is_main_process and batch_idx == 0:
                                print(f"[WARNING] Could not load ground truth files for IoU evaluation.")
                                print(f"[WARNING] Missing files: {missing_files[:3]}..." if len(missing_files) > 3 else f"[WARNING] Missing files: {missing_files}")
                                print(f"[WARNING] Skipping IoU evaluation. This is expected when using cached latents without original .npz files.")
                        else:
                            # If we loaded some but not all, warn but continue with what we have
                            if len(gt_occ_list) < len(all_seq_gtocc_path):
                                if is_main_process and batch_idx == 0:
                                    print(f"[WARNING] Loaded {len(gt_occ_list)}/{len(all_seq_gtocc_path)} ground truth files. Some files missing.")
                            
                            gt_occ = torch.as_tensor(np.stack(gt_occ_list)).unsqueeze(0)
                            pred_occ = torch.concat(pred_occs_near, dim=1)

                            if val_loader.dataset.sem_mode:
                                mIoU_counter._after_step(pred_occ, gt_occ)

                                # assume empty is the last label
                                # Use free_label_id if available, otherwise len(label_name) - 1
                                free_label_id = getattr(val_loader.dataset, 'free_label_id', len(label_name) - 1)
                                pred_occ[pred_occ != free_label_id] = 1
                                pred_occ[pred_occ == free_label_id] = 0
                                gt_occ[gt_occ != free_label_id] = 1
                                gt_occ[gt_occ == free_label_id] = 0

                            IoU_counter._after_step(pred_occ, gt_occ)
                    else:
                        gt_occ = val_disp_dict['gt_occ'].detach().cpu()
                        pred_occ = torch.concat(pred_occs_near, dim=1)

                        if val_loader.dataset.sem_mode:
                            mIoU_counter._after_step(pred_occ, gt_occ)

                            # assume empty is the last label
                            # Use free_label_id if available, otherwise len(label_name) - 1
                            free_label_id = getattr(val_loader.dataset, 'free_label_id', len(label_name) - 1)
                            pred_occ[pred_occ != free_label_id] = 1
                            pred_occ[pred_occ == free_label_id] = 0
                            gt_occ[gt_occ != free_label_id] = 1
                            gt_occ[gt_occ == free_label_id] = 0

                        IoU_counter._after_step(pred_occ, gt_occ)
                    #all_miou, cate_miou = mIoU_counter._after_epoch()
                    #all_iou, cate_iou = IoU_counter._after_epoch()
                    #print()

            dist.barrier()

            if len(pred_occs_all) > 0 and fid_eval_path is not None:
                pred_occs_all = torch.concat(pred_occs_all, dim=1).squeeze(0).numpy().astype(np.uint8)
                # Load ground truth - use same key as dataset (voxel_label) or fallback to semantics
                npz_label_key = getattr(val_loader.dataset, 'npz_label_key', 'voxel_label')
                gt_occ_list = []
                for x in all_paths[0][cond_length:]:
                    file_path = x[0] if isinstance(x, list) else x
                    labels = np.load(file_path)[npz_label_key].copy()
                    # Remap free label from 23 (ground truth format) to 15 (model format)
                    free_label_id = getattr(val_loader.dataset, 'free_label_id', 15)
                    if free_label_id == 15:
                        labels[labels == 23] = 15
                    gt_occ_list.append(labels)
                gt_occ_all = np.stack(gt_occ_list)
                np.save(fid_eval_path + f'/gt_{str(batch_idx).zfill(4)}.npy', gt_occ_all)
                np.save(fid_eval_path + f'/pred_{str(batch_idx).zfill(4)}.npy', pred_occs_all)

            if is_main_process:
                progress.update(val_task, advance=1)
                console_live.update(Group(progress, format_disp_dict(tb_dict)))

    dist.barrier(device_ids=[rank])
    all_miou, cate_miou = mIoU_counter._after_epoch()
    all_iou, cate_iou = IoU_counter._after_epoch()
    avg_dict = metrics_mean_counter.compute()

    if is_main_process:
        console_live.update(Group(progress))
        progress.remove_task(val_task)

        avg_dict["all_miou"] = all_miou
        avg_dict["all_iou"] = all_iou

        if val_loader.dataset.sem_mode:
            cate_miou = np.mean(cate_miou, axis=0)
            cate_miou = dict(zip(label_name, cate_miou*100))
            avg_dict['cate_miou'] = cate_miou

    else:
        avg_dict = {}
    return avg_dict

if __name__ == "__main__":
    print()