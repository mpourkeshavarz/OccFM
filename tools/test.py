import torch
import numpy as np
import time
from rich.console import Group

from tools.common_utils.display_utils import format_disp_dict
from tools.common_utils.common_utils import accumulate_disp_dict
from forecast.utils.eval_utils import multi_step_MeanIou


def setup_occ_comparsion(label_name, frame):

    unique_labels = np.asarray([x for x in range(17)])  # 17 stand for empty
    unique_label_str = [label_name[l] for l in unique_labels]
    IoU_counter = multi_step_MeanIou([1], -100, ['occupied'], 'vox', times=frame)
    IoU_counter.reset()
    mIoU_counter = multi_step_MeanIou(unique_labels, -100, unique_label_str, 'sem', times=frame)
    mIoU_counter.reset()
    return IoU_counter, mIoU_counter


def val_model(model, val_loader, model_func, progress, console_live, use_amp=False, eval_iou=False, eval_fps=False):

    val_loss_all, val_fps = [], []
    label_name = val_loader.dataset.label_name
    model.eval()
    model.count_fps = eval_fps

    val_task = progress.add_task(description="Eval samples", total=len(val_loader))
    IoU_counter, mIoU_counter = setup_occ_comparsion(label_name, val_loader.dataset.sequence_length)

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):

            with torch.amp.autocast('cuda', enabled=use_amp):

                val_loss, tb_dict, val_disp_dict = model_func(model, batch)
                val_loss_all.append(tb_dict)

                if eval_fps:
                    val_fps.append(val_disp_dict['time'])

                if eval_iou:
                    pred_occ, gt_occ = val_disp_dict['pred_occ'].detach().cpu(), \
                        val_disp_dict['gt_occ'].detach().cpu()
                    if val_loader.dataset.sem_mode:
                        mIoU_counter._after_step(pred_occ, gt_occ)
                        pred_occ[pred_occ != len(label_name)] = 1
                        pred_occ[pred_occ == len(label_name)] = 0
                        gt_occ[gt_occ != len(label_name)] = 1
                        gt_occ[gt_occ == len(label_name)] = 0
                    IoU_counter._after_step(pred_occ, gt_occ)

            progress.update(val_task, advance=1)
            disp_table = format_disp_dict(tb_dict)
            console_live.update(Group(progress, disp_table))

    console_live.update(Group(progress))
    progress.remove_task(val_task)
    avg_dict = accumulate_disp_dict(val_loss_all)

    avg_dict['all_miou'] = mIoU_counter._after_epoch()
    avg_dict['all_iou'] = IoU_counter._after_epoch()
    avg_dict['time'] = np.mean(val_fps)

    model.count_fps = False
    return avg_dict

if __name__ == "__main__":
    print()