import torch
from torch.nn.utils import clip_grad_norm_
from rich.live import Live
from tools.test import val_model
from rich.console import Group
from contextlib import nullcontext
from tools.common_utils.display_utils import format_disp_dict, save_eval_results_by_epoch
from torch.nn.parallel import DistributedDataParallel as DDP
from tools.common_utils.logging import wandb_log_train_step, wandb_log_val_step

import torch.distributed as dist
import os
import numpy as np
import wandb

def single_loop_training(model, model_func, batch, use_amp, scaler, optimizer, lr_scheduler, max_grad,
                         cond_length=None, ema_model=None):

    batch['cond_length'] = cond_length
    with torch.amp.autocast('cuda', enabled=use_amp):
        loss, tb_dict, disp_dict = model_func(model, batch)

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    clip_grad_norm_(model.parameters(), max_grad)
    scaler.step(optimizer)
    scaler.update()
    lr_scheduler.step()

    if ema_model is not None:
        ema_model.update_parameters(model.module)

    return tb_dict

def auto_regressive_training(model, model_func, batch, use_amp, scaler, optimizer, lr_scheduler, max_grad,
                             cond_length, roll_out_step=1, ema_model=None):

    forecast_length = batch['x_sampled'].shape[1] - cond_length

    # Process roll_out_step frames at a time instead of 1 frame at a time
    # With teacher forcing, we use ground truth, so we don't need to replace predictions
    for forecast_idx in range(0, forecast_length, roll_out_step):
        # Process cond_length + roll_out_step frames (e.g., 10 + 10 = 20 frames)
        end_idx = forecast_idx + cond_length + roll_out_step
        temp_data_dict = {'paths': [x[forecast_idx:end_idx] for x in batch['paths']],
                          'trajectory': batch['trajectory'][:, forecast_idx:end_idx],
                          'x_sampled': batch['x_sampled'][:, forecast_idx:end_idx],
                          'cond_length': cond_length}

        with torch.amp.autocast('cuda', enabled=use_amp):
            loss, tb_dict, disp_dict = model_func(model, temp_data_dict)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        clip_grad_norm_(model.parameters(), max_grad)
        scaler.step(optimizer)
        scaler.update()
        lr_scheduler.step()

        if ema_model is not None:
            ema_model.update_parameters(model.module)

    return tb_dict


def train_model(model, optimizer, train_loader, val_loader, lr_scheduler, start_epoch, optim_cfg,
                model_func, ckpt_path, console, progress, ema_model=None, wandb_logger=None,
                eval_interval=1, use_amp=True, loss_monitor=None, is_main_process=None, rank=None,
                ckpt_save_interval_batches=None, output_dir=None):
    # DDP after load parameters
    # encoder/decoder param also include but no grad
    model_update_func = auto_regressive_training if getattr(model, 'auto_regressive', False) else single_loop_training
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    train_loader.sampler.set_epoch(start_epoch)
    scaler = torch.amp.GradScaler(enabled=use_amp, init_scale=optim_cfg.get('LOSS_SCALE_FP16', 10))
    historical_losses, saved_ckpts = [], []

    if is_main_process:
        epoch_task = progress.add_task(description="Epoch", total=optim_cfg.NUM_EPOCHS,
                                       completed=start_epoch if start_epoch > 0 else start_epoch+1)

    live_ctx = Live(console=console, refresh_per_second=4, transient=True) if is_main_process else nullcontext()

    # if is_main_process and wandb_logger is not None:
    #     wandb.watch(model.module, log="gradients", log_freq=200)

    with live_ctx as live:
        if is_main_process:
            print(f"[DEBUG] train_loader length: {len(train_loader)}")
            if len(train_loader) == 0:
                print(f"[ERROR] train_loader is EMPTY! This will cause training to skip immediately to evaluation!")
        
        for epoch in range(start_epoch, optim_cfg.NUM_EPOCHS):
            # 每轮新建 step task
            if is_main_process:
                step_task = progress.add_task(description="Samples", total=len(train_loader))
                print(f"[DEBUG] Starting epoch {epoch}, train_loader has {len(train_loader)} batches")

            batch_count = 0
            for batch_idx, batch in enumerate(train_loader):
                batch_count += 1
                model.train()
                optimizer.zero_grad()

                tb_dict = model_update_func(model, model_func, batch, use_amp, scaler, optimizer, lr_scheduler,
                                            optim_cfg.GRAD_NORM_CLIP,
                                            cond_length=getattr(train_loader.dataset, 'hist_length', 0),
                                            roll_out_step=getattr(train_loader.dataset, 'roll_out_step', 1),
                                            ema_model=ema_model)

                if is_main_process:
                    progress.update(step_task, advance=1)
                    tb_dict['lr'] = optimizer.param_groups[0]['lr']
                    disp_table = format_disp_dict(tb_dict)
                    live.update(Group(progress, disp_table))
                    wandb_log_train_step(model.module.global_step, epoch, tb_dict) if wandb_logger is not None else None
                    
                    # Save checkpoint after N batches if configured
                    if ckpt_save_interval_batches is not None and (batch_idx + 1) % ckpt_save_interval_batches == 0:
                        # Synchronize before saving checkpoint to ensure all ranks are ready
                        dist.barrier()
                        current_epoch = epoch + 1
                        ckpt_file = f'epoch={str(current_epoch).zfill(6)}_batch={str(batch_idx + 1).zfill(6)}.ckpt'
                        ckpt_path_full = ckpt_path + ckpt_file
                        torch.save({
                            'state_dict': model.state_dict(),
                            'ema_model': ema_model.state_dict() if ema_model is not None else None,
                            'optimizer_states': [optimizer.state_dict()],
                            'epoch': current_epoch,
                            'scaler_state_dict': scaler.state_dict() if use_amp else None,
                            'lr_scheduler': lr_scheduler.state_dict()
                        }, ckpt_path_full)
                        console.print(f"[green]💾  Saved batch checkpoint:[/] {ckpt_file}")
                        dist.barrier()

            if is_main_process:
                print(f"[DEBUG] Completed epoch {epoch}: processed {batch_count} batches")
                if batch_count == 0:
                    print(f"[ERROR] Epoch {epoch} had 0 batches! Training is being skipped!")
                progress.update(epoch_task, advance=1)
                progress.remove_task(step_task)

            # ----------- Save checkpoint every epoch -----------
            current_epoch = epoch + 1
            dist.barrier()
            
            if is_main_process:
                # Save checkpoint every epoch (not tied to validation)
                ckpt_file = f'epoch={str(current_epoch).zfill(6)}.ckpt'
                ckpt_path_full = ckpt_path + ckpt_file
                try:
                    torch.save({
                        'state_dict': model.state_dict(),
                        'ema_model': ema_model.state_dict() if ema_model is not None else None,
                        'optimizer_states': [optimizer.state_dict()],
                        'epoch': current_epoch,
                        'scaler_state_dict': scaler.state_dict() if use_amp else None,
                        'lr_scheduler': lr_scheduler.state_dict()
                    }, ckpt_path_full)
                    console.print(f"[green]💾  Saved epoch checkpoint:[/] {ckpt_file}")
                except Exception as e:
                    console.print(f"[red]Error saving checkpoint: {e}[/red]")
            
            dist.barrier()

            # ----------- validation -----------
            if val_loader is not None and current_epoch % eval_interval == 0:
                # Check if validation loader has samples
                if len(val_loader) == 0:
                    if is_main_process:
                        print(f"Warning: Validation loader is empty (len={len(val_loader)}). Skipping validation.")
                    dist.barrier()
                    continue

                eval_model = ema_model if ema_model is not None else model
                teach_forcing = getattr(eval_model.module, 'teach_forcing', False)
                
                # Determine if this is a CFM model (has transition_model)
                test_cfm = hasattr(eval_model.module, 'transition_model')

                val_avg_loss = val_model(eval_model, val_loader, model_func, progress, live, teach_forcing=teach_forcing,
                                         use_amp=use_amp, rank=rank, is_main_process=is_main_process,
                                         eval_iou=True, test_cfm=test_cfm)

                if is_main_process:
                    # Save IoU evaluation results to file
                    if output_dir is not None and 'all_miou' in val_avg_loss:
                        eval_file = save_eval_results_by_epoch(val_avg_loss, output_dir, current_epoch)
                        console.print(f"[green]📊 Evaluation results saved to: {eval_file}[/green]")
                    
                    wandb_log_val_step(model.module.global_step, epoch, val_avg_loss) if wandb_logger is not None else None
                    
                    # Get loss value - assert error if expected loss key is missing
                    loss_key = loss_monitor if loss_monitor is not None else 'loss'
                    if loss_key not in val_avg_loss:
                        available_keys = list(val_avg_loss.keys())
                        error_msg = (f"Expected loss key '{loss_key}' not found in validation results. "
                                    f"Available keys: {available_keys}")
                        console.print(f"[red]Error: {error_msg}[/red]")
                        dist.barrier()
                        raise KeyError(error_msg)
                    
                    current_loss = val_avg_loss[loss_key]
                    historical_losses.append([current_epoch, current_loss])

                    # Always save first checkpoint, then check top-3 for subsequent ones
                    should_save = False
                    if len(historical_losses) <= 3:
                        # Always save first 3 checkpoints
                        should_save = True
                    else:
                        # Check if current loss is in top-3
                        sorted_losses = sorted([x[1] for x in historical_losses])
                        top3_losses = sorted_losses[:3]
                        should_save = current_loss in top3_losses

                    if should_save:
                        # Save best checkpoint with different name to avoid overwriting regular epoch checkpoint
                        ckpt_file = f'epoch={str(current_epoch).zfill(6)}_best.ckpt'
                        ckpt_path_full = ckpt_path + ckpt_file

                        try:
                            torch.save({
                                'state_dict': model.state_dict(),
                                'ema_model': ema_model.state_dict() if ema_model is not None else None,
                                'optimizer_states': [optimizer.state_dict()],
                                'epoch': current_epoch,
                                'scaler_state_dict': scaler.state_dict() if use_amp else None,
                                'lr_scheduler': lr_scheduler.state_dict()
                            }, ckpt_path_full)

                            saved_ckpts.append((current_loss, ckpt_path_full))  # 记录新模型

                            # 保留 top-3：如果多于 3 个，删掉最差的那个
                            if len(saved_ckpts) > 3:
                                worst = max(saved_ckpts, key=lambda p: p[0])  # 按 loss 最大找最差
                                if os.path.exists(worst[1]):
                                    os.remove(worst[1])  # 删除文件
                                saved_ckpts.remove(worst)  # 从列表移除
                                console.print(f"[red]🗑️  Removed best ckpt:[/] {worst[1]}")

                            console.print(f"[green]💾  Saved best checkpoint:[/] {ckpt_file}  ({loss_key}={current_loss:.6f})")
                        except Exception as e:
                            console.print(f"[red]Error saving best checkpoint: {e}[/red]")
                    else:
                        console.print(f"[yellow]Validation {loss_key}={current_loss:.6f} not in top-3, best ckpt not saved.[/]")