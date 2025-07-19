import torch
from torch.nn.utils import clip_grad_norm_
from rich.progress import (
    Progress, TextColumn, BarColumn, TimeRemainingColumn,
    TimeElapsedColumn, SpinnerColumn
)
from rich.live import Live
from tools.test import val_model
from rich.console import Group
from contextlib import nullcontext

from tools.common_utils.display_utils import format_disp_dict
import torch.distributed as dist
from rich.console import Console
import os


def train_model(model, optimizer, train_loader, val_loader, lr_scheduler, start_epoch, optim_cfg,
                model_func, ckpt_path, console, progress,
                eval_interval=1, use_amp=True, loss_monitor=None, is_main_process=None, rank=None):

    train_loader.sampler.set_epoch(start_epoch)
    scaler = torch.amp.GradScaler(enabled=use_amp, init_scale=optim_cfg.get('LOSS_SCALE_FP16', 10))
    historical_losses, saved_ckpts = [], []

    if is_main_process:
        epoch_task = progress.add_task(description="Epoch", total=optim_cfg.NUM_EPOCHS,
                                       completed=start_epoch if start_epoch > 0 else start_epoch+1)

    live_ctx = Live(console=console, refresh_per_second=4, transient=True) if is_main_process else nullcontext()

    with live_ctx as live:
        for epoch in range(start_epoch, optim_cfg.NUM_EPOCHS):
            # 每轮新建 step task
            if is_main_process:
                step_task = progress.add_task(description="Samples", total=len(train_loader))

            for batch_idx, batch in enumerate(train_loader):
                model.train()
                optimizer.zero_grad()

                with torch.amp.autocast('cuda', enabled=use_amp):
                    loss, tb_dict, disp_dict = model_func(model, batch)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), optim_cfg.GRAD_NORM_CLIP)
                scaler.step(optimizer)
                scaler.update()
                lr_scheduler.step()

                if is_main_process:
                    progress.update(step_task, advance=1)
                    tb_dict['lr'] = optimizer.param_groups[0]['lr']
                    disp_table = format_disp_dict(tb_dict)
                    live.update(Group(progress, disp_table))

            if is_main_process:
                progress.update(epoch_task, advance=1)
                progress.remove_task(step_task)

            # ----------- validation -----------
            dist.barrier(device_ids=[rank])

            current_epoch = epoch + 1
            if val_loader is not None and current_epoch % eval_interval == 0:
                val_avg_loss = val_model(model, val_loader, model_func, progress, live, use_amp=use_amp, rank=rank, is_main_process=is_main_process)

                if is_main_process:
                    current_loss = val_avg_loss[loss_monitor if loss_monitor is not None else 'loss']
                    historical_losses.append([current_epoch, current_loss])

                    if current_loss in sorted([x[1] for x in historical_losses])[:3]:
                        ckpt_file = f'epoch={str(current_epoch).zfill(6)}.ckpt'
                        ckpt_path_full = ckpt_path / ckpt_file

                        torch.save({
                            'state_dict': model.state_dict(),
                            'optimizer_states': [optimizer.state_dict()],
                            'epoch': current_epoch,
                            'scaler_state_dict': scaler.state_dict() if use_amp else None,
                        }, ckpt_path_full)

                        saved_ckpts.append((current_loss, ckpt_path_full))  # 记录新模型

                        # 保留 top-3：如果多于 3 个，删掉最差的那个
                        if len(saved_ckpts) > 3:
                            worst = max(saved_ckpts, key=lambda p: p[0])  # 按 loss 最大找最差
                            os.remove(worst[1])  # 删除文件
                            saved_ckpts.remove(worst)  # 从列表移除
                            console.print(f"[red]🗑️  Removed ckpt:[/] {worst[1].name}")

                        console.print(f"[green]💾  Saved:[/] {ckpt_file}  (loss {current_loss:.4f})")

                    else:
                        console.print(f"[yellow]Validation loss {current_loss:.4f} not in top-3, ckpt not saved.[/]")