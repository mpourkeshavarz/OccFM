import torch
from torch.nn.utils import clip_grad_norm_
from rich.progress import (
    Progress, TextColumn, BarColumn, TimeRemainingColumn,
    TimeElapsedColumn, SpinnerColumn
)
from rich.live import Live
from tools.test import val_model
from rich.console import Group

from tools.common_utils.display_utils import format_disp_dict

def train_model(model, optimizer, train_loader, val_loader, lr_scheduler, start_epoch, optim_cfg,
                model_func, ckpt_path, console, progress, eval_interval=1, use_amp=True, loss_monitor=None):

    scaler = torch.amp.GradScaler(enabled=use_amp, init_scale=optim_cfg.get('LOSS_SCALE_FP16', 2.0 ** 16))
    historical_losses = []

    epoch_task = progress.add_task(description="Epoch", total=optim_cfg.NUM_EPOCHS - start_epoch,
                                   completed=start_epoch if start_epoch > 0 else start_epoch+1)

    with Live(console=console, refresh_per_second=4, transient=True) as live:
        for epoch in range(start_epoch, optim_cfg.NUM_EPOCHS):
            # 每轮新建 step task
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

                progress.update(step_task, advance=1)

                tb_dict['lr'] = optimizer.param_groups[0]['lr']
                disp_table = format_disp_dict(tb_dict)
                live.update(Group(progress, disp_table))

            progress.update(epoch_task, advance=1)
            progress.remove_task(step_task)

            # ----------- validation -----------
            current_epoch = epoch + 1
            if val_loader is not None and current_epoch % eval_interval == 0:
                val_avg_loss = val_model(model, val_loader, model_func, progress, live, use_amp=use_amp)
                current_loss = val_avg_loss[loss_monitor if loss_monitor is not None else 'loss']
                historical_losses.append([current_epoch, current_loss])

                top3 = sorted([x[1] for x in historical_losses])[:3]
                if current_loss in top3:
                    ckpt_file = f'epoch={str(current_epoch).zfill(6)}.ckpt'
                    checkpoint = {
                        'state_dict': model.state_dict(),
                        'optimizer_states': [optimizer.state_dict()],
                        'epoch': current_epoch,
                        'scaler_state_dict': scaler.state_dict() if use_amp else None,
                    }
                    torch.save(checkpoint, ckpt_path / ckpt_file)
                    console.print(
                        f"[bold green]💾 Top-3 validation loss detected (loss = {current_loss:.4f}), checkpoint saved to:[/bold green] [cyan]{ckpt_file}[/cyan]"
                    )
                else:
                    console.print(f"[yellow]Validation loss = {current_loss:.4f} not in top-3, checkpoint not saved.[/yellow]")
