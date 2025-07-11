import torch
from torch.nn.utils import clip_grad_norm_
from rich.progress import (
    Progress, TextColumn, BarColumn, TimeRemainingColumn,
    TimeElapsedColumn, SpinnerColumn
)
from rich.live import Live
from rich.table import Table
from rich.console import Group

def format_disp_dict(disp_dict):
    """将 tensor 字典格式化为 rich table"""
    table = Table(show_header=True, title="Current Loss", expand=True)
    table.add_column("Metric", justify="right")
    table.add_column("Value", justify="left")

    for key, val in disp_dict.items():
        if isinstance(val, torch.Tensor):
            val = val.item()
        table.add_row(key, f"{val:.4f}")
    return table

def train_model(model, optimizer, train_loader, lr_scheduler, start_epoch, optim_cfg,
                model_func, use_amp=False):

    scaler = torch.amp.GradScaler(enabled=use_amp, init_scale=optim_cfg.get('LOSS_SCALE_FP16', 2.0 ** 16))

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.fields[desc]}", justify="right"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        transient=False,
    )

    epoch_task = progress.add_task("epoch", total=optim_cfg.NUM_EPOCHS, desc="Epochs")
    step_task = None
    disp_table = Table()

    # Live 显示整合 group（progress + loss table）
    with Live(refresh_per_second=4, transient=True) as live:
        for epoch in range(optim_cfg.NUM_EPOCHS):
            # 创建新的 step 任务
            if step_task is not None:
                progress.remove_task(step_task)
            step_task = progress.add_task("step", total=len(train_loader), desc=f"Epoch {epoch + 1}")

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

                cur_lr = optimizer.param_groups[0]['lr']
                disp_dict['lr'] = cur_lr

                disp_table = format_disp_dict(disp_dict)
                live.update(Group(progress, disp_table))  # 更新 group 内容

            progress.update(epoch_task, advance=1)