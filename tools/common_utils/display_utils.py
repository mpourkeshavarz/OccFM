import numpy as np
from pathlib import Path
from rich.table import Table
from rich.align import Align
from rich.panel import Panel
from rich.columns import Columns
from rich.box import SIMPLE_HEAD
from rich import box

from rich.console import Console

from rich.progress import (
    Progress, TextColumn, BarColumn, TimeRemainingColumn,
    TimeElapsedColumn, SpinnerColumn
)

def format_disp_dict(disp_dict):
    table = Table(
        title="Current Loss",
        show_header=True,
        header_style="bold yellow",
        box=SIMPLE_HEAD,
        expand=False,
        pad_edge=False,
        padding=(0, 2),
    )

    table.add_column("Metric", justify="left", style="cyan")
    table.add_column("Value", justify="right", style="bold white")

    for key, val in disp_dict.items():
        if abs(val) < 1e-3 or abs(val) >= 1e4:
            value_str = f"{val:.5e}"
        else:
            value_str = f"{val:.4f}"

        table.add_row(str(key), value_str)

    panel = Panel(
        Align.center(table, vertical="middle"),
        border_style="dim",
        padding=(1, 4),
        width=60,  # 控制总宽度
    )

    return panel

def setup_loggers():

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        TextColumn("[green]•[bold]{task.completed}[/] / {task.total}"),
        BarColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        transient=False,
    )

    console = Console(force_terminal=True, legacy_windows=True)

    return progress, console


def show_eval(avg_dict, console, output_dir=None):
    all_miou, all_iou = avg_dict['all_miou'], avg_dict['all_iou']
    time_avg = avg_dict['time']
    mean_miou, mean_iou = np.mean(all_miou), np.mean(all_iou)

    if 'cate_miou' in avg_dict and isinstance(avg_dict['cate_miou'], dict):
        extra_table = Table(title="MIOU on each category", box=box.SIMPLE_HEAVY)
        for key in avg_dict['cate_miou'].keys():
            extra_table.add_column(str(key), justify="center", style="bold white")
        extra_table.add_row(*[f"{v:.2f}" if isinstance(v, float) else str(v) for v in avg_dict['cate_miou'].values()])
        console.print(extra_table)
        
        # Save category mIoU to txt file if output_dir is provided
        if output_dir is not None:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            txt_file = output_path / 'category_miou.txt'
            with open(txt_file, 'w') as f:
                f.write("mIoU on each category:\n")
                f.write("=" * 50 + "\n")
                for key, value in avg_dict['cate_miou'].items():
                    f.write(f"{key}: {value:.2f}\n")
                f.write("=" * 50 + "\n")
                f.write(f"Mean mIoU: {mean_miou:.4f}\n")
            console.print(f"[green]Category mIoU saved to: {txt_file}[/green]")

    summary_panels = [
        Panel.fit(f"[bold green]{mean_miou:.4f}[/bold green]", title="Mean mIoU", border_style="magenta"),
        Panel.fit(f"[bold yellow]{mean_iou:.4f}[/bold yellow]", title="Mean IoU", border_style="cyan"),
        Panel.fit(f"[bold yellow]{time_avg:.4f}[/bold yellow]", title="Avg latency", border_style="orange1"),
    ]

    console.print(Columns(summary_panels))

    # 表格：Per-frame IoU 和 mIoU
    table = Table(title="Per-frame IoU and mIoU", box=box.ROUNDED, show_lines=True)

    table.add_column("Frame Index", justify="right", style="bold cyan")
    table.add_column("mIoU", justify="right", style="green")
    table.add_column("IoU", justify="right", style="yellow")

    num_frames = min(len(all_miou), len(all_iou))
    for idx in range(num_frames):
        table.add_row(str(idx), f"{all_miou[idx]:.4f}", f"{all_iou[idx]:.4f}")

    console.print(table)


def save_eval_results_by_epoch(avg_dict, output_dir, epoch):
    """
    Save evaluation results (IoU, mIoU) to a text file named by epoch number.
    
    Args:
        avg_dict: Dictionary containing evaluation metrics
        output_dir: Directory to save the results
        epoch: Epoch number (1-indexed)
    """
    if 'all_miou' not in avg_dict or 'all_iou' not in avg_dict:
        return
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    all_miou = avg_dict['all_miou']
    all_iou = avg_dict['all_iou']
    mean_miou = np.mean(all_miou)
    mean_iou = np.mean(all_iou)
    
    txt_file = output_path / f'eval_epoch_{str(epoch).zfill(6)}.txt'
    
    with open(txt_file, 'w') as f:
        f.write(f"Evaluation Results - Epoch {epoch}\n")
        f.write("=" * 60 + "\n\n")
        
        # Summary metrics
        f.write("Summary Metrics:\n")
        f.write("-" * 60 + "\n")
        f.write(f"Mean mIoU: {mean_miou:.6f}\n")
        f.write(f"Mean IoU: {mean_iou:.6f}\n")
        
        if 'time' in avg_dict:
            f.write(f"Avg latency: {avg_dict['time']:.6f}\n")
        
        # Loss metrics if available
        for key, value in avg_dict.items():
            if 'loss' in key.lower():
                f.write(f"{key}: {value:.6f}\n")
        
        f.write("\n" + "=" * 60 + "\n\n")
        
        # Per-frame IoU and mIoU
        f.write("Per-frame Results:\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'Frame Index':<15} {'mIoU':<15} {'IoU':<15}\n")
        f.write("-" * 60 + "\n")
        
        num_frames = min(len(all_miou), len(all_iou))
        for idx in range(num_frames):
            f.write(f"{idx:<15} {all_miou[idx]:<15.6f} {all_iou[idx]:<15.6f}\n")
        
        # Category mIoU if available
        if 'cate_miou' in avg_dict and isinstance(avg_dict['cate_miou'], dict):
            f.write("\n" + "=" * 60 + "\n\n")
            f.write("Category-wise mIoU:\n")
            f.write("-" * 60 + "\n")
            for key, value in avg_dict['cate_miou'].items():
                f.write(f"{key}: {value:.6f}\n")
        
        f.write("\n" + "=" * 60 + "\n")
    
    return txt_file