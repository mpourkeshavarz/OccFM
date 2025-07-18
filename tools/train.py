import argparse
import glob
import os
import shutil
from os.path import isfile

from forecast.utils import common_utils
from easydict import EasyDict
from pathlib import Path
from forecast.config import cfg_from_yaml_file
from forecast.datasets import build_dataloader, reset_batch_size
from forecast.models import build_network, model_fn_decorator

from train_utils.optimization import build_optimizer, build_scheduler
from train_utils.train_utils import train_model
from common_utils.display_utils import setup_loggers, show_eval
from rich.live import Live
from tools.test import val_model

from rich.console import Console
from cache_vae import cache_model
import torch.distributed as dist

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')

    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=None, required=False, help='number of epochs to train for')
    parser.add_argument('--workers', type=int, default=6, help='number of workers for dataloader')

    # recover training only, train from last ckpt, if have finished, just eval val loss
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    # for pretrain weight load, train from scratch
    parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained_model')
    parser.add_argument('--fix_random_seed', action='store_true', default=False, help='')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')

    parser.add_argument('--cache_mode', action='store_true', default=False, help='')
    parser.add_argument('--mu_sigma_cache', action='store_true', default=False, help='')

    parser.add_argument('--save_path', type=str, default=None, help='checkpoint to start from')

    cfg = EasyDict()
    cfg.ROOT_DIR = (Path(__file__).resolve().parent / '../').resolve()
    cfg.LOCAL_RANK = 0

    return parser.parse_args(), cfg

if __name__ == '__main__':
    args, cfg = parse_config()
    console = Console()

    if args.fix_random_seed:
        common_utils.set_random_seed(666)

    recover_training = False
    if getattr(args, 'ckpt', None) is not None and isfile(args.ckpt):
        ori_run_name = '/'.join(args.ckpt.rstrip('/').split('/')[:-2])
        ori_yaml_path = glob.glob(ori_run_name + '/*.yaml')
        assert len(ori_yaml_path) == 1 and ori_yaml_path[0].split('/')[-1] == args.cfg_file.split('/')[-1], "YAML confliction"
        args.cfg_file = ori_yaml_path[0]
        recover_training = True

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    if not recover_training:
        output_dir = cfg.ROOT_DIR / 'logs' / cfg.TAG / args.extra_tag
        ckpt_dir = output_dir / 'ckpt'
        output_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(args.cfg_file, output_dir)
    else:
        output_dir = ori_run_name

    print("----------- Create dataloader & network & optimizer -----------")

    batch_size = args.batch_size if args.batch_size is not None \
        else cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU * 1

    model = build_network(model_cfg=cfg.MODEL, loss_cfg=cfg.LOSS)
    model.cuda()

    train_set, train_loader = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG, batch_size=batch_size, num_workers=args.workers,
        cache_mode=cfg.CACHE_MODE
    )

    val_set, val_loader = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG, batch_size=batch_size, num_workers=args.workers,
        cache_mode=cfg.CACHE_MODE, training=False
    )

    optimizer = build_optimizer(cfg.OPTIMIZATION, model)
    scheduler = build_scheduler(optimizer, cfg.OPTIMIZATION, training_length_ep=len(train_set))

    progress, console = setup_loggers()

    if recover_training:
        model_status = model.recover_training(args.ckpt)
        optimizer.load_state_dict(model_status['optimizer_states'][0])
        scheduler.step(model_status['epoch'] * len(train_set))
        if cfg.OPTIMIZATION.NUM_EPOCHS > model_status['epoch']:
            train_model(model, optimizer, train_loader, val_loader, scheduler, console=console, progress=progress,
                        ckpt_path=output_dir + '/ckpt/', start_epoch=model_status['epoch'], optim_cfg=cfg.OPTIMIZATION,
                        eval_interval=cfg.EVAL_INTERVAL, model_func=model_fn_decorator(), loss_monitor=cfg.LOSS_MONITOR)
        else:
            console.print(
                "[bold magenta]✔️ All training epochs completed. The model is fully trained and ready for evaluation or deployment.[/bold magenta]"
            )

    else:
        train_model(model, optimizer, train_loader, val_loader, scheduler, console=console, progress=progress,
                    ckpt_path=output_dir / 'ckpt', start_epoch=0, optim_cfg=cfg.OPTIMIZATION,
                    eval_interval=cfg.EVAL_INTERVAL, model_func=model_fn_decorator(), loss_monitor=cfg.LOSS_MONITOR)


    with Live(console=console, refresh_per_second=4, transient=True) as live:

        train_loader = reset_batch_size(train_loader, 1)
        val_loader = reset_batch_size(val_loader, 1)

        # old weight have trouble when eval with amp
        val_avg_loss = val_model(model, val_loader, model_fn_decorator(), progress, live, use_amp=False, eval_iou=True, eval_fps=True)
        show_eval(val_avg_loss, console)

        cache_latent = args.cache_mode or cfg.CACHE_MODE
        cache_mu_sigma = args.mu_sigma_cache or cfg.MU_SIGMA_CACHE
        cache_model(model, [train_loader, val_loader], model_fn_decorator(), progress, console=console,
                    cache_mode=cache_latent, mu_sigma_cache=cache_mu_sigma, save_path=args.save_path)