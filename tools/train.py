import argparse

from forecast.utils import common_utils
from easydict import EasyDict
from pathlib import Path
from forecast.config import cfg_from_yaml_file
from forecast.datasets import build_dataloader
from forecast.models import build_network, model_fn_decorator

from train_utils.optimization import build_optimizer, build_scheduler
from train_utils.train_utils import train_model

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')

    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=None, required=False, help='number of epochs to train for')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')

    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--fix_random_seed', action='store_true', default=False, help='')

    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')

    cfg = EasyDict()
    cfg.ROOT_DIR = (Path(__file__).resolve().parent / '../').resolve()
    cfg.LOCAL_RANK = 0

    return parser.parse_args(), cfg

if __name__ == '__main__':
    args, cfg = parse_config()

    if args.fix_random_seed:
        common_utils.set_random_seed(666)

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    output_dir = cfg.ROOT_DIR / 'logs' / cfg.TAG / args.extra_tag
    ckpt_dir = output_dir / 'ckpt'
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print("----------- Create dataloader & network & optimizer -----------")

    batch_size = args.batch_size if args.batch_size is not None \
        else cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU * 1

    train_set, train_loader = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG, batch_size=batch_size, num_workers=args.workers,
        cache_mode=cfg.CACHE_MODE
    )

    model = build_network(model_cfg=cfg.MODEL, loss_cfg=cfg.LOSS)
    model.cuda()

    optimizer = build_optimizer(cfg.OPTIMIZATION, model)
    scheduler = build_scheduler(optimizer, cfg.OPTIMIZATION, len(train_set))

    train_model(model, optimizer, train_loader, scheduler, 0, optim_cfg=cfg.OPTIMIZATION,model_func=model_fn_decorator())

