import wandb
from datetime import datetime
import socket
import requests

def wandb_log_val_step(global_step, epoch, avg_dict, eval_iou=False):

    log_dict = {f"train/{k}": float(v) for k, v in avg_dict.items() if 'loss' in k}
    log_dict['epoch'] = epoch + 1
    log_dict['step'] = global_step

    if eval_iou:
        raise NotImplementedError

    wandb.log(log_dict, step=global_step)


def wandb_log_train_step(global_step, epoch, tb_dict):
    log_dict = {f"train/{k}": float(v) for k, v in tb_dict.items()}
    log_dict['epoch'] = epoch + 1
    log_dict['step'] = global_step
    wandb.log(log_dict, step=global_step)

def create_wandb_logger(cfg, args):

    if not args.wandb_offline:
        try:
            r = requests.get("http://worldtimeapi.org/api/timezone/Etc/UTC")
            utc_time = datetime.fromisoformat(r.json()["utc_datetime"].replace("Z", "+00:00"))
            current_time = 'UTC: ' + utc_time.strftime("%Y-%m-%d %H:%M:%S")
        except Exception as e:
            current_time = 'Local time: ' + datetime.now().strftime("%Y-%m-%d-%H:%M")
    else:
        current_time = 'Local time: ' + datetime.now().strftime("%Y-%m-%d-%H:%M")

    wandb_mode = 'offline' if args.wandb_offline else 'online'
    hostname = socket.gethostname()

    run_name = f"{cfg.TAG}-{args.extra_tag}-{hostname}-{current_time}"

    wandb_run = wandb.init(project=cfg.NAME, entity=args.wandb_entity, name=run_name, mode=wandb_mode,
        config={
            "cfg_file": args.cfg_file,
            "batch_size": args.batch_size,
            "amp": args.amp,
            "use_ema": args.use_ema,
            "tag": args.extra_tag
        },
        dir=str(cfg.ROOT_DIR / 'logs' / cfg.TAG / args.extra_tag)
    )
    print(f"[W&B] Logging initialized in {wandb_mode} mode.")
    return wandb_run
