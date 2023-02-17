import os, torch
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.profiler import SimpleProfiler
from datetime import datetime

now = datetime.now()
name = now.strftime("%m%d%Y_%H:%M:%S")


def get_logger(args):
    if args.mode =='train':
        logger_cfg = get_logger_cfg(args)
        logger_cfg.update({"name": name})
        logger = WandbLogger(**logger_cfg)
        return {"logger": logger}
    else:
        return {}


def get_checkpoint(args):
    if args.mode == 'train':
        ckpt_cfg = get_checkpoint_cfg(args)
        checkpoint_callback = ModelCheckpoint(**ckpt_cfg)
        return {"callbacks": [checkpoint_callback]}
    if args.mode == 'test':
        return {}


def get_profiler(args):
    if args.mode == 'train':
        profiler_cfg = get_profiler_cfg(args)
        profiler = SimpleProfiler(**profiler_cfg)
        return {"profiler": profiler}
    if args.mode == 'test':
        return {}


def set_arguments_pl(args):
    pl_args = {}
    accelerator_cfg = get_accelerator_cfg(args)
    logger = get_logger(args)
    ckpt_callback = get_checkpoint(args)
    profiler = get_profiler(args)
    train_val_cfg = get_train_val_cfg(args)
    resume_cfg = get_resume_cfg(args)

    pl_args.update(
        **accelerator_cfg,
        **logger,
        **ckpt_callback,
        **profiler,
        **train_val_cfg,
        **resume_cfg,
    )
    # pl_args.update(**accelerator_cfg, **train_val_cfg)
    return pl_args


"""
configurations for accelerator: single gpu or ddp
"""


def get_accelerator_cfg(args):
    cfg = {}
    if args.accelerator == "gpu":
        cfg.update({"accelerator": "gpu", "devices": 1})
    elif args.accelerator == "ddp":
        node_str = os.environ["SLURM_JOB_NODELIST"].replace("[", "").replace("]", "")
        nodes = node_str.split(",")
        world_size = len(nodes)
        gpus = torch.cuda.device_count()
        cfg.update(
            {
                "strategy": "ddp",
                "accelerator": "gpu",
                "num_nodes": world_size,
                "devices": gpus,
            }
        )
    return cfg


"""
configurations for train validation
"""


def get_train_val_cfg(args):
    cfg = {}
    if args.mode == 'train':
        cfg.update(
            {
                "max_epochs": args.epoch,
                "log_every_n_steps": args.log_steps,
                "val_check_interval": args.val_check_interval,
            }
        )
    return cfg



"""
configurations for logger
"""


def get_logger_cfg(args):
    cfg = {}
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir, exist_ok=True)
    cfg.update(
        {
            "save_dir": args.log_dir,
            "offline": args.offline,
            "reinit": args.reinit,
            "project": args.project_name,
        }
    )
    return cfg


"""
configurations for checkpoint
"""


def get_checkpoint_cfg(args):
    cfg = {}
    if not os.path.exists(os.path.join(args.checkpoint_dir, name)):
        os.makedirs(os.path.join(args.checkpoint_dir, name), exist_ok=True)

    cfg.update(
        {
            "dirpath": os.path.join(args.checkpoint_dir, name),
            "save_top_k": args.save_top_k,
            "monitor": args.monitor,
            "mode": args.monitor_mode,
            "filename": args.filename,
        }
    )
    return cfg


"""
configurations for profiler
"""


def get_profiler_cfg(args):
    cfg = {}
    if not os.path.exists(args.profiler_path):
        os.makedirs(args.profiler_path, exist_ok=True)
    cfg.update({"dirpath": args.profiler_path, "filename": args.profiler_name})
    return cfg


"""
configurations for resume training
"""


def get_resume_cfg(args):
    cfg = {}
    if args.mode == 'train':
        if args.resume:
            cfg.update({"resume_from_checkpoint": args.resume_dir})
    return cfg
