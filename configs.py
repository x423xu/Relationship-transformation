import argparse

parser = argparse.ArgumentParser(
    prog="Relationship transformation",
    description="Project for ICCV 2023 xxx",
)
parser.add_argument("--project_name", default="Relationship transformation", type=str)
parser.add_argument("--mode", default="train", type=str, choices=["train", "test"])
parser.add_argument(
    "--algorithm",
    default="direct_map",
    type=str,
    choices=["direct_map", "map_with_graph"],
    help="",
)

"""
args for dataloader
"""
parser.add_argument("--train_val_test_dir", default="train_val_test.npy", type=str)
parser.add_argument("--dataset", default="real_estate", type=str)
parser.add_argument(
    "--cameras_dir", default="/home/xxy/Documents/data/RealEstate10K", type=str
)
parser.add_argument(
    "--frames_dir",
    default="/home/xxy/Documents/data/RealEstate10K/benchmark_frames",
    type=str,
)
parser.add_argument("--W", default=256, type=int, help="width of input image")
parser.add_argument(
    "--pre_factor", default=2, type=int, help="prefactor for dataloader"
)
parser.add_argument("--num_workers", default=10, type=int, help="number of workers")

"""
args for train
"""
parser.add_argument("--epoch", default=100, type=int)
parser.add_argument("--train_size", default=8, type=int)
parser.add_argument("--val_size", default=8, type=int)
parser.add_argument("--scheduler_step", default=1000, type=int)
parser.add_argument("--scheduler_frequency", default=1, type=int)
parser.add_argument("--log_steps", default=5, type=int)
parser.add_argument("--val_check_interval", default=1, type=float)
parser.add_argument("--scheduler_gamma", default=0.5, type=float)
parser.add_argument("--lr", "--learning_rate", default=1e-4, type=float)
parser.add_argument(
    "--accelerator",
    default="gpu",
    type=str,
    choices=["gpu", "ddp", "cpu"],
    help="gpu: single gpu, ddp: distributed data parallel",
)


"""
args for test
"""
parser.add_argument("--test_size", default=8, type=int)
parser.add_argument(
    "--test_model_dir",
    default="./experiments/checkpoints/Relationship transformation-epoch=22-val_loss=0.03.ckpt",
    type=str,
)

"""
args for model
"""

"""
args for metrics
"""
parser.add_argument("--losses", default=["structure_loss", "value_loss"], nargs="+", type=str)
parser.add_argument("--losses_weights", default=[1, 1], nargs="+", type = int)
# parser.add_argument('--losses', default=['value_loss'], nargs='+')
# parser.add_argument('--losses_weights', default=[1], nargs='+')
parser.add_argument("--simple_loss", default=False, type=bool)
parser.add_argument("--K", default=[5, 10, 20], type=int, nargs="+", help="reall @ k")
"""
args for logger
"""
parser.add_argument("--log_dir", default="./experiments/logs", type=str)
parser.add_argument("--offline", default=False, type=bool)
parser.add_argument("--reinit", default=True, type=bool)

"""
args for checkpoint
"""
parser.add_argument("--checkpoint_dir", default="./experiments/checkpoints", type=str)
parser.add_argument("--save_top_k", default=5, type=int)
parser.add_argument("--monitor", default="val_loss", type=str)
parser.add_argument("--monitor_mode", default="min", type=str)
parser.add_argument(
    "--resume_dir",
    default="experiments/checkpoints/Relationship transformation-epoch=00-val_loss=0.00.ckpt",
    type=str,
)
parser.add_argument("--resume", default=False, type=bool)
"""
args for profiler
"""
parser.add_argument("--profiler_path", default="./experiments/profiler", type=str)
parser.add_argument("--profiler_name", default="perf_log", type=str)

args = parser.parse_args()
args.filename = args.project_name + "-{epoch:02d}-{val_loss:.2f}"
from pprint import pprint

pprint(vars(args))
