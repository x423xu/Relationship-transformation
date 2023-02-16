module load gcc/9.3.0 opencv python scipy-stack cuda
virtualenv --no-download reltrans
source reltrans/bin/activate
pip install --no-index h5py==3.1.0
pip install --no-index matplotlib
pip install --no-index pytorch3d
pip install --no-index pytorch_lightning==1.6.1
pip install --no-index scikit_image
pip install --no-index scipy==1.10.0
pip install --no-index tensorboardX
pip install --no-index torchvision
pip install --no-index wandb
export NCCL_BLOCKING_WAIT=1

python main.py --offline True --cameras_dir /scratch/xiaoyu/reltrans/datasets/RealEstate10K --frames_dir /scratch/xiaoyu/reltrans/datasets/RealEstate10K/benchmark_frames/
srun python main.py --offline True --cameras_dir /scratch/xiaoyu/reltrans/datasets/RealEstate10K --frames_dir /scratch/xiaoyu/reltrans/datasets/RealEstate10K/benchmark_frames/ --accelerator ddp
