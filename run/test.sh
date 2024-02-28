#!/bin/bash
#SBATCH -p gpu1
#SBATCH --gpus=1
alias ll='ls -al'  # 快捷键
module load anaconda/anaconda3-2022.10  # 加载conda
module load cuda/11.1.0  # 加载cuda
module load gcc-11

source activate DiffusionVID  # 激活环境
which pip

cd /mnt/nfs/data/home/1120220334/pro/DiffusionVID  # 打开项目

# 多程序运行，后面加&，最后加wait
python tools/test_net.py --config-file configs/vid_R_101_DiffusionVID.yaml MODEL.WEIGHT models/DiffusionVID_R101.pth
