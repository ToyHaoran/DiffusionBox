#!/bin/bash
#SBATCH -p gpu1
#SBATCH --gpus=1
alias ll='ls -al'  # 快捷键  # 无需指定结点SBATCH --nodelist=aiwkr3
module load anaconda/anaconda3-2022.10  # 加载conda
module load cuda/11.1.0  # 加载cuda
module load gcc-11

source activate DiffusionVID  # 激活环境
which pip

cd /mnt/nfs/data/home/1120220334/pro/DiffusionVID  # 打开项目
#python tools/train_net.py --config-file configs/vid_R_101_DiffusionVID.yaml OUTPUT_DIR training_dir/DiffusionVID_R_101_Adamw
#python tools/train_net.py --config-file configs/vid_R_101_DiffusionVID.yaml OUTPUT_DIR training_dir/DiffusionVID_R_101
#python tools/train_net.py --config-file configs/vid_R_101_DiffusionVID.yaml OUTPUT_DIR training_dir/DiffusionVID_R_101_float16
python tools/train_net.py --config-file configs/vid_SwinB.yaml OUTPUT_DIR training_dir/DiffusionVID_SwinB
