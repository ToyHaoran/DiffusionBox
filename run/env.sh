#!/bin/bash
#SBATCH -p gpu1
#SBATCH --gpus=1
alias ll='ls -al'  # 快捷键
module load anaconda/anaconda3-2022.10  # 加载conda
module load cuda/11.1.0  # 加载cuda
module load gcc-11

#which pip
#conda remove -n DiffusionVID --all -y # 删除conda环境，删不掉就手动删除。
#conda create -n DiffusionVID  python=3.8 -y # 创建conda环境 (需要更新清华源或阿里源，自行解决)
source activate DiffusionVID  # 激活环境
which pip

# 安装pytorch，因为网络问题，需要先conda安装，然后再pip再安装，都试试，最终还是用pip的包。
#pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple  # 换源
#pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 --extra-index-url https://download.pytorch.org/whl/cu111
#pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
# pip下载不下来，使用conda下载
#conda install pytorch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 -c pytorch

## 安装依赖环境
#cd /mnt/nfs/data/home/1120220334/pro/detectron2
#pip install -v -e .

#cd /mnt/nfs/data/home/1120220334/pro/cityscapesScripts
#python setup.py build_ext install
#
#cd /mnt/nfs/data/home/1120220334/pro/DiffusionVID  # https://github.com/sdroh1027/DiffusionVID.git
#pip install -r requirements.txt
#python setup.py build develop

which pip
pip list
