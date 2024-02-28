## Installation

### Requirements:

- PyTorch >= 1.3 (we recommend 1.8.1 for better speed)
- torchvision
- detectron2
- cocoapi
- cityscapesScripts
- apex
- GCC >= 4.9
- CUDA >= 9.2
- pip packages in [requirements.txt](requirements.txt)

### 安装步骤
直接用AutoDl创建的基础环境，必须使用2080Ti。(3080安装失败)
### Step 1 (Option 2): Install torch with conda (下面的这些直接用base环境即可，必须使用2080Ti，使用3080安装失败)

```bash
# first, make sure that your conda is setup properly with the right environment
# for that, check that `which conda`, `which pip` and `which python` points to the
# right path. From a clean conda env, this is what you need to do
# or you can use pytorch docker to easily setup the environment

conda create --name DAFA -y python=3.8
source activate DAFA

# this installs the right pip and dependencies for the fresh python
conda install ipython pip

# follow PyTorch installation in https://pytorch.org/get-started/locally/
# we give the instructions for CUDA 10.0
# conda install pytorch=1.3.0 torchvision cudatoolkit=10.0 -c pytorch
conda install pytorch=1.8.1 torchvision=0.9.1 cudatoolkit=11.1 -c pytorch
# 安装失败，没有11.1

sudo apt-get update
```

### Step 2: Package Installation

```bash
cd <your install dir>
export INSTALL_DIR=$PWD

# install detectron2:
cd $INSTALL_DIR
git clone https://github.com/facebookresearch/detectron2.git
python -m pip install -e detectron2

# install cityscapesScripts(mmtracking环境已安装，跳过)
cd $INSTALL_DIR
git clone https://github.com/mcordts/cityscapesScripts.git
cd cityscapesScripts/
python setup.py build_ext install

# install apex (skip when you use nvcr docker) 
cd $INSTALL_DIR
git clone https://github.com/NVIDIA/apex.git
cd apex
python setup.py install --cuda_ext --cpp_ext
# (安装失败，直接pip install apex，结果还是不行，版本太老了。)
# 安装失败，各种错误，放弃。把代码中的from apex import amp引用全部注释掉。

# clone our lib:
cd $INSTALL_DIR
git clone https://github.com/sdroh1027/DiffusionVID.git
cd DiffusionVID

# Then install pip packages:
pip install -r requirements.txt

# install the lib with symbolic links
python setup.py build develop  

unset INSTALL_DIR
```
