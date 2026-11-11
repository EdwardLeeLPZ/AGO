# Installation

AGO requires Python 3.10, PyTorch 2.1.1 with CUDA 11.8, and a source installation of mmdetection3d.

## Docker

```shell
docker build -t ago -f docker/Dockerfile docker
docker run --gpus all -it --shm-size=32g -v $PWD:/workspace ago
```

## Conda

```shell
conda create -n ago python=3.10 -y
conda activate ago

pip install torch==2.1.1 torchvision==0.16.1 --index-url https://download.pytorch.org/whl/cu118
pip install numpy==1.26.4 matplotlib==3.5.3 openmim
mim install "mmengine==0.10.7" "mmcv==2.1.0" "mmdet==3.3.0"
pip install "mmsegmentation==1.0.0"

git clone https://github.com/open-mmlab/mmdetection3d.git
git -C mmdetection3d checkout 962f0937
pip install -e mmdetection3d

export CUDA_HOME=/usr/local/cuda-11.8  # required to build FeatUp
pip install git+https://github.com/mhamilton723/CLIP.git@dce9d43cc659cb9a013745b378c78077fde6aac0
git clone https://github.com/mhamilton723/FeatUp.git
git -C FeatUp checkout 6b5a6c0
pip install -e FeatUp

pip install timm==1.0.14 einops==0.8.2 nltk==3.10.4 pytorch-lightning==2.6.6 torchmetrics==1.9.0
```

The first run downloads FeatUp through `torch.hub`, the MaskCLIP upsampler and CLIP ViT-B/16 weights to `$TORCH_HOME`, and the NLTK word list.
