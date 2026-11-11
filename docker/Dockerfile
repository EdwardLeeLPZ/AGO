# ------------------------------------------------------------------------
# AGO
# Copyright (c) 2025 Peizheng Li. All Rights Reserved.
# ------------------------------------------------------------------------

FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ca-certificates curl git wget \
    libgl1-mesa-dev libglib2.0-0 libsm6 libxext6 ffmpeg \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN curl -sLo /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-py310_24.11.1-0-Linux-x86_64.sh \
    && bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh
ENV PATH=/opt/conda/bin:$PATH

WORKDIR /opt
RUN pip install --no-cache-dir torch==2.1.1 torchvision==0.16.1 --index-url https://download.pytorch.org/whl/cu118
RUN pip install --no-cache-dir numpy==1.26.4 matplotlib==3.5.3 openmim \
    && mim install "mmengine==0.10.7" "mmcv==2.1.0" "mmdet==3.3.0" \
    && pip install --no-cache-dir "mmsegmentation==1.0.0"
RUN git clone https://github.com/open-mmlab/mmdetection3d.git \
    && git -C mmdetection3d checkout 962f0937 \
    && pip install --no-cache-dir -e mmdetection3d
ENV FORCE_CUDA=1
ENV CUDA_HOME=/usr/local/cuda-11.8/
ENV TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6;8.9;9.0+PTX"
RUN pip install --no-cache-dir git+https://github.com/mhamilton723/CLIP.git@dce9d43cc659cb9a013745b378c78077fde6aac0 \
    && git clone https://github.com/mhamilton723/FeatUp.git \
    && git -C FeatUp checkout 6b5a6c0 \
    && pip install --no-cache-dir -e FeatUp
RUN pip install --no-cache-dir timm==1.0.14 einops==0.8.2 nltk==3.10.4 pytorch-lightning==2.6.6 torchmetrics==1.9.0

WORKDIR /workspace
