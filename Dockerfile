FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-devel

WORKDIR /workspace

ENV TORCH_CUDA_ARCH_LIST="8.9;8.6;8.0;9.0"

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git \
        libgl1-mesa-glx \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip \
    && pip install --no-cache-dir \
        pyhocon \
        open3d \
        scipy \
        pandas \
        trimesh \
        PyMCubes \
        libigl \
        opencv-python \
        python-igraph \
        spconv-cu126 \
        yacs \
        h5py \
        tensorboardX

COPY extensions ./extensions

RUN pip install --no-cache-dir --no-build-isolation ./extensions/chamfer_dist
