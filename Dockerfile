# CUDA 12.8 build: its PyTorch wheels ship sm_120 (Blackwell) kernels, which the
# cu126 image lacks. Required for RTX 50-series (5080/5090, sm_120). The cu126
# image only covered up to sm_90, so those GPUs hit "no kernel image is available".
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel

WORKDIR /workspace

# Arch list for the Chamfer CUDA extension build. Covers Ampere (8.0/8.6),
# Ada / RTX 4090 (8.9), Hopper (9.0) and Blackwell / RTX 5080-5090 (12.0).
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;12.0"

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git \
        libgl1-mesa-glx \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip \
    && pip install --no-cache-dir \
        pyyaml \
        tqdm \
        huggingface_hub \
        pyhocon \
        open3d \
        scipy \
        pandas \
        trimesh \
        opencv-python \
        scikit-image \
        Pillow \
        segmentation-models-pytorch==0.3.3
# (torchvision ships in the base pytorch image. pyyaml/tqdm/pyhocon are the registration
#  runtime deps and huggingface_hub auto-downloads the SpineDepth dataset + segmentation
#  model; the rest are the reconstruction package deps -- see reconstruction/requirements.txt)

COPY extensions ./extensions

RUN pip install --no-cache-dir --no-build-isolation ./extensions/chamfer_dist
