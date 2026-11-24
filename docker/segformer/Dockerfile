# SegFormer 하늘 마스크 추출 환경 — 책 03-2
# DriveStudio의 마스크 추출 스크립트가 mmcv-full 1.2.7(PyTorch 1.8)을 요구하므로 별도 이미지로 분리
FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive \
    TORCH_CUDA_ARCH_LIST="8.6" \
    MMCV_WITH_OPS=1 \
    FORCE_CUDA=1 \
    MAX_JOBS=8 \
    PIP_NO_CACHE_DIR=1
RUN apt-key del 7fa2af80 2>/dev/null; rm -f /etc/apt/sources.list.d/cuda.list /etc/apt/sources.list.d/nvidia-ml.list; \
    apt-get update && apt-get install -y --no-install-recommends git libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install timm==0.3.2 opencv-python-headless==4.5.5.64 attrs tqdm imageio scikit-image omegaconf gdown \
    && pip install mmcv-full==1.2.7
RUN git clone https://github.com/NVlabs/SegFormer /opt/SegFormer && cd /opt/SegFormer && pip install .
# 공식 체크포인트(Cityscapes b5). DriveStudio 문서의 백업 링크를 사용
RUN mkdir -p /opt/SegFormer/pretrained && cd /opt/SegFormer/pretrained \
    && gdown 1e7DECAH0TRtPZM6hTqRGoboq1XPqSmuj -O segformer.b5.1024x1024.city.160k.pth
