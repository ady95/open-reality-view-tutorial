# DriveStudio (Street Gaussians / OmniRe) 실행 환경 — 책 03-2, 03-3
# DriveStudio 요구사항: PyTorch 2.0 + CUDA 11.7, gsplat 1.3.0 고정
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive \
    TORCH_CUDA_ARCH_LIST="8.6" \
    MAX_JOBS=8 \
    PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential ninja-build libgl1 libglib2.0-0 libegl1 \
    && rm -rf /var/lib/apt/lists/*

ARG DRIVESTUDIO_COMMIT=e59bda4
RUN git clone --recursive https://github.com/ziyc/drivestudio.git /opt/drivestudio \
    && cd /opt/drivestudio && git checkout ${DRIVESTUDIO_COMMIT} && git submodule update --init --recursive

WORKDIR /opt/drivestudio
# torch/torchvision은 베이스 이미지 것을 쓰고 나머지 의존성만 설치
RUN grep -vE '^(--extra-index-url|torch==|torchvision==)' requirements.txt > /tmp/req.txt \
    && pip install -r /tmp/req.txt nuscenes-devkit  # matplotlib==3.5.3 고정과 맞는 devkit 버전을 pip가 고른다
RUN pip install --no-build-isolation git+https://github.com/nerfstudio-project/gsplat.git@v1.3.0
RUN pip install --no-build-isolation git+https://github.com/facebookresearch/pytorch3d.git@v0.7.7
RUN pip install --no-build-isolation git+https://github.com/NVlabs/nvdiffrast.git@v0.3.3
RUN cd third_party/smplx && pip install -e .

ENV PYTHONPATH=/opt/drivestudio

# imageio-ffmpeg 내장 바이너리에는 H.264 인코더가 없어 평가 영상 저장이 실패한다 → 시스템 ffmpeg 사용
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*
ENV IMAGEIO_FFMPEG_EXE=/usr/bin/ffmpeg
