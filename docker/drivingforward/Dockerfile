# DrivingForward (Feed-forward 3DGS) 실행 환경 — 책 04-2
# 코드: MIT License. 단, 래스터라이저(graphdeco diff-gaussian-rasterization)는 Inria 비상업 라이선스
# 공식 설치: Python 3.8 + PyTorch 1.12.0 + CUDA 11.3
FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends \
        git python3.8 python3.8-dev python3-pip libgl1 libglib2.0-0 ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.8 /usr/bin/python && python -m pip install --upgrade "pip<24.1"

RUN pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 \
        --extra-index-url https://download.pytorch.org/whl/cu113

# requirements.txt에서 추론·평가에 필요한 것만 같은 버전으로 (jupyter 등은 뺌)
RUN pip install numpy==1.24.4 einops==0.7.0 e3nn==0.4.0 jaxtyping==0.2.19 typeguard==4.3.0 \
        lpips==0.1.4 nuscenes-devkit==1.1.9 matplotlib==3.5.3 Pillow==9.5.0 pyquaternion==0.9.5 \
        PyYAML==6.0.1 scikit-image==0.21.0 scipy==1.10.1 tensorboardX==2.6.2.2 tqdm==4.66.5 \
        opencv-python-headless==4.10.0.84 protobuf==3.20.3 yacs==0.1.8 fvcore==0.1.5.post20221221 \
        iopath==0.1.10 "imageio[ffmpeg]==2.35.1" gdown==5.2.0

# pytorch3d는 transforms(순수 파이썬)만 쓴다. requirements의 0.3.0 대신 torch 1.12와 맞는 0.7.2를 빌드
ENV TORCH_CUDA_ARCH_LIST="8.6"
RUN FORCE_CUDA=1 pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.2"

ARG DF_COMMIT=5d0a2c7
RUN git clone https://github.com/fangzhou2000/DrivingForward.git /opt/DrivingForward \
    && cd /opt/DrivingForward && git checkout ${DF_COMMIT} \
    && git submodule update --init --recursive
# docker build 중에는 GPU가 보이지 않으므로 TORCH_CUDA_ARCH_LIST로 컴파일 대상을 지정한다
RUN cd /opt/DrivingForward/models/gaussian/gaussian-splatting \
    && pip install --no-build-isolation submodules/diff-gaussian-rasterization

# 서브모듈(packnet_sfm, dgp)이 자기 패키지를 최상위 이름으로 import하므로 경로를 더하고, 그쪽 의존성을 넣는다
ENV PYTHONPATH=/opt/DrivingForward/external/packnet_sfm:/opt/DrivingForward/external/dgp
RUN pip install pandas==2.0.3 xarray==2023.1.0 diskcache==5.6.3 boto3==1.34.74 tenacity==8.2.3

WORKDIR /opt/DrivingForward
