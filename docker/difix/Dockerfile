# Difix / Difix-ref (1-step 확산 보정) 실행 환경 — 책 03-5, 03-6
# 코드·가중치 라이선스: NVIDIA License (비상업 연구·평가 용도만 허용)
FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends git libgl1 libglib2.0-0 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

ARG DIFIX_COMMIT=c76edc5
RUN git clone https://github.com/nv-tlabs/Difix3D.git /opt/Difix3D \
    && cd /opt/Difix3D && git checkout ${DIFIX_COMMIT}

# requirements.txt의 고정 버전 + torch 2.1.2에 맞는 xformers
RUN pip install xformers==0.0.23.post1 peft==0.9.0 diffusers==0.25.1 huggingface-hub==0.25.1 \
        transformers==4.38.0 einops lpips "imageio[ffmpeg]" torchmetrics==1.3.2 accelerate==0.27.2 \
        "numpy<2" opencv-python-headless==4.9.0.80

ENV PYTHONPATH=/opt/Difix3D/src HF_HOME=/hf
WORKDIR /opt/Difix3D
