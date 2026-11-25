# 05-1 실습 환경: Vista (OpenDriveLab/Vista, commit cc9821b) 주행 월드 모델 추론
# 빌드: docker build -t orv/vista:cc9821b docker/vista
# 저장소 권장 환경(PyTorch 2.0.1, CUDA 11.7)에 맞추고, 학습·시각화 도구(deepspeed, wandb, streamlit 등)는 뺐다.
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
# tzdata가 시간대를 물어보며 빌드가 멈추는 것을 막는다
ENV DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC

RUN apt-get update && apt-get install -y --no-install-recommends git ffmpeg libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir xformers==0.0.21 \
 && pip install --no-cache-dir \
      einops==0.6.1 fire==0.5.0 imageio==2.31.1 imageio-ffmpeg==0.4.8 kornia==0.6.9 \
      natsort==8.4.0 omegaconf==2.3.0 open-clip-torch==2.20.0 opencv-python-headless==4.8.0.76 \
      pytorch-lightning==2.0.1 safetensors==0.4.1 scipy==1.10.1 timm==0.9.2 \
      transformers==4.19.1 tokenizers==0.12.1 torchmetrics==1.0.1 tqdm fsspec \
      "numpy<2" huggingface_hub==0.17.3 matplotlib==3.7.2 \
      git+https://github.com/openai/CLIP.git

RUN git clone https://github.com/OpenDriveLab/Vista.git /opt/Vista \
 && cd /opt/Vista && git checkout cc9821b
WORKDIR /opt/Vista
