# 05장 Track C 실습 환경: 비전 인코더(DINOv2·SigLIP 2), 이미지 VAE, 소형 Video DiT 학습
# 빌드: docker build -t orv/genvideo:1.0 docker/genvideo
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

RUN pip install --no-cache-dir \
      transformers==4.57.1 \
      diffusers==0.35.2 \
      accelerate==1.10.1 \
      safetensors==0.6.2 \
      einops==0.8.1 \
      timm==1.0.20 \
      pillow==11.3.0 \
      scikit-image==0.25.2 \
      lpips==0.1.4 \
      matplotlib==3.10.6 \
      sentencepiece==0.2.1 \
      protobuf==6.32.1 \
      imageio[ffmpeg]==2.37.0

ENV HF_HOME=/hf_cache
WORKDIR /work
