# 05-2 실습 환경: Self-Forcing (guandeh17/Self-Forcing, commit 33593df) 추론
# 빌드: docker build -t orv/selfforcing:33593df docker/selfforcing
# 학습용 패키지(TensorRT, wandb, onnx 등)는 추론에 필요 없어 뺐다.
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

RUN apt-get update && apt-get install -y --no-install-recommends git ffmpeg libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
      diffusers==0.31.0 transformers==4.49.0 tokenizers==0.21.0 accelerate==1.1.1 \
      numpy==1.24.4 opencv-python-headless==4.10.0.84 av==13.1.0 imageio imageio-ffmpeg \
      easydict ftfy omegaconf einops tqdm sentencepiece "huggingface_hub[cli]==0.29.3" \
      open_clip_torch==2.29.0 pydantic==2.10.6 lmdb matplotlib scikit-image

# FlashAttention 2 미리 빌드된 휠 (torch 2.5, CUDA 12, Python 3.11)
RUN pip install --no-cache-dir \
      https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl

RUN git clone https://github.com/guandeh17/Self-Forcing.git /opt/Self-Forcing \
 && cd /opt/Self-Forcing && git checkout 33593df
WORKDIR /opt/Self-Forcing
