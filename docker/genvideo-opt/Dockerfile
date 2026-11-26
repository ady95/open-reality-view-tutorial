# 07장 추론 최적화 실습 환경: 05장 이미지 + torch.compile(Triton)용 C 컴파일러 + ONNX/TensorRT + torchao(INT8)
# 빌드: docker build -t orv/genvideo:1.0 docker/genvideo && docker build -t orv/genvideo-opt:1.0 docker/genvideo-opt
# torchao는 0.9 이상이어야 transformers 4.57과 함께 import된다 (0.7은 Float8WeightOnlyConfig 부재로 실패)
FROM orv/genvideo:1.0

RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir \
      onnx==1.17.0 \
      onnxscript==0.1.0 \
      tensorrt-cu12==10.7.0 \
      torchao==0.9.0

ENV HF_HOME=/hf_cache
WORKDIR /work
