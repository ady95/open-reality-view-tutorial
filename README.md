# open-reality-view-tutorial

위키독스 책 **「테슬라 리얼리티뷰 따라하기」**([https://wikidocs.net/book/22323](https://wikidocs.net/book/22323))의 예제 코드 저장소입니다.

실제 차량이나 촬영 장비 없이, 공개 주행 데이터·CARLA·오픈소스 AI 모델만으로 차량 뒤 상공 가상 카메라 시점의 주행 영상을 만드는 세 가지 방법을 구현하고 비교합니다.

| 경로 | 방법 | 책 |
|---|---|---|
| Track A | 최적화 기반 3D Gaussian Splatting + Difix 보정 | 03장 |
| Track B | Feed-forward 3DGS (DrivingForward 등) | 04장 |
| Track C | 시각 특징 조건부 Video DiT + Flow Matching / MeanFlow + 자기회귀 생성 | 05장 |

> 이 프로젝트는 Tesla, Inc.와 관련이 없는 독립적인 학습·연구용 자료입니다. 연구용 시각화이며 차량 제어나 운전 중 판단에 사용하지 마십시오.

## 기준 환경

- NVIDIA RTX 3090 24GB × 1, RAM 64GB, NVMe SSD 1~2TB
- Ubuntu 22.04 / 24.04, NVIDIA 드라이버, Docker, NVIDIA Container Toolkit
- 프로젝트별 Docker 컨테이너 (책 02-1 참조)

## 빠른 확인

```bash
git clone https://github.com/ady95/open-reality-view-tutorial.git ~/work/open-reality-view-tutorial
cd ~/work/open-reality-view-tutorial
docker run --rm --gpus "device=0" -v $PWD:/work -w /work \
  pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime python env/gpu_check.py
```

## 디렉터리 구성

| 경로 | 내용 | 책 |
|---|---|---|
| `env/` | GPU 환경 점검 스크립트 | 02-1 |
| `docker/` | 프로젝트별 Dockerfile | 02장~ |
| `datasets/` | 공개 데이터셋 어댑터 (nuScenes, Waymo, PandaSet 등) | 02-2, 02-5 |
| `simulation/` | CARLA 데이터 생성 | 02-4 |
| `geometry/` | 좌표 변환, 추격 카메라 포즈, 깊이 워핑 | 02-3 |
| `reconstruction/` | DriveStudio · 3DGS · Difix | 03장 |
| `feedforward/` | DrivingForward 등 | 04장 |
| `world_models/` | Cosmos · Self-Forcing · Oasis | 05-1, 05-2 |
| `models/` | Conditional Video DiT · MeanFlow 구현 | 05-3 ~ 05-6 |
| `training/` | 학습 파이프라인 | 05장, 06장 |
| `inference/` | 스트리밍 추론 | 05-6, 07장 |
| `optimization/` | BF16 · INT8 · TensorRT | 07-1 |
| `evaluation/` | 공통 평가 지표와 벤치마크 | 07-2 |
| `viewer/` | Open Reality View Lab 통합 뷰어 | 03-6, 08-1 |
| `experiments/` | 실험 설정과 결과 기록 | 전체 |

책의 진행에 맞춰 디렉터리를 채워 갑니다. 아직 비어 있는 디렉터리는 해당 장이 공개될 때 추가됩니다.

## 라이선스

이 저장소의 코드는 [MIT License](LICENSE)를 따릅니다.

모델 가중치와 데이터셋은 이 저장소에 포함하지 않으며, 각 장의 준비 스크립트가 공식 배포처에서 내려받습니다. 가중치·데이터셋은 코드와 라이선스가 다를 수 있습니다. 예를 들어 Difix/Difix-ref는 NVIDIA 비상업 라이선스, Argoverse 2는 CC BY-NC-SA 4.0입니다. 사용 전 각 배포처의 조건을 확인하십시오.
