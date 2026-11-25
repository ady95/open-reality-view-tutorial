#!/usr/bin/env bash
# 05장 학습·평가용 CARLA 데이터 일괄 수집 (책 05-4 참조).
# CARLA 서버가 떠 있는 상태에서 예제 저장소 루트에서 실행한다.
#   bash simulation/collect_batch.sh
# 학습: Town10HD_Opt, 시드 1~14, 날씨 순환 / 평가: Town10HD_Opt 시드 100, Town15 시드 0
set -euo pipefail
TICKS=${TICKS:-1000}
WEATHERS=(ClearNoon CloudyNoon ClearSunset WetNoon CloudySunset WetCloudyNoon SoftRainNoon)

collect() {  # $1=town $2=seed $3=weather $4=out
  if [ -f "$4/meta.jsonl" ] && [ "$(wc -l < "$4/meta.jsonl")" -ge "$TICKS" ]; then
    echo "skip $4"; return
  fi
  rm -rf "$4"
  docker run --rm --net=host -u "$(id -u):$(id -g)" -e HOME=/tmp \
    -v "$PWD":/work -v ~/work/carla_dist:/dist -w /work \
    pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime \
    sh -c "pip install -q --user /dist/carla-1.0-cp311-*.whl pillow >/dev/null 2>&1 && \
           python simulation/carla_collect.py --town $1 --seed $2 --weather $3 \
             --ticks $TICKS --warmup 30 --out $4"
}

for s in $(seq 1 14); do
  w=${WEATHERS[$(( (s - 1) % ${#WEATHERS[@]} ))]}
  collect Town10HD_Opt "$s" "$w" "data/carla/train/s$(printf %02d "$s")"
done
collect Town10HD_Opt 100 ClearNoon data/carla/test/town10_s100
collect Town15 0 ClearNoon data/carla/test/town15_s0
