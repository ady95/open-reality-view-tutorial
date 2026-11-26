"""06-2: CARLA 수집 런의 추격 카메라 프레임을 MP4로 내보낸다 (Cosmos-Transfer 입력용).

사용:
    python simulation/export_clip.py --run data/carla/test/town10_s100 --start 400 --frames 93 \
        --out outputs/06-2/input/town10_chase_400.mp4
"""
import argparse
import json
import os

import imageio
import numpy as np
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--cam", default="CHASE")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--frames", type=int, default=93, help="Cosmos-Transfer2.5 한 청크 = 93프레임")
    ap.add_argument("--size", default="848,480")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    w, h = (int(x) for x in a.size.split(","))
    metas = [json.loads(line) for line in open(os.path.join(a.run, "meta.jsonl"))]
    frames = []
    for m in metas[a.start:a.start + a.frames]:
        p = os.path.join(a.run, "frames", f"{m['frame']:08d}", f"{a.cam}.jpg")
        frames.append(np.asarray(Image.open(p).convert("RGB").resize((w, h), Image.BICUBIC)))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    imageio.mimwrite(a.out, frames, fps=a.fps, quality=9, macro_block_size=16)
    print(a.out, len(frames), "frames", f"{w}x{h}")


if __name__ == "__main__":
    main()
