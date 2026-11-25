"""05-4: CARLA 수집 데이터를 학습용 텐서로 미리 바꿔 둔다.

입력 카메라 6장 → 동결 DINOv2 특징(카메라당 8×14 토큰으로 평균 풀링)
추격 카메라 정답 → 256×144로 줄인 뒤 SD VAE 잠재(4×18×32)
자차 포즈 → 이전 프레임 대비 이동량(전진 m, 좌우 m, yaw 도)

사용:
    python world_models/precompute.py --runs data/carla/train/* data/carla/test/* --out data/carla_pre
출력 (런마다):
    <out>/<런 이름>/feats.npy    (N, 6, 112, 384) float16
    <out>/<런 이름>/latents.npy  (N, 4, 18, 32)   float16  (스케일 0.18215 적용 전 원값)
    <out>/<런 이름>/motion.npy   (N, 3)           float32
    <out>/<런 이름>/frames.json  프레임 번호 목록
"""
import argparse
import json
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL
from PIL import Image
from transformers import AutoModel

CAMS = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"]
ENC_HW = (224, 392)     # DINOv2 입력 (14의 배수) → 16×28 패치
POOL_HW = (8, 14)       # 카메라당 토큰 격자 (2×2 평균 풀링)
CHASE_WH = (256, 144)   # 추격 카메라 정답 해상도 → VAE 잠재 32×18
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load(path, wh):
    im = Image.open(path).convert("RGB").resize(wh, Image.BICUBIC)
    return torch.from_numpy(np.array(im)).permute(2, 0, 1).float() / 255.0


def ego_motion(prev, cur):
    """이전 자차 포즈 기준으로 본 현재 포즈의 이동량. CARLA는 왼손 좌표계(y 오른쪽)이므로 y 부호를 뒤집어
    nuScenes와 같은 '왼쪽이 +' 규약으로 맞춘다."""
    dx, dy = cur["x"] - prev["x"], cur["y"] - prev["y"]
    yaw = math.radians(prev["yaw"])
    fwd = dx * math.cos(yaw) + dy * math.sin(yaw)
    right = -dx * math.sin(yaw) + dy * math.cos(yaw)
    dyaw = (cur["yaw"] - prev["yaw"] + 180.0) % 360.0 - 180.0
    return [fwd, -right, -dyaw]


@torch.no_grad()
def siglip2_only(args, dev):
    """SigLIP 2 base(patch16): 224×400 입력 → 14×25 패치 → 4×7로 적응 평균 풀링 (DINOv2 실험과 토큰 수를 맞춘다)."""
    enc = AutoModel.from_pretrained("google/siglip2-base-patch16-256", dtype=torch.float16).vision_model.to(dev).eval()
    for run in args.runs:
        od = os.path.join(args.out, os.path.basename(run.rstrip("/")))
        path = os.path.join(od, "feats_siglip2.npy")
        if os.path.exists(path):
            continue
        metas = [json.loads(l) for l in open(os.path.join(run, "meta.jsonl"))]
        out = np.zeros((len(metas), 6, 28, 768), np.float16)
        for s in range(0, len(metas), args.batch):
            fd = [os.path.join(run, "frames", f"{m['frame']:08d}") for m in metas[s:s + args.batch]]
            x = torch.stack([load(os.path.join(d, f"{c}.jpg"), (400, 224)) for d in fd for c in CAMS])
            x = ((x - 0.5) / 0.5).to(dev).half()
            tok = enc(pixel_values=x, interpolate_pos_encoding=True).last_hidden_state   # (B*6, 350, 768)
            tok = F.adaptive_avg_pool2d(tok.reshape(-1, 14, 25, 768).permute(0, 3, 1, 2).float(), (4, 7))
            out[s:s + len(fd)] = tok.flatten(2).transpose(1, 2).reshape(len(fd), 6, 28, 768).half().cpu().numpy()
        np.save(path, out)
        print(os.path.basename(od), "siglip2", out.shape)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", default="data/carla_pre")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--siglip2", action="store_true",
                    help="05-3 비교용: SigLIP 2 특징만 추가로 뽑아 feats_siglip2.npy (N, 6, 28, 768)로 저장")
    args = ap.parse_args()
    dev = "cuda"
    if args.siglip2:
        return siglip2_only(args, dev)
    enc = AutoModel.from_pretrained("facebook/dinov2-small", dtype=torch.float16).to(dev).eval()
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16).to(dev).eval()
    for run in args.runs:
        name = os.path.basename(run.rstrip("/"))
        od = os.path.join(args.out, name)
        if os.path.exists(os.path.join(od, "motion.npy")):
            print("skip", name)
            continue
        os.makedirs(od, exist_ok=True)
        metas = [json.loads(l) for l in open(os.path.join(run, "meta.jsonl"))]
        n = len(metas)
        feats = np.lib.format.open_memmap(os.path.join(od, "feats.npy"), "w+", np.float16, (n, 6, 112, 384))
        lats = np.zeros((n, 4, 18, 32), np.float16)
        for s in range(0, n, args.batch):
            ms = metas[s:s + args.batch]
            fd = [os.path.join(run, "frames", f"{m['frame']:08d}") for m in ms]
            x = torch.stack([load(os.path.join(d, f"{c}.jpg"), ENC_HW[::-1]) for d in fd for c in CAMS])
            x = ((x - MEAN) / STD).to(dev).half()
            tok = enc(pixel_values=x).last_hidden_state[:, 1:]                      # (B*6, 448, 384)
            tok = tok.reshape(-1, 16, 28, 384).permute(0, 3, 1, 2)
            tok = F.avg_pool2d(tok, 2).flatten(2).transpose(1, 2)                   # (B*6, 112, 384)
            feats[s:s + len(ms)] = tok.reshape(len(ms), 6, 112, 384).cpu().numpy()
            y = torch.stack([load(os.path.join(d, "CHASE.jpg"), CHASE_WH) for d in fd]).to(dev).half()
            lats[s:s + len(ms)] = vae.encode(y * 2 - 1).latent_dist.mean.cpu().numpy()
        feats.flush()
        motion = np.array([[0.0, 0.0, 0.0]] + [ego_motion(metas[i - 1]["ego"], metas[i]["ego"]) for i in range(1, n)],
                          np.float32)
        np.save(os.path.join(od, "latents.npy"), lats)
        np.save(os.path.join(od, "motion.npy"), motion)
        json.dump([m["frame"] for m in metas], open(os.path.join(od, "frames.json"), "w"))
        print(name, n, "frames | speed mean %.2f m/s" % (np.linalg.norm(motion[:, :2], axis=1).mean() * 10))


if __name__ == "__main__":
    main()
