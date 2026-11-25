"""05-3: 동결 비전 인코더(DINOv2·SigLIP 2)로 카메라 6장의 특징을 뽑고 속도·메모리·토큰 수를 잰다.

사용:
    python world_models/encoder_bench.py --frame data/carla/run0/frames/<frame> --out outputs/05-3
출력:
    <out>/bench.json      인코더별 파라미터 수, 토큰 수, 지연(ms), 최대 메모리
    <out>/pca_<name>.jpg  6개 카메라 특징의 PCA 3성분을 RGB로 칠한 그림
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel

CAMS = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"]

# (이름, HF 모델 ID, 입력 크기 H×W, 패치 크기, 정규화 평균, 표준편차)
ENCODERS = {
    "dinov2-s": ("facebook/dinov2-small", (224, 392), 14, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    "dinov2-b": ("facebook/dinov2-base", (224, 392), 14, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    "dinov2-s-reg": ("facebook/dinov2-with-registers-small", (224, 392), 14, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    "siglip2-b": ("google/siglip2-base-patch16-256", (224, 400), 16, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
}


def load_images(frame_dir, size, mean, std, device):
    H, W = size
    ims = []
    for c in CAMS:
        im = Image.open(os.path.join(frame_dir, f"{c}.jpg")).convert("RGB").resize((W, H), Image.BICUBIC)
        ims.append(torch.from_numpy(np.asarray(im)).permute(2, 0, 1).float() / 255.0)
    x = torch.stack(ims).to(device)
    m = torch.tensor(mean, device=device).view(1, 3, 1, 1)
    s = torch.tensor(std, device=device).view(1, 3, 1, 1)
    return (x - m) / s


class Encoder(torch.nn.Module):
    """패치 토큰만 돌려주는 얇은 래퍼. 출력: (카메라 수, 토큰 수, 채널)"""

    def __init__(self, name):
        super().__init__()
        hf_id, self.size, self.patch, self.mean, self.std = ENCODERS[name]
        self.kind = "siglip" if "siglip" in name else "dinov2"
        model = AutoModel.from_pretrained(hf_id, torch_dtype=torch.float16)
        self.net = model.vision_model if self.kind == "siglip" else model
        self.n_skip = 0 if self.kind == "siglip" else 1 + getattr(model.config, "num_register_tokens", 0)

    @torch.no_grad()
    def forward(self, x):
        if self.kind == "siglip":
            # SigLIP 2 기본 모델은 위치 임베딩을 16×16 격자로 학습했다. 다른 종횡비는 보간해서 넣는다.
            out = self.net(pixel_values=x.half(), interpolate_pos_encoding=True)
        else:
            out = self.net(pixel_values=x.half())
        return out.last_hidden_state[:, self.n_skip:]


def pca_image(tokens, gh, gw):
    """모든 카메라의 토큰을 함께 PCA해 같은 색이 같은 의미가 되게 한다."""
    n, t, c = tokens.shape
    f = tokens.float().reshape(-1, c)
    f = f - f.mean(0, keepdim=True)
    _, _, v = torch.pca_lowrank(f, q=3)
    p = f @ v[:, :3]
    lo, hi = p.quantile(0.01, dim=0), p.quantile(0.99, dim=0)
    p = ((p - lo) / (hi - lo)).clamp(0, 1).reshape(n, gh, gw, 3)
    p = F.interpolate(p.permute(0, 3, 1, 2), scale_factor=8, mode="nearest").permute(0, 2, 3, 1)
    rows = [torch.cat(list(p[i:i + 3]), dim=1) for i in (0, 3)]
    return (torch.cat(rows, dim=0).cpu().numpy() * 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", required=True, help="카메라 6장이 있는 폴더 (CAM_*.jpg)")
    ap.add_argument("--out", default="outputs/05-3")
    ap.add_argument("--encoders", default=",".join(ENCODERS))
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = "cuda"
    results = {}
    for name in args.encoders.split(","):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        enc = Encoder(name).to(dev).eval()
        x = load_images(args.frame, enc.size, enc.mean, enc.std, dev)
        for _ in range(10):  # 예열
            tok = enc(x)
        torch.cuda.synchronize()
        ts = []
        for _ in range(args.iters):
            t0 = time.perf_counter()
            tok = enc(x)
            torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1000)
        gh, gw = enc.size[0] // enc.patch, enc.size[1] // enc.patch
        results[name] = dict(
            params_m=round(sum(p.numel() for p in enc.parameters()) / 1e6, 1),
            input=f"6x{enc.size[0]}x{enc.size[1]}", grid=f"{gh}x{gw}",
            tokens_per_cam=tok.shape[1], dim=tok.shape[2], tokens_total=tok.shape[0] * tok.shape[1],
            ms_mean=round(float(np.mean(ts)), 2), ms_p95=round(float(np.percentile(ts, 95)), 2),
            peak_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2))
        print(name, results[name])
        Image.fromarray(pca_image(tok, gh, gw)).save(os.path.join(args.out, f"pca_{name}.jpg"), quality=90)
        del enc
    json.dump(results, open(os.path.join(args.out, "bench.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
