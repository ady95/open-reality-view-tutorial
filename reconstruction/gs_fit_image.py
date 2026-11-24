"""3D 가우시안으로 사진 한 장 맞추기 — 책 03-1.

카메라 앞 공간에 3D 가우시안 N개를 뿌리고, gsplat 래스터라이저로 그린 결과가 사진과 같아지도록
위치·크기·회전·색·투명도를 경사하강법으로 학습한다. 3DGS의 "미분 가능한 래스터화"를 가장 작게 체험하는 예제다.

DriveStudio 컨테이너(gsplat 1.3.0)에서 실행:
    python /work/reconstruction/gs_fit_image.py --image <jpg> --out /work/outputs/03-1
"""
import argparse
import json
import math
import os
import time

import imageio
import numpy as np
import torch
import torch.nn.functional as F
from gsplat import rasterization


def psnr(a, b):
    return float(-10 * torch.log10(F.mse_loss(a, b)))


def fit(gt, n, iters, seed=0):
    torch.manual_seed(seed)
    dev = gt.device
    H, W, _ = gt.shape
    fx = W / (2 * math.tan(math.radians(60) / 2))
    K = torch.tensor([[fx, 0, W / 2], [0, fx, H / 2], [0, 0, 1]], device=dev)[None]
    viewmat = torch.eye(4, device=dev)[None]  # 카메라는 원점에서 +z를 본다

    # 카메라 앞 z = 2~4 범위에 무작위로 뿌린다
    bd = 2.0
    means = torch.empty(n, 3, device=dev).uniform_(-bd, bd)
    means[:, 2] = means[:, 2].abs() + 2.0
    params = {
        "means": torch.nn.Parameter(means),
        "scales": torch.nn.Parameter(torch.full((n, 3), math.log(0.02), device=dev)),
        "quats": torch.nn.Parameter(F.normalize(torch.randn(n, 4, device=dev), dim=-1)),
        "rgbs": torch.nn.Parameter(torch.logit(torch.rand(n, 3, device=dev).clamp(0.05, 0.95))),
        "opacities": torch.nn.Parameter(torch.logit(torch.full((n,), 0.5, device=dev))),
    }
    opt = torch.optim.Adam(params.values(), lr=0.01)

    def render():
        out, _, _ = rasterization(
            params["means"], F.normalize(params["quats"], dim=-1), torch.exp(params["scales"]),
            torch.sigmoid(params["opacities"]), torch.sigmoid(params["rgbs"]), viewmat, K, W, H)
        return out[0]

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        loss = F.l1_loss(render(), gt)
        opt.zero_grad()
        loss.backward()
        opt.step()
    torch.cuda.synchronize()
    sec = time.perf_counter() - t0
    with torch.no_grad():
        img = render().clamp(0, 1)
    return img, psnr(img, gt), sec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default="/work/outputs/03-1")
    ap.add_argument("--counts", default="1000,10000,100000")
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--width", type=int, default=533)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    img = imageio.imread(args.image)[..., :3]
    h = round(img.shape[0] * args.width / img.shape[1])
    gt = torch.tensor(img, dtype=torch.float32, device="cuda").permute(2, 0, 1)[None] / 255
    gt = F.interpolate(gt, size=(h, args.width), mode="area")[0].permute(1, 2, 0).contiguous()
    imageio.imwrite(os.path.join(args.out, "gt.jpg"), (gt.cpu().numpy() * 255).astype(np.uint8), quality=92)

    results = []
    for n in [int(c) for c in args.counts.split(",")]:
        torch.cuda.reset_peak_memory_stats()
        out, p, sec = fit(gt, n, args.iters)
        mem = torch.cuda.max_memory_allocated() / 2**30
        imageio.imwrite(os.path.join(args.out, f"fit_{n}.jpg"), (out.cpu().numpy() * 255).astype(np.uint8), quality=92)
        results.append(dict(gaussians=n, iters=args.iters, psnr=round(p, 2), seconds=round(sec, 1), peak_mem_gib=round(mem, 2)))
        print(results[-1])
    json.dump(results, open(os.path.join(args.out, "fit_results.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
