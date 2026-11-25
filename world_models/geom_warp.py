"""05-6: 기하학적 재투영 기준선. 시작 직전 추격 화면 한 장을, 정답 깊이와 카메라 포즈로 이후 시점에 재투영한다.

생성 모델 없이 '기하만으로' 다음 화면을 얼마나 맞출 수 있는지 재는 실험이다.
대상 프레임의 정답 깊이를 쓰므로(역방향 와핑) 실제 차량에서는 얻을 수 없는 오라클 조건이다.

사용:
    python world_models/geom_warp.py --runs town10_s100,town15_s0 --out outputs/05-6/geom
"""
import argparse
import json
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

W, H = 256, 144
FOV = 90.0


def carla_matrix(t):
    """CARLA Transform.get_matrix와 같은 4×4 행렬 (UE 왼손 좌표계, 도 단위 회전)."""
    cy, sy = math.cos(math.radians(t["yaw"])), math.sin(math.radians(t["yaw"]))
    cr, sr = math.cos(math.radians(t["roll"])), math.sin(math.radians(t["roll"]))
    cp, sp = math.cos(math.radians(t["pitch"])), math.sin(math.radians(t["pitch"]))
    return np.array([[cp * cy, cy * sp * sr - sy * cr, -cy * sp * cr - sy * sr, t["x"]],
                     [sy * cp, sy * sp * sr + cy * cr, -sy * sp * cr + cy * sr, t["y"]],
                     [sp, -cp * sr, cp * cr, t["z"]],
                     [0, 0, 0, 1]])


def load_rgb(p):
    return np.asarray(Image.open(p).convert("RGB").resize((W, H), Image.BICUBIC))


def load_depth(p):
    d = np.asarray(Image.open(p)).astype(np.float64)
    m = (d[..., 0] + d[..., 1] * 256 + d[..., 2] * 256 ** 2) / (256 ** 3 - 1) * 1000.0
    return np.asarray(Image.fromarray(m.astype(np.float32)).resize((W, H), Image.NEAREST))


def warp(src_rgb, src_pose, dst_depth, dst_pose):
    """대상 시점의 각 픽셀을 깊이로 3D 점으로 올린 뒤, 원본 카메라에 투영해 색을 가져온다."""
    f = W / (2 * math.tan(math.radians(FOV) / 2))
    u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    x = dst_depth                                  # CARLA 깊이 = 카메라 전방(x) 거리
    pts = np.stack([x, (u - W / 2) * x / f, -(v - H / 2) * x / f, np.ones_like(x)], -1).reshape(-1, 4)
    T = np.linalg.inv(carla_matrix(src_pose)) @ carla_matrix(dst_pose)
    q = pts @ T.T
    valid = q[:, 0] > 0.1
    us = f * q[:, 1] / np.maximum(q[:, 0], 1e-6) + W / 2
    vs = -f * q[:, 2] / np.maximum(q[:, 0], 1e-6) + H / 2
    valid &= (us >= 0) & (us < W) & (vs >= 0) & (vs < H)
    grid = torch.from_numpy(np.stack([us / W * 2 - 1, vs / H * 2 - 1], -1).reshape(1, H, W, 2)).float()
    img = torch.from_numpy(src_rgb).permute(2, 0, 1)[None].float()
    out = F.grid_sample(img, grid, mode="bilinear", align_corners=False)[0].permute(1, 2, 0).numpy()
    out[~valid.reshape(H, W)] = 0
    return out.round().clip(0, 255).astype(np.uint8), valid.reshape(H, W)


def psnr(a, b):
    return 10 * np.log10(255.0 ** 2 / np.mean((a.astype(np.float64) - b) ** 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/carla/test")
    ap.add_argument("--runs", default="town10_s100,town15_s0")
    ap.add_argument("--starts", default="100,400,700")
    ap.add_argument("--length", type=int, default=200)
    ap.add_argument("--out", default="outputs/05-6/geom")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    summary = {}
    for name in args.runs.split(","):
        run = os.path.join(args.raw, name)
        metas = [json.loads(l) for l in open(os.path.join(run, "meta.jsonl"))]
        fdir = lambda i: os.path.join(run, "frames", f"{metas[i]['frame']:08d}")
        # (1) 한 프레임 앞 재투영: 매 프레임 t-1 → t
        one, one_copy, one_cov = [], [], []
        for i in range(1, len(metas), 5):
            gt = load_rgb(os.path.join(fdir(i), "CHASE.jpg"))
            prev = load_rgb(os.path.join(fdir(i - 1), "CHASE.jpg"))
            w, val = warp(prev, metas[i - 1]["sensors"]["CHASE"], load_depth(os.path.join(fdir(i), "chase_depth.png")),
                          metas[i]["sensors"]["CHASE"])
            w = np.where(val[..., None], w, prev)       # 재투영할 수 없는 픽셀은 직전 화면으로 채운다
            one.append(psnr(w, gt))
            one_copy.append(psnr(prev, gt))
            one_cov.append(val.mean())
        # (2) 긴 구간: 시작 직전 한 장을 이후 k프레임에 재투영
        curves, copies, covs = [], [], []
        for s in [int(x) for x in args.starts.split(",")]:
            src = load_rgb(os.path.join(fdir(s - 1), "CHASE.jpg"))
            c, cp, cv = [], [], []
            for k in range(args.length):
                i = s + k
                gt = load_rgb(os.path.join(fdir(i), "CHASE.jpg"))
                w, val = warp(src, metas[s - 1]["sensors"]["CHASE"],
                              load_depth(os.path.join(fdir(i), "chase_depth.png")), metas[i]["sensors"]["CHASE"])
                c.append(psnr(np.where(val[..., None], w, src), gt))
                cp.append(psnr(src, gt))
                cv.append(val.mean())
                if k in (0, 40, 100, 196) and s == 400:
                    Image.fromarray(np.concatenate([gt, w], 0)).save(
                        os.path.join(args.out, f"warp_{name}_{s}_{k:03d}.jpg"), quality=92)
            curves.append(c)
            copies.append(cp)
            covs.append(cv)
        c, cp, cv = np.array(curves), np.array(copies), np.array(covs)
        seg = lambda a: [round(float(a[:, x:x + 8].mean()), 3) for x in (0, 40, 96, 192)]
        summary[name] = dict(one_step_warp_psnr=round(float(np.mean(one)), 2),
                             one_step_copy_psnr=round(float(np.mean(one_copy)), 2),
                             one_step_coverage=round(float(np.mean(one_cov)), 3),
                             long_warp_psnr_mean=round(float(c.mean()), 2), long_copy_psnr_mean=round(float(cp.mean()), 2),
                             long_warp_seg=seg(c), long_copy_seg=seg(cp), long_coverage_seg=seg(cv),
                             warp_curve=c.mean(0).round(3).tolist(), coverage_curve=cv.mean(0).round(3).tolist())
        print(name, {k: v for k, v in summary[name].items() if "curve" not in k}, flush=True)
    json.dump(summary, open(os.path.join(args.out, "geom.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
