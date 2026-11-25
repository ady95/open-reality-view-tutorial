"""05-4~05-6: 청크 하나(문맥 2 + 생성 4프레임)를 만드는 데 드는 시간을 단계별로 잰다. GPU를 혼자 쓸 때 실행한다.

단계: DINOv2-S로 카메라 6장 인코딩(생성 4프레임 분량) → Video DiT NFE회 → SD VAE로 4프레임 복원
사용:
    python world_models/bench_dit.py --ckpt outputs/05-4/fm_s/ckpt.pt --nfe 1,2,4,8,25 --out outputs/05-4/bench.json
"""
import argparse
import json
import time

import numpy as np
import torch
from diffusers import AutoencoderKL
from transformers import AutoModel

from dit import VideoDiT
from eval_dit import decode, sample


def timed(fn, iters):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    return round(float(np.median(ts)), 2)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--nfe", default="1,2,4,8,25")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    ck = torch.load(args.ckpt, map_location="cuda")
    sd = ck["ema"]
    model = VideoDiT(ck["args"]["size"], cam_tokens=sd["cam_pos"].shape[1] // 6,
                     feat_dim=sd["cam_in.1.weight"].shape[1]).cuda().eval()
    model.load_state_dict(sd)
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16).cuda().eval()
    enc = AutoModel.from_pretrained("facebook/dinov2-small", dtype=torch.float16).cuda().eval()
    C, K = ck["args"]["ctx"], ck["args"]["chunk"]
    ctx = torch.randn(1, C, 4, 18, 32, device="cuda")
    feats = torch.randn(1, K, 6, sd["cam_pos"].shape[1] // 6, sd["cam_in.1.weight"].shape[1], device="cuda")
    motion = torch.zeros(1, C + K, 3, device="cuda")
    imgs = torch.randn(K * 6, 3, 224, 392, device="cuda", dtype=torch.float16)
    z = torch.randn(1, K, 4, 18, 32, device="cuda")
    res = dict(ckpt=args.ckpt, gpu=torch.cuda.get_device_name(), frames_per_chunk=K,
               encoder_ms=timed(lambda: enc(pixel_values=imgs), args.iters),
               vae_decode_ms=timed(lambda: decode(vae, z), args.iters), dit_ms={})
    for n in [int(x) for x in args.nfe.split(",")]:
        res["dit_ms"][n] = timed(lambda: sample(model, ctx, feats, motion, n), args.iters)
    res["chunk_ms"] = {n: round(res["encoder_ms"] + res["vae_decode_ms"] + d, 2) for n, d in res["dit_ms"].items()}
    res["fps"] = {n: round(K / (ms / 1000), 1) for n, ms in res["chunk_ms"].items()}
    torch.cuda.reset_peak_memory_stats()
    sample(model, ctx, feats, motion, 1)
    res["dit_peak_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    print(json.dumps(res, indent=1))
    json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
