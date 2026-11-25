"""05-2: Self-Forcing(Wan2.1-T2V-1.3B 기반 자기회귀 비디오 생성) 속도·메모리 측정.

Self-Forcing 저장소 루트(/opt/Self-Forcing)에서 실행한다. 가중치는 wan_models/, checkpoints/에 둔다.
    python /work/world_models/selfforcing_bench.py --out /work/outputs/05-2 --runs 3
출력:
    <out>/bench.json   블록(청크)별 생성 시간, VAE 복원 시간, 최대 메모리
    <out>/sample.mp4   생성 영상 (832×480, 16 fps, 81프레임)
"""
import argparse
import json
import os
import time

import torch
from omegaconf import OmegaConf
from torchvision.io import write_video

from demo_utils.memory import DynamicSwapInstaller, gpu
from pipeline import CausalInferencePipeline

PROMPT = ("A dashcam-style video from a car driving through a sunny city street. "
          "The camera follows behind and slightly above a white sedan, with buildings, trees, "
          "parked cars and road markings passing by. Realistic, steady motion.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--frames", type=int, default=21, help="잠재 프레임 수 (21 → 영상 81프레임)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    cfg = OmegaConf.merge(OmegaConf.load("configs/default_config.yaml"), OmegaConf.load("configs/self_forcing_dmd.yaml"))
    torch.set_grad_enabled(False)
    t0 = time.perf_counter()
    pipe = CausalInferencePipeline(cfg, device="cuda")
    sd = torch.load("checkpoints/self_forcing_dmd.pt", map_location="cpu")
    pipe.generator.load_state_dict(sd["generator_ema"])
    pipe = pipe.to(dtype=torch.bfloat16)
    # 텍스트 인코더(umt5-xxl, 약 11GB)를 GPU에 상주시키면 24GB에서는 VAE 복원 단계에서 메모리가 모자란다.
    # 저장소의 inference.py와 같이 필요할 때만 GPU로 올리는 방식(DynamicSwapInstaller)을 쓴다.
    DynamicSwapInstaller.install_model(pipe.text_encoder, device=gpu)
    pipe.generator.to("cuda")
    pipe.vae.to("cuda")
    load_s = time.perf_counter() - t0
    params = {k: round(sum(p.numel() for p in getattr(pipe, k).parameters()) / 1e9, 3)
              for k in ("generator", "text_encoder", "vae")}
    print("loaded", load_s, params, flush=True)
    results = []
    for i in range(args.runs):
        torch.cuda.reset_peak_memory_stats()
        g = torch.Generator(device="cuda").manual_seed(i)
        noise = torch.randn([1, args.frames, 16, 60, 104], device="cuda", dtype=torch.bfloat16, generator=g)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        video = pipe.inference(noise=noise, text_prompts=[PROMPT], profile=True, low_memory=True)
        torch.cuda.synchronize()
        total = time.perf_counter() - t0
        n = video.shape[1]
        results.append(dict(total_s=round(total, 2), frames=n, fps=round(n / total, 2),
                            peak_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2)))
        print(results[-1], flush=True)
        pipe.vae.model.clear_cache()
        if i == 0:
            write_video(os.path.join(args.out, "sample.mp4"), (video[0].permute(0, 2, 3, 1) * 255).byte().cpu(), fps=16)
    json.dump(dict(load_s=round(load_s, 1), params_b=params, runs=results, prompt=PROMPT,
                   gpu=torch.cuda.get_device_name()), open(os.path.join(args.out, "bench.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
