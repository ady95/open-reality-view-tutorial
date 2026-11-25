"""Difix / Difix-ref 1-step 확산 보정과 평가 — 책 03-5.

3DGS 렌더 영상을 NVIDIA Difix(SD-Turbo 기반 1-step 확산 모델)로 보정하고 세 가지를 잰다.
  1) 학습에서 뺀 시점(정답 사진 있음): 보정 전후 PSNR / SSIM / LPIPS, 그리고 "오히려 나빠진 픽셀" 비율
  2) 추격 시점(정답 없음): 이웃 프레임 차이로 본 깜빡임(시간 일관성)
  3) 프레임당 처리 시간과 최대 GPU 메모리

Difix 컨테이너(orv/difix:c76edc5)에서 실행:
    python /work/reconstruction/difix_refine.py heldout --root /work/outputs/03-5 \
        --images /work/data/drivestudio/nuscenes/processed_10Hz/mini/000/images
    python /work/reconstruction/difix_refine.py chase --root /work/outputs/03-5
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import torch
from PIL import Image

PROMPT = "remove degradation"
WORK_SIZE = (1024, 576)  # Difix 학습 해상도(16:9). 입력을 이 크기로 바꿔 보정한 뒤 원래 크기로 되돌린다


def load_pipe(model_id):
    from pipeline_difix import DifixPipeline
    pipe = DifixPipeline.from_pretrained(model_id, trust_remote_code=True)
    pipe.set_progress_bar_config(disable=True)
    return pipe.to("cuda")


def refine(pipe, img, ref=None):
    """PIL 이미지 한 장을 1-step으로 보정한다 (Difix 공식 설정: timestep 199, guidance 0)."""
    size = img.size
    kw = dict(image=img.resize(WORK_SIZE, Image.LANCZOS), num_inference_steps=1, timesteps=[199],
              guidance_scale=0.0, generator=torch.Generator("cuda").manual_seed(0))
    if ref is not None:
        kw["ref_image"] = ref.resize(WORK_SIZE, Image.LANCZOS)
    out = pipe(PROMPT, **kw).images[0]
    return out.resize(size, Image.LANCZOS)


def run_dir(pipe, names, src_dir, out_dir, ref_dir=None):
    os.makedirs(out_dir, exist_ok=True)
    times = []
    torch.cuda.reset_peak_memory_stats()
    for n in names:
        img = Image.open(os.path.join(src_dir, n)).convert("RGB")
        ref = Image.open(os.path.join(ref_dir, n)).convert("RGB") if ref_dir else None
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = refine(pipe, img, ref)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        out.save(os.path.join(out_dir, n))
    ms = np.array(times[3:] or times) * 1000  # 앞 3장은 warm-up
    return dict(frames=len(names), ms_mean=round(float(ms.mean()), 1),
                peak_mem_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2))


def to_t(path):
    a = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255
    return torch.from_numpy(a).permute(2, 0, 1)[None].cuda()


def heldout(args):
    """정답 사진이 있는 시점에서 보정 전후를 비교한다."""
    import lpips
    from torchmetrics.functional import structural_similarity_index_measure as ssim

    root = args.root
    names = sorted(os.path.basename(p) for p in glob.glob(f"{root}/heldout/render/*.png"))
    # Difix-ref 기준 이미지: 같은 카메라의 바로 이전 프레임(학습에 쓴 실제 사진). 정답 사진 자체는 쓰지 않는다
    ref_dir = f"{root}/heldout/ref"
    os.makedirs(ref_dir, exist_ok=True)
    for n in names:
        t, c = int(n[1:4]), int(n[6])
        tr = t - 1 if t > 0 else t + 1
        size = Image.open(f"{root}/heldout/gt/{n}").size
        Image.open(f"{args.images}/{tr:03d}_{c}.jpg").convert("RGB").resize(size, Image.LANCZOS).save(f"{ref_dir}/{n}")

    perf = {}
    for key, model_id, rdir in (("difix", "nvidia/difix", None), ("difix_ref", "nvidia/difix_ref", ref_dir)):
        pipe = load_pipe(model_id)
        perf[key] = run_dir(pipe, names, f"{root}/heldout/render", f"{root}/heldout/{key}", rdir)
        del pipe
        torch.cuda.empty_cache()

    lp = lpips.LPIPS(net="alex", verbose=False).cuda()
    rows = {k: {"psnr": [], "ssim": [], "lpips": [], "worse_px": []} for k in ("render", "difix", "difix_ref")}
    for n in names:
        gt = to_t(f"{root}/heldout/gt/{n}")
        base_err = None
        for k in rows:
            x = to_t(f"{root}/heldout/{k}/{n}")
            err = (x - gt).abs().mean(1)  # 픽셀별 오차
            rows[k]["psnr"].append(float(-10 * torch.log10(((x - gt) ** 2).mean())))
            rows[k]["ssim"].append(float(ssim(x, gt, data_range=1.0)))
            rows[k]["lpips"].append(float(lp(x * 2 - 1, gt * 2 - 1)))
            if k == "render":
                base_err = err
            # 보정 후 오차가 렌더보다 0.05(8bit 기준 약 13) 이상 커진 픽셀 = 정답에서 멀어진 픽셀
            rows[k]["worse_px"].append(float(((err - base_err) > 0.05).float().mean()))
    summary = {k: {m: round(float(np.mean(v)), 4) for m, v in r.items()} for k, r in rows.items()}
    result = dict(pairs=len(names), work_size=WORK_SIZE, metrics=summary, perf=perf)
    json.dump(result, open(f"{root}/heldout_metrics.json", "w"), indent=1)
    print(json.dumps(result, indent=1))


def flicker(frames_dir):
    """이웃 프레임 사이 평균 절대 차이(0~255). 카메라가 움직이므로 0이 될 수는 없고, 상대 비교용이다."""
    fs = sorted(glob.glob(f"{frames_dir}/*.png"))
    prev, diffs = None, []
    for f in fs:
        a = np.asarray(Image.open(f).convert("RGB"), dtype=np.float32)
        if prev is not None:
            diffs.append(np.abs(a - prev).mean())
        prev = a
    return round(float(np.mean(diffs)), 2)


def chase(args):
    """정답이 없는 추격 시점을 보정하고 깜빡임을 비교한다."""
    import imageio
    root = args.root
    names = sorted(os.path.basename(p) for p in glob.glob(f"{root}/chase/*.png"))
    perf = {}
    for key, model_id, rdir in (("difix", "nvidia/difix", None), ("difix_ref", "nvidia/difix_ref", f"{root}/chase_ref")):
        pipe = load_pipe(model_id)
        perf[key] = run_dir(pipe, names, f"{root}/chase", f"{root}/chase_{key}", rdir)
        del pipe
        torch.cuda.empty_cache()
        w = imageio.get_writer(f"{root}/chase_{key}.mp4", fps=10, macro_block_size=1)
        for n in names:
            w.append_data(np.asarray(Image.open(f"{root}/chase_{key}/{n}")))
        w.close()
    fl = {k: flicker(f"{root}/{d}") for k, d in (("render", "chase"), ("difix", "chase_difix"), ("difix_ref", "chase_difix_ref"))}
    result = dict(frames=len(names), work_size=WORK_SIZE, flicker_mean_abs_diff=fl, perf=perf)
    json.dump(result, open(f"{root}/chase_metrics.json", "w"), indent=1)
    print(json.dumps(result, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["heldout", "chase"])
    ap.add_argument("--root", default="/work/outputs/03-5")
    ap.add_argument("--images", default="/work/data/drivestudio/nuscenes/processed_10Hz/mini/000/images")
    args = ap.parse_args()
    {"heldout": heldout, "chase": chase}[args.mode](args)


if __name__ == "__main__":
    main()
