"""05-6·07-2: 롤아웃 비교 영상을 만든다. 2×2 화면(1280×720)에 정답과 방법 세 가지를 같은 시각으로 나란히 놓는다.

각 칸은 256×144 생성 결과를 2.5배 확대한 것이다. 10Hz로 수집한 데이터이므로 10fps 실시간으로 저장한다.
생성 칸 아래의 점 4개는 지금 프레임이 청크(4프레임)의 몇 번째인지 보여 준다 (청크 경계의 튐 확인용).

사용 (저장소 루트에서, 한글 글꼴 경로 지정):
    python evaluation/make_videos.py --font /usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc --out outputs/videos
"""
import argparse
import os
import sys

import imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "world_models"))
from eval_dit import gt_image, load_run, sample  # noqa: E402
from rv_bench import load_model, to_img  # noqa: E402
from train_dit import SCALE, pool_feats  # noqa: E402

PW, PH = 640, 360          # 칸 하나 크기 (256×144의 2.5배)
FPS = 10

# 영상 이름 → (맵, 시작 프레임, 제목, [(칸 이름, 방법)])
VIDEOS = {
    "05-6_town10_fm_vs_meanflow": ("town10_s100", 100, "20초 자기회귀 롤아웃 — 학습한 맵(Town10)",
                                   [("시뮬레이션 GT", "gt"), ("정지 화면 (시작 직전 프레임)", "frozen"),
                                    ("Flow Matching NFE 4", "fm4"), ("MeanFlow NFE 2", "mf2")]),
    "05-6_town15_unseen": ("town15_s0", 400, "20초 자기회귀 롤아웃 — 처음 보는 맵(Town15)",
                           [("시뮬레이션 GT", "gt"), ("정지 화면 (시작 직전 프레임)", "frozen"),
                            ("Flow Matching NFE 4", "fm4"), ("MeanFlow NFE 2", "mf2")]),
    "07-2_nfe_and_taesd": ("town10_s100", 400, "환각과 깜빡임 비교 — 걸음 수와 경량 디코더",
                           [("시뮬레이션 GT", "gt"), ("MeanFlow NFE 1", "mf1"),
                            ("MeanFlow NFE 2", "mf2"), ("MeanFlow NFE 2 + TAESD", "mf2_taesd")]),
}


@torch.no_grad()
def rollout(meth, run, s, length, models, decs):
    if meth == "gt":
        return [gt_image(run, s + k) for k in range(length)]
    if meth == "frozen":
        return [decs["sd"](run["lat"][s - 1:s].cuda())[0]] * length
    ck, nfe, dec = dict(fm4=("outputs/05-4/fm_s/ckpt.pt", 4, "sd"), mf1=("outputs/05-5/mf_s/ckpt.pt", 1, "sd"),
                        mf2=("outputs/05-5/mf_s/ckpt.pt", 2, "sd"), mf2_taesd=("outputs/05-5/mf_s/ckpt.pt", 2, "tae"))[meth]
    if ck not in models:
        models[ck] = load_model(ck)
    model, targs = models[ck]
    C, K, pool = targs["ctx"], targs["chunk"], targs.get("cam_pool", 2)
    ctx, frames = run["lat"][s - C:s][None].cuda(), []
    for i in range(s, s + length - K + 1, K):
        f = pool_feats(torch.from_numpy(np.ascontiguousarray(run["feats"][i:i + K])), pool)[None].cuda().float()
        z = sample(model, ctx, f, run["motion"][i - C:i + K][None].cuda(), nfe, 0.0, seed=i)  # 07-2와 같은 시드
        frames += decs[dec](z[0])
        ctx = torch.cat([ctx, z], 1)[:, -C:]
    return frames


def panel(img, label, font, small, k=None, t=None):
    im = Image.fromarray(img).resize((PW, PH), Image.BICUBIC).convert("RGBA")
    ov = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    w = d.textlength(label, font=font)
    d.rounded_rectangle([10, 10, 26 + w, 52], 6, fill=(0, 0, 0, 160))
    d.text((18, 14), label, font=font, fill=(255, 255, 255, 255))
    if t is not None:
        txt = f"{t:4.1f}초"
        tw = d.textlength(txt, font=font)
        d.rounded_rectangle([PW - 26 - tw, 10, PW - 10, 52], 6, fill=(0, 0, 0, 160))
        d.text((PW - 18 - tw, 14), txt, font=font, fill=(255, 255, 255, 255))
    if k is not None:  # 청크 안 위치: ●○○○
        d.rounded_rectangle([10, PH - 40, 168, PH - 10], 6, fill=(0, 0, 0, 160))
        d.text((18, PH - 38), "청크", font=small, fill=(255, 255, 255, 255))
        for j in range(4):
            x = 78 + j * 22
            d.ellipse([x, PH - 32, x + 14, PH - 18], fill=(255, 210, 0, 255) if j == k else (120, 120, 120, 255))
    return np.asarray(Image.alpha_composite(im, ov).convert("RGB"))


def title_card(title, font, small):
    im = Image.new("RGB", (PW * 2, PH * 2), (16, 16, 16))
    d = ImageDraw.Draw(im)
    lines = [(title, font), ("", small),
             ("CARLA 1.0 시뮬레이션 · 카메라 6대 특징으로 추격 시점 생성 · RTX 3090", small),
             ("처음 2프레임만 정답 문맥, 이후 20초는 자기가 만든 화면을 이어 붙임", small),
             ("256×144 생성 결과를 2.5배 확대 · 10Hz 실시간 재생", small),
             ("「테슬라 리얼리티뷰 따라하기」 실험 영상 (테슬라 공식 자료가 아닌 오픈소스 재현)", small)]
    y = 230
    for txt, f in lines:
        d.text(((PW * 2 - d.textlength(txt, font=f)) / 2, y), txt, font=f, fill=(235, 235, 235))
        y += 64 if f is font else 44
    return np.asarray(im)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre", default="data/carla_pre")
    ap.add_argument("--raw", default="data/carla/test")
    ap.add_argument("--font", required=True, help="한글 글꼴 (.ttf/.ttc)")
    ap.add_argument("--font_index", type=int, default=1, help="Noto CJK .ttc에서 한국어는 1번")
    ap.add_argument("--videos", default=",".join(VIDEOS))
    ap.add_argument("--length", type=int, default=200)
    ap.add_argument("--thumb", type=int, default=60, help="정지 그림으로 저장할 프레임 번호")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    font = ImageFont.truetype(args.font, 28, index=args.font_index)
    small = ImageFont.truetype(args.font, 20, index=args.font_index)
    from diffusers import AutoencoderKL, AutoencoderTiny
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16).cuda().eval()
    tae = AutoencoderTiny.from_pretrained("madebyollin/taesd", torch_dtype=torch.float16).cuda().eval()
    decs = dict(sd=lambda z: to_img(vae.decode(z.half() / SCALE).sample), tae=lambda z: to_img(tae.decode(z.half()).sample))
    models = {}
    for name in args.videos.split(","):
        mapname, s, title, cells = VIDEOS[name]
        run = load_run(os.path.join(args.pre, mapname), os.path.join(args.raw, mapname))
        streams = [rollout(m, run, s, args.length, models, decs) for _, m in cells]
        n = min(len(x) for x in streams)
        frames = [title_card(title, font, small)] * (FPS * 3)
        for i in range(n):
            ps = [panel(st[i], lab, font, small, k=None if m in ("gt", "frozen") else i % 4, t=i / FPS if m == "gt" else None)
                  for st, (lab, m) in zip(streams, cells)]
            fr = np.concatenate([np.concatenate(ps[:2], 1), np.concatenate(ps[2:], 1)], 0)
            frames.append(fr)
            if i == args.thumb:
                Image.fromarray(fr).save(os.path.join(args.out, f"{name}_t{i:03d}.jpg"), quality=90)
        imageio.mimwrite(os.path.join(args.out, f"{name}.mp4"), frames, fps=FPS, quality=9, macro_block_size=8)
        print(name, len(frames), "frames", flush=True)


if __name__ == "__main__":
    main()
