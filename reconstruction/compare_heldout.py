"""두 재구성 모델(예: Street Gaussians와 OmniRe)의 학습에서 뺀 시점 렌더를 사람·차량 영역별로 비교한다 (책 03-3).

render_chase_drivestudio.py --heldout_dir 로 만든 두 폴더(render/, gt/)와 DriveStudio 전처리 결과의
동적 물체 마스크(dynamic_masks/human, vehicle)를 쓴다. 마스크 안쪽 픽셀만으로 PSNR을 잰다.

    python reconstruction/compare_heldout.py --a outputs/03-5/heldout --b outputs/03-3/heldout \
        --masks data/drivestudio/nuscenes/processed_10Hz/mini/000/dynamic_masks --out outputs/03-3/compare
"""
import argparse
import glob
import json
import os

import numpy as np
from PIL import Image


def load(p, size=None):
    im = Image.open(p)
    if size is not None and im.size != size:
        im = im.resize(size, Image.NEAREST if im.mode == "L" else Image.LANCZOS)
    return np.asarray(im).astype(np.float32) / 255.0


def masked_psnr(x, y, m):
    if m.sum() == 0:
        return None
    mse = (((x - y) ** 2).mean(-1) * m).sum() / m.sum()
    return float(10 * np.log10(1.0 / max(mse, 1e-10)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="기준 모델의 heldout 폴더")
    ap.add_argument("--b", required=True, help="비교 모델의 heldout 폴더")
    ap.add_argument("--names", default="StreetGS,OmniRe")
    ap.add_argument("--masks", required=True, help="dynamic_masks 폴더 (human/, vehicle/)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--crops", type=int, default=3, help="사람이 가장 많이 보이는 장면 몇 개를 잘라 저장할지")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    na, nb = args.names.split(",")

    rows = []
    for fp in sorted(glob.glob(os.path.join(args.a, "render", "*.png"))):
        name = os.path.basename(fp)                      # t010_c0.png
        t, c = int(name[1:4]), int(name.split("_c")[1][:-4])
        gt = load(os.path.join(args.a, "gt", name))
        size = (gt.shape[1], gt.shape[0])
        ra, rb = load(fp, size), load(os.path.join(args.b, "render", name), size)
        row = dict(name=name)
        for cls in ("human", "vehicle"):
            m = load(os.path.join(args.masks, cls, f"{t:03d}_{c}.png"), size) > 0.5
            row[f"{cls}_px"] = int(m.sum())
            row[f"{cls}_{na}"] = masked_psnr(ra, gt, m)
            row[f"{cls}_{nb}"] = masked_psnr(rb, gt, m)
        rows.append(row)

    summary = {}
    for cls in ("human", "vehicle"):
        sel = [r for r in rows if r[f"{cls}_px"] > 0]
        # 픽셀 가중 평균이 아니라 이미지별 PSNR의 평균 (DriveStudio 평가 방식과 같은 집계)
        summary[cls] = dict(images=len(sel),
                            **{n: round(float(np.mean([r[f"{cls}_{n}"] for r in sel])), 2) for n in (na, nb)},
                            b_better=sum(r[f"{cls}_{nb}"] > r[f"{cls}_{na}"] for r in sel))

    # 사람이 가장 많이 보이는 장면을 GT | A | B 로 잘라 붙인다
    for r in sorted(rows, key=lambda r: -r["human_px"])[:args.crops]:
        name = r["name"]
        t, c = int(name[1:4]), int(name.split("_c")[1][:-4])
        gt = Image.open(os.path.join(args.a, "gt", name)).convert("RGB")
        m = load(os.path.join(args.masks, "human", f"{t:03d}_{c}.png"), gt.size) > 0.5
        ys, xs = np.nonzero(m)
        cx, cy = int(xs.mean()), int(ys.mean())
        w, h = 200, 150
        box = (max(0, min(cx - w // 2, gt.width - w)), max(0, min(cy - h // 2, gt.height - h)))
        box = box + (box[0] + w, box[1] + h)
        tiles = [gt, Image.open(os.path.join(args.a, "render", name)).convert("RGB").resize(gt.size),
                 Image.open(os.path.join(args.b, "render", name)).convert("RGB").resize(gt.size)]
        strip = Image.new("RGB", (w * 3 * 2, h * 2), "white")
        for k, im in enumerate(tiles):
            strip.paste(im.crop(box).resize((w * 2, h * 2), Image.LANCZOS), (k * w * 2, 0))
        strip.save(os.path.join(args.out, f"crop_{name[:-4]}.jpg"), quality=92)

    json.dump(dict(summary=summary, per_image=rows), open(os.path.join(args.out, "compare.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
