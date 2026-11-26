"""06-1·06-2: 실험 결과 MP4를 YouTube용 비교 영상(1920×1080)으로 다시 묶는다. 모델을 다시 돌리지 않는다.

  real   : 06-1 실사 롤아웃. eval_dit --video가 만든 영상(위 Pseudo-GT, 아래 생성)에서 학습 설정 5가지를 3×2로 배치
  cosmos : 06-2 Cosmos-Transfer2.5. 입력·윤곽선·출력을 나란히 놓고, 원속 → 바뀐 곳 표시(정지) → 0.5배속 순서로 이어 붙임

사용 (저장소 루트에서):
    python evaluation/make_videos_06.py real   --font /usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc --out outputs/videos
    python evaluation/make_videos_06.py cosmos --font /usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc --out outputs/videos
"""
import argparse
import os

import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H = 1920, 1080
FPS = 10
BG = (16, 16, 16)
NOTE = "「테슬라 리얼리티뷰 따라하기」 실험 영상 (테슬라 공식 자료가 아닌 오픈소스 재현)"


def fonts(path, idx):
    return {s: ImageFont.truetype(path, s, index=idx) for s in (22, 26, 30, 44)}


def label(d, xy, text, f, align="left"):
    x, y = xy
    w = d.textlength(text, font=f)
    if align == "right":
        x -= w + 16
    d.rounded_rectangle([x, y, x + w + 16, y + f.size + 14], 6, fill=(0, 0, 0, 170))
    d.text((x + 8, y + 4), text, font=f, fill=(255, 255, 255, 255))


def paste(canvas, img, box):
    x0, y0, x1, y1 = box
    canvas.paste(Image.fromarray(img).resize((x1 - x0, y1 - y0), Image.BICUBIC), (x0, y0))


def title_card(lines, F):
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    y = 340
    for i, t in enumerate(lines + ["", NOTE]):
        f = F[44] if i == 0 else F[26]
        d.text(((W - d.textlength(t, font=f)) / 2, y), t, font=f, fill=(235, 235, 235))
        y += 80 if i == 0 else 48
    return np.asarray(im)


def read(path):
    return [f for f in imageio.get_reader(path)]


# ------------------------------------------------------------------ 06-1
REAL = [("실사 Pseudo-GT (3DGS 재구성)", None), ("CARLA만", "carla_only"), ("혼합 25%", "mix25"),
        ("혼합 50%", "mix50"), ("CARLA 30k + 혼합 50% 미세조정", "ft50"), ("실사만", "real_only")]


def make_real(args, F):
    frames = [title_card(["Sim-to-Real: 실사 장면에서의 20초 생성 비교 (nuScenes)",
                          "CARLA만 학습한 모델과 실사 Pseudo-GT를 섞어 학습한 모델의 자기회귀 롤아웃",
                          "256×144 생성 결과를 2.5배 확대 · 10Hz 실시간 재생 · 장면당 16초"], F)] * (FPS * 3)
    for scene in args.scenes.split(","):
        streams = []
        for _, d in REAL:
            v = read(os.path.join(args.real_dir, d or "mix25", "real_roll", f"roll_scene-{scene}_10.mp4"))
            streams.append([f[:144] if d is None else f[144:] for f in v])   # 위: Pseudo-GT, 아래: 생성
        n = min(len(s) for s in streams)
        for i in range(n):
            im = Image.new("RGB", (W, H), BG)
            for j, ((lab, _), st) in enumerate(zip(REAL, streams)):
                x, y = (j % 3) * 640, 120 + (j // 3) * 360
                paste(im, st[i], (x, y, x + 640, y + 360))
            d = ImageDraw.Draw(im, "RGBA")
            for j, (lab, _) in enumerate(REAL):
                label(d, ((j % 3) * 640 + 10, 130 + (j // 3) * 360), lab, F[26])
            label(d, (630, 130), f"{i / FPS:4.1f}초", F[26], align="right")
            d.text((24, 30), f"nuScenes scene-{scene} · 처음 2프레임만 정답 문맥, 이후는 자기가 만든 화면을 이어 붙임",
                   font=F[30], fill=(235, 235, 235))
            d.text((24, 862), "· 정답은 실제 사진이 아니라 3DGS로 재구성한 추격 시점(Pseudo-GT)입니다. 화면 위쪽의 번짐은 재구성의 한계입니다.",
                   font=F[22], fill=(200, 200, 200))
            d.text((24, 900), "· CARLA만 학습한 모델은 시간이 지나면 CARLA풍 색 덩어리로 바뀌고, 실사를 섞을수록 도로 구조가 오래 유지됩니다.",
                   font=F[22], fill=(200, 200, 200))
            d.text((24, 938), "· '실사만'은 처음부터 어둡고 흐린 평균 화면을 내기 때문에 PSNR이 높게 나옵니다 (좋은 결과로 읽으면 안 됨).",
                   font=F[22], fill=(200, 200, 200))
            d.text((24, 1010), NOTE, font=F[22], fill=(150, 150, 150))
            fr = np.asarray(im)
            frames.append(fr)
            if i == args.thumb:
                Image.fromarray(fr).save(os.path.join(args.out, f"06-1_real_rollout_{scene}_t{i:03d}.jpg"), quality=90)
    out = os.path.join(args.out, "06-1_real_rollout.mp4")
    imageio.mimwrite(out, frames, fps=FPS, quality=9, macro_block_size=8)
    print(out, len(frames), "frames")


# ------------------------------------------------------------------ 06-2
# 46번째 프레임에서 입력과 출력이 달라진 곳 (848×480 원본 좌표)
BOXES = [((188, 134, 334, 256), "구급차 줄무늬: 주황 → 검정", "below"),
         ((318, 128, 458, 182), "교차로 노란 표시 → 흰 격자", "right")]


def make_cosmos(args, F):
    src = read(os.path.join(args.cosmos_dir, "input", "town10_chase_400.mp4"))
    edge = read(os.path.join(args.cosmos_dir, "out_edge_2gpu", "town10_chase_400_control_edge.mp4"))
    gen = read(os.path.join(args.cosmos_dir, "out_edge_2gpu", "town10_chase_400.mp4"))
    n = min(len(src), len(edge), len(gen))
    sx = 960 / 848

    def compose(i, mode, boxes=False):
        im = Image.new("RGB", (W, H), BG)
        paste(im, src[i], (0, 0, 960, 540))
        paste(im, gen[i], (960, 0, 1920, 540))
        paste(im, edge[i], (0, 540, 960, 1080))
        d = ImageDraw.Draw(im, "RGBA")
        if boxes:
            for (x0, y0, x1, y1), txt, where in BOXES:
                for ox in (0, 960):
                    d.rectangle([ox + x0 * sx, y0 * sx, ox + x1 * sx, y1 * sx], outline=(255, 210, 0, 255), width=4)
                xy = (960 + x0 * sx, y1 * sx + 6) if where == "below" else (960 + x1 * sx + 6, y0 * sx)
                label(d, xy, txt, F[22])
        label(d, (10, 10), "입력: CARLA 시뮬레이션", F[30])
        label(d, (970, 10), "출력: Cosmos-Transfer2.5 (실사풍)", F[30])
        label(d, (10, 550), "제어 신호: 자동 추출한 윤곽선", F[30])
        label(d, (1910, 10), f"{i + 1:2d}/{n}프레임 · {mode}", F[26], align="right")
        lines = [("Cosmos-Transfer2.5-2B · 윤곽선(edge) 제어", F[30]), ("", F[22]),
                 ("· 848×480, 93프레임(10Hz, 9.3초), 확산 35단계", F[26]),
                 ("· RTX 3090 1장: 모델 불러오기 중 메모리 부족으로 실패", F[26]),
                 ("· RTX 3090 2장: 텍스트 인코더만 두 번째 GPU로 옮겨 성공", F[26]),
                 ("   GPU별 최대 19.1GB·16.6GB, 확산 7분 22초", F[26]), ("", F[22]),
                 ("· 형태(차·차선·맨홀 위치)는 윤곽선을 따라 유지", F[26]),
                 ("· 노란 노면 표시·구급차 무늬처럼 의미가 바뀐 곳이 있음", F[26]),
                 ("  → 학습 데이터로는 Pseudo-GT로만 사용", F[26])]
        y = 580
        for t, f in lines:
            d.text((1000, y), t, font=f, fill=(235, 235, 235))
            y += 44 if f.size >= 26 else 20
        d.text((1000, 1036), NOTE, font=F[22], fill=(150, 150, 150))
        return np.asarray(im)

    frames = [title_card(["CARLA 시뮬레이션을 실사풍으로: Cosmos-Transfer2.5",
                          "RTX 3090 2장에서 생성 · 형태는 지키고 겉모습만 바꾸는 생성형 데이터 확장",
                          "원속 재생 → 달라진 곳 표시 → 0.5배속 재생"], F)] * (FPS * 3)
    frames += [compose(i, "원속") for i in range(n)]
    frames += [compose(46, "정지: 달라진 곳", boxes=True)] * (FPS * 5)
    for i in range(n):
        fr = compose(i, "0.5배속")
        frames += [fr, fr]
    Image.fromarray(compose(46, "정지: 달라진 곳", boxes=True)).save(
        os.path.join(args.out, "06-2_cosmos_transfer_t046.jpg"), quality=90)
    out = os.path.join(args.out, "06-2_cosmos_transfer.mp4")
    imageio.mimwrite(out, frames, fps=FPS, quality=9, macro_block_size=8)
    print(out, len(frames), "frames")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["real", "cosmos"])
    ap.add_argument("--font", required=True)
    ap.add_argument("--font_index", type=int, default=1, help="Noto CJK .ttc에서 한국어는 1번")
    ap.add_argument("--real_dir", default="outputs/06-1")
    ap.add_argument("--scenes", default="0916,0103")
    ap.add_argument("--cosmos_dir", default="outputs/06-2")
    ap.add_argument("--thumb", type=int, default=60)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    F = fonts(args.font, args.font_index)
    (make_real if args.what == "real" else make_cosmos)(args, F)


if __name__ == "__main__":
    main()
