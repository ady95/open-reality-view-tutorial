"""나만의 리얼리티뷰 1.0 — 추격 시점 프레임에 자차 모델을 합성해 최종 영상을 만든다 (책 03-6).

추격 카메라는 자차에 고정돼 있으므로 자차는 매 프레임 화면의 같은 자리에 같은 모습으로 보인다.
그래서 저폴리 자차 모델을 한 번만 그려(RGBA) 모든 프레임에 덧씌운다. 외부 3D 모델 파일은 쓰지 않는다.

    python /work/viewer/compose_realityview.py --frames /work/outputs/03-5/chase_difix_ref \
        --ref /work/outputs/03-5/chase_ref --preset book --out /work/outputs/03-6/realityview_v1.mp4
"""
import argparse
import glob
import json
import math
import os

import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

# 추격 카메라 프리셋 (render_chase_drivestudio.py와 같은 값): 자차 기준 x, z[m], 숙임각, 가로 화각[도]
PRESETS = {"book": (-7.0, 3.5, 18.0, 90.0), "dashsplat": (-14.1, 5.5, 11.6, 79.3)}

BODY = (0.88, 0.89, 0.91)   # 밝은 회색 차체
GLASS = (0.18, 0.21, 0.26)  # 어두운 유리
TIRE = (0.08, 0.08, 0.09)
TAIL = (0.85, 0.08, 0.10)   # 후미등


def box(x0, x1, y0, y1, z0, z1, top_shrink=(0.0, 0.0, 0.0)):
    """직육면체(윗면을 앞뒤 sx0/sx1, 좌우 sy만큼 줄이면 사다리꼴 객실)의 꼭짓점 8개와 사각면 6개."""
    sx0, sx1, sy = top_shrink
    v = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                  [x0 + sx0, y0 + sy, z1], [x1 - sx1, y0 + sy, z1], [x1 - sx1, y1 - sy, z1], [x0 + sx0, y1 - sy, z1]])
    f = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    return v, f


def wheel(cx, cy, r=0.33, w=0.22, n=12):
    """y축 방향으로 누운 n각 기둥."""
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = np.stack([cx + r * np.cos(a), np.zeros(n), r + r * np.sin(a)], 1)
    v = np.concatenate([ring + [0, cy - w / 2, 0], ring + [0, cy + w / 2, 0]])
    # 면은 바깥에서 볼 때 반시계 방향 (뒷면 제거에 쓰임)
    f = [tuple(range(n)), tuple(range(n, 2 * n))[::-1]] + [(n + i, n + (i + 1) % n, (i + 1) % n, i) for i in range(n)]
    return v, f


def ego_mesh():
    """자차 좌표계(뒷바퀴 축 중심 지면이 원점, x 앞, y 왼쪽, z 위) 기준 저폴리 세단. 길이 4.65m, 폭 1.85m."""
    parts = [(box(-0.95, 3.70, -0.92, 0.92, 0.28, 0.86), BODY),
             (box(-0.55, 2.45, -0.86, 0.86, 0.86, 1.42, top_shrink=(0.55, 0.85, 0.12)), GLASS),
             (box(-0.30, 1.70, -0.75, 0.75, 1.42, 1.44), BODY),  # 지붕
             (box(-0.97, -0.95, 0.50, 0.86, 0.66, 0.78), TAIL),  # 후미등
             (box(-0.97, -0.95, -0.86, -0.50, 0.66, 0.78), TAIL)]
    for cx in (0.0, 2.85):
        for cy in (-0.82, 0.82):
            parts.append((wheel(cx, cy), TIRE))
    return parts


def camera(preset, W, H):
    cx, cz, pitch, hfov = PRESETS[preset]
    p = math.radians(pitch)
    z = np.array([math.cos(p), 0.0, -math.sin(p)])
    x = np.array([0.0, -1.0, 0.0])
    R = np.stack([x, np.cross(z, x), z], 1)  # 카메라 → 자차 회전 (OpenCV 축)
    f = W / (2 * math.tan(math.radians(hfov) / 2))
    return R, np.array([cx, 0.0, cz]), f


def render_ego(preset, W, H, ss=3):
    """자차 RGBA 오버레이를 ss배 크게 그린 뒤 줄여 계단 현상을 없앤다. 화가 알고리즘(먼 면부터)."""
    R, t, f = camera(preset, W, H)
    Ws, Hs, fs = W * ss, H * ss, f * ss
    img = Image.new("RGBA", (Ws, Hs), (0, 0, 0, 0))

    def proj(P):
        c = (P - t) @ R  # 자차 → 카메라
        return np.stack([fs * c[:, 0] / c[:, 2] + Ws / 2, fs * c[:, 1] / c[:, 2] + Hs / 2], 1), c[:, 2]

    # 바닥 그림자: 차 아래 흐릿한 사각형
    sh = Image.new("L", (Ws, Hs), 0)
    g, _ = proj(np.array([[-1.1, -1.05, 0.01], [3.85, -1.05, 0.01], [3.85, 1.05, 0.01], [-1.1, 1.05, 0.01]]))
    ImageDraw.Draw(sh).polygon([tuple(q) for q in g], fill=150)
    sh = sh.filter(ImageFilter.GaussianBlur(6 * ss))
    img.paste(Image.new("RGBA", (Ws, Hs), (0, 0, 0, 255)), (0, 0), sh)

    light = np.array([-0.4, 0.3, 0.87])
    light /= np.linalg.norm(light)
    faces = []
    for (v, fl), col in ego_mesh():
        for face in fl:
            P = v[list(face)]
            n = np.cross(P[1] - P[0], P[2] - P[0])
            if np.linalg.norm(n) < 1e-9:
                continue
            n /= np.linalg.norm(n)
            if np.dot(n, P.mean(0) - (R @ np.zeros(3) + t)) >= 0:  # 뒷면 제거
                continue
            uv, depth = proj(P)
            shade = 0.45 + 0.55 * max(0.0, float(np.dot(n, light)))
            faces.append((depth.mean(), uv, tuple(int(255 * min(1, c * shade)) for c in col)))
    draw = ImageDraw.Draw(img)
    for _, uv, c in sorted(faces, key=lambda q: -q[0]):
        draw.polygon([tuple(q) for q in uv], fill=c + (255,))
    return img.resize((W, H), Image.LANCZOS)


def path_overlay(preset, W, H, ego_to_world, i, horizon=30, half_width=0.9, ss=2):
    """i번째 프레임에서 앞으로 horizon프레임(10Hz면 3초) 동안 자차가 지나갈 노면 띠를 그린다.
    녹화 데이터이므로 '계획 경로' 대신 실제로 지나간 궤적을 쓴다."""
    R, t, f = camera(preset, W, H)
    inv = np.linalg.inv(ego_to_world[i])
    pts = [(inv @ ego_to_world[j])[:3, 3] for j in range(i, min(i + horizon, len(ego_to_world)))]
    if len(pts) < 2:
        return None
    pts = np.array(pts)
    pts[:, 2] = 0.02  # 노면 위
    d = np.gradient(pts[:, :2], axis=0)
    n = np.stack([-d[:, 1], d[:, 0]], 1)
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
    left, right = pts.copy(), pts.copy()
    left[:, :2] += n * half_width
    right[:, :2] -= n * half_width
    poly = np.concatenate([left, right[::-1]])
    c = (poly - t) @ R
    if (c[:, 2] <= 0.1).any():
        return None
    uv = np.stack([f * c[:, 0] / c[:, 2] + W / 2, f * c[:, 1] / c[:, 2] + H / 2], 1) * ss
    img = Image.new("RGBA", (W * ss, H * ss), (0, 0, 0, 0))
    ImageDraw.Draw(img).polygon([tuple(q) for q in uv], fill=(40, 120, 255, 110))
    return img.resize((W, H), Image.LANCZOS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True, help="추격 시점 PNG 폴더 (렌더 또는 Difix 보정 결과)")
    ap.add_argument("--ref", default=None, help="같은 시각 전방 카메라 사진 폴더. 주면 아래에 나란히 붙인다")
    ap.add_argument("--preset", choices=sorted(PRESETS), default="book")
    ap.add_argument("--out", required=True)
    ap.add_argument("--save_frames", default="95")
    ap.add_argument("--traj", default=None, help="trajectory.json을 주면 앞으로 3초 경로를 노면에 그린다")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.frames, "*.png")))
    W, H = Image.open(files[0]).size
    ego = render_ego(args.preset, W, H)
    ego.save(os.path.splitext(args.out)[0] + "_ego.png")
    cover = float((np.asarray(ego)[..., 3] > 128).mean())  # 자차(그림자 포함)가 가리는 화면 비율

    poses = np.array(json.load(open(args.traj))["ego_to_world"]) if args.traj else None
    keep = {int(i) for i in args.save_frames.split(",")}
    w = imageio.get_writer(args.out, fps=10, macro_block_size=1, quality=8)
    for i, fp in enumerate(files):
        frame = Image.open(fp).convert("RGBA")
        if poses is not None:
            band = path_overlay(args.preset, W, H, poses, i)
            if band is not None:
                frame.alpha_composite(band)
        frame.alpha_composite(ego)  # 경로 위에 자차
        frame = frame.convert("RGB")
        if args.ref:
            ref = Image.open(os.path.join(args.ref, os.path.basename(fp))).convert("RGB")
            ref = ref.resize((W, round(ref.height * W / ref.width)), Image.LANCZOS)
            both = Image.new("RGB", (W, H + ref.height))
            both.paste(frame, (0, 0))
            both.paste(ref, (0, H))
            frame = both
        w.append_data(np.asarray(frame))
        if i in keep:
            frame.save(os.path.splitext(args.out)[0] + f"_{i:03d}.jpg", quality=92)
    w.close()
    stats = dict(frames=len(files), size=[W, H], preset=args.preset, ego_cover_ratio=round(cover, 3))
    json.dump(stats, open(os.path.splitext(args.out)[0] + ".json", "w"), indent=1)
    print(stats)


if __name__ == "__main__":
    main()
