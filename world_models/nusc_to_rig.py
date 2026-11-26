"""06-1: 3DGS로 재구성한 nuScenes 장면을 CARLA 수집 데이터와 같은 폴더 형식으로 바꾼다.

입력 카메라 6장 = nuScenes 실제 영상 (DriveStudio 전처리 결과, 10Hz 보간)
추격 카메라 정답 = Street Gaussians 렌더링 (Pseudo-GT) + 자차 스프라이트 합성
자차 포즈      = render_chase_drivestudio.py --traj_json 이 저장한 ego_to_world

3DGS는 자차 자신을 재구성하지 못한다(자기 카메라에 찍히지 않음). CARLA 정답에는 늘 자차가 보이므로,
CARLA 추격 영상에서 '매 프레임 같은 자리·같은 모습'인 화소만 골라 자차 스프라이트를 만들어 덧씌운다.

사용:
    # 1) CARLA 추격 영상에서 자차 스프라이트 만들기 (한 번)
    python world_models/nusc_to_rig.py sprite --run data/carla/train/s01 --out data/real/ego_sprite.png
    # 2) 장면 변환
    python world_models/nusc_to_rig.py convert --scene_dir data/drivestudio/nuscenes/processed_10Hz/mini/001 \
        --chase_dir outputs/06-1/chase/001 --traj outputs/06-1/chase/001/traj.json \
        --sprite data/real/ego_sprite.png --out data/real/raw/scene-0103
"""
import argparse
import glob
import json
import math
import os

import numpy as np
from PIL import Image, ImageFilter

# DriveStudio 파일 번호 → nuScenes 카메라 이름 (datasets/nuscenes/nuscenes_preprocess.py 의 cam_list 순서)
DS_CAMS = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK_LEFT", "CAM_BACK_RIGHT", "CAM_BACK"]
CHASE_WH = (512, 288)


def ground_depth(W, H, fov=90.0, height=3.5, pitch=18.0):
    """평평한 지면이라면 각 화소에서 잴 CARLA 깊이(카메라 전방 거리). 지면을 보지 않는 화소는 무한대."""
    f = W / (2 * math.tan(math.radians(fov) / 2))
    v = np.arange(H)[:, None] + 0.5
    zc = -(v - H / 2) / f
    th = math.radians(pitch)
    down = math.sin(th) - math.cos(th) * zc            # 광선의 아래 방향 성분 (전방 성분 1 기준)
    d = np.where(down > 1e-3, height / np.maximum(down, 1e-3), np.inf)
    return np.repeat(d, W, 1)


def make_sprite(run, out, n=200):
    """자차 화소 = 지면이 있어야 할 거리보다 확연히 가까운 화소. 매 프레임 거의 같은 자리에 있다.
    (처음에는 화소 색의 표준편차로 골라 보았으나, 자차에도 그림자·햇빛이 바뀌어 실패했다)"""
    dirs = sorted(glob.glob(os.path.join(run, "frames", "*")))
    dirs = dirs[:: max(1, len(dirs) // n)][:n]
    W, H = CHASE_WH
    g = ground_depth(W, H)
    rgbs, masks = [], []
    for d in dirs:
        rgbs.append(np.asarray(Image.open(os.path.join(d, "CHASE.jpg")).convert("RGB").resize(CHASE_WH, Image.BICUBIC)))
        dep = np.asarray(Image.open(os.path.join(d, "chase_depth.png")).convert("RGB")).astype(np.float64)
        dep = (dep[..., 0] + dep[..., 1] * 256 + dep[..., 2] * 256 ** 2) / (256 ** 3 - 1) * 1000.0
        dep = np.asarray(Image.fromarray(dep.astype(np.float32)).resize(CHASE_WH, Image.NEAREST))
        masks.append(dep < 0.8 * g)
    masks = np.stack(masks)
    # 가까운 차량이 지나갈 때도 있으므로, 90% 이상의 프레임에서 가까웠던 화소만 자차로 본다
    mask = masks.mean(0) > 0.9
    H2, W2 = mask.shape
    keep = np.zeros_like(mask)
    keep[2 * H2 // 5:, W2 // 4: 3 * W2 // 4] = True
    mask &= keep
    med = np.median(np.stack(rgbs), 0)
    m = Image.fromarray(mask.astype(np.uint8) * 255).filter(ImageFilter.MedianFilter(3)).filter(ImageFilter.GaussianBlur(0.7))
    rgba = np.dstack([med.round().clip(0, 255).astype(np.uint8), np.asarray(m)])
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    Image.fromarray(rgba).save(out)
    print("sprite", out, "frames", len(dirs), "coverage %.3f" % (np.asarray(m) > 127).mean())


def yaw_deg(R):
    return math.degrees(math.atan2(R[1, 0], R[0, 0]))


def convert(a):
    traj = json.load(open(a.traj))
    ego = np.array(traj["ego_to_world"])                     # (N, 4, 4) nuScenes 오른손 좌표계
    sprite = Image.open(a.sprite).convert("RGBA").resize(CHASE_WH, Image.BICUBIC) if a.sprite else None
    chase = sorted(glob.glob(os.path.join(a.chase_dir, "*.png")))
    n = min(len(chase), len(ego))
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "meta.jsonl"), "w") as f:
        for i in range(n):
            fd = os.path.join(a.out, "frames", f"{i:08d}")
            os.makedirs(fd, exist_ok=True)
            for k, cam in enumerate(DS_CAMS):
                Image.open(os.path.join(a.scene_dir, "images", f"{i:03d}_{k}.jpg")).convert("RGB") \
                    .resize((800, 450), Image.BICUBIC).save(os.path.join(fd, f"{cam}.jpg"), quality=92)
            im = Image.open(chase[i]).convert("RGB").resize(CHASE_WH, Image.BICUBIC)
            if sprite is not None:
                im.paste(sprite, (0, 0), sprite)
            im.save(os.path.join(fd, "CHASE.jpg"), quality=92)
            # precompute.py 는 CARLA(왼손, y 오른쪽) 포즈를 기대하므로 y와 yaw 부호를 뒤집어 저장한다
            T = ego[i]
            pose = dict(x=float(T[0, 3]), y=float(-T[1, 3]), yaw=float(-yaw_deg(T[:3, :3])))
            f.write(json.dumps(dict(frame=i, ego=pose)) + "\n")
    print(os.path.basename(a.out.rstrip("/")), n, "frames")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sprite")
    s.add_argument("--run", required=True)
    s.add_argument("--out", required=True)
    c = sub.add_parser("convert")
    c.add_argument("--scene_dir", required=True)
    c.add_argument("--chase_dir", required=True)
    c.add_argument("--traj", required=True)
    c.add_argument("--sprite", default="")
    c.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "sprite":
        make_sprite(a.run, a.out)
    else:
        convert(a)


if __name__ == "__main__":
    main()
