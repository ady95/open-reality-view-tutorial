"""좌표계 실습: 센서 → 자차 → 월드 변환, 카메라 시각차 보정, 추격 카메라 렌더링.

1) LiDAR 점을 전방 카메라 영상에 투영해 캘리브레이션을 눈으로 확인한다.
2) 카메라마다 자기 촬영 시각의 자차 포즈를 쓰지 않으면 투영이 몇 픽셀 어긋나는지 잰다.
3) 장면 전체의 LiDAR 점에 카메라 색을 입혀, 차 뒤 상공 추격 카메라에서 렌더링한다.

사용:
    python geometry/chase_camera.py --dataroot data/nuscenes --scene 0 --out outputs/02-3
"""
import argparse
import os
import time

import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from PIL import Image, ImageDraw
from pyquaternion import Quaternion

CAMS = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
        "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"]

# 추격 카메라: 자차 기준 뒤 7m, 위 3.5m, 아래로 18도. 자차 좌표계는 x 앞, y 왼쪽, z 위
CHASE_OFFSET = np.array([-7.0, 0.0, 3.5])
CHASE_PITCH_DEG = 18.0
CHASE_W, CHASE_H, CHASE_FX = 800, 450, 400.0  # 수평 화각 90도


def make_T(translation, rotation):
    """평행이동(x,y,z)과 쿼터니언(w,x,y,z)으로 4x4 변환 행렬을 만든다."""
    T = np.eye(4)
    T[:3, :3] = Quaternion(rotation).rotation_matrix
    T[:3, 3] = translation
    return T


def sensor_to_ego(nusc, sd):
    cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
    return make_T(cs["translation"], cs["rotation"])


def ego_to_global(nusc, sd):
    pose = nusc.get("ego_pose", sd["ego_pose_token"])
    return make_T(pose["translation"], pose["rotation"])


def transform(T, pts):
    """(3,N) 점에 4x4 변환을 적용한다."""
    return T[:3, :3] @ pts + T[:3, 3:4]


def project(pts_cam, K, w, h, near=0.5):
    """카메라 좌표(x 오른쪽, y 아래, z 앞) 점을 픽셀로 투영한다."""
    z = pts_cam[2]
    uv = (K @ pts_cam)[:2] / np.maximum(z, 1e-6)
    m = (z > near) & (uv[0] >= 0) & (uv[0] < w) & (uv[1] >= 0) & (uv[1] < h)
    return uv, z, m


def chase_extrinsic():
    """추격 카메라 → 자차 변환. 카메라 축(x 오른쪽, y 아래, z 앞)을 자차 좌표로 표현한다."""
    p = np.radians(CHASE_PITCH_DEG)
    z_axis = np.array([np.cos(p), 0.0, -np.sin(p)])  # 앞으로, 약간 아래로
    x_axis = np.array([0.0, -1.0, 0.0])               # 자차의 오른쪽
    y_axis = np.cross(z_axis, x_axis)                 # 아래쪽
    T = np.eye(4)
    T[:3, :3] = np.stack([x_axis, y_axis, z_axis], axis=1)
    T[:3, 3] = CHASE_OFFSET
    return T


def lidar_in_global(nusc, sample):
    sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    pts = LidarPointCloud.from_file(nusc.get_sample_data_path(sd["token"])).points[:3]
    pts_ego = transform(sensor_to_ego(nusc, sd), pts)
    on_car = (pts_ego[0] > -1.2) & (pts_ego[0] < 4.2) & (np.abs(pts_ego[1]) < 1.2)
    pts_ego = pts_ego[:, ~on_car]  # 자차 차체에 맞은 점 제거
    return transform(ego_to_global(nusc, sd), pts_ego), sd


def camera_model(nusc, sample, cam, ego_pose_from=None):
    """월드 → 카메라 변환과 내부 파라미터. ego_pose_from을 주면 그 sample_data의 자차 포즈를 쓴다."""
    sd = nusc.get("sample_data", sample["data"][cam])
    K = np.array(nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])["camera_intrinsic"])
    T_e2g = ego_to_global(nusc, ego_pose_from or sd)
    T_g2c = np.linalg.inv(sensor_to_ego(nusc, sd)) @ np.linalg.inv(T_e2g)
    return sd, K, T_g2c


def overlay_front(nusc, sample, out_path):
    pts_g, _ = lidar_in_global(nusc, sample)
    sd, K, T_g2c = camera_model(nusc, sample, "CAM_FRONT")
    img = Image.open(nusc.get_sample_data_path(sd["token"])).convert("RGB")
    uv, z, m = project(transform(T_g2c, pts_g), K, *img.size)
    draw = ImageDraw.Draw(img)
    for (u, v), d in zip(uv[:, m].T, z[m]):
        t = min(d / 40.0, 1.0)  # 가까우면 빨강, 멀면 파랑
        draw.ellipse([u - 2, v - 2, u + 2, v + 2], fill=(int(255 * (1 - t)), 60, int(255 * t)))
    img.save(out_path, quality=90)
    return int(m.sum())


def timing_error(nusc, samples):
    """LiDAR 시각의 자차 포즈를 그대로 쓸 때와 카메라 시각의 포즈를 쓸 때의 투영 차이(픽셀)."""
    rows = {c: {"dt": [], "err": []} for c in CAMS}
    for sample in samples:
        pts_g, lidar_sd = lidar_in_global(nusc, sample)
        for cam in CAMS:
            sd, K, T_ok = camera_model(nusc, sample, cam)
            _, _, T_naive = camera_model(nusc, sample, cam, ego_pose_from=lidar_sd)
            uv1, _, m1 = project(transform(T_ok, pts_g), K, 1600, 900)
            uv2, _, m2 = project(transform(T_naive, pts_g), K, 1600, 900)
            m = m1 & m2
            rows[cam]["dt"].append((sd["timestamp"] - lidar_sd["timestamp"]) / 1000)
            rows[cam]["err"].append(np.linalg.norm(uv1[:, m] - uv2[:, m], axis=0).mean())
    return rows


def colored_point_map(nusc, samples):
    """장면의 모든 키프레임 LiDAR 점에, 같은 시점 카메라 영상의 색을 입힌다."""
    all_pts, all_rgb = [], []
    for sample in samples:
        pts_g, _ = lidar_in_global(nusc, sample)
        rgb = np.zeros((3, pts_g.shape[1]), dtype=np.uint8)
        seen = np.zeros(pts_g.shape[1], dtype=bool)
        for cam in CAMS:
            sd, K, T_g2c = camera_model(nusc, sample, cam)
            img = np.asarray(Image.open(nusc.get_sample_data_path(sd["token"])).convert("RGB"))
            uv, _, m = project(transform(T_g2c, pts_g), K, 1600, 900)
            idx = np.where(m)[0]
            rgb[:, idx] = img[uv[1, idx].astype(int), uv[0, idx].astype(int)].T
            seen[idx] = True
        all_pts.append(pts_g[:, seen])
        all_rgb.append(rgb[:, seen])
    return np.concatenate(all_pts, axis=1), np.concatenate(all_rgb, axis=1)


def render_chase(nusc, sample, pts_g, rgb, out_path, splat=2):
    _, lidar_sd = lidar_in_global(nusc, sample)
    T_c2g = ego_to_global(nusc, lidar_sd) @ chase_extrinsic()
    T_g2c = np.linalg.inv(T_c2g)
    K = np.array([[CHASE_FX, 0, CHASE_W / 2], [0, CHASE_FX, CHASE_H / 2], [0, 0, 1]])
    uv, z, m = project(transform(T_g2c, pts_g), K, CHASE_W, CHASE_H)
    order = np.argsort(-z[m])  # 먼 점부터 그려서 가까운 점이 덮어쓰게 한다 (단순 z-버퍼)
    u, v, col = uv[0, m][order].astype(int), uv[1, m][order].astype(int), rgb[:, m][:, order]
    canvas = np.zeros((CHASE_H, CHASE_W, 3), dtype=np.uint8)
    filled = np.zeros((CHASE_H, CHASE_W), dtype=bool)
    for du in range(splat):
        for dv in range(splat):
            uu, vv = np.clip(u + du, 0, CHASE_W - 1), np.clip(v + dv, 0, CHASE_H - 1)
            canvas[vv, uu] = col.T
            filled[vv, uu] = True
    img = Image.fromarray(canvas)
    draw_ego_box(img, T_g2c @ ego_to_global(nusc, lidar_sd), K)
    img.save(out_path, quality=90)
    return 1.0 - filled.mean()


def draw_ego_box(img, T_ego2cam, K):
    """자차 위치에 3D 상자(길이 4.6m, 폭 1.9m, 높이 1.6m)를 선으로 그린다."""
    xs, ys, zs = (-1.0, 3.6), (-0.95, 0.95), (0.0, 1.6)
    corners = np.array([[x, y, z] for x in xs for y in ys for z in zs]).T
    uv, _, _ = project(transform(T_ego2cam, corners), K, 10**6, 10**6, near=0.1)
    draw = ImageDraw.Draw(img)
    for i in range(8):
        for j in range(i + 1, 8):
            if bin(i ^ j).count("1") == 1:  # 한 축만 다른 꼭짓점끼리 = 모서리
                draw.line([tuple(uv[:, i]), tuple(uv[:, j])], fill=(80, 160, 255), width=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataroot", default="data/nuscenes")
    ap.add_argument("--scene", type=int, default=0)
    ap.add_argument("--out", default="outputs/02-3")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    nusc = NuScenes(version="v1.0-mini", dataroot=args.dataroot, verbose=False)
    scene = nusc.scene[args.scene]
    samples, tok = [], scene["first_sample_token"]
    while tok:
        samples.append(nusc.get("sample", tok))
        tok = samples[-1]["next"]
    mid = samples[len(samples) // 2]
    print(f"scene {scene['name']} | keyframes {len(samples)}")

    n = overlay_front(nusc, mid, os.path.join(args.out, "lidar_on_cam_front.jpg"))
    print(f"[1] LiDAR points projected on CAM_FRONT: {n}")

    rows = timing_error(nusc, samples)
    print("[2] camera time offset vs LiDAR, and projection error if LiDAR ego pose is reused")
    for cam in CAMS:
        dt, err = np.array(rows[cam]["dt"]), np.array(rows[cam]["err"])
        print(f"    {cam:16s} dt {dt.mean():+6.1f} ms | error mean {err.mean():5.2f} px, max {err.max():5.2f} px")

    t0 = time.perf_counter()
    pts, rgb = colored_point_map(nusc, samples)
    t1 = time.perf_counter()
    holes = render_chase(nusc, mid, pts, rgb, os.path.join(args.out, "chase_view_points.jpg"))
    t2 = time.perf_counter()
    print(f"[3] colored points {pts.shape[1]:,} (build {t1 - t0:.1f} s) | chase render {t2 - t1:.2f} s | empty pixels {holes:.1%}")


if __name__ == "__main__":
    main()
