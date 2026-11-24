"""공통 데이터 포맷: nuScenes와 CARLA를 같은 장면 매니페스트로 묶고, 좌표 변환과 학습 분할을 검증한다.

좌표 규약 (모든 소스 공통)
  월드·자차: 오른손 좌표계, 자차는 x 앞 / y 왼쪽 / z 위
  카메라   : OpenCV 규약, x 오른쪽 / y 아래 / z 앞
  포즈     : T_world_cam, T_world_ego (4x4, 카메라·자차 → 월드). 카메라마다 자기 촬영 시각의 포즈

사용:
  python datasets/unified.py build-nuscenes --dataroot data/nuscenes --out data/unified
  python datasets/unified.py build-carla --run data/carla/run0 --out data/unified --split val
  python datasets/unified.py check --root data/unified
"""
import argparse
import glob
import json
import math
import os

import numpy as np
from PIL import Image

CAMS = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
        "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"]

# Unreal 카메라 축(x 앞, y 오른쪽, z 위) ← OpenCV 카메라 축(x 오른쪽, y 아래, z 앞)
R_UE_FROM_CV = np.array([[0, 0, 1], [1, 0, 0], [0, -1, 0]], dtype=float)
S_FLIP_Y = np.diag([1.0, -1.0, 1.0, 1.0])  # 왼손 ↔ 오른손 (y 부호 반전)


# ---------------------------------------------------------------- nuScenes

def build_nuscenes(dataroot, out):
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.splits import create_splits_scenes
    from pyquaternion import Quaternion

    def make_T(t, q):
        T = np.eye(4)
        T[:3, :3] = Quaternion(q).rotation_matrix
        T[:3, 3] = t
        return T

    nusc = NuScenes(version="v1.0-mini", dataroot=dataroot, verbose=False)
    official = create_splits_scenes()
    split_of = {s: "train" for s in official["mini_train"]}
    split_of.update({s: "val" for s in official["mini_val"]})
    os.makedirs(out, exist_ok=True)
    for scene in nusc.scene:
        log = nusc.get("log", scene["log_token"])
        frames, cams, tok = [], {}, scene["first_sample_token"]
        while tok:
            sample = nusc.get("sample", tok)
            lidar = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
            ego = nusc.get("ego_pose", lidar["ego_pose_token"])
            fr = dict(t=sample["timestamp"] / 1e6, T_world_ego=make_T(ego["translation"], ego["rotation"]).tolist(), cams={})
            for cam in CAMS:
                sd = nusc.get("sample_data", sample["data"][cam])
                cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
                pose = nusc.get("ego_pose", sd["ego_pose_token"])  # 이 카메라의 촬영 시각 포즈
                T = make_T(pose["translation"], pose["rotation"]) @ make_T(cs["translation"], cs["rotation"])
                fr["cams"][cam] = dict(path=os.path.join(dataroot, sd["filename"]),  # 저장소 루트 기준 상대 경로
                                       t=sd["timestamp"] / 1e6, T_world_cam=T.tolist())
                cams[cam] = dict(K=cs["camera_intrinsic"], width=sd["width"], height=sd["height"])
            frames.append(fr)
            tok = sample["next"]
        write_scene(out, dict(scene_id=f"nuscenes/{scene['name']}", source="nuscenes",
                              split=split_of[scene["name"]], location=log["location"],
                              ego_origin="rear_axle", chase_gt=None, cameras=cams, frames=frames))
    print(f"nuScenes: {len(nusc.scene)} scenes written")


# ---------------------------------------------------------------- CARLA

def carla_matrix(d):
    """CARLA Transform(위치, roll/pitch/yaw 도)를 Unreal 왼손 좌표계 4x4 행렬로 (carla.Transform.get_matrix와 같은 식)."""
    cy, sy = math.cos(math.radians(d["yaw"])), math.sin(math.radians(d["yaw"]))
    cr, sr = math.cos(math.radians(d["roll"])), math.sin(math.radians(d["roll"]))
    cp, sp = math.cos(math.radians(d["pitch"])), math.sin(math.radians(d["pitch"]))
    return np.array([
        [cp * cy, cy * sp * sr - sy * cr, -cy * sp * cr - sy * sr, d["x"]],
        [cp * sy, sy * sp * sr + cy * cr, -sy * sp * cr + cy * sr, d["y"]],
        [sp, -cp * sr, cp * cr, d["z"]],
        [0, 0, 0, 1.0]])


def carla_cam_pose(d):
    """CARLA 센서 월드 포즈 → 오른손 월드 기준 OpenCV 카메라 포즈."""
    T_cv = carla_matrix(d)
    T_cv[:3, :3] = T_cv[:3, :3] @ R_UE_FROM_CV
    return S_FLIP_Y @ T_cv


def carla_ego_pose(d):
    """CARLA 차량 월드 포즈 → 오른손 월드·오른손 자차(x 앞, y 왼쪽, z 위) 포즈."""
    return S_FLIP_Y @ carla_matrix(d) @ S_FLIP_Y


def carla_cam_pose_naive(d):
    """흔한 실수: 위치 y만 뒤집고 yaw·roll 부호는 그대로 둔 변환 (비교용)."""
    d2 = dict(d, y=-d["y"])
    T = carla_matrix(d2)
    T[:3, :3] = T[:3, :3] @ R_UE_FROM_CV
    return T


def build_carla(run, out, split):
    rig = json.load(open(os.path.join(run, "rig.json")))
    frames = []
    for line in open(os.path.join(run, "meta.jsonl")):
        rec = json.loads(line)
        fdir = os.path.join(run, "frames", f"{rec['frame']:08d}")  # 저장소 루트 기준 상대 경로
        fr = dict(t=rec["sim_time"], T_world_ego=carla_ego_pose(rec["ego"]).tolist(), cams={})
        for cam in CAMS:
            fr["cams"][cam] = dict(path=os.path.join(fdir, f"{cam}.jpg"), t=rec["sim_time"],
                                   T_world_cam=carla_cam_pose(rec["sensors"][cam]).tolist(),
                                   raw=rec["sensors"][cam])
        fr["chase"] = dict(path=os.path.join(fdir, "CHASE.jpg"), depth=os.path.join(fdir, "chase_depth.png"),
                           semantic=os.path.join(fdir, "chase_semantic.png"),
                           T_world_cam=carla_cam_pose(rec["sensors"]["CHASE"]).tolist(), raw=rec["sensors"]["CHASE"])
        frames.append(fr)
    cams = {c: dict(K=rig["cameras"][c]["K"], width=rig["cameras"][c]["w"], height=rig["cameras"][c]["h"]) for c in CAMS}
    cams["CHASE"] = dict(K=rig["cameras"]["CHASE"]["K"], width=rig["cameras"]["CHASE"]["w"], height=rig["cameras"]["CHASE"]["h"])
    name = os.path.basename(os.path.normpath(run))
    write_scene(out, dict(scene_id=f"carla/{name}", source="carla", split=split, location=rig["map"],
                          ego_origin="bbox_center", chase_gt="sim_gt", cameras=cams, frames=frames))
    print(f"CARLA: {name} | {len(frames)} frames written")


def write_scene(out, scene):
    path = os.path.join(out, scene["scene_id"].replace("/", "__") + ".json")
    json.dump(scene, open(path, "w"))


# ---------------------------------------------------------------- 검증

def carla_depth_to_meters(png_path):
    d = np.asarray(Image.open(png_path)).astype(np.float64)
    return (d[..., 0] + d[..., 1] * 256 + d[..., 2] * 256**2) / (256**3 - 1) * 1000.0


def warp_check(scene, pose_fn, step=10, max_depth=60.0):
    """추격 카메라 깊이로 3D 점을 복원해 CAM_FRONT에 투영하고, 두 영상의 색 차이를 잰다.
    좌표 변환이 맞으면 같은 표면을 보므로 색 차이가 작다."""
    Kc = np.array(scene["cameras"]["CHASE"]["K"])
    Kf = np.array(scene["cameras"]["CAM_FRONT"]["K"])
    wf, hf = scene["cameras"]["CAM_FRONT"]["width"], scene["cameras"]["CAM_FRONT"]["height"]
    errs, hits = [], []
    for fr in scene["frames"][::step]:
        depth = carla_depth_to_meters(fr["chase"]["depth"])
        chase = np.asarray(Image.open(fr["chase"]["path"]), dtype=np.float64)
        front = np.asarray(Image.open(fr["cams"]["CAM_FRONT"]["path"]), dtype=np.float64)
        h, w = depth.shape
        v, u = np.mgrid[0:h:2, 0:w:2]
        z = depth[v, u]
        # CARLA 깊이는 광축 방향 거리(z)이므로 그대로 역투영
        pts = np.linalg.inv(Kc) @ np.stack([u.ravel() * 1.0, v.ravel() * 1.0, np.ones(u.size)]) * z.ravel()
        T_wc = pose_fn(fr["chase"]["raw"])
        T_wf = pose_fn(fr["cams"]["CAM_FRONT"]["raw"])
        T_ego = np.linalg.inv(carla_ego_pose_any(fr)) @ T_wc
        p_ego = T_ego[:3, :3] @ pts + T_ego[:3, 3:4]
        on_ego = (np.abs(p_ego[0]) < 2.6) & (np.abs(p_ego[1]) < 1.2) & (p_ego[2] < 1.8)  # 자차 차체 제외
        keep = (z.ravel() < max_depth) & ~on_ego
        T = np.linalg.inv(T_wf) @ T_wc
        q = T[:3, :3] @ pts[:, keep] + T[:3, 3:4]
        uvf = (Kf @ q)[:2] / np.maximum(q[2], 1e-6)
        m = (q[2] > 0.5) & (uvf[0] >= 0) & (uvf[0] < wf) & (uvf[1] >= 0) & (uvf[1] < hf)
        src = chase[v.ravel()[keep][m], u.ravel()[keep][m]]
        dst = front[uvf[1, m].astype(int), uvf[0, m].astype(int)]
        errs.append(np.abs(src - dst).mean())
        hits.append(m.mean())
    return float(np.mean(errs)), float(np.mean(hits))


def carla_ego_pose_any(fr):
    return np.array(fr["T_world_ego"])


def leakage_report(scenes):
    by_split = {}
    for s in scenes:
        by_split.setdefault(s["split"], []).append(s)
    ids = [s["scene_id"] for s in scenes]
    assert len(ids) == len(set(ids)), "같은 scene_id가 두 번 등장"
    print("splits:", {k: len(v) for k, v in by_split.items()})
    if "train" in by_split and "val" in by_split:
        tr_loc = {s["location"] for s in by_split["train"]}
        for s in by_split["val"]:
            xy_val = np.array([np.array(f["T_world_ego"])[:2, 3] for f in s["frames"]])
            best = np.inf
            for t in by_split["train"]:
                if t["location"] != s["location"]:
                    continue
                xy_tr = np.array([np.array(f["T_world_ego"])[:2, 3] for f in t["frames"]])
                d = np.linalg.norm(xy_val[:, None] - xy_tr[None], axis=2)
                best = min(best, d.min())
            same = s["location"] in tr_loc
            print(f"  val {s['scene_id']:24s} location {s['location']:22s} "
                  f"same-location train: {'yes' if same else 'no '} | nearest train ego position {best:7.1f} m")


def check(root):
    scenes = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(root, "*.json")))]
    print(f"scenes {len(scenes)} | frames {sum(len(s['frames']) for s in scenes)}")
    for s in scenes:
        n_img = sum(len(f["cams"]) for f in s["frames"])
        missing = sum(not os.path.exists(c["path"]) for f in s["frames"] for c in f["cams"].values())
        print(f"  {s['scene_id']:24s} {s['split']:5s} frames {len(s['frames']):4d} images {n_img:5d} "
              f"missing {missing} chase_gt {s['chase_gt']}")
    leakage_report(scenes)
    for s in scenes:
        if s["source"] == "carla":
            err_ok, hit_ok = warp_check(s, carla_cam_pose)
            err_bad, hit_bad = warp_check(s, carla_cam_pose_naive)
            print(f"[warp chase->front] {s['scene_id']}: correct pose  color error {err_ok:5.1f} (overlap {hit_ok:.1%})")
            print(f"[warp chase->front] {s['scene_id']}: naive y-flip  color error {err_bad:5.1f} (overlap {hit_bad:.1%})")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("build-nuscenes")
    a.add_argument("--dataroot", default="data/nuscenes")
    a.add_argument("--out", default="data/unified")
    b = sub.add_parser("build-carla")
    b.add_argument("--run", required=True)
    b.add_argument("--out", default="data/unified")
    b.add_argument("--split", default="train")
    c = sub.add_parser("check")
    c.add_argument("--root", default="data/unified")
    args = ap.parse_args()
    if args.cmd == "build-nuscenes":
        build_nuscenes(args.dataroot, args.out)
    elif args.cmd == "build-carla":
        build_carla(args.run, args.out, args.split)
    else:
        check(args.root)


if __name__ == "__main__":
    main()
