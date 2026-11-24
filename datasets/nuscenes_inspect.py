"""nuScenes mini 구조 살펴보기: 장면 수, 카메라 6채널, 해상도, 캘리브레이션, 자차 포즈.

사용:
    python datasets/nuscenes_inspect.py --dataroot data/nuscenes
"""
import argparse
from collections import Counter

from nuscenes.nuscenes import NuScenes
from PIL import Image

CAMS = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
        "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataroot", default="data/nuscenes")
    args = ap.parse_args()

    nusc = NuScenes(version="v1.0-mini", dataroot=args.dataroot, verbose=False)
    print(f"scenes {len(nusc.scene)} | samples(keyframes) {len(nusc.sample)} | sample_data {len(nusc.sample_data)}")

    channels = Counter(sd["channel"] for sd in nusc.sample_data if sd["is_key_frame"])
    print("keyframe sample_data per channel:")
    for ch in sorted(channels):
        print(f"  {ch:18s} {channels[ch]}")

    scene = nusc.scene[0]
    print(f"\nscene[0] {scene['name']} | {scene['nbr_samples']} keyframes | {scene['description']}")
    sample = nusc.get("sample", scene["first_sample_token"])
    for cam in CAMS:
        sd = nusc.get("sample_data", sample["data"][cam])
        cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
        w, h = Image.open(nusc.get_sample_data_path(sd["token"])).size
        fx = cs["camera_intrinsic"][0][0]
        t = [round(v, 2) for v in cs["translation"]]
        print(f"  {cam:16s} {w}x{h}  fx={fx:7.1f}  t(ego)={t}")

    sd = nusc.get("sample_data", sample["data"]["CAM_FRONT"])
    pose = nusc.get("ego_pose", sd["ego_pose_token"])
    print(f"\nego_pose(global) t={[round(v, 2) for v in pose['translation']]} q={[round(v, 3) for v in pose['rotation']]}")

    stamps = [nusc.get("sample_data", sample["data"][c])["timestamp"] for c in CAMS]
    print(f"camera timestamp spread in one keyframe: {(max(stamps) - min(stamps)) / 1000:.1f} ms")

    keys = [s for s in nusc.sample if s["scene_token"] == scene["token"]]
    ts = sorted(s["timestamp"] for s in keys)
    gaps = [(b - a) / 1e6 for a, b in zip(ts, ts[1:])]
    print(f"keyframe interval: mean {sum(gaps) / len(gaps):.3f} s")

    # 키프레임 사이의 sweep까지 포함한 전체 카메라 프레임 (영상 모델 학습에 쓰는 연속 프레임)
    allcam = Counter(sd["channel"] for sd in nusc.sample_data if sd["channel"].startswith("CAM"))
    print(f"all CAM_FRONT frames (keyframe + sweep): {allcam['CAM_FRONT']}")
    sd = nusc.get("sample_data", sample["data"]["CAM_FRONT"])
    ts = []
    while sd:
        ts.append(sd["timestamp"])
        sd = nusc.get("sample_data", sd["next"]) if sd["next"] else None
    dur = (ts[-1] - ts[0]) / 1e6
    print(f"scene[0] CAM_FRONT: {len(ts)} frames over {dur:.1f} s -> {(len(ts) - 1) / dur:.1f} fps")


if __name__ == "__main__":
    main()
