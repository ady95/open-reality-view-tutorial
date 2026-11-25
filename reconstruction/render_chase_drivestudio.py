"""학습된 DriveStudio(Street Gaussians / OmniRe) 장면을 차 뒤 상공 추격 카메라로 렌더링한다 — 책 03-2.

추격 카메라 포즈 = (매 프레임 전방 카메라의 camera→world) · (전방 카메라 → 자차)^-1 · (자차 → 추격 카메라)
자차 → 추격 카메라는 02-3에서 정의한 값(뒤 7m, 위 3.5m, 아래 18도)을 그대로 쓴다.

DriveStudio 컨테이너 안에서 실행:
    python /work/reconstruction/render_chase_drivestudio.py \
        --resume_from <output_root>/<project>/<run>/checkpoint_final.pth \
        --nusc_root data/nuscenes/raw --scene_name scene-0061 --out /work/outputs/03-2
"""
import argparse
import json
import math
import os
import time

import imageio
import numpy as np
import torch
from omegaconf import OmegaConf
from pyquaternion import Quaternion

from datasets.base.pixel_source import get_rays
from datasets.driving_dataset import DrivingDataset
from utils.misc import import_str

CHASE_OFFSET = np.array([-7.0, 0.0, 3.5])  # 자차 기준: 뒤 7m, 위 3.5m
CHASE_PITCH_DEG = 18.0


def chase_extrinsic():
    """추격 카메라 → 자차 변환 (OpenCV 카메라 축, nuScenes 자차 좌표계). 02-3과 같은 정의."""
    p = np.radians(CHASE_PITCH_DEG)
    z_axis = np.array([np.cos(p), 0.0, -np.sin(p)])
    x_axis = np.array([0.0, -1.0, 0.0])
    y_axis = np.cross(z_axis, x_axis)
    T = np.eye(4)
    T[:3, :3] = np.stack([x_axis, y_axis, z_axis], axis=1)
    T[:3, 3] = CHASE_OFFSET
    return T


def front_cam_to_ego(nusc_root, scene_name):
    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes(version="v1.0-mini", dataroot=nusc_root, verbose=False)
    scene = next(s for s in nusc.scene if s["name"] == scene_name)
    sample = nusc.get("sample", scene["first_sample_token"])
    sd = nusc.get("sample_data", sample["data"]["CAM_FRONT"])
    cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
    T = np.eye(4)
    T[:3, :3] = Quaternion(cs["rotation"]).rotation_matrix
    T[:3, 3] = cs["translation"]
    return T


def build_render_item(c2w, K, H, W, frame_idx, normed_time, device):
    x, y = torch.meshgrid(torch.arange(W, device=device), torch.arange(H, device=device), indexing="xy")
    origins, viewdirs, direction_norm = get_rays(x.flatten(), y.flatten(), c2w, K)
    cam_infos = {"camera_to_world": c2w, "intrinsics": K,
                 "height": torch.tensor([H], device=device), "width": torch.tensor([W], device=device)}
    image_infos = {
        "origins": origins.reshape(H, W, 3), "viewdirs": viewdirs.reshape(H, W, 3),
        "direction_norm": direction_norm.reshape(H, W, 1),
        "img_idx": torch.full((H, W), frame_idx, dtype=torch.long, device=device),
        "frame_idx": torch.full((H, W), frame_idx, dtype=torch.long, device=device),
        "normed_time": torch.full((H, W), normed_time, dtype=torch.float32, device=device),
        "pixel_coords": torch.stack([y.float() / H, x.float() / W], dim=-1),
    }
    return image_infos, cam_infos


def export_heldout(trainer, dataset, out_dir, device):
    """학습에 쓰지 않은 시점(test_timesteps)의 6개 카메라 렌더와 실제 사진을 PNG 쌍으로 저장한다."""
    for sub in ("render", "gt"):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)
    n = dataset.num_img_timesteps
    with torch.no_grad():
        for t in dataset.test_timesteps:
            for c, cam in dataset.pixel_source.camera_data.items():
                gt = cam.images[t]
                H, W = gt.shape[:2]
                c2w = cam.cam_to_worlds[t].to(device)
                K = cam.intrinsics[t].to(device)
                image_infos, cam_infos = build_render_item(c2w, K, H, W, int(t), int(t) / max(n - 1, 1), device)
                out = trainer(image_infos=image_infos, camera_infos=cam_infos, novel_view=True)
                name = f"t{int(t):03d}_c{int(c)}.png"
                imageio.imwrite(os.path.join(out_dir, "render", name),
                                (out["rgb"].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8))
                imageio.imwrite(os.path.join(out_dir, "gt", name), (gt.cpu().numpy() * 255).astype(np.uint8))
    print("held-out pairs:", len(dataset.test_timesteps) * len(dataset.pixel_source.camera_data))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume_from", required=True)
    ap.add_argument("--nusc_root", default="data/nuscenes/raw")
    ap.add_argument("--scene_name", default="scene-0061")
    ap.add_argument("--out", default="/work/outputs/03-2")
    ap.add_argument("--fov", type=float, default=90.0)
    ap.add_argument("--width", type=int, default=800)
    ap.add_argument("--height", type=int, default=450)
    ap.add_argument("--save_frames", default="0,95,190")
    ap.add_argument("--png_dir", default=None, help="추격 시점 전 프레임을 PNG로 저장 (03-5 Difix 입력)")
    ap.add_argument("--heldout_dir", default=None,
                    help="학습에서 뺀 시점의 렌더/원본 쌍을 저장 (03-5 정량 평가). 저장 후 종료")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda")

    log_dir = os.path.dirname(args.resume_from)
    cfg = OmegaConf.load(os.path.join(log_dir, "config.yaml"))
    dataset = DrivingDataset(data_cfg=cfg.data)
    trainer = import_str(cfg.trainer.type)(
        **cfg.trainer, num_timesteps=dataset.num_img_timesteps, model_config=cfg.model,
        num_train_images=len(dataset.train_image_set), num_full_images=len(dataset.full_image_set),
        test_set_indices=dataset.test_timesteps, scene_aabb=dataset.get_aabb().reshape(2, 3), device=device)
    trainer.resume_from_checkpoint(ckpt_path=args.resume_from, load_only_model=True)
    trainer.set_eval()

    if args.heldout_dir:
        export_heldout(trainer, dataset, args.heldout_dir, device)
        return

    front_c2w = dataset.pixel_source.camera_data[0].cam_to_worlds.to(device)  # cam 0 = CAM_FRONT
    T_front_chase = np.linalg.inv(front_cam_to_ego(args.nusc_root, args.scene_name)) @ chase_extrinsic()
    T_front_chase = torch.tensor(T_front_chase, dtype=torch.float32, device=device)
    W, H = args.width, args.height
    f = W / (2 * math.tan(math.radians(args.fov) / 2))
    K = torch.tensor([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], dtype=torch.float32, device=device)

    n = front_c2w.shape[0]
    keep = {int(i) for i in args.save_frames.split(",")}
    writer = imageio.get_writer(os.path.join(args.out, "chase.mp4"), fps=10)
    times = []
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for i in range(n):
            image_infos, cam_infos = build_render_item(front_c2w[i] @ T_front_chase, K, H, W, i, i / max(n - 1, 1), device)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = trainer(image_infos=image_infos, camera_infos=cam_infos, novel_view=True)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
            rgb = (out["rgb"].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
            writer.append_data(rgb)
            if args.png_dir:
                os.makedirs(args.png_dir, exist_ok=True)
                imageio.imwrite(os.path.join(args.png_dir, f"{i:03d}.png"), rgb)
                # Difix-ref 기준 이미지: 같은 시각의 전방 카메라 실제 사진
                ref_dir = args.png_dir.rstrip("/") + "_ref"
                os.makedirs(ref_dir, exist_ok=True)
                ref = dataset.pixel_source.camera_data[0].images[i]
                imageio.imwrite(os.path.join(ref_dir, f"{i:03d}.png"), (ref.cpu().numpy() * 255).astype(np.uint8))
            if i in keep:
                imageio.imwrite(os.path.join(args.out, f"chase_{i:03d}.jpg"), rgb, quality=92)
    writer.close()
    ms = np.array(times[5:]) * 1000  # 앞 5프레임은 warm-up
    stats = dict(frames=n, width=W, height=H, fov=args.fov,
                 render_ms_mean=round(float(ms.mean()), 2), render_ms_p95=round(float(np.percentile(ms, 95)), 2),
                 peak_mem_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2))
    json.dump(stats, open(os.path.join(args.out, "render_stats.json"), "w"), indent=1)
    print(stats)


if __name__ == "__main__":
    main()
