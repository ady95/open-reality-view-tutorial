"""DrivingForward(SF 모드) 사전학습 모델로 nuScenes mini를 추론한다 — 책 04-2.

SF(single-frame) 모드는 '한 시각의 카메라 6장'만으로 가우시안을 만든다. 장면별 학습이 없다.

  eval  : 키프레임마다 6장 → 가우시안 → 다음 sample_data 시각(약 0.08초 뒤)의 6개 카메라 시점을 렌더링해
          실제 사진과 비교한다 (DrivingForward 논문의 SF 평가 방식). 단계별 지연과 메모리도 잰다.
  chase : 전방 카메라의 모든 프레임(약 12Hz)마다 6장 → 가우시안 → 03장과 같은 추격 카메라로 렌더링한다.
          매 프레임 그 시각의 사진만 쓰므로 과거·미래 프레임에 의존하지 않는다.

컨테이너 안에서 실행한다 (작업 폴더 /opt/DrivingForward):
    python /work/feedforward/df_infer.py eval  --scenes scene-0103,scene-0553,scene-0796,scene-0916 --out /work/outputs/04-2/eval
    python /work/feedforward/df_infer.py chase --scenes scene-0916 --out /work/outputs/04-2/chase_scene-0916
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "/opt/DrivingForward")
torch.backends.cudnn.benchmark = False

import utils  # noqa: E402  (DrivingForward)
import dataset.nuscenes_dataset as nd  # noqa: E402
from nuscenes.nuscenes import NuScenes  # noqa: E402

VERSION = "v1.0-mini"
# 저장소는 v1.0-trainval을 코드에 고정해 두었다. mini로 바꿔 끼운다
nd.NuScenes = lambda version, dataroot, verbose: NuScenes(VERSION, dataroot, verbose)

from models import DrivingForwardModel  # noqa: E402
from models.gaussian.GaussianRender import pts2render  # noqa: E402
from models.gaussian.gaussian_renderer import render  # noqa: E402
from models.gaussian.utils import focal2fov, getProjectionMatrix  # noqa: E402
from trainer import DrivingForwardTrainer  # noqa: E402

CAMS = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK_LEFT", "CAM_BACK_RIGHT", "CAM_BACK"]
# 03장과 같은 추격 카메라 프리셋: 자차 기준 x, z[m], 숙임각, 가로 화각[도]
PRESETS = {"book": (-7.0, 3.5, 18.0, 90.0), "dashsplat": (-14.1, 5.5, 11.6, 79.3)}
NO_DEVICE = ["idx", "dataset_idx", "sensor_name", "filename", "token"]


class FrameDataset(nd.NuScenesdataset):
    """sample 토큰 대신 '카메라별 sample_data 토큰 묶음'을 받아 같은 전처리로 입력을 만든다."""

    def __init__(self, items, **kw):
        super().__init__(split="eval_SF", **kw)  # eval_SF: 뒤 문맥=현재 프레임, 앞 문맥=다음 프레임
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        sample = []
        for cam in self.cameras:
            cs = self.dataset.get("sample_data", it[cam])
            data = {"idx": idx, "token": it["name"], "sensor_name": cam, "contexts": [-1, 1],
                    "filename": cs["filename"], "rgb": self.get_current("rgb", cs),
                    "intrinsics": self.get_current("intrinsics", cs),
                    "extrinsics": self.get_current("extrinsics", cs),
                    "ego_pose": self.get_cam_T_cam(cs),
                    "mask": self.mask_loader(self.mask_path, "", cam),
                    "rgb_context": self.get_context("rgb", cs)}
            sample.append(data)
        sample = [self.data_transform(s) for s in sample]
        sample = [nd.transform_mask_sample(s, self.data_transform) for s in sample]
        sample = nd.stack_sample(sample)
        return nd.align_dataset(sample, self.scales, [-1, 1])


def scene_frames(nusc, name, every_sweep):
    """장면의 입력 프레임 목록. every_sweep이면 전방 카메라의 모든 sample_data(약 12Hz), 아니면 키프레임(2Hz)."""
    scene = next(s for s in nusc.scene if s["name"] == name)
    first = nusc.get("sample", scene["first_sample_token"])
    chains = {}
    for cam in CAMS:
        sd, lst = nusc.get("sample_data", first["data"][cam]), []
        while True:
            lst.append((sd["timestamp"], sd["token"], sd["is_key_frame"], sd["next"]))
            if not sd["next"]:
                break
            sd = nusc.get("sample_data", sd["next"])
        chains[cam] = lst
    items = []
    for ts, tok, key, nxt in chains["CAM_FRONT"]:
        if not nxt or (not every_sweep and not key):
            continue  # 마지막 프레임은 '다음 프레임'이 없어 평가·문맥 구성이 안 된다
        it = {"name": f"{name}_{len(items):04d}", "timestamp": ts, "CAM_FRONT": tok}
        for cam in CAMS[1:]:
            cand = [c for c in chains[cam] if c[3]]  # 다음 프레임이 있는 것만
            it[cam] = min(cand, key=lambda c: abs(c[0] - ts))[1]
        items.append(it)
    return items


def to_device(inputs):
    for k, v in inputs.items():
        if k in NO_DEVICE:
            continue
        if isinstance(v, list) and v and torch.is_tensor(v[0]):
            inputs[k] = [x.float().cuda() for x in v]
        elif torch.is_tensor(v):
            inputs[k] = v.float().cuda()
    return inputs


@torch.no_grad()
def infer(model, inputs):
    """깊이 네트워크 → 가우시안 네트워크. 추론에 쓰지 않는 포즈 네트워크는 건너뛴다(평가는 실제 포즈 사용)."""
    inputs["extrinsics_inv"] = torch.inverse(inputs["extrinsics"])
    outputs = {("cam", c): {} for c in range(model.num_cams)}
    feats = model.predict_depth(inputs)
    for c in range(model.num_cams):
        outputs[("cam", c)].update({("cam_T_cam", 0, 1): inputs[("cam_T_cam", 0, 1)][:, c],
                                    ("cam_T_cam", 0, -1): inputs[("cam_T_cam", 0, -1)][:, c]})
        outputs[("cam", c)].update(feats[("cam", c)])
    model.compute_depth_maps(inputs, outputs)
    model.gs_net = model.models["gs_net"]
    for c in range(model.num_cams):
        model.get_gaussian_data(inputs, outputs, c)
    return outputs


def collect(outputs, n_cams):
    """카메라 6대의 픽셀별 가우시안을 하나로 모은다 (자차 좌표계: x 앞, y 왼쪽, z 위)."""
    keys = [("xyz", 3), ("rot_maps", 4), ("scale_maps", 3), ("opacity_maps", 1)]
    g = {k: [] for k, _ in keys}
    g["sh"] = []
    for c in range(n_cams):
        o = outputs[("cam", c)]
        valid = o[("pts_valid", 0, 0)][0]
        g["xyz"].append(o[("xyz", 0, 0)][0][valid])
        for k, d in keys[1:]:
            g[k].append(o[(k, 0, 0)][0].permute(1, 2, 0).reshape(-1, d)[valid])
        sh = o[("sh_maps", 0, 0)][0]
        g["sh"].append(sh.reshape(-1, sh.shape[-2], sh.shape[-1]).permute(0, 2, 1).contiguous()[valid])
    return {k: torch.cat(v, 0) for k, v in g.items()}


def chase_camera(preset, W, H, device):
    cx, cz, pitch, hfov = PRESETS[preset]
    p = math.radians(pitch)
    z = np.array([math.cos(p), 0.0, -math.sin(p)])
    x = np.array([0.0, -1.0, 0.0])
    c2e = np.eye(4)
    c2e[:3, :3] = np.stack([x, np.cross(z, x), z], 1)  # OpenCV 축(x 오른쪽, y 아래, z 앞)
    c2e[:3, 3] = [cx, 0.0, cz]
    f = W / (2 * math.tan(math.radians(hfov) / 2))
    K = torch.tensor([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], dtype=torch.float32)
    wvt = torch.tensor(np.linalg.inv(c2e), dtype=torch.float32).T.to(device)
    proj = getProjectionMatrix(znear=0.01, zfar=80.0, K=K, h=H, w=W).T.to(device)
    full = (wvt.unsqueeze(0).bmm(proj.unsqueeze(0))).squeeze(0)
    return dict(novel_FovX=focal2fov(f, W), novel_FovY=focal2fov(f, H), novel_height=H, novel_width=W,
                novel_world_view_transform=wvt, novel_full_proj_transform=full,
                novel_camera_center=wvt.inverse()[3, :3])


def move_gaussians(g, T):
    """과거 프레임의 가우시안을 4x4 변환 T(과거 자차 → 현재 자차)로 옮긴다. 위치·회전·SH 방향을 함께 돌린다."""
    from pytorch3d.transforms import matrix_to_quaternion, quaternion_multiply
    from models.gaussian.utils import rotate_sh
    T = torch.tensor(T, dtype=torch.float32, device=g["xyz"].device)
    R, t = T[:3, :3], T[:3, 3]
    q = matrix_to_quaternion(R)[None].expand(g["rot_maps"].shape[0], 4)  # (w, x, y, z): 3DGS와 같은 순서
    sh = rotate_sh(g["sh"].permute(0, 2, 1), R[None, None]).permute(0, 2, 1).contiguous()
    return dict(xyz=g["xyz"] @ R.T + t, rot_maps=quaternion_multiply(q, g["rot_maps"]),
                scale_maps=g["scale_maps"], opacity_maps=g["opacity_maps"], sh=sh)


def render_chase(g, cam, bg):
    return render(pts_xyz=g["xyz"], pts_rgb=None, rotations=g["rot_maps"], scales=g["scale_maps"],
                  opacity=g["opacity_maps"], shs=g["sh"], bg_color=bg, **cam)


def save(t, path):
    Image.fromarray((t.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)).save(path)


def timed(fn):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    r = fn()
    torch.cuda.synchronize()
    return r, (time.perf_counter() - t0) * 1000


def summarize(ms, skip=3):
    a = np.array(ms[skip:] if len(ms) > skip else ms)
    return dict(mean=round(float(a.mean()), 2), p50=round(float(np.percentile(a, 50)), 2),
                p95=round(float(np.percentile(a, 95)), 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["eval", "chase"])
    ap.add_argument("--scenes", required=True)
    ap.add_argument("--weights", default="/weights/weights_SF")
    ap.add_argument("--data", default="/data/nuscenes/")
    ap.add_argument("--out", required=True)
    ap.add_argument("--preset", choices=sorted(PRESETS), default="book")
    ap.add_argument("--size", default="800x450", help="추격 시점 렌더 크기")
    ap.add_argument("--bg", default="1,1,1", help="가우시안이 없는 곳의 배경색 (저장소 기본값: 흰색)")
    ap.add_argument("--accum", default="", help="chase: 합칠 과거 프레임 간격 목록 (예: 6,12,18 = 약 0.5·1.0·1.5초 전)")
    ap.add_argument("--input_size", default=None, help="입력 크기 HxW (기본: 설정 파일의 352x640)")
    ap.add_argument("--save_every", type=int, default=10, help="eval에서 이미지를 저장할 간격")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.set_grad_enabled(False)  # 추론만 한다 (렌더러가 기울기용 텐서를 만들지 않게)

    cfg = utils.get_config("./configs/nuscenes/main.yaml", mode="eval", weight_path=args.weights,
                           novel_view_mode="SF")
    if args.input_size:  # 114x228 가중치(weight_SF_114_228)는 입력 크기도 바꿔야 한다
        cfg["training"]["height"], cfg["training"]["width"] = (int(v) for v in args.input_size.split("x"))
    cfg["data"]["data_path"] = args.data
    cfg["data"]["log_path"] = os.path.join(args.out, "log")  # 기본값 ./results는 컨테이너 안에서 쓰기 불가
    cfg["data"]["val_requirements"] = "(gt_pose, gt_ego_pose, mask)"  # LiDAR 깊이 정답은 만들지 않음
    cfg["eval"]["eval_num_workers"] = 0
    model = DrivingForwardModel(cfg, 0)  # 생성 중 데이터셋을 한 번 읽는다 (mini로 바꿔 끼운 상태)
    model.load_weights()
    model.set_eval()
    base = model._dataloaders["eval"].dataset
    kw = dict(path=args.data, cameras=CAMS, back_context=1, forward_context=1,
              data_transform=base.data_transform, depth_type=None, with_pose=True, with_ego_pose=True,
              with_mask=True)

    every_sweep = args.mode == "chase"
    items = []
    for s in args.scenes.split(","):
        items += scene_frames(base.dataset, s, every_sweep)
    ds = FrameDataset(items, **kw)
    print(f"frames: {len(ds)}")

    W, H = (int(v) for v in args.size.split("x"))
    bg = [float(v) for v in args.bg.split(",")]
    trainer = DrivingForwardTrainer(cfg, 0, use_tb=False) if args.mode == "eval" else None
    cam = chase_camera(args.preset, W, H, "cuda") if args.mode == "chase" else None
    if args.mode == "chase":
        os.makedirs(os.path.join(args.out, "frames"), exist_ok=True)
        os.makedirs(os.path.join(args.out, "ref"), exist_ok=True)

    torch.cuda.reset_peak_memory_stats()
    t_net, t_render, n_gauss, rows, poses, coverage, flicker = [], [], [], [], [], [], []
    t_accum, history, prev = [], {}, None
    accum = [int(v) for v in args.accum.split(",")] if args.accum else []
    for i in range(len(ds)):
        inputs = to_device({k: (v.unsqueeze(0) if torch.is_tensor(v) else
                                [x.unsqueeze(0) for x in v] if isinstance(v, list) and v and torch.is_tensor(v[0])
                                else torch.from_numpy(v).unsqueeze(0) if isinstance(v, np.ndarray) else v)
                            for k, v in ds[i].items()})
        outputs, ms = timed(lambda: infer(model, inputs))
        t_net.append(ms)
        if args.mode == "eval":
            def novel():
                for c in range(model.num_cams):
                    outputs[("cam", c)][("gaussian_color", 1, 0)] = pts2render(
                        inputs=inputs, outputs=outputs, cam_num=model.num_cams, novel_cam=c,
                        novel_frame_id=1, bg_color=bg, mode="SF")
            _, ms = timed(novel)
            t_render.append(ms / model.num_cams)  # 시점 1개당
            psnr, ssim, lpips = trainer.compute_reconstruction_metrics(inputs, outputs)
            # 입력 시각과 평가 시각(다음 프레임) 사이의 시간·자차 이동 거리
            n = base.dataset
            sd0 = n.get("sample_data", items[i]["CAM_FRONT"])
            sd1 = n.get("sample_data", sd0["next"])
            p0 = np.array(n.get("ego_pose", sd0["ego_pose_token"])["translation"])
            p1 = np.array(n.get("ego_pose", sd1["ego_pose_token"])["translation"])
            rows.append(dict(name=items[i]["name"], psnr=float(psnr), ssim=float(ssim), lpips=float(lpips),
                             dt_s=(sd1["timestamp"] - sd0["timestamp"]) / 1e6, move_m=float(np.linalg.norm(p1 - p0))))
            if i % args.save_every == 0:
                for c in (0, 3):
                    save(outputs[("cam", c)][("gaussian_color", 1, 0)][0], f"{args.out}/{items[i]['name']}_c{c}.png")
                    save(inputs[("color", 1, 0)][0, c], f"{args.out}/{items[i]['name']}_c{c}_gt.png")
                    save(inputs[("color", 0, 0)][0, c], f"{args.out}/{items[i]['name']}_c{c}_input.png")
        else:
            g = collect(outputs, model.num_cams)
            sd = base.dataset.get("sample_data", items[i]["CAM_FRONT"])
            ep = base.dataset.get("ego_pose", sd["ego_pose_token"])
            T = np.eye(4)
            T[:3, :3] = nd.Quaternion(ep["rotation"]).rotation_matrix
            T[:3, 3] = ep["translation"]
            poses.append(T.tolist())
            if accum:  # 과거 프레임의 가우시안을 현재 자차 좌표로 옮겨 합친다 (미래 프레임은 쓰지 않음)
                history[i] = g
                for k in [k for k in history if k < i - max(accum)]:
                    del history[k]

                def merged():
                    parts = [g] + [move_gaussians(history[i - o], np.linalg.inv(T) @ np.array(poses[i - o]))
                                   for o in accum if i - o in history]
                    return {k: torch.cat([p[k] for p in parts], 0) for k in g}
                g, ms = timed(merged)
                t_accum.append(ms)
            n_gauss.append(int(g["xyz"].shape[0]))
            img, ms = timed(lambda: render_chase(g, cam, bg))
            t_render.append(ms)
            # 흰 배경과 검은 배경으로 한 번씩 그려 차이를 보면 '가우시안이 덮지 못한 정도'를 알 수 있다
            diff = (render_chase(g, cam, [1.0, 1.0, 1.0]) - render_chase(g, cam, [0.0, 0.0, 0.0])).mean(0)
            coverage.append(float(1.0 - diff.clamp(0, 1).mean()))
            prev = img if i == 0 else prev
            flicker.append(float((img - prev).abs().mean() * 255))
            prev = img
            save(img, f"{args.out}/frames/{i:03d}.png")
            save(inputs[("color", 0, 0)][0, 0], f"{args.out}/ref/{i:03d}.png")

    stats = dict(mode=args.mode, scenes=args.scenes, frames=len(ds), weights=os.path.basename(args.weights),
                 input_size=[int(cfg["training"]["height"]), int(cfg["training"]["width"])],
                 net_ms=summarize(t_net), render_ms=summarize(t_render),
                 peak_mem_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2))
    if args.mode == "eval":
        for k in ("psnr", "ssim", "lpips", "dt_s", "move_m"):
            stats[k] = round(float(np.mean([r[k] for r in rows])), 4)
        per_scene = {}
        for r in rows:
            per_scene.setdefault(r["name"].rsplit("_", 1)[0], []).append(r["psnr"])
        stats["psnr_per_scene"] = {k: round(float(np.mean(v)), 2) for k, v in per_scene.items()}
        json.dump(rows, open(f"{args.out}/per_frame.json", "w"), indent=1)
    else:
        ts = [it["timestamp"] for it in items]
        stats.update(preset=args.preset, render_size=[W, H], gaussians_mean=int(np.mean(n_gauss)),
                     coverage_mean=round(float(np.mean(coverage)), 4), accum=accum,
                     accum_ms=summarize(t_accum) if t_accum else None,
                     flicker_mean=round(float(np.mean(flicker[1:])), 2),  # 03-5와 같은 이웃 프레임 차이(0~255)
                     input_fps=round((len(ts) - 1) / ((ts[-1] - ts[0]) / 1e6), 2))
        json.dump({"fps": stats["input_fps"], "ego_to_world": poses}, open(f"{args.out}/trajectory.json", "w"))
    json.dump(stats, open(f"{args.out}/stats.json", "w"), indent=1)
    print(stats)


if __name__ == "__main__":
    main()
