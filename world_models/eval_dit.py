"""05-4~05-6: 학습한 Video DiT로 추격 시점을 생성하고 시뮬레이션 GT와 비교한다.

두 가지 평가:
  chunk   : 문맥 프레임을 정답으로 주고 다음 K프레임 한 청크만 생성 (모델 자체 화질)
  rollout : 처음 C프레임만 정답으로 주고 이후는 자기가 만든 프레임을 문맥으로 이어 붙여 생성 (05-6, 누적 오류)

사용:
    python world_models/eval_dit.py --ckpt outputs/05-4/fm_s/ckpt.pt --nfe 1,2,4,8,25 --out outputs/05-4/fm_s/eval
    python world_models/eval_dit.py --ckpt ... --mode rollout --nfe 2 --length 200 --out ...
"""
import argparse
import json
import os
import time

import lpips
import numpy as np
import torch
from diffusers import AutoencoderKL
from PIL import Image
from skimage.metrics import structural_similarity

from dit import VideoDiT
from train_dit import SCALE, pool_feats

CHASE_WH = (256, 144)


def load_run(pre_dir, raw_dir, feat_file="feats.npy"):
    return dict(lat=torch.from_numpy(np.load(os.path.join(pre_dir, "latents.npy")).astype(np.float32)) * SCALE,
                feats=np.load(os.path.join(pre_dir, feat_file), mmap_mode="r"),
                motion=torch.from_numpy(np.load(os.path.join(pre_dir, "motion.npy"))),
                frames=json.load(open(os.path.join(pre_dir, "frames.json"))), raw=raw_dir)


def gt_image(run, i):
    p = os.path.join(run["raw"], "frames", f"{run['frames'][i]:08d}", "CHASE.jpg")
    return np.asarray(Image.open(p).convert("RGB").resize(CHASE_WH, Image.BICUBIC))


@torch.no_grad()
def sample(model, ctx, feats, motion, nfe, ctx_t=0.0, seed=0):
    """잡음에서 시작해 nfe번 모델을 불러 K프레임을 만든다. 구간 [t_next, t]의 평균 속도 u(z, r=t_next, t)로
    한 번에 건너뛴다. Flow Matching 모델(r=t만 학습)은 이것이 곧 오일러 적분이다."""
    B, C = ctx.shape[:2]
    K = feats.shape[1]
    g = torch.Generator(device="cuda").manual_seed(seed)
    z = torch.randn((B, K) + ctx.shape[2:], device="cuda", generator=g)
    ctx_tt = torch.full((B, C), ctx_t, device="cuda")
    if ctx_t > 0:
        ctx = (1 - ctx_t) * ctx + ctx_t * torch.randn(ctx.shape, device="cuda", generator=g)
    ts = torch.linspace(1.0, 0.0, nfe + 1, device="cuda")
    for i in range(nfe):
        t, r = ts[i].expand(B), ts[i + 1].expand(B)
        tt = torch.cat([ctx_tt, t[:, None].expand(B, K)], 1)
        rr = torch.cat([ctx_tt, r[:, None].expand(B, K)], 1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            u = model(torch.cat([ctx, z], 1), tt, rr, feats, motion)[:, C:].float()
        z = z - (ts[i] - ts[i + 1]) * u
    return z


@torch.no_grad()
def decode(vae, z):
    B, K = z.shape[:2]
    with torch.autocast("cuda", dtype=torch.float16):
        x = vae.decode(z.reshape(B * K, *z.shape[2:]) / SCALE).sample
    return ((x.float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy()


def metrics(pred, gt, lp):
    psnr = 10 * np.log10(255.0 ** 2 / np.mean((pred.astype(np.float64) - gt) ** 2))
    ssim = structural_similarity(pred, gt, channel_axis=2, data_range=255)
    to_t = lambda a: torch.from_numpy(a).permute(2, 0, 1)[None].float().cuda() / 127.5 - 1
    return psnr, ssim, float(lp(to_t(pred), to_t(gt)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--pre", default="data/carla_pre")
    ap.add_argument("--raw", default="data/carla/test")
    ap.add_argument("--runs", default="town10_s100,town15_s0")
    ap.add_argument("--mode", choices=["chunk", "rollout"], default="chunk")
    ap.add_argument("--nfe", default="1,2,4,8,25")
    ap.add_argument("--stride", type=int, default=20, help="chunk 평가에서 시작 프레임 간격")
    ap.add_argument("--length", type=int, default=200, help="rollout 길이(프레임)")
    ap.add_argument("--starts", default="100,400,700", help="rollout 시작 프레임")
    ap.add_argument("--ctx_t", type=float, default=0.0, help="문맥 프레임에 넣을 노이즈 세기 (05-6)")
    ap.add_argument("--no_motion", action="store_true")
    ap.add_argument("--moving", type=float, default=2.0, help="'움직이는 구간'으로 볼 자차 속도 하한 (m/s)")
    ap.add_argument("--save", type=int, default=6, help="저장할 예시 수")
    ap.add_argument("--video", action="store_true", help="rollout 전 구간을 MP4로 저장 (정답 위, 생성 아래)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    ck = torch.load(args.ckpt, map_location="cuda")
    targs = ck["args"]
    C, K, pool = targs["ctx"], targs["chunk"], targs.get("cam_pool", 2)
    feat_file = targs.get("feat_file", "feats.npy")
    no_cam = targs.get("no_cam", False)
    args.no_motion = args.no_motion or targs.get("no_motion", False)
    cam_pos = ck["ema"]["cam_pos"]
    feat_dim = ck["ema"]["cam_in.1.weight"].shape[1]
    model = VideoDiT(targs["size"], cam_tokens=cam_pos.shape[1] // 6, feat_dim=feat_dim).cuda().eval()
    model.load_state_dict(ck["ema"])
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16).cuda().eval()
    lp = lpips.LPIPS(net="alex", verbose=False).cuda()
    summary = dict(ckpt=args.ckpt, step=ck["step"], mode=args.mode, ctx_t=args.ctx_t)

    def inputs(run, i):
        f = pool_feats(torch.from_numpy(np.ascontiguousarray(run["feats"][i:i + K])), pool)[None].cuda().float()
        if no_cam:
            f = torch.zeros_like(f)
        m = run["motion"][i - C:i + K][None].cuda()
        return f, (torch.zeros_like(m) if args.no_motion else m)

    for name in args.runs.split(","):
        run = load_run(os.path.join(args.pre, name), os.path.join(args.raw, name), feat_file)
        N = len(run["frames"])
        if args.mode == "chunk":
            starts = list(range(C, N - K, args.stride))
            res = {}
            # VAE 상한: 정답 잠재를 그대로 복원했을 때의 화질
            vals = [metrics(decode(vae, run["lat"][i:i + 1][None].cuda())[0], gt_image(run, i), lp) for i in starts]
            res["vae_ceiling"] = np.mean(vals, 0).round(4).tolist()
            # 복사 기준선: 문맥의 마지막 정답 프레임을 K프레임 내내 그대로 보여 줬을 때
            vals = []
            for i in starts:
                last = decode(vae, run["lat"][i - 1:i][None].cuda())[0]
                vals += [metrics(last, gt_image(run, i + k), lp) for k in range(K)]
            # 자차 속도(m/s, 10Hz 기준). 신호 대기처럼 서 있는 구간은 복사만으로도 점수가 높으므로 따로 집계한다
            speed = np.array([float(run["motion"][i + k, :2].norm()) * 10 for i in starts for k in range(K)])
            moving = speed >= args.moving
            res["copy_last"] = np.mean(vals, 0).round(4).tolist()
            res["copy_last_moving"] = np.mean(np.array(vals)[moving], 0).round(4).tolist()
            res["moving_ratio"] = round(float(moving.mean()), 3)
            print(name, "vae_ceiling", res["vae_ceiling"], "copy_last", res["copy_last"],
                  "copy_last_moving", res["copy_last_moving"], "moving_ratio", res["moving_ratio"], flush=True)
            for nfe in [int(n) for n in args.nfe.split(",")]:
                vals, times = [], []
                for j, i in enumerate(starts):
                    f, m = inputs(run, i)
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    z = sample(model, run["lat"][i - C:i][None].cuda(), f, m, nfe, args.ctx_t, seed=j)
                    torch.cuda.synchronize()
                    times.append((time.perf_counter() - t0) * 1000)
                    imgs = decode(vae, z)
                    for k in range(K):
                        vals.append(metrics(imgs[k], gt_image(run, i + k), lp))
                    if j < args.save and nfe == int(args.nfe.split(",")[-1]) or (j < args.save and nfe <= 2):
                        row = np.concatenate([np.concatenate([gt_image(run, i + k), imgs[k]], 0) for k in range(K)], 1)
                        Image.fromarray(row).save(os.path.join(args.out, f"{name}_{i:04d}_nfe{nfe}.jpg"), quality=92)
                res[f"nfe{nfe}"] = dict(psnr_ssim_lpips=np.mean(vals, 0).round(4).tolist(),
                                       moving=np.mean(np.array(vals)[moving], 0).round(4).tolist(),
                                       dit_ms_per_chunk=round(float(np.median(times[3:])), 2))
                print(name, nfe, res[f"nfe{nfe}"], flush=True)
            res["n_chunks"] = len(starts)
            summary[name] = res
        else:
            nfe = int(args.nfe.split(",")[0])
            curves, copies, speeds = [], [], []
            for s in [int(x) for x in args.starts.split(",")]:
                if s + args.length > N:
                    continue
                ctx = run["lat"][s - C:s][None].cuda()
                first = decode(vae, run["lat"][s - 1:s][None].cuda())[0]   # 복사 기준선: 시작 직전 프레임에 멈춘 화면
                curve, copy, frames = [], [], []
                for i in range(s, s + args.length - K + 1, K):
                    f, m = inputs(run, i)
                    z = sample(model, ctx, f, m, nfe, args.ctx_t, seed=i)
                    imgs = decode(vae, z)
                    for k in range(K):
                        gt = gt_image(run, i + k)
                        curve.append(metrics(imgs[k], gt, lp))
                        copy.append(metrics(first, gt, lp))
                        frames.append(imgs[k])
                    ctx = torch.cat([ctx, z], 1)[:, -C:]          # 자기가 만든 마지막 C프레임이 다음 문맥
                curves.append(np.array(curve))
                copies.append(np.array(copy))
                speeds.append(float(run["motion"][s:s + len(curve), :2].norm(dim=1).mean()) * 10)
                if args.video:
                    # 위: 시뮬레이션 GT, 아래: 생성. 10Hz로 수집했으므로 10fps로 저장
                    import imageio
                    imageio.mimwrite(os.path.join(args.out, f"roll_{name}_{s}.mp4"),
                                     [np.concatenate([gt_image(run, s + k), fr], 0) for k, fr in enumerate(frames)],
                                     fps=10, quality=8, macro_block_size=8)
                for k in (0, 40, 100, 196):
                    if k < len(frames):
                        Image.fromarray(np.concatenate([gt_image(run, s + k), frames[k]], 0)).save(
                            os.path.join(args.out, f"roll_{name}_{s}_{k:03d}.jpg"), quality=92)
            c, cp = np.stack(curves), np.stack(copies)           # (시작 수, 길이, 3)
            summary[name] = dict(mean_curve=c.mean(0).round(4).tolist(), copy_curve=cp.mean(0).round(4).tolist(),
                                 starts=len(curves), mean_speed_mps=[round(v, 2) for v in speeds])
            seg = lambda a: [round(float(a[:, x:x + 8, 0].mean()), 2) for x in (0, 40, 96, 192) if x < a.shape[1]]
            print(name, "rollout PSNR @0-8/40-48/96-104/192-200:", seg(c), "| frozen copy:", seg(cp),
                  "| speed", summary[name]["mean_speed_mps"], flush=True)
    json.dump(summary, open(os.path.join(args.out, f"{args.mode}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
