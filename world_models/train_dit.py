"""05-4~05-6: 카메라 특징 조건부 Video DiT 학습 (Flow Matching / MeanFlow / iMF).

사용 예:
    python world_models/train_dit.py --mode fm --out outputs/05-4/fm_s       # 05-4 Flow Matching
    python world_models/train_dit.py --mode meanflow --out outputs/05-5/mf_s  # 05-5 MeanFlow (원 논문 목표식)
    python world_models/train_dit.py --mode imf --out outputs/05-5/imf_s      # 05-5 iMF (v-loss, 경계조건 탄젠트)

시간 규약은 MeanFlow 논문과 같다: z_t = (1 - t) x + t e,  t = 0 데이터, t = 1 노이즈,  v = e - x.
"""
import argparse
import copy
import glob
import json
import math
import os
import time

import numpy as np
import torch
from torch.func import jvp
from torch.utils.data import DataLoader, Dataset

import dit
from dit import VideoDiT

SCALE = 0.18215  # SD VAE 잠재 스케일


class ChunkDataset(Dataset):
    """런마다 연속 프레임 C + K장을 잘라 (문맥, 생성 대상)으로 돌려준다."""

    def __init__(self, root, runs, C, K, cam_pool=2, feat_file="feats.npy", pre=0):
        # pre > 0 (05-6 자기 문맥 학습): 앞쪽에 프레임 pre장을 더 붙여, 모델이 그 구간을 직접 생성해 문맥으로 쓰게 한다
        self.C, self.K, self.pool, self.pre, self.items, self.runs = C, K, cam_pool, pre, [], []
        for ri, d in enumerate(sorted(sum([glob.glob(os.path.join(root, r)) for r in runs], []))):
            lat = np.load(os.path.join(d, "latents.npy"))
            self.runs.append(dict(name=os.path.basename(d), lat=lat,
                                  feats=np.load(os.path.join(d, feat_file), mmap_mode="r"),
                                  motion=np.load(os.path.join(d, "motion.npy"))))
            self.items += [(ri, i) for i in range(0, len(lat) - pre - C - K + 1)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        ri, i = self.items[idx]
        n = self.pre + self.C + self.K
        r, s = self.runs[ri], slice(i, i + n)
        feats = pool_feats(torch.from_numpy(np.ascontiguousarray(r["feats"][i + self.C:i + n])), self.pool)
        return (torch.from_numpy(r["lat"][s].astype(np.float32)) * SCALE, feats,
                torch.from_numpy(r["motion"][s]))


def pool_feats(f, pool):
    """(K, 6, 112, 384) 카메라당 8×14 토큰을 pool×pool 평균 풀링. pool=2면 카메라당 4×7 = 28토큰."""
    if pool == 1:
        return f
    K = f.shape[0]
    g = f.float().reshape(K * 6, 8, 14, -1).permute(0, 3, 1, 2)
    g = torch.nn.functional.avg_pool2d(g, pool, ceil_mode=True)
    return g.flatten(2).transpose(1, 2).reshape(K, 6, -1, f.shape[-1]).half()


def logit_normal(n, mean, std, device):
    return torch.sigmoid(torch.randn(n, device=device) * std + mean)


def sample_t_r(n, ratio, device, mean=-0.4, std=1.0):
    """t ≥ r. ratio 비율만 r < t (평균 속도 구간), 나머지는 r = t (순간 속도)."""
    a, b = logit_normal(n, mean, std, device), logit_normal(n, mean, std, device)
    t, r = torch.maximum(a, b), torch.minimum(a, b)
    keep = torch.rand(n, device=device) < ratio
    return t, torch.where(keep, r, t)


def adaptive_loss(err, p, c=1e-3):
    """MeanFlow 적응 가중: 샘플마다 sg(1 / (||err||² + c)^p) · ||err||²"""
    sq = err.pow(2).flatten(1).mean(1)
    if p == 0:
        return sq.mean()
    w = 1.0 / (sq.detach() + c).pow(p)
    return (w * sq).mean()


@torch.no_grad()
def dudt_sg(fn, z, r, t, tangent_v):
    """du/dt = v·∂u/∂z + ∂u/∂t 를 JVP로 구한다. 목표식에서 stop-gradient로 쓰이므로 기울기 없이 계산한다.
    fused 어텐션에는 JVP 규칙이 없어 이 구간만 행렬곱 어텐션을 쓴다."""
    dit.FUSED["on"] = False
    try:
        _, d = jvp(fn, (z, r, t), (tangent_v, torch.zeros_like(r), torch.ones_like(t)))
    finally:
        dit.FUSED["on"] = True
    return d.float()


def make_fn(model, ctx, feats, motion, C, ctx_t):
    """생성 프레임 잠재 z와 스칼라 시각 (r, t)만 입력으로 받는 함수. JVP는 이 함수에 대해 계산한다."""
    def fn(z, r, t):
        B, K = z.shape[:2]
        full = torch.cat([ctx, z], dim=1)
        tt = torch.cat([ctx_t, t[:, None].expand(B, K)], dim=1)
        rr = torch.cat([ctx_t, r[:, None].expand(B, K)], dim=1)
        return model(full, tt, rr, feats, motion)[:, C:]
    return fn


@torch.no_grad()
def generate_chunk(model, ctx, feats, motion, nfe):
    """문맥 ctx 다음 K프레임을 nfe번의 평가로 생성 (05-6 자기 문맥 학습용, eval_dit.sample과 같은 적분)."""
    B, C = ctx.shape[:2]
    K = feats.shape[1]
    z = torch.randn((B, K) + ctx.shape[2:], device=ctx.device)
    zc = torch.zeros(B, C, device=ctx.device)
    ts = torch.linspace(1.0, 0.0, nfe + 1, device=ctx.device)
    with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
        for i in range(nfe):
            t, r = ts[i].expand(B), ts[i + 1].expand(B)
            tt = torch.cat([zc, t[:, None].expand(B, K)], 1)
            rr = torch.cat([zc, r[:, None].expand(B, K)], 1)
            z = z - (ts[i] - ts[i + 1]) * model(torch.cat([ctx, z], 1), tt, rr, feats, motion)[:, C:].float()
    return z


def train_step(model, batch, args):
    lat, feats, motion = (x.cuda(non_blocking=True) for x in batch)
    C, K = args.ctx, args.chunk
    if args.self_ctx > 0:
        # 앞 K프레임을 모델이 직접 생성하고, 그 마지막 C프레임을 정답 문맥 대신 쓴다 (일부 샘플만)
        pre_lat, pre_feats, pre_motion = lat[:, :C], feats[:, :K], motion[:, :C + K]
        lat, feats, motion = lat[:, K:], feats[:, K:], motion[:, K:]
        use = torch.rand(lat.shape[0], device=lat.device) < args.self_ctx
        if use.any():
            gen = generate_chunk(model, pre_lat[use], pre_feats[use], pre_motion[use], args.self_nfe)
            lat = lat.clone()
            lat[use, :C] = gen[:, K - C:]
    ctx, x = lat[:, :C], lat[:, C:]
    B = x.shape[0]
    # 문맥 프레임 노이즈 보강 (05-6): 앞 프레임을 일부러 흐려서 '자기가 만든 앞 프레임'에 대비한다
    if args.ctx_noise > 0:
        tau = torch.rand(B, device=x.device) * args.ctx_noise
        tau = torch.where(torch.rand(B, device=x.device) < 0.5, tau, torch.zeros_like(tau))
        ctx = (1 - tau.view(B, 1, 1, 1, 1)) * ctx + tau.view(B, 1, 1, 1, 1) * torch.randn_like(ctx)
        ctx_t = tau[:, None].expand(B, C)
    else:
        ctx_t = torch.zeros(B, C, device=x.device)
    if args.no_motion:
        motion = torch.zeros_like(motion)
    if args.no_cam:
        feats = torch.zeros_like(feats)
    e = torch.randn_like(x)
    ratio = 0.0 if args.mode == "fm" else args.ratio
    t, r = sample_t_r(B, ratio, x.device)
    tb = t.view(B, 1, 1, 1, 1)
    z = (1 - tb) * x + tb * e
    v = e - x
    fn = make_fn(model, ctx, feats, motion, C, ctx_t)
    # cache_enabled=False: no_grad 구간에서 만든 bf16 가중치 사본이 캐시되면, 뒤의 기울기 계산 구간이
    # 그 사본을 재사용해 기울기가 0이 된다 (iMF에서 실제로 겪은 문제)
    with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
        if args.mode == "fm":
            return adaptive_loss(fn(z, t, t).float() - v, 0.0)
        # (t - r) = 0 인 샘플은 du/dt 항이 사라지므로 r < t 인 샘플만 JVP를 계산한다
        dudt = torch.zeros_like(x)
        m = t > r
        if m.any():
            fn_m = make_fn(model, ctx[m], feats[m], motion[m], C, ctx_t[m])
            if args.mode == "meanflow":
                tangent = v[m]                                  # 원 논문: 정답 속도 e - x
            else:
                with torch.no_grad():                           # iMF: 경계조건 v_θ = u_θ(z, t, t)
                    tangent = fn_m(z[m], t[m], t[m])
            dudt[m] = dudt_sg(fn_m, z[m], r[m], t[m], tangent)
        u = fn(z, r, t).float()
        h = (t - r).view(B, 1, 1, 1, 1)
        if args.mode == "meanflow":
            return adaptive_loss(u - (v - h * dudt), args.p)   # u를 sg(v - (t-r)·du/dt)에 회귀
        return adaptive_loss(u + h * dudt - v, args.p)         # iMF: V = u + (t-r)·sg(du/dt)를 v에 회귀


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/carla_pre")
    ap.add_argument("--runs", default="s*", help="학습 런 패턴 (쉼표 구분)")
    ap.add_argument("--mode", choices=["fm", "meanflow", "imf"], default="fm")
    ap.add_argument("--size", default="S")
    ap.add_argument("--ctx", type=int, default=2, help="문맥 프레임 수 C")
    ap.add_argument("--chunk", type=int, default=4, help="한 번에 생성할 프레임 수 K")
    ap.add_argument("--ratio", type=float, default=0.25, help="r != t 샘플 비율 (MeanFlow 25%%, iMF 50%%)")
    ap.add_argument("--p", type=float, default=1.0, help="적응 가중 지수")
    ap.add_argument("--ctx_noise", type=float, default=0.0, help="문맥 노이즈 보강 최대 세기 (0이면 끔)")
    ap.add_argument("--no_motion", action="store_true", help="자차 이동량 조건을 끈다 (비교 실험)")
    ap.add_argument("--cam_pool", type=int, default=2, help="카메라 토큰 풀링 (1: 그대로, 2: 8×14 → 4×7 = 28)")
    ap.add_argument("--feat_file", default="feats.npy", help="feats.npy(DINOv2) 또는 feats_siglip2.npy")
    ap.add_argument("--no_cam", action="store_true", help="카메라 특징을 0으로 (비교 실험: 앞 프레임만 보고 생성)")
    ap.add_argument("--snap", default="", help="따로 저장할 스텝 (쉼표 구분, 예: 10000)")
    ap.add_argument("--self_ctx", type=float, default=0.0,
                    help="05-6: 문맥을 모델이 직접 생성한 프레임으로 바꾸는 샘플 비율 (0이면 끔)")
    ap.add_argument("--self_nfe", type=int, default=1, help="자기 문맥을 만들 때의 평가 횟수 (MeanFlow는 1)")
    ap.add_argument("--data2", default="", help="06-1: 섞을 두 번째 전처리 폴더 (실사 Pseudo-GT)")
    ap.add_argument("--runs2", default="*", help="06-1: 두 번째 폴더에서 쓸 런 (쉼표 구분 glob)")
    ap.add_argument("--mix", type=float, default=0.0, help="06-1: 배치 중 두 번째 데이터의 비율 (1이면 그것만)")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--accum", type=int, default=1, help="배치를 나눠 처리할 횟수 (메모리 절약)")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--ema", type=float, default=0.9995)
    ap.add_argument("--init", default=None, help="이어서 학습할 체크포인트")
    ap.add_argument("--save_every", type=int, default=5000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    json.dump(vars(args), open(os.path.join(args.out, "args.json"), "w"), indent=1)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True

    ds = ChunkDataset(args.data, args.runs.split(","), args.ctx, args.chunk, args.cam_pool, args.feat_file,
                      pre=args.chunk if args.self_ctx > 0 else 0)
    sampler = None
    if args.data2:
        # 06-1 데이터 혼합: 두 번째 데이터(실사 Pseudo-GT)를 샘플 수와 무관하게 mix 비율로 뽑는다
        ds2 = ChunkDataset(args.data2, args.runs2.split(","), args.ctx, args.chunk, args.cam_pool, args.feat_file)
        n1, n2 = len(ds), len(ds2)
        if args.mix >= 1.0:
            ds, n1 = ds2, 0
            w = torch.ones(n2, dtype=torch.double)
        else:
            ds = torch.utils.data.ConcatDataset([ds, ds2])
            w = torch.cat([torch.full((n1,), (1 - args.mix) / n1, dtype=torch.double),
                           torch.full((n2,), args.mix / n2, dtype=torch.double)])
        sampler = torch.utils.data.WeightedRandomSampler(w, num_samples=len(w), replacement=True)
        print(f"mix {args.mix} | set1 {n1} samples | set2 {n2} samples", flush=True)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=sampler is None, sampler=sampler, num_workers=6,
                    drop_last=True, pin_memory=True, persistent_workers=True)
    f0 = ds[0][1]
    model = VideoDiT(args.size, cam_tokens=f0.shape[2], feat_dim=f0.shape[3]).cuda()
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location="cuda")["ema"])
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"runs {len(getattr(ds, 'runs', []))} | samples {len(ds)} | params {n_params / 1e6:.1f}M | mode {args.mode}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0, betas=(0.9, 0.99))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 1000))
    log = open(os.path.join(args.out, "log.jsonl"), "a")
    step, t0, losses = 0, time.time(), []
    torch.cuda.reset_peak_memory_stats()
    while step < args.steps:
        for batch in dl:
            # MeanFlow·iMF는 JVP 때문에 메모리를 더 쓴다. 배치를 accum개로 나눠 기울기를 모은다 (유효 배치는 같다)
            opt.zero_grad(set_to_none=True)
            loss = 0.0
            for part in zip(*(x.chunk(args.accum) for x in batch)):
                l = train_step(model, part, args) / args.accum
                l.backward()
                loss = loss + l.detach()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.lerp_(pm, 1 - args.ema)
            losses.append(loss.item())
            step += 1
            if step % 100 == 0:
                el = time.time() - t0
                rec = dict(step=step, loss=float(np.mean(losses)), gnorm=float(gn), it_s=step / el,
                           elapsed_min=el / 60, peak_gib=torch.cuda.max_memory_allocated() / 2**30)
                log.write(json.dumps(rec) + "\n")
                log.flush()
                if step % 1000 == 0:
                    print(rec, flush=True)
                losses = []
            if step % args.save_every == 0 or step == args.steps:
                torch.save(dict(ema=ema.state_dict(), args=vars(args), step=step),
                           os.path.join(args.out, "ckpt.pt"))
            if str(step) in args.snap.split(","):
                torch.save(dict(ema=ema.state_dict(), args=vars(args), step=step),
                           os.path.join(args.out, f"ckpt_{step}.pt"))
            if step >= args.steps:
                break


if __name__ == "__main__":
    main()
