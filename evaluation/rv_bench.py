"""07-2: 리얼리티뷰 인과적(causal) 벤치마크. CARLA 시뮬레이션 GT로 화질·기하·시간 일관성·환각·지연을 한 번에 잰다.

프로토콜
  - 입력: 처음 C프레임만 정답 추격 화면(문맥), 이후는 카메라 6대 특징과 자차 이동량만 받는다 (미래 프레임 사용 금지)
  - 출력: 200프레임(20초, 10Hz) 롤아웃. 시작 프레임 100/400/700 × 테스트 맵 2개 = 1,200프레임
  - 지표 (프레임마다, 256×144)
      화질     : PSNR, LPIPS(AlexNet)
      기하     : Depth Anything V2-Small로 추정한 상대 깊이를 정답 깊이에 축척·이동 정렬한 뒤 AbsRel, δ<1.25
      의미     : SegFormer-B2(Cityscapes) 분할을 CARLA 정답 분할과 대조한 mIoU
      환각     : 차량·보행자 '물체 덩어리'가 정답에 없는데 생긴 수(허위)와 정답에 있는데 사라진 수(누락), 100프레임당.
                 주 지표(_rel)는 정답 화면을 같은 분할 모델로 나눈 결과와 대조한다 (분할 모델의 CARLA 영역 오차 상쇄)
      시간     : tLPIPS = |LPIPS(g_t, g_t-1) − LPIPS(y_t, y_t-1)|, 이웃 프레임 차이(깜빡임)와 청크 경계에서의 비율
      지연     : 청크(4프레임) 생성+복원 시간 P50/P95, 생성 중 추가로 잡은 GPU 메모리 최대값
  - 기준 행: 정답 화면(추정기 자체의 오차 바닥), VAE 복원(디코더 상한), 정지 화면(시작 직전 프레임 고정)

사용 (저장소 루트에서):
    python evaluation/rv_bench.py --out outputs/07-2/causal.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "world_models"))
from dit import VideoDiT  # noqa: E402
from eval_dit import CHASE_WH, gt_image, load_run, sample  # noqa: E402
from train_dit import SCALE, pool_feats  # noqa: E402

W, H = CHASE_WH
OBJ = list(range(11, 19))          # Cityscapes trainId: person, rider, car, truck, bus, train, motorcycle, bicycle
MEAN = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)


def carla_to_city(sem):
    """CARLA 의미 태그(1~19는 Cityscapes 순서와 같음, 24 차선 표시 = 도로) → trainId. 나머지는 255(무시)."""
    out = np.full(sem.shape, 255, np.uint8)
    m = (sem >= 1) & (sem <= 19)
    out[m] = sem[m] - 1
    out[sem == 24] = 0
    return out


def load_gt_aux(run, i):
    d = os.path.join(run["raw"], "frames", f"{run['frames'][i]:08d}")
    dep = np.asarray(Image.open(os.path.join(d, "chase_depth.png"))).astype(np.float64)
    dep = (dep[..., 0] + dep[..., 1] * 256 + dep[..., 2] * 256 ** 2) / (256 ** 3 - 1) * 1000.0
    dep = np.asarray(Image.fromarray(dep.astype(np.float32)).resize((W, H), Image.NEAREST))
    sem = np.asarray(Image.open(os.path.join(d, "chase_semantic.png")).resize((W, H), Image.NEAREST))
    return dep, carla_to_city(sem)


def load_segformer(repo, cls):
    """이 체크포인트는 .bin 파일뿐이라 transformers 4.57이 torch 2.6 미만에서 from_pretrained를 거부한다.
    설정으로 빈 모델을 만들고 가중치는 weights_only=True로 직접 읽는다."""
    from huggingface_hub import hf_hub_download
    from transformers import AutoConfig
    m = cls(AutoConfig.from_pretrained(repo))
    m.load_state_dict(torch.load(hf_hub_download(repo, "pytorch_model.bin"), map_location="cpu", weights_only=True))
    return m.cuda().eval()


class Judges:
    """평가용 모델 묶음: LPIPS, 깊이 추정, 의미 분할."""

    def __init__(self):
        import lpips
        from transformers import AutoModelForDepthEstimation, SegformerForSemanticSegmentation
        self.lp = lpips.LPIPS(net="alex", verbose=False).cuda()
        self.depth = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf").cuda().eval()
        self.seg = load_segformer("nvidia/segformer-b2-finetuned-cityscapes-1024-1024", SegformerForSemanticSegmentation)

    @staticmethod
    def to_t(imgs):
        return torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).float().cuda() / 255.0

    @torch.no_grad()
    def lpips(self, a, b):
        return self.lp(self.to_t(a) * 2 - 1, self.to_t(b) * 2 - 1).flatten().cpu().numpy()

    @torch.no_grad()
    def disparity(self, imgs):
        x = (F.interpolate(self.to_t(imgs), (294, 518), mode="bilinear", align_corners=False) - MEAN) / STD
        d = self.depth(pixel_values=x).predicted_depth[:, None]
        return F.interpolate(d, (H, W), mode="bilinear", align_corners=False)[:, 0].cpu().numpy()

    @torch.no_grad()
    def segment(self, imgs):
        x = (F.interpolate(self.to_t(imgs), (576, 1024), mode="bilinear", align_corners=False) - MEAN) / STD
        lg = self.seg(pixel_values=x).logits
        return F.interpolate(lg, (H, W), mode="bilinear", align_corners=False).argmax(1).byte().cpu().numpy()


def depth_err(disp, dep, sem):
    """상대 시차를 정답 시차(1/깊이)에 최소제곱으로 축척·이동 정렬한 뒤 깊이 AbsRel과 δ<1.25. 하늘과 80m 밖은 제외."""
    v = (dep < 80) & (sem != 10)
    if v.sum() < 100:
        return np.nan, np.nan
    gt_disp = 1.0 / dep[v]
    A = np.stack([disp[v], np.ones(v.sum())], 1)
    s, t = np.linalg.lstsq(A, gt_disp, rcond=None)[0]
    pd = 1.0 / np.clip(s * disp[v] + t, 1 / 80, None)
    ratio = np.maximum(pd / dep[v], dep[v] / pd)
    return float(np.mean(np.abs(pd - dep[v]) / dep[v])), float(np.mean(ratio < 1.25))


def blobs(mask, min_area=40):
    lab, n = ndimage.label(mask)
    if n == 0:
        return []
    areas = ndimage.sum(mask, lab, range(1, n + 1))
    return [lab == (k + 1) for k in range(n) if areas[k] >= min_area]


def halluc(pred, gt):
    """물체 덩어리 단위 환각. 허위: 예측 덩어리의 90% 이상이 정답 물체(3px 여유) 밖. 누락: 정답 덩어리의 90% 이상이 예측 물체 밖.
    정답이 '무시(255)'인 곳이 절반 넘는 예측 덩어리는 판정하지 않는다 (광고판·소품 등 CARLA '기타' 태그)."""
    p_obj, g_obj = np.isin(pred, OBJ), np.isin(gt, OBJ)
    g_dil, p_dil = ndimage.binary_dilation(g_obj, iterations=3), ndimage.binary_dilation(p_obj, iterations=3)
    ign = gt == 255
    fp = sum(1 for b in blobs(p_obj) if (b & g_dil).sum() < 0.1 * b.sum() and (b & ign).sum() < 0.5 * b.sum())
    fn = sum(1 for b in blobs(g_obj) if (b & p_dil).sum() < 0.1 * b.sum())
    return fp, fn


class Acc:
    """한 방법의 프레임별 지표를 모은다."""

    def __init__(self):
        self.v = {k: [] for k in ("psnr", "lpips", "absrel", "d125", "fp", "fn", "fp_rel", "fn_rel", "tlpips", "flick", "flick_gt", "bnd")}
        self.conf = np.zeros((19, 19), np.int64)       # CARLA 정답 분할 대비
        self.conf_rel = np.zeros((19, 19), np.int64)   # 정답 화면을 같은 분할 모델로 나눈 결과 대비

    def add_chunk(self, J, gen, gts, deps, sems, prev_g, prev_y, first_idx):
        lp = J.lpips(gen, gts)
        disp = J.disparity(gen)
        seg = J.segment(gen)
        seg_y = J.segment(gts)
        for k, (g, y) in enumerate(zip(gen, gts)):
            mse = np.mean((g.astype(np.float64) - y) ** 2)
            self.v["psnr"].append(10 * np.log10(255.0 ** 2 / max(mse, 1e-10)))
            self.v["lpips"].append(float(lp[k]))
            a, d = depth_err(disp[k], deps[k], sems[k])
            self.v["absrel"].append(a)
            self.v["d125"].append(d)
            fp, fn = halluc(seg[k], sems[k])
            self.v["fp"].append(fp)
            self.v["fn"].append(fn)
            fp, fn = halluc(seg[k], seg_y[k])
            self.v["fp_rel"].append(fp)
            self.v["fn_rel"].append(fn)
            m = sems[k] != 255
            self.conf += np.bincount(sems[k][m].astype(np.int64) * 19 + seg[k][m], minlength=361).reshape(19, 19)
            self.conf_rel += np.bincount(seg_y[k].astype(np.int64).ravel() * 19 + seg[k].ravel(), minlength=361).reshape(19, 19)
        # 시간 지표: 이 청크의 각 프레임과 바로 앞 프레임
        pg = ([prev_g] if prev_g is not None else []) + list(gen)
        py = ([prev_y] if prev_y is not None else []) + list(gts)
        if len(pg) > 1:
            lg, ly = J.lpips(pg[1:], pg[:-1]), J.lpips(py[1:], py[:-1])
            for k in range(len(pg) - 1):
                self.v["tlpips"].append(abs(float(lg[k]) - float(ly[k])))
                self.v["flick"].append(float(np.mean(np.abs(pg[k + 1].astype(np.float32) - pg[k]))))
                self.v["flick_gt"].append(float(np.mean(np.abs(py[k + 1].astype(np.float32) - py[k]))))
                self.v["bnd"].append(prev_g is not None and k == 0 and first_idx)

    def summary(self):
        a = {k: np.array(v, dtype=np.float64) for k, v in self.v.items()}
        miou = lambda c: np.diag(c) / np.maximum(c.sum(0) + c.sum(1) - np.diag(c), 1)
        iou, iou_rel = miou(self.conf), miou(self.conf_rel)
        present, present_rel = self.conf.sum(1) > 0, self.conf_rel.sum(1) > 0
        bnd = a["bnd"].astype(bool)
        ratio = a["flick"] / np.maximum(a["flick_gt"], 1e-6)
        n = len(a["psnr"])
        return dict(frames=n, psnr=round(float(a["psnr"].mean()), 2), lpips=round(float(a["lpips"].mean()), 3),
                    absrel=round(float(np.nanmean(a["absrel"])), 3), delta125=round(float(np.nanmean(a["d125"])), 3),
                    miou=round(float(iou[present].mean()), 3), miou_rel=round(float(iou_rel[present_rel].mean()), 3),
                    car_iou_rel=round(float(iou_rel[13]), 3),
                    obj_iou={c: round(float(iou[c]), 3) for c in (11, 13) if present[c]},
                    false_obj_per100=round(float(a["fp"].sum()) / n * 100, 1),
                    missed_obj_per100=round(float(a["fn"].sum()) / n * 100, 1),
                    false_obj_rel_per100=round(float(a["fp_rel"].sum()) / n * 100, 1),
                    missed_obj_rel_per100=round(float(a["fn_rel"].sum()) / n * 100, 1),
                    tlpips=round(float(a["tlpips"].mean()), 4) if len(a["tlpips"]) else None,
                    flicker=round(float(a["flick"].mean()), 2) if len(a["flick"]) else None,
                    flicker_gt=round(float(a["flick_gt"].mean()), 2) if len(a["flick_gt"]) else None,
                    flicker_ratio_boundary=round(float(ratio[bnd].mean()), 2) if bnd.any() else None,
                    flicker_ratio_inside=round(float(ratio[~bnd].mean()), 2) if (~bnd).any() else None)


def load_model(path):
    ck = torch.load(path, map_location="cuda")
    sd = ck["ema"]
    m = VideoDiT(ck["args"]["size"], cam_tokens=sd["cam_pos"].shape[1] // 6,
                 feat_dim=sd["cam_in.1.weight"].shape[1]).cuda().eval()
    m.load_state_dict(sd)
    return m, ck["args"]


def to_img(x):
    return list(((x.float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy())


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre", default="data/carla_pre")
    ap.add_argument("--raw", default="data/carla/test")
    ap.add_argument("--runs", default="town10_s100,town15_s0")
    ap.add_argument("--starts", default="100,400,700")
    ap.add_argument("--length", type=int, default=200)
    ap.add_argument("--methods", default="gt,vae,frozen,fm4,mf1,mf2,mf2_selfctx,mf2_taesd")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    from diffusers import AutoencoderKL, AutoencoderTiny
    J = Judges()
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16).cuda().eval()
    tae = AutoencoderTiny.from_pretrained("madebyollin/taesd", torch_dtype=torch.float16).cuda().eval()
    sd_dec = lambda z: to_img(vae.decode(z.half() / SCALE).sample)
    tae_dec = lambda z: to_img(tae.decode(z.half()).sample)   # TAESD는 스케일된 잠재를 그대로 받는다

    # 방법 이름 → (체크포인트, NFE, 문맥 노이즈, 디코더)
    spec = dict(fm4=("outputs/05-4/fm_s/ckpt.pt", 4, 0.0, sd_dec),
                mf1=("outputs/05-5/mf_s/ckpt.pt", 1, 0.0, sd_dec),
                mf2=("outputs/05-5/mf_s/ckpt.pt", 2, 0.0, sd_dec),
                mf2_selfctx=("outputs/05-6/mf_selfctx/ckpt.pt", 2, 0.1, sd_dec),
                mf2_taesd=("outputs/05-5/mf_s/ckpt.pt", 2, 0.0, tae_dec))
    res = json.load(open(args.out)) if os.path.exists(args.out) else {}
    models = {}
    for meth in args.methods.split(","):
        acc, times = Acc(), []
        peak = 0.0
        if meth in spec:
            ck, nfe, ctx_t, dec = spec[meth]
            if ck not in models:
                models[ck] = load_model(ck)
            model, targs = models[ck]
        else:
            targs = models[next(iter(models))][1] if models else torch.load(spec["mf2"][0], map_location="cpu")["args"]
        C, K, pool = targs["ctx"], targs["chunk"], targs.get("cam_pool", 2)
        for name in args.runs.split(","):
            run = load_run(os.path.join(args.pre, name), os.path.join(args.raw, name), targs.get("feat_file", "feats.npy"))
            for s in [int(x) for x in args.starts.split(",")]:
                if s + args.length > len(run["frames"]):
                    continue
                ctx = run["lat"][s - C:s][None].cuda()
                frozen = sd_dec(run["lat"][s - 1:s].cuda())[0]
                prev_g = prev_y = None
                for i in range(s, s + args.length - K + 1, K):
                    gts = [gt_image(run, i + k) for k in range(K)]
                    aux = [load_gt_aux(run, i + k) for k in range(K)]
                    if meth == "gt":
                        gen = gts
                    elif meth == "vae":
                        gen = sd_dec(run["lat"][i:i + K].cuda())
                    elif meth == "frozen":
                        gen = [frozen] * K
                    else:
                        f = pool_feats(torch.from_numpy(np.ascontiguousarray(run["feats"][i:i + K])), pool)[None].cuda().float()
                        mo = run["motion"][i - C:i + K][None].cuda()
                        torch.cuda.synchronize()
                        torch.cuda.reset_peak_memory_stats()
                        base = torch.cuda.memory_allocated()
                        t0 = time.perf_counter()
                        z = sample(model, ctx, f, mo, nfe, ctx_t, seed=i)
                        gen = dec(z[0])                                  # GPU→CPU 복사까지 포함
                        times.append((time.perf_counter() - t0) * 1000)
                        peak = max(peak, (torch.cuda.max_memory_allocated() - base) / 2**20)
                        ctx = torch.cat([ctx, z], 1)[:, -C:]           # 자기가 만든 마지막 C프레임이 다음 문맥
                    acc.add_chunk(J, gen, gts, [a[0] for a in aux], [a[1] for a in aux], prev_g, prev_y, True)
                    prev_g, prev_y = gen[-1], gts[-1]
        r = acc.summary()
        if times:
            t = np.array(times[3:])
            r.update(chunk_ms_p50=round(float(np.percentile(t, 50)), 1), chunk_ms_p95=round(float(np.percentile(t, 95)), 1),
                     peak_extra_mib=round(peak, 1))
        res[meth] = r
        print(meth, r, flush=True)
        json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
