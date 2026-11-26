"""07-1: 05장 리얼리티뷰 파이프라인(DINOv2-S → Video DiT → VAE)을 RTX 3090에서 단계별로 최적화하고 속도·화질을 함께 잰다.

단계마다 여러 변형을 같은 입력으로 돌려 비교한다. GPU를 혼자 쓸 때 실행한다.
  dit    : fp32 / bf16 / fp16 자동 혼합 정밀도, torch.compile, CUDA 그래프, TensorRT FP16, torchao INT8
  vae    : SD VAE fp16 / torch.compile / TensorRT FP16, 경량 디코더 TAESD
  enc    : DINOv2-S fp16 / torch.compile / TensorRT FP16
  attn   : softmax를 근사 지수 함수로 바꿨을 때의 화질 (차량용 NPU처럼 exp 연산이 비싼 환경 가정)
  e2e    : 기준 조합과 최적 조합의 청크 전체 지연 P50/P95

사용 (저장소 루트에서):
    python optimization/opt_bench.py --ckpt outputs/05-5/mf_s/ckpt.pt --nfe 2 --parts dit,vae,enc,attn,e2e \
        --out outputs/07-1/bench.json
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "world_models"))
import dit as dit_mod  # noqa: E402
from dit import VideoDiT  # noqa: E402
from train_dit import SCALE, pool_feats  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


# ---------------------------------------------------------------- 측정 도구
def timed(fn, iters, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    ts = np.array(ts)
    return dict(p50=round(float(np.percentile(ts, 50)), 2), p95=round(float(np.percentile(ts, 95)), 2))


def peak_of(fn):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    fn()
    torch.cuda.synchronize()
    return round((torch.cuda.max_memory_allocated() - base) / 2**20, 1)  # 실행 중 추가로 잡은 메모리 MiB


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 99.0 if mse == 0 else 10 * math.log10(255.0 ** 2 / mse)


# ---------------------------------------------------------------- TensorRT
class TRTModule:
    """ONNX → TensorRT 엔진(FP16)을 만들어 torch 텐서로 바로 실행한다. 입력 모양은 고정."""

    def __init__(self, onnx_path, engine_path, fp16=True, calib=None):
        import tensorrt as trt
        self.trt = trt
        logger = trt.Logger(trt.Logger.WARNING)
        if not os.path.exists(engine_path):
            builder = trt.Builder(logger)
            net = builder.create_network(0)
            parser = trt.OnnxParser(net, logger)
            with open(onnx_path, "rb") as f:
                if not parser.parse(f.read()):
                    raise RuntimeError(parser.get_error(0))
            cfg = builder.create_builder_config()
            cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
            if fp16:
                cfg.set_flag(trt.BuilderFlag.FP16)
            if calib is not None:  # INT8: 실제 입력 몇십 개로 각 텐서의 값 범위를 재서 축척을 정한다 (엔트로피 보정)
                cfg.set_flag(trt.BuilderFlag.INT8)
                cfg.int8_calibrator = make_calibrator(trt, calib)
            t0 = time.time()
            blob = builder.build_serialized_network(net, cfg)
            self.build_s = round(time.time() - t0, 1)
            with open(engine_path, "wb") as f:
                f.write(blob)
        else:
            self.build_s = None
        self.engine = trt.Runtime(logger).deserialize_cuda_engine(open(engine_path, "rb").read())
        self.ctx = self.engine.create_execution_context()
        self.names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.inputs = [n for n in self.names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        self.outputs = [n for n in self.names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
        dt = {trt.float32: torch.float32, trt.float16: torch.float16}
        self.out_bufs = {n: torch.empty(tuple(self.engine.get_tensor_shape(n)), device="cuda",
                                        dtype=dt[self.engine.get_tensor_dtype(n)]) for n in self.outputs}

    def __call__(self, *args):
        keep = []
        for n, a in zip(self.inputs, args):
            a = a.contiguous().float() if self.engine.get_tensor_dtype(n) == self.trt.float32 else a.contiguous().half()
            keep.append(a)
            self.ctx.set_tensor_address(n, a.data_ptr())
        for n, b in self.out_bufs.items():
            self.ctx.set_tensor_address(n, b.data_ptr())
        self.ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        return self.out_bufs[self.outputs[0]]


def make_calibrator(trt, batches):
    """batches: [{입력 이름: float32 CUDA 텐서}] 목록을 TensorRT 보정기로 감싼다."""
    class Calib(trt.IInt8EntropyCalibrator2):
        def __init__(self):
            super().__init__()
            self.i = 0

        def get_batch_size(self):
            return 1

        def get_batch(self, names):
            if self.i >= len(batches):
                return None
            b = batches[self.i]
            self.i += 1
            return [int(b[n].data_ptr()) for n in names]

        def read_calibration_cache(self):
            return None

        def write_calibration_cache(self, cache):
            return None
    return Calib()


def export_onnx(module, args, path, names):
    if not os.path.exists(path):
        torch.onnx.export(module, args, path, input_names=names, output_names=["out"], opset_version=17,
                          do_constant_folding=True)
    return path


# ---------------------------------------------------------------- 파이프라인 부품
def load_dit(ckpt):
    ck = torch.load(ckpt, map_location="cuda")
    sd = ck["ema"]
    m = VideoDiT(ck["args"]["size"], cam_tokens=sd["cam_pos"].shape[1] // 6,
                 feat_dim=sd["cam_in.1.weight"].shape[1]).cuda().eval()
    m.load_state_dict(sd)
    return m, ck["args"]


def make_sampler(step_fn, C, K, nfe):
    """잡음에서 nfe번 step_fn(z_all, t, r)을 불러 K프레임 잠재를 만든다 (eval_dit.sample과 같은 적분)."""
    ts = torch.linspace(1.0, 0.0, nfe + 1, device="cuda")

    def run(ctx, noise, feats, motion):
        z = noise
        for i in range(nfe):
            tt = torch.cat([torch.zeros(1, C, device="cuda"), ts[i].expand(1, K)], 1)
            rr = torch.cat([torch.zeros(1, C, device="cuda"), ts[i + 1].expand(1, K)], 1)
            torch.compiler.cudagraph_mark_step_begin()  # CUDA 그래프(reduce-overhead) 출력 재사용 표시
            u = step_fn(torch.cat([ctx, z], 1), tt, rr, feats, motion)[:, C:].float()
            z = z - (ts[i] - ts[i + 1]) * u
        return z
    return run


def amp_step(model, dtype):
    def f(z, t, r, feats, motion):
        if dtype is None:
            return model(z, t, r, feats, motion)
        with torch.autocast("cuda", dtype=dtype):
            return model(z, t, r, feats, motion)
    return f


class Half(torch.nn.Module):
    """ONNX 내보내기용: 가중치를 fp16으로 바꾸지 않고 TensorRT의 FP16 모드에 맡긴다."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, z, t, r, feats, motion):
        return self.m(z, t, r, feats, motion)


def to_img(x):
    return ((x.float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy()


# ---------------------------------------------------------------- 평가 데이터
def load_eval(pre, raw, runs, C, K, pool, stride):
    from PIL import Image
    items = []
    for name in runs.split(","):
        d = os.path.join(pre, name)
        lat = torch.from_numpy(np.load(os.path.join(d, "latents.npy")).astype(np.float32)) * SCALE
        feats = np.load(os.path.join(d, "feats.npy"), mmap_mode="r")
        motion = torch.from_numpy(np.load(os.path.join(d, "motion.npy")))
        frames = json.load(open(os.path.join(d, "frames.json")))
        for i in range(C, len(frames) - K, stride):
            f = pool_feats(torch.from_numpy(np.ascontiguousarray(feats[i:i + K])), pool)[None].cuda().float()
            gt = [np.asarray(Image.open(os.path.join(raw, name, "frames", f"{frames[i + k]:08d}", "CHASE.jpg"))
                             .convert("RGB").resize((256, 144), Image.BICUBIC)) for k in range(K)]
            items.append(dict(ctx=lat[i - C:i][None].cuda(), feats=f, motion=motion[i - C:i + K][None].cuda(),
                              lat_gt=lat[i:i + K].cuda(), gt=gt))
    return items


def quality(items, gen_fn, dec_fn, lp, ref=None):
    """gen_fn(item, seed) → 잠재, dec_fn(잠재 (K,4,18,32)) → (K,H,W,3). ref가 있으면 기준 출력과의 일치도(PSNR)도."""
    vals, agree, outs = [], [], []
    for j, it in enumerate(items):
        z = gen_fn(it, j)
        imgs = dec_fn(z[0])
        outs.append(imgs)
        for k, im in enumerate(imgs):
            gt = it["gt"][k]
            tt = lambda a: torch.from_numpy(a).permute(2, 0, 1)[None].float().cuda() / 127.5 - 1
            vals.append([psnr(im, gt), float(lp(tt(im), tt(gt)))])
            if ref is not None:
                agree.append(psnr(im, ref[j][k]))
    r = dict(psnr=round(float(np.mean([v[0] for v in vals])), 3), lpips=round(float(np.mean([v[1] for v in vals])), 4))
    if ref is not None:
        r["psnr_vs_ref"] = round(float(np.mean(agree)), 2)
    return r, outs


# ---------------------------------------------------------------- 메인
@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--nfe", type=int, default=2)
    ap.add_argument("--parts", default="dit,vae,enc,attn,e2e")
    ap.add_argument("--pre", default="data/carla_pre")
    ap.add_argument("--raw", default="data/carla/test")
    ap.add_argument("--runs", default="town10_s100,town15_s0")
    ap.add_argument("--stride", type=int, default=20)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--work", default="outputs/07-1/engines", help="ONNX·TensorRT 엔진 저장 폴더")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.work, exist_ok=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    parts = args.parts.split(",")
    import lpips
    from diffusers import AutoencoderKL, AutoencoderTiny
    lp = lpips.LPIPS(net="alex", verbose=False).cuda()

    model, targs = load_dit(args.ckpt)
    C, K, pool = targs["ctx"], targs["chunk"], targs.get("cam_pool", 2)
    items = load_eval(args.pre, args.raw, args.runs, C, K, pool, args.stride)
    res = dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__, ckpt=args.ckpt, nfe=args.nfe,
               n_chunks=len(items), frames_per_chunk=K)
    print("eval chunks", len(items), flush=True)

    noise = {j: torch.randn((1, K, 4, 18, 32), device="cuda", generator=torch.Generator("cuda").manual_seed(j))
             for j in range(len(items))}
    it0 = items[0]
    dummy = (it0["ctx"], noise[0], it0["feats"], it0["motion"])

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16).cuda().eval()
    sd_dec = lambda z: to_img(vae.decode(z.half() / SCALE).sample)
    if os.path.exists(args.out):
        res.update(json.load(open(args.out)))

    def save():
        json.dump(res, open(args.out, "w"), indent=1)

    # ------------------------------------------------ DiT
    ref_imgs = None
    if "dit" in parts or "attn" in parts:
        base = make_sampler(amp_step(model, torch.bfloat16), C, K, args.nfe)
        _, ref_imgs = quality(items, lambda it, j: base(it["ctx"], noise[j], it["feats"], it["motion"]), sd_dec, lp)
    if "dit" in parts:
        out = {}
        variants = [("fp32", amp_step(model, None)), ("tf32", None), ("bf16", amp_step(model, torch.bfloat16)),
                    ("fp16", amp_step(model, torch.float16))]
        for name, step in variants:
            if name == "tf32":
                torch.backends.cuda.matmul.allow_tf32 = True
                step = amp_step(model, None)
            s = make_sampler(step, C, K, args.nfe)
            q, _ = quality(items, lambda it, j: s(it["ctx"], noise[j], it["feats"], it["motion"]), sd_dec, lp, ref_imgs)
            out[name] = dict(**timed(lambda: s(*dummy), args.iters), peak_mib=peak_of(lambda: s(*dummy)), **q)
            torch.backends.cuda.matmul.allow_tf32 = False
            print("dit", name, out[name], flush=True)
            res["dit"] = out
            save()
        # torch.compile: 커널 융합(default)과 CUDA 그래프까지 쓰는 reduce-overhead
        for mode in ["default", "reduce-overhead"]:
            torch._dynamo.reset()
            cm = torch.compile(model, mode=mode, fullgraph=True)
            s = make_sampler(amp_step(cm, torch.bfloat16), C, K, args.nfe)
            t0 = time.time()
            s(*dummy)
            torch.cuda.synchronize()
            ctime = round(time.time() - t0, 1)
            q, _ = quality(items, lambda it, j: s(it["ctx"], noise[j], it["feats"], it["motion"]), sd_dec, lp, ref_imgs)
            out[f"bf16+compile:{mode}"] = dict(**timed(lambda: s(*dummy), args.iters), compile_s=ctime, **q)
            print("dit compile", mode, out[f"bf16+compile:{mode}"], flush=True)
            res["dit"] = out
            save()
        torch._dynamo.reset()
        # TensorRT FP16
        try:
            z_all = torch.cat([dummy[0], dummy[1]], 1)
            tt = torch.zeros(1, C + K, device="cuda")
            onnx = export_onnx(Half(model).float(), (z_all, tt, tt, dummy[2], dummy[3]),
                               os.path.join(args.work, "dit.onnx"), ["z", "t", "r", "feats", "motion"])
            trt_m = TRTModule(onnx, os.path.join(args.work, "dit_fp16.engine"))
            s = make_sampler(lambda z, t, r, f, m: trt_m(z, t, r, f, m), C, K, args.nfe)
            q, _ = quality(items, lambda it, j: s(it["ctx"], noise[j], it["feats"], it["motion"]), sd_dec, lp, ref_imgs)
            out["tensorrt-fp16"] = dict(**timed(lambda: s(*dummy), args.iters), build_s=trt_m.build_s, **q)
            print("dit trt", out["tensorrt-fp16"], flush=True)
        except Exception as e:  # 실패도 기록한다
            out["tensorrt-fp16"] = dict(error=repr(e)[:300])
            print("dit trt failed", e, flush=True)
        # torchao INT8: 가중치만 INT8(W8A16), 가중치·활성값 모두 INT8(W8A8 동적)
        try:
            import copy

            from torchao.quantization import int8_dynamic_activation_int8_weight, int8_weight_only, quantize_
            for name, cfg in [("int8-weight-only", int8_weight_only), ("int8-dynamic(w8a8)", int8_dynamic_activation_int8_weight)]:
                qm = copy.deepcopy(model).to(torch.bfloat16)
                quantize_(qm, cfg())
                s = make_sampler(amp_step(qm, torch.bfloat16), C, K, args.nfe)
                q, _ = quality(items, lambda it, j: s(it["ctx"], noise[j], it["feats"], it["motion"]), sd_dec, lp, ref_imgs)
                out[name] = dict(**timed(lambda: s(*dummy), args.iters), **q)
                torch._dynamo.reset()
                cq = torch.compile(qm, mode="default")
                s2 = make_sampler(amp_step(cq, torch.bfloat16), C, K, args.nfe)
                s2(*dummy)
                out[name + "+compile"] = dict(**timed(lambda: s2(*dummy), args.iters))
                print("dit", name, out[name], out[name + "+compile"], flush=True)
                del qm, cq
                torch._dynamo.reset()
        except Exception as e:
            out["int8"] = dict(error=repr(e)[:300])
            print("int8 failed", e, flush=True)
        res["dit"] = out
        save()

    # ------------------------------------------------ TensorRT INT8 (보정 기반, 차량용 칩의 INT8 추론을 흉내)
    if "trt8" in parts:
        out = res.get("dit", {})
        if ref_imgs is None:
            base = make_sampler(amp_step(model, torch.bfloat16), C, K, args.nfe)
            _, ref_imgs = quality(items, lambda it, j: base(it["ctx"], noise[j], it["feats"], it["motion"]), sd_dec, lp)
        try:
            # 보정 입력: 평가에 쓰지 않는 시드로 실제 샘플링 중 DiT가 받는 입력을 그대로 기록한다
            calib = []

            def rec(z, t, r, f, m):
                calib.append(dict(z=z.float().contiguous(), t=t.float().contiguous(), r=r.float().contiguous(),
                                  feats=f.float().contiguous(), motion=m.float().contiguous()))
                return amp_step(model, torch.bfloat16)(z, t, r, f, m)
            s = make_sampler(rec, C, K, args.nfe)
            for j, it in enumerate(items[::2]):
                s(it["ctx"], torch.randn_like(noise[0]), it["feats"], it["motion"])
            onnx = os.path.join(args.work, "dit.onnx")
            if not os.path.exists(onnx):
                z_all = torch.cat([dummy[0], dummy[1]], 1)
                tt = torch.zeros(1, C + K, device="cuda")
                export_onnx(Half(model).float(), (z_all, tt, tt, dummy[2], dummy[3]), onnx, ["z", "t", "r", "feats", "motion"])
            trt_8 = TRTModule(onnx, os.path.join(args.work, "dit_int8.engine"), calib=calib)
            s = make_sampler(lambda z, t, r, f, m: trt_8(z, t, r, f, m), C, K, args.nfe)
            q, _ = quality(items, lambda it, j: s(it["ctx"], noise[j], it["feats"], it["motion"]), sd_dec, lp, ref_imgs)
            out["tensorrt-int8"] = dict(**timed(lambda: s(*dummy), args.iters), build_s=trt_8.build_s,
                                        calib_inputs=len(calib), **q)
            print("dit trt int8", out["tensorrt-int8"], flush=True)
        except Exception as e:
            out["tensorrt-int8"] = dict(error=repr(e)[:300])
            print("trt int8 failed", e, flush=True)
        res["dit"] = out
        save()

    # ------------------------------------------------ softmax 근사
    if "attn" in parts:
        out = {}
        orig = dit_mod.attention

        def make_attn(n_sq):
            def f(q, k, v):
                w = (q @ k.transpose(-1, -2)).float() * (q.shape[-1] ** -0.5)
                w = w - w.amax(-1, keepdim=True)
                if n_sq is None:
                    e = torch.exp(w)
                else:  # exp(x) ≈ (1 + x/2^n)^(2^n): 덧셈·곱셈과 제곱 n번만으로 근사
                    e = torch.clamp(1 + w / (2 ** n_sq), min=0)
                    for _ in range(n_sq):
                        e = e * e
                p = e / e.sum(-1, keepdim=True)
                return p.to(v.dtype) @ v
            return f
        for name, fn in [("sdpa", orig), ("exact-math", make_attn(None)), ("approx-2^6", make_attn(6)),
                         ("approx-2^4", make_attn(4)), ("approx-2^2", make_attn(2))]:
            dit_mod.attention = fn
            s = make_sampler(amp_step(model, torch.bfloat16), C, K, args.nfe)
            q, _ = quality(items, lambda it, j: s(it["ctx"], noise[j], it["feats"], it["motion"]), sd_dec, lp, ref_imgs)
            out[name] = dict(**timed(lambda: s(*dummy), max(20, args.iters // 5)), **q)
            print("attn", name, out[name], flush=True)
        dit_mod.attention = orig
        res["attn"] = out
        save()

    # ------------------------------------------------ VAE 디코더
    if "vae" in parts:
        out = {}
        zK = items[0]["lat_gt"]
        tae = AutoencoderTiny.from_pretrained("madebyollin/taesd", torch_dtype=torch.float16).cuda().eval()
        # TAESD는 SD 확산 공간의 잠재(스케일 0.18215를 곱한 값)를 그대로 받는다 (scaling_factor = 1)
        tae_dec = lambda z: to_img(tae.decode(z.half()).sample)
        decs = [("sd-vae-fp16", sd_dec), ("taesd-fp16", tae_dec)]
        torch._dynamo.reset()
        cdec = torch.compile(vae.decoder, mode="default")
        pq = vae.post_quant_conv

        def sd_c(z):
            return to_img(cdec(pq(z.half() / SCALE)))
        decs.append(("sd-vae-fp16+compile", sd_c))
        try:
            class Dec(torch.nn.Module):
                def __init__(self, v):
                    super().__init__()
                    self.v = v

                def forward(self, z):
                    return self.v.decode(z / SCALE).sample
            onnx = export_onnx(Dec(AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").cuda().eval()),
                               (zK.float(),), os.path.join(args.work, "vae_dec.onnx"), ["z"])
            trt_v = TRTModule(onnx, os.path.join(args.work, "vae_dec_fp16.engine"))
            decs.append(("sd-vae-tensorrt-fp16", lambda z: to_img(trt_v(z.float()))))
        except Exception as e:
            out["sd-vae-tensorrt-fp16"] = dict(error=repr(e)[:300])
            print("vae trt failed", e, flush=True)
        gen = make_sampler(amp_step(model, torch.bfloat16), C, K, args.nfe)
        lat_gen = [gen(it["ctx"], noise[j], it["feats"], it["motion"]) for j, it in enumerate(items)]
        for name, dec in decs:
            # 화질 두 가지: 정답 잠재를 복원(디코더 상한), 생성 잠재를 복원(실제 파이프라인)
            ceil, _ = quality(items, lambda it, j: it["lat_gt"][None], dec, lp)
            gq, _ = quality(items, lambda it, j: lat_gen[j], dec, lp)
            # 시간에는 GPU→CPU 복사와 uint8 변환까지 포함한다 (화면에 내보내는 비용)
            out[name] = dict(**timed(lambda: dec(zK), args.iters), ceiling=ceil, generated=gq)
            print("vae", name, out[name], flush=True)
        res["vae"] = out
        res["vae_params_m"] = dict(sd_decoder=round(sum(p.numel() for p in vae.decoder.parameters()) / 1e6, 1),
                                   taesd_decoder=round(sum(p.numel() for p in tae.decoder.parameters()) / 1e6, 2))
        save()

    # ------------------------------------------------ 인코더
    if "enc" in parts:
        from transformers import AutoModel
        out = {}
        enc = AutoModel.from_pretrained("facebook/dinov2-small", dtype=torch.float16).cuda().eval()
        imgs = torch.randn(K * 6, 3, 224, 392, device="cuda", dtype=torch.float16)
        ref = enc(pixel_values=imgs).last_hidden_state.float()
        cos = lambda a: round(float(torch.nn.functional.cosine_similarity(a.float(), ref, dim=-1).mean()), 5)
        out["dinov2-s-fp16"] = dict(**timed(lambda: enc(pixel_values=imgs), args.iters), cos=1.0)
        torch._dynamo.reset()
        ce = torch.compile(enc, mode="default")
        ce(pixel_values=imgs)
        out["dinov2-s-fp16+compile"] = dict(**timed(lambda: ce(pixel_values=imgs), args.iters),
                                            cos=cos(ce(pixel_values=imgs).last_hidden_state))
        try:
            class E(torch.nn.Module):
                def __init__(self, e):
                    super().__init__()
                    self.e = e

                def forward(self, x):
                    return self.e(pixel_values=x).last_hidden_state
            e32 = AutoModel.from_pretrained("facebook/dinov2-small", attn_implementation="eager").cuda().eval()
            onnx = export_onnx(E(e32), (imgs.float(),), os.path.join(args.work, "dinov2s.onnx"), ["x"])
            trt_e = TRTModule(onnx, os.path.join(args.work, "dinov2s_fp16.engine"))
            out["dinov2-s-tensorrt-fp16"] = dict(**timed(lambda: trt_e(imgs), args.iters), cos=cos(trt_e(imgs)),
                                                 build_s=trt_e.build_s)
        except Exception as e:
            out["dinov2-s-tensorrt-fp16"] = dict(error=repr(e)[:300])
        for k, v in out.items():
            print("enc", k, v, flush=True)
        res["enc"] = out
        save()

    # ------------------------------------------------ 청크 전체: 기준 vs 최적
    if "e2e" in parts:
        from transformers import AutoModel
        enc = AutoModel.from_pretrained("facebook/dinov2-small", dtype=torch.float16).cuda().eval()
        tae = AutoencoderTiny.from_pretrained("madebyollin/taesd", torch_dtype=torch.float16).cuda().eval()
        imgs = torch.randn(K * 6, 3, 224, 392, device="cuda", dtype=torch.float16)

        def feats_from(h):  # (24, 1+448, 384) → (1, K, 6, 28, 384): 05-4 전처리와 같은 풀링
            g = h[:, 1:].reshape(K * 6, 16, 28, -1).permute(0, 3, 1, 2).float()
            g = torch.nn.functional.avg_pool2d(g, 2)          # 16×28 → 8×14 (전처리)
            g = torch.nn.functional.avg_pool2d(g, pool, ceil_mode=True)  # 8×14 → 4×7 (학습 시 풀링)
            return g.flatten(2).transpose(1, 2).reshape(1, K, 6, -1, g.shape[1])

        def pipeline(enc_fn, step, dec):
            s = make_sampler(step, C, K, args.nfe)

            def run():
                f = feats_from(enc_fn(imgs))
                z = s(dummy[0], dummy[1], f, dummy[3])
                return dec(z[0])
            return run
        out = {}
        base = pipeline(lambda x: enc(pixel_values=x).last_hidden_state, amp_step(model, torch.bfloat16), sd_dec)
        out["baseline(05-6)"] = timed(base, args.iters * 3)
        torch._dynamo.reset()
        ce = torch.compile(enc, mode="default")
        cm = torch.compile(model, mode="reduce-overhead", fullgraph=True)
        ctae = torch.compile(tae.decoder, mode="default")
        opt = pipeline(lambda x: ce(pixel_values=x).last_hidden_state, amp_step(cm, torch.bfloat16),
                       lambda z: to_img(ctae(z.half())))
        opt()
        out["optimized"] = timed(opt, args.iters * 3)
        out["peak_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        for k in ("baseline(05-6)", "optimized"):
            out[k]["fps_p50"] = round(K / (out[k]["p50"] / 1000), 1)
            out[k]["fps_p95"] = round(K / (out[k]["p95"] / 1000), 1)
        print("e2e", out, flush=True)
        res["e2e"] = out
        save()
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
