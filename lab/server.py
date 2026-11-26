"""08-1: Open Reality View Lab — 세 Track의 추격 시점을 한 화면에서 전환·비교하는 통합 뷰어 서버.

같은 장면(nuScenes scene-0916)을 세 방식으로 보여 준다.
  A  3DGS 사전 재구성 (03장 Street Gaussians, 06-1 Pseudo-GT 렌더) — 기록 재생
  B  Feed-forward 재구성 (04장 DrivingForward)                   — 기록 재생
  C  생성 모델 (06-1 혼합 25% Video DiT)                          — 서버에서 실시간 생성
     매 청크마다 실제 카메라 6장 JPEG를 읽고 DINOv2-S → DiT(compile + CUDA 그래프) → TAESD로 만든다.

시계는 하나다. 서버가 켜진 시각부터 장면 시간이 흐르고 끝나면 처음으로 돌아간다. 모든 화면은 이 시계를 따른다.
  정렬 모드      : 세 화면이 같은 장면 시각의 프레임을 보여 준다 (녹화된 입력이므로 미리 계산 가능)
  실시간 제약 모드: 실제 차량이었다면 그 시각에 화면에 떠 있었을 프레임을 보여 준다
                   B는 실측 처리 시간(프레임당 약 178ms)마다 최신 입력 하나만 처리, C는 청크 버퍼링 + 계산 시간만큼 늦게 표시

외부 패키지 없이 표준 라이브러리 HTTP 서버로 MJPEG(multipart/x-mixed-replace)를 보낸다.
  GET /                      뷰어 (index.html)
  GET /stream?track=A&rt=0   MJPEG 스트림
  GET /stats                 Track별 실측 정보와 C의 실시간 지연 통계 (JSON)
  GET /bench                 07-1·07-2 벤치마크 결과 (JSON)

사용 (저장소 루트에서, orv/genvideo-opt 컨테이너):
    python lab/server.py --port 8100
"""
import argparse
import collections
import io
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "world_models"))
from dit import VideoDiT  # noqa: E402
from train_dit import SCALE, pool_feats  # noqa: E402

CAMS = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"]  # precompute.py와 같은 순서
OUT_WH = (512, 288)
FPS = 10.0


def jpeg(img, wh=OUT_WH, q=85):
    im = img if isinstance(img, Image.Image) else Image.fromarray(img)
    if im.size != wh:
        im = im.resize(wh, Image.BICUBIC)
    b = io.BytesIO()
    im.convert("RGB").save(b, "JPEG", quality=q)
    return b.getvalue()


class Clock:
    def __init__(self, duration):
        self.t0, self.duration = time.time(), duration

    def now(self):
        return (time.time() - self.t0) % self.duration

    def loop(self):
        return int((time.time() - self.t0) // self.duration)


# ------------------------------------------------------------------ Track A·B: 기록 재생
class Recorded:
    def __init__(self, files, fps, info):
        self.frames = [jpeg(Image.open(f)) for f in files]     # 미리 JPEG로 바꿔 두고 보내기만 한다
        self.fps, self.info = fps, info

    def frame_at(self, t, rt):
        return self.frames[min(int(t * self.fps), len(self.frames) - 1)]


class RecordedFeedForward(Recorded):
    """실시간 제약: 처리기는 한 번에 하나씩, 끝나면 그 순간의 최신 입력 프레임을 잡는다."""

    def __init__(self, files, fps, info, latency_s):
        super().__init__(files, fps, info)
        self.schedule, t = [], 0.0          # (표시 가능 시각, 프레임 번호)
        while t < len(self.frames) / fps:
            k = min(int(t * fps), len(self.frames) - 1)
            t += latency_s
            self.schedule.append((t, k))

    def frame_at(self, t, rt):
        if not rt:
            return super().frame_at(t, rt)
        ready = [k for ts, k in self.schedule if ts <= t]
        return self.frames[ready[-1]] if ready else None


# ------------------------------------------------------------------ Track C: 실시간 생성
class LiveGenerator:
    def __init__(self, a, clock):
        from diffusers import AutoencoderTiny
        from transformers import AutoModel
        self.a, self.clock = a, clock
        ck = torch.load(a.ckpt, map_location="cuda")
        sd, self.targs = ck["ema"], ck["args"]
        self.C, self.K, self.pool = self.targs["ctx"], self.targs["chunk"], self.targs.get("cam_pool", 2)
        self.nfe = a.nfe
        model = VideoDiT(self.targs["size"], cam_tokens=sd["cam_pos"].shape[1] // 6,
                         feat_dim=sd["cam_in.1.weight"].shape[1]).cuda().eval()
        model.load_state_dict(sd)
        enc = AutoModel.from_pretrained("facebook/dinov2-small", dtype=torch.float16).cuda().eval()
        tae = AutoencoderTiny.from_pretrained("madebyollin/taesd", torch_dtype=torch.float16).cuda().eval()
        # 07-1의 최적화 구성
        self.dit = torch.compile(model, mode="reduce-overhead", fullgraph=True)
        self.enc = torch.compile(enc, mode="default")
        self.dec = torch.compile(tae.decoder, mode="default")
        pre = os.path.join(a.pre, a.scene)
        self.lat = torch.from_numpy(np.load(os.path.join(pre, "latents.npy")).astype(np.float32)) * SCALE
        self.motion = torch.from_numpy(np.load(os.path.join(pre, "motion.npy")))
        self.frame_ids = json.load(open(os.path.join(pre, "frames.json")))
        self.N = len(self.frame_ids)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)
        self.buf, self.buf_loop = {}, -1
        self.stats = collections.deque(maxlen=200)
        self.lock = threading.Lock()
        t0 = time.time()
        self.warmup()
        self.warmup_s = round(time.time() - t0, 1)
        torch.cuda.reset_peak_memory_stats()

    def load_cams(self, idx):
        """청크 K프레임 × 카메라 6장의 원본 JPEG(DriveStudio 전처리 800×450)를 읽어 DINOv2 입력 224×392로 만든다. 방식 세 가지:
          pil    : CPU에서 전체 해상도로 풀고 바이큐빅 축소 (precompute.py와 같은 전처리)
          draft  : JPEG의 DCT 축소 디코딩으로 1/2 크기만 푼 뒤 축소 (PIL draft)
          nvjpeg : 파일 바이트만 CPU에서 읽고 GPU에서 디코딩·축소 (torchvision.io.decode_jpeg)"""
        paths = [os.path.join(self.a.raw, self.a.scene, "frames", f"{self.frame_ids[i]:08d}", f"{c}.jpg")
                 for i in idx for c in CAMS]
        if self.a.decode == "nvjpeg":
            from torchvision.io import decode_jpeg
            data = [torch.frombuffer(bytearray(open(p, "rb").read()), dtype=torch.uint8) for p in paths]
            imgs = torch.stack(decode_jpeg(data, device="cuda"))
            return F.interpolate(imgs.float(), (224, 392), mode="bicubic", antialias=True, align_corners=False).clamp(0, 255)
        ims = []
        for p in paths:
            im = Image.open(p)
            if self.a.decode == "draft":
                im.draft("RGB", (392, 224))  # 800×450 → 400×225로 풀림
            ims.append(torch.from_numpy(np.asarray(im.convert("RGB").resize((392, 224), Image.BICUBIC))))
        return torch.stack(ims).permute(0, 3, 1, 2)

    @torch.no_grad()
    def chunk(self, ctx, i, seed):
        K, C = self.K, self.C
        t0 = time.perf_counter()
        x = self.load_cams(range(i, i + K))
        t1 = time.perf_counter()
        x = ((x.cuda().half() / 255.0) - self.mean) / self.std
        self.last_feat = x
        tok = self.enc(pixel_values=x).last_hidden_state[:, 1:]
        tok = F.avg_pool2d(tok.reshape(-1, 16, 28, 384).permute(0, 3, 1, 2), 2).flatten(2).transpose(1, 2)
        feats = pool_feats(tok.reshape(K, 6, 112, 384).float(), self.pool)[None]
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        g = torch.Generator(device="cuda").manual_seed(seed)
        z = torch.randn((1, K) + ctx.shape[2:], device="cuda", generator=g)
        ts = torch.linspace(1.0, 0.0, self.nfe + 1, device="cuda")
        mo = self.motion[i - C:i + K][None].cuda()
        for s in range(self.nfe):
            tt = torch.cat([torch.zeros(1, C, device="cuda"), ts[s].expand(1, K)], 1)
            rr = torch.cat([torch.zeros(1, C, device="cuda"), ts[s + 1].expand(1, K)], 1)
            torch.compiler.cudagraph_mark_step_begin()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                u = self.dit(torch.cat([ctx, z], 1), tt, rr, feats, mo)[:, C:].float()
            z = z - (ts[s] - ts[s + 1]) * u
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        img = self.dec(z[0].half())
        img = ((img.float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy()
        t4 = time.perf_counter()
        ms = lambda a, b: round((b - a) * 1000, 2)
        return z, list(img), dict(load=ms(t0, t1), enc=ms(t1, t2), dit=ms(t2, t3), dec=ms(t3, t4), total=ms(t0, t4))

    def warmup(self):
        ctx = self.lat[0:self.C][None].cuda()
        for _ in range(3):
            self.chunk(ctx, self.C, 0)

    def run(self):
        """장면을 처음부터 롤아웃한다. 처음 C프레임만 Pseudo-GT 잠재를 문맥으로 쓰고 이후는 자기 출력을 이어 붙인다.
        장면이 한 바퀴 돌 때마다 새로 생성해 지연 통계를 계속 갱신한다."""
        while True:
            loop = self.clock.loop()
            ctx = self.lat[0:self.C][None].cuda()
            buf = {k: jpeg(Image.open(os.path.join(self.a.raw, self.a.scene, "frames", f"{self.frame_ids[k]:08d}",
                                                   "CHASE.jpg"))) for k in range(self.C)}
            for i in range(self.C, self.N - self.K + 1, self.K):
                z, imgs, st = self.chunk(ctx, i, seed=i)
                ctx = torch.cat([ctx, z], 1)[:, -self.C:]
                for k, im in enumerate(imgs):
                    buf[i + k] = jpeg(im)
                with self.lock:
                    self.stats.append(st)
                    if loop != self.buf_loop:
                        self.buf, self.buf_loop = buf, loop
                    else:
                        self.buf.update(buf)
            while self.clock.loop() == loop:   # 다음 바퀴까지 대기 (계산은 장면 길이보다 훨씬 짧다)
                time.sleep(0.05)

    def lag_s(self):
        with self.lock:
            tot = [s["total"] for s in self.stats]
        comp = float(np.percentile(tot, 50)) / 1000 if tot else 0.0
        return (self.K - 1) / FPS + comp     # 청크 마지막 입력을 기다리는 시간 + 계산

    def frame_at(self, t, rt):
        k = int((t - (self.lag_s() if rt else 0.0)) * FPS)
        with self.lock:
            if k < 0:
                return None
            while k >= 0 and k not in self.buf:
                k -= 1
            return self.buf.get(k)

    def summary(self):
        with self.lock:
            st = list(self.stats)
        if not st:
            return {}
        out = {k: dict(p50=round(float(np.percentile([s[k] for s in st], 50)), 1),
                       p95=round(float(np.percentile([s[k] for s in st], 95)), 1)) for k in st[0]}
        out["chunks"] = len(st)
        out["peak_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        out["lag_ms"] = round(self.lag_s() * 1000)
        out["warmup_s"] = self.warmup_s
        return out


# ------------------------------------------------------------------ HTTP
def make_handler(tracks, info, bench, index_html):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def send_json(self, obj):
            b = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            u = urlparse(self.path)
            q = parse_qs(u.query)
            if u.path == "/":
                b = open(index_html, "rb").read()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(b)
            elif u.path == "/stats":
                self.send_json(dict(info=info, live=tracks["C"].summary(), scene_t=round(clock.now(), 2)))
            elif u.path == "/bench":
                self.send_json(bench)
            elif u.path == "/stream":
                tr, rt = tracks[q.get("track", ["A"])[0]], q.get("rt", ["0"])[0] == "1"
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                blank = jpeg(Image.new("RGB", OUT_WH, (20, 20, 20)))
                try:
                    while True:
                        b = tr.frame_at(clock.now(), rt) or blank
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(b))
                        self.wfile.write(b + b"\r\n")
                        time.sleep(1.0 / FPS / 2)
                except (BrokenPipeError, ConnectionResetError):
                    pass
            else:
                self.send_error(404)
    return H


def main():
    global clock
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="scene-0916")
    ap.add_argument("--raw", default="data/real/raw")
    ap.add_argument("--pre", default="data/real_pre")
    ap.add_argument("--chase_a", default="outputs/06-1/chase/006", help="Track A 렌더 통계 (render_stats.json)")
    ap.add_argument("--chase_b", default="outputs/04-2/chase_scene-0916", help="Track B 프레임과 stats.json")
    ap.add_argument("--ckpt", default="outputs/06-1/mix25/ckpt.pt")
    ap.add_argument("--nfe", type=int, default=4)
    ap.add_argument("--decode", choices=["pil", "draft", "nvjpeg"], default="nvjpeg", help="카메라 JPEG 디코딩 방식")
    ap.add_argument("--port", type=int, default=8100)
    a = ap.parse_args()

    frames = sorted(os.listdir(os.path.join(a.raw, a.scene, "frames")))
    clock = Clock(len(frames) / FPS)
    sa = json.load(open(os.path.join(a.chase_a, "render_stats.json")))
    sb = json.load(open(os.path.join(a.chase_b, "stats.json")))
    lat_b = (sb["net_ms"]["p50"] + sb["render_ms"]["p50"]) / 1000
    info = {
        "A": dict(name="3DGS 사전 재구성", chapter="03장 · 06-1", protocol="오프라인", source="기록 재생",
                  first="장면별 학습 54~65분", per_frame=f"렌더 {sa['render_ms_mean']}ms (P95 {sa['render_ms_p95']}ms, 512×288)",
                  input="녹화 구간 전체 (미래 포함)", empty="없음"),
        "B": dict(name="Feed-forward (DrivingForward)", chapter="04장", protocol="인과적", source="기록 재생",
                  first="0.18초", per_frame=f"추론 {sb['net_ms']['p50']}ms + 렌더 {sb['render_ms']['p50']}ms",
                  input="그 순간의 카메라 6장", empty=f"약 {round((1 - sb['coverage_mean']) * 100)}%",
                  realtime_fps=round(1 / lat_b, 1)),
        "C": dict(name="생성 모델 (Video DiT, 혼합 25%)", chapter="05~07장 · 06-1", protocol="인과적", source="서버 실시간 생성",
                  first="문맥 2프레임 + 청크 1개", per_frame="아래 실시간 통계", input="지금까지의 카메라 6장 + 자차 이동량",
                  empty="없음 (모두 생성)", nfe=a.nfe),
    }
    t0 = time.time()
    tracks = {
        "A": Recorded([os.path.join(a.raw, a.scene, "frames", f, "CHASE.jpg") for f in frames], FPS, info["A"]),
        "B": RecordedFeedForward(sorted(os.path.join(a.chase_b, "frames", f) for f in os.listdir(os.path.join(a.chase_b, "frames"))),
                                 sb["input_fps"], info["B"], lat_b),
    }
    tracks["C"] = LiveGenerator(a, clock)
    print(f"ready in {time.time() - t0:.1f}s (C warmup {tracks['C'].warmup_s}s)", flush=True)
    threading.Thread(target=tracks["C"].run, daemon=True).start()
    bench = {}
    for k, p in [("opt", "outputs/07-1/bench.json"), ("causal", "outputs/07-2/causal.json")]:
        if os.path.exists(p):
            bench[k] = json.load(open(p))
    srv = ThreadingHTTPServer(("0.0.0.0", a.port), make_handler(tracks, info, bench, os.path.join(ROOT, "lab", "index.html")))
    srv.daemon_threads = True
    srv.serve_forever()


if __name__ == "__main__":
    main()
