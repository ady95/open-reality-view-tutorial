"""05-4~05-6: 카메라 특징 조건부 소형 Video DiT.

한 번에 '문맥 프레임 C장 + 생성할 프레임 K장'의 추격 시점 잠재를 다룬다.
- 프레임마다 노이즈 시각(t)과 구간 길이(t - r)를 따로 받는다 (Diffusion Forcing식 프레임별 노이즈).
  문맥 프레임은 t = 0(깨끗함) 또는 작은 노이즈, 생성 프레임은 학습 중 뽑은 t.
- r = t 이면 순간 속도 v를, r < t 이면 MeanFlow의 평균 속도 u를 예측한다. 같은 네트워크로
  Flow Matching(05-4)과 MeanFlow(05-5)를 모두 학습할 수 있다.
- 조건: 프레임마다 카메라 6대의 DINOv2 토큰(6×112)을 교차 어텐션으로, 자차 이동량을 adaLN으로 넣는다.

평소에는 fused 어텐션을 쓰고, MeanFlow 학습의 JVP 구간에서만 행렬곱으로 직접 계산한다.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

CONFIGS = {
    "S": dict(dim=384, depth=12, heads=6),
    "B": dict(dim=768, depth=12, heads=12),
}


def timestep_embedding(t, dim, max_period=10000.0):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t.float()[..., None] * 1000.0 * freqs
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


FUSED = {"on": True}  # JVP를 계산할 때만 False로 바꾼다


def attention(q, k, v):
    # q, k, v: (B, heads, N, d)
    if FUSED["on"]:
        return F.scaled_dot_product_attention(q, k, v)
    # fused 커널에는 forward-mode 자동미분(JVP) 규칙이 없어 행렬곱으로 직접 계산한다
    w = (q @ k.transpose(-1, -2)) * (q.shape[-1] ** -0.5)
    return w.softmax(dim=-1) @ v


class MHA(nn.Module):
    def __init__(self, dim, heads, kv_dim=None):
        super().__init__()
        self.h = heads
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(kv_dim or dim, dim * 2)
        self.o = nn.Linear(dim, dim)
        self.qn = nn.LayerNorm(dim // heads, elementwise_affine=False)
        self.kn = nn.LayerNorm(dim // heads, elementwise_affine=False)

    def forward(self, x, ctx=None):
        B, N, D = x.shape
        ctx = x if ctx is None else ctx
        q = self.q(x).view(B, N, self.h, -1).transpose(1, 2)
        k, v = self.kv(ctx).view(B, ctx.shape[1], 2, self.h, -1).permute(2, 0, 3, 1, 4)
        out = attention(self.qn(q), self.kn(k), v)
        return self.o(out.transpose(1, 2).reshape(B, N, D))


class Block(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.n1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = MHA(dim, heads)
        self.n2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.cross = MHA(dim, heads)
        self.n3 = nn.LayerNorm(dim, elementwise_affine=False)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(approximate="tanh"), nn.Linear(dim * 4, dim))
        self.ada = nn.Linear(dim, dim * 9)
        nn.init.zeros_(self.ada.weight)
        nn.init.zeros_(self.ada.bias)

    def forward(self, x, cond, cam):
        # x: (B, F, T, D) 프레임별 공간 토큰, cond: (B, F, 1, D) 프레임 조건, cam: (B*K, M, D) 생성 프레임의 카메라 토큰
        s1, b1, g1, s2, b2, g2, s3, b3, g3 = self.ada(F.silu(cond)).chunk(9, dim=-1)
        B, F_, T, D = x.shape
        h = (self.n1(x) * (1 + s1) + b1).reshape(B, F_ * T, D)               # 시공간 전체 자기 어텐션
        x = x + g1 * self.attn(h).reshape(B, F_, T, D)
        K = cam.shape[0] // B                                                 # 문맥 프레임은 카메라를 보지 않는다
        h = (self.n2(x[:, -K:]) * (1 + s2[:, -K:]) + b2[:, -K:]).reshape(B * K, T, D)
        upd = (g2[:, -K:] * self.cross(h, cam).reshape(B, K, T, D))           # 프레임마다 자기 카메라 6대만 본다
        x = torch.cat([x[:, :-K], x[:, -K:] + upd], dim=1)
        x = x + g3 * self.mlp(self.n3(x) * (1 + s3) + b3)
        return x


class VideoDiT(nn.Module):
    def __init__(self, size="S", latent_ch=4, latent_hw=(18, 32), patch=2, feat_dim=384, cam_tokens=28,
                 n_cams=6, max_frames=16):
        super().__init__()
        c = CONFIGS[size]
        D = self.dim = c["dim"]
        self.p, self.ch, self.hw = patch, latent_ch, latent_hw
        self.gh, self.gw = latent_hw[0] // patch, latent_hw[1] // patch
        self.T = self.gh * self.gw
        self.x_in = nn.Linear(latent_ch * patch * patch, D)
        self.pos = nn.Parameter(torch.zeros(1, 1, self.T, D))
        self.frame_pos = nn.Parameter(torch.zeros(1, max_frames, 1, D))
        nn.init.normal_(self.pos, std=0.02)
        nn.init.normal_(self.frame_pos, std=0.02)
        self.t_mlp = nn.Sequential(nn.Linear(256, D), nn.SiLU(), nn.Linear(D, D))
        self.dt_mlp = nn.Sequential(nn.Linear(256, D), nn.SiLU(), nn.Linear(D, D))
        self.m_mlp = nn.Sequential(nn.Linear(3, D), nn.SiLU(), nn.Linear(D, D))
        self.cam_in = nn.Sequential(nn.LayerNorm(feat_dim), nn.Linear(feat_dim, D))
        self.cam_pos = nn.Parameter(torch.zeros(1, n_cams * cam_tokens, D))
        nn.init.normal_(self.cam_pos, std=0.02)
        self.blocks = nn.ModuleList([Block(D, c["heads"]) for _ in range(c["depth"])])
        self.out_norm = nn.LayerNorm(D, elementwise_affine=False)
        self.out_ada = nn.Linear(D, 2 * D)
        self.out = nn.Linear(D, latent_ch * patch * patch)
        for m in (self.out_ada, self.out):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def patchify(self, z):  # (B, F, C, H, W) → (B, F, T, C*p*p)
        B, F_, C, H, W = z.shape
        p = self.p
        z = z.reshape(B, F_, C, H // p, p, W // p, p).permute(0, 1, 3, 5, 2, 4, 6)
        return z.reshape(B, F_, self.T, C * p * p)

    def unpatchify(self, x):  # (B, F, T, C*p*p) → (B, F, C, H, W)
        B, F_ = x.shape[:2]
        p = self.p
        x = x.reshape(B, F_, self.gh, self.gw, self.ch, p, p).permute(0, 1, 4, 2, 5, 3, 6)
        return x.reshape(B, F_, self.ch, self.gh * p, self.gw * p)

    def forward(self, z, t, r, feats, motion):
        """z: (B, F, C, H, W) 문맥+생성 프레임 잠재, t·r: (B, F) 프레임별 시각,
        feats: (B, K, 6, M, 384) 생성할 K프레임의 카메라 토큰, motion: (B, F, 3) 이전 프레임 대비 자차 이동량.
        반환: (B, F, C, H, W) 속도(r=t) 또는 평균 속도(r<t). 문맥 프레임 출력은 쓰지 않는다."""
        B, F_ = z.shape[:2]
        x = self.x_in(self.patchify(z)) + self.pos + self.frame_pos[:, :F_]          # (B, F, T, D)
        cond = (self.t_mlp(timestep_embedding(t, 256)) + self.dt_mlp(timestep_embedding(t - r, 256))
                + self.m_mlp(motion))[:, :, None]                                     # (B, F, 1, D)
        K = feats.shape[1]
        cam = self.cam_in(feats.reshape(B * K, -1, feats.shape[-1])) + self.cam_pos
        for blk in self.blocks:
            x = blk(x, cond, cam)
        s, b = self.out_ada(F.silu(cond)).chunk(2, dim=-1)
        return self.unpatchify(self.out(self.out_norm(x) * (1 + s) + b))
