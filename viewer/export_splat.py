"""DriveStudio 체크포인트의 배경 가우시안을 웹 뷰어용 .splat 파일로 내보낸다 — 책 03-6.

.splat 형식(가우시안 하나당 32바이트, 웹 스플랫 뷰어들이 공통으로 읽는 형식):
    위치 float32×3 | 크기 float32×3 | 색 RGBA uint8×4 | 회전 쿼터니언(w,x,y,z) uint8×4
보는 방향에 따른 색(SH 고차항)은 버리고 기본색(DC)만 남긴다. 파일이 약 1/7로 줄어든다.

DriveStudio 컨테이너에서 실행 (GPU 불필요):
    python /work/viewer/export_splat.py \
        --ckpt /work/outputs/drivestudio/nuscenes/streetgs_scene0061/checkpoint_final.pth \
        --traj /work/outputs/03-6/trajectory.json --out /work/outputs/03-6/scene.splat
"""
import argparse
import json

import numpy as np
import torch

SH_C0 = 0.28209479177387814  # 0차 구면 조화 함수 상수


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--traj", required=True, help="render_chase_drivestudio.py --traj_json 결과")
    ap.add_argument("--out", required=True)
    ap.add_argument("--radius", type=float, default=80.0, help="자차 궤적에서 이 거리(m) 안의 가우시안만 남긴다")
    ap.add_argument("--min_opacity", type=float, default=0.05)
    args = ap.parse_args()

    bg = torch.load(args.ckpt, map_location="cpu")["models"]["Background"]
    xyz = bg["_means"].numpy()
    scale = np.exp(bg["_scales"].numpy())
    quat = bg["_quats"].numpy()
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    rgb = np.clip(0.5 + SH_C0 * bg["_features_dc"].numpy(), 0, 1)
    alpha = 1 / (1 + np.exp(-bg["_opacities"].numpy()[:, 0]))

    # 궤적 근처·충분히 불투명한 가우시안만 남긴다 (먼 하늘 쪽 랜덤 점 제거)
    traj = np.array(json.load(open(args.traj))["ego_to_world"])[:, :3, 3]
    d2 = np.full(len(xyz), np.inf)
    for p in traj[::5]:
        d2 = np.minimum(d2, ((xyz[:, :2] - p[:2]) ** 2).sum(1))
    keep = (d2 < args.radius ** 2) & (alpha > args.min_opacity)
    # 보이는 기여가 큰 것부터 저장 (뷰어가 앞부분만 읽어도 형태가 잡힌다)
    order = np.argsort(-(scale.prod(1) * alpha)[keep])
    idx = np.nonzero(keep)[0][order]

    buf = np.zeros(len(idx), dtype=[("xyz", "<f4", 3), ("scale", "<f4", 3), ("rgba", "u1", 4), ("rot", "u1", 4)])
    buf["xyz"] = xyz[idx]
    buf["scale"] = scale[idx]
    buf["rgba"][:, :3] = (rgb[idx] * 255).round()
    buf["rgba"][:, 3] = (alpha[idx] * 255).round()
    buf["rot"] = np.clip(quat[idx] * 128 + 128, 0, 255).round()  # DriveStudio 쿼터니언 순서도 (w, x, y, z)
    buf.tofile(args.out)
    print(dict(total=len(xyz), kept=len(idx), mib=round(buf.nbytes / 2**20, 1)))


if __name__ == "__main__":
    main()
