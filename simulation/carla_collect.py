"""CARLA 데이터 수집: 입력 카메라 6대 + 가상 추격 카메라(RGB·깊이·의미 분할) 동기 녹화.

CARLA 서버가 떠 있는 상태에서 실행한다 (책 02-4 참조).

사용:
    python simulation/carla_collect.py --ticks 300 --out data/carla/run0
출력:
    <out>/frames/<frame>/<CAM>.jpg, chase_depth.png, chase_semantic.png
    <out>/meta.jsonl   프레임마다 시각·자차 포즈·센서 포즈·내부 파라미터
"""
import argparse
import json
import math
import os
import queue
import random
import time

import carla
import numpy as np
from PIL import Image

W, H = 800, 450  # 입력 카메라: nuScenes 1600x900의 절반

# CARLA는 왼손 좌표계: x 앞, y 오른쪽, z 위 (nuScenes 자차 좌표와 y 부호가 반대)
# (이름, x, y, z, yaw, fov) — nuScenes 장착 위치를 참고해 배치. CARLA 차량 원점은 차체 중앙(바닥)
INPUT_CAMS = [
    ("CAM_FRONT_LEFT", 1.5, -0.5, 1.5, -55, 70),
    ("CAM_FRONT", 1.7, 0.0, 1.5, 0, 70),
    ("CAM_FRONT_RIGHT", 1.5, 0.5, 1.5, 55, 70),
    ("CAM_BACK_LEFT", 1.0, -0.5, 1.6, -110, 70),
    ("CAM_BACK", -2.4, 0.0, 1.2, 180, 90),  # 차량 원점이 차체 중앙이므로 뒤 범퍼 쪽으로
    ("CAM_BACK_RIGHT", 1.0, 0.5, 1.6, 110, 70),
]
CHASE = dict(x=-7.0, z=3.5, pitch=-18.0, fov=90, w=800, h=450)


def intrinsic(w, h, fov):
    f = w / (2.0 * math.tan(math.radians(fov) / 2.0))
    return [[f, 0.0, w / 2.0], [0.0, f, h / 2.0], [0.0, 0.0, 1.0]]


def tf_dict(t):
    return dict(x=t.location.x, y=t.location.y, z=t.location.z,
                roll=t.rotation.roll, pitch=t.rotation.pitch, yaw=t.rotation.yaw)


def spawn_camera(world, bp_name, vehicle, transform, w, h, fov):
    bp = world.get_blueprint_library().find(bp_name)
    bp.set_attribute("image_size_x", str(w))
    bp.set_attribute("image_size_y", str(h))
    bp.set_attribute("fov", str(fov))
    return world.spawn_actor(bp, transform, attach_to=vehicle, attachment_type=carla.AttachmentType.Rigid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--town", default=None, help="예: Town10HD_Opt. 지정하지 않으면 현재 맵 사용")
    ap.add_argument("--ticks", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--dt", type=float, default=0.1, help="시뮬레이션 고정 시간 간격(초)")
    ap.add_argument("--npc", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--weather", default=None, help="예: ClearNoon, CloudySunset. carla.WeatherParameters의 프리셋 이름")
    ap.add_argument("--out", default="data/carla/run0")
    ap.add_argument("--nosave", action="store_true", help="저장하지 않고 시뮬레이션·센서 수신 속도만 잰다")
    args = ap.parse_args()
    random.seed(args.seed)

    client = carla.Client(args.host, args.port)
    client.set_timeout(120.0)
    world = client.load_world(args.town) if args.town else client.get_world()
    if args.weather:
        world.set_weather(getattr(carla.WeatherParameters, args.weather))
    print("server", client.get_server_version(), "| client", client.get_client_version(),
          "| map", world.get_map().name, "| weather", args.weather or "current")

    original = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = args.dt
    world.apply_settings(settings)
    tm = client.get_trafficmanager()
    tm.set_synchronous_mode(True)
    tm.set_random_device_seed(args.seed)

    actors = []
    try:
        bpl = world.get_blueprint_library()
        spawns = world.get_map().get_spawn_points()
        random.shuffle(spawns)
        ego_bp = (bpl.filter("vehicle.tesla.model3") or bpl.filter("vehicle.*"))[0]
        ego = world.spawn_actor(ego_bp, spawns[0])
        actors.append(ego)
        ego.set_autopilot(True, tm.get_port())
        for sp in spawns[1:1 + args.npc]:
            npc = world.try_spawn_actor(random.choice(bpl.filter("vehicle.*")), sp)
            if npc:
                npc.set_autopilot(True, tm.get_port())
                actors.append(npc)
        print("ego", ego.type_id, "| npc", len(actors) - 1)

        sensors, queues, meta_cam = {}, {}, {}
        for name, x, y, z, yaw, fov in INPUT_CAMS:
            tf = carla.Transform(carla.Location(x=x, y=y, z=z), carla.Rotation(yaw=yaw))
            sensors[name] = spawn_camera(world, "sensor.camera.rgb", ego, tf, W, H, fov)
            meta_cam[name] = dict(K=intrinsic(W, H, fov), w=W, h=H)
        chase_tf = carla.Transform(carla.Location(x=CHASE["x"], z=CHASE["z"]), carla.Rotation(pitch=CHASE["pitch"]))
        for name, bp in [("CHASE", "sensor.camera.rgb"), ("chase_depth", "sensor.camera.depth"),
                         ("chase_semantic", "sensor.camera.semantic_segmentation")]:
            sensors[name] = spawn_camera(world, bp, ego, chase_tf, CHASE["w"], CHASE["h"], CHASE["fov"])
        meta_cam["CHASE"] = dict(K=intrinsic(CHASE["w"], CHASE["h"], CHASE["fov"]), w=CHASE["w"], h=CHASE["h"])
        actors.extend(sensors.values())
        for name, s in sensors.items():
            queues[name] = queue.Queue()
            s.listen(queues[name].put)

        os.makedirs(os.path.join(args.out, "frames"), exist_ok=True)
        meta_f = open(os.path.join(args.out, "meta.jsonl"), "w")
        json.dump(dict(cameras=meta_cam, dt=args.dt, map=world.get_map().name, weather=args.weather, seed=args.seed),
open(os.path.join(args.out, "rig.json"), "w"), indent=1)

        saved, mismatched, t_start = 0, 0, None
        for i in range(args.warmup + args.ticks):
            frame = world.tick()
            data = {}
            for name, q in queues.items():
                d = q.get(timeout=30.0)
                while d.frame < frame:  # 이전 틱에서 남은 데이터는 버린다
                    d = q.get(timeout=30.0)
                data[name] = d
            if any(d.frame != frame for d in data.values()):
                mismatched += 1
                continue
            if i < args.warmup:
                continue
            if t_start is None:
                t_start = time.perf_counter()
            if args.nosave:
                saved += 1
                continue
            fdir = os.path.join(args.out, "frames", f"{frame:08d}")
            os.makedirs(fdir, exist_ok=True)
            for name, d in data.items():
                arr = np.frombuffer(d.raw_data, dtype=np.uint8).reshape(d.height, d.width, 4)
                if name == "chase_depth":  # 깊이는 RGB 24비트에 인코딩되어 있으므로 무손실 PNG
                    Image.fromarray(arr[:, :, 2::-1]).save(os.path.join(fdir, "chase_depth.png"))
                elif name == "chase_semantic":  # 클래스 ID는 R 채널
                    Image.fromarray(arr[:, :, 2]).save(os.path.join(fdir, "chase_semantic.png"))
                else:
                    Image.fromarray(arr[:, :, 2::-1]).save(os.path.join(fdir, f"{name}.jpg"), quality=92)
            snap = world.get_snapshot()
            rec = dict(frame=frame, sim_time=snap.timestamp.elapsed_seconds,
                       ego=tf_dict(ego.get_transform()),
                       sensors={n: tf_dict(data[n].transform) for n in list(meta_cam)})
            meta_f.write(json.dumps(rec) + "\n")
            saved += 1
        wall = time.perf_counter() - t_start
        meta_f.close()
        if args.nosave:
            print(f"[nosave] timesteps {saved} | mismatched {mismatched} | wall {wall:.1f} s | {saved / wall:.2f} timesteps/s")
            return
        size = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(args.out) for f in fs)
        print(f"saved timesteps {saved} | mismatched {mismatched} | wall {wall:.1f} s "
              f"| {saved / wall:.2f} timesteps/s | disk {size / 2**20:.0f} MiB ({size / saved / 2**20:.2f} MiB/timestep)")
    finally:
        world.apply_settings(original)
        tm.set_synchronous_mode(False)
        for a in reversed(actors):
            if a.is_alive:
                if isinstance(a, carla.Sensor):
                    a.stop()
                a.destroy()


if __name__ == "__main__":
    main()
