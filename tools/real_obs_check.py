#!/usr/bin/env python3
"""What does the POLICY see on the real robot? Grab real depth frames from the bridge and turn
them into observations with PepperNavEnv's own _depth_obs (same downsampling, clipping and
invalid-pixel handling as training), then report delivered frame rate and observation stats.

    python3 tools/real_obs_check.py --out docs/calib_.../real_obs [--seconds 15] [--sim-frames npz]

Needs pepper_bridge_node.py running with publish_depth on. Moves nothing."""
import argparse, sys, time
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seconds", type=float, default=15.0)
    ap.add_argument("--sim-frames", default=None, help="a viz/frames.npz to compare against")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    import rclpy, yaml
    from sensor_msgs.msg import Image, CameraInfo
    from rl.env.pepper_nav_env import PepperNavEnv
    cfg = yaml.safe_load((REPO / "rl/config/env.yaml").read_text())

    class Shell:            # the attributes _depth_obs reads, nothing else
        pass
    env = Shell(); env.cfg = cfg; env.randomiser = None
    env.dw, env.dh = cfg["observation"]["depth_width"], cfg["observation"]["depth_height"]
    env.ros = Shell()

    rclpy.init(); node = rclpy.create_node("real_obs_check")
    frames, stamps, info = [], [], {}
    def on_img(m):
        if "enc" not in info:
            info.update(enc=m.encoding, w=m.width, h=m.height, frame=m.header.frame_id, step=m.step)
        dt = np.float32 if m.encoding == "32FC1" else np.uint16
        d = np.frombuffer(bytes(m.data), dtype=dt).reshape(m.height, m.width).astype(np.float32)
        if m.encoding != "32FC1":
            d = np.where(d > 0, d / 1000.0, np.inf)
        frames.append(d); stamps.append(time.time())
    def on_info(m):
        info.setdefault("cam_k", list(m.k)); info.setdefault("cam_wh", (m.width, m.height))
    node.create_subscription(Image, "/pepper_robot/depth_camera/image_raw", on_img, 10)
    node.create_subscription(CameraInfo, "/pepper_robot/depth_camera/camera_info", on_info, 10)
    t0 = time.time()
    while time.time() - t0 < a.seconds:
        rclpy.spin_once(node, timeout_sec=0.05)
    node.destroy_node(); rclpy.shutdown()
    if not frames:
        sys.exit("no depth frames received -- is the bridge running with publish_depth on?")

    obs = []
    for d in frames:
        env.ros._depth = d
        obs.append(PepperNavEnv._depth_obs(env)[0])
    obs = np.array(obs)
    fps = (len(stamps) - 1) / (stamps[-1] - stamps[0]) if len(stamps) > 1 else 0.0
    gaps = np.diff(stamps) if len(stamps) > 1 else np.array([0.0])
    raw = np.array(frames)
    fin = np.isfinite(raw)
    print(f"message: encoding {info.get('enc')}, {info.get('w')}x{info.get('h')}, frame '{info.get('frame')}'")
    print(f"camera_info: {info.get('cam_wh')}, fx {info.get('cam_k', [0])[0]:.1f}" if "cam_k" in info else "camera_info: none received")
    print(f"delivered: {len(frames)} frames in {a.seconds:.0f} s = {fps:.1f} fps; longest gap {gaps.max():.2f} s "
          f"(policy acts at {cfg['task']['control_hz']} Hz)")
    print(f"raw metres: finite {fin.mean()*100:.1f}%, range {np.nanmin(np.where(fin, raw, np.nan)):.2f}-"
          f"{np.nanmax(np.where(fin, raw, np.nan)):.2f} m")

    def stats(o):
        o = o.astype(float)
        return (f"mean {o.mean():5.1f}  far(255) {np.mean(o == 255)*100:4.1f}%  near(0) {np.mean(o == 0)*100:4.1f}%  "
                f"top-row mean {o[:, :6].mean():5.1f}  bottom-row mean {o[:, -6:].mean():5.1f}")
    print(f"REAL obs ({obs.shape[1]}x{obs.shape[2]}): " + stats(obs))
    if a.sim_frames:
        F = np.load(a.sim_frames); S = np.concatenate([F[k] for k in F.files])
        print(f"SIM  obs ({S.shape[1]}x{S.shape[2]}): " + stats(S))
    np.savez_compressed(out / "real_obs.npz", obs=obs, raw_first=raw[0], stamps=np.array(stamps))
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    k = [0, len(obs) // 2, len(obs) - 1]
    fig, ax = plt.subplots(1, 4, figsize=(15, 3.4))
    ax[0].imshow(np.where(np.isfinite(raw[0]), raw[0], np.nan), cmap="viridis_r"); ax[0].set_title("real raw depth (m)")
    for j, i in enumerate(k):
        ax[j + 1].imshow(obs[i], cmap="viridis_r", vmin=0, vmax=255); ax[j + 1].set_title(f"policy obs, frame {i}")
    for x in ax: x.axis("off")
    plt.tight_layout(); plt.savefig(out / "real_obs.png", dpi=90)
    print(f"wrote {out}/real_obs.png and real_obs.npz")


if __name__ == "__main__":
    main()
