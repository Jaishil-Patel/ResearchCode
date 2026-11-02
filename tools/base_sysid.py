#!/usr/bin/env python3
"""Base system identification: run the SAME velocity-command profiles on the real Pepper and in
the simulator, and compare how each responds. This is what makes "the sim base behaves like the
real one" a measured statement rather than a hope.

    # real robot: naoqi_shim.py + pepper_bridge_node.py running. Run the bridge with odometry at 50 Hz, not its 20 Hz default:
    #     python3 tools/pepper_bridge_node.py --ros-args -p odom_rate:=50.0
    # At 20 Hz a 0.1 s rise time reads as ~0.2 s (measured 1 Oct 2026 against the fake robot),
    # and the rise time is what sets the simulated base's acceleration limit.
    python3 tools/base_sysid.py --source real --out docs/base_sysid_real.json

    # simulator: a world running with Pepper spawned and bridge.launch.py up
    python3 tools/base_sysid.py --source sim --world room_standard --out docs/calib_20261002_0948/base_sysid_sim.json

    python3 tools/base_sysid.py --compare docs/base_sysid_real.json docs/calib_20261002_0948/base_sysid_sim.json

Per profile it reports: steady-state speed and turn rate as a fraction of the command, response
latency (command to first 10% of the step), and 10-90% rise time (which sets the acceleration
limit to give the simulated base). Each profile runs twice so the spread is visible.

Pose sources differ by necessity: the simulator's ground truth (/pepper_robot/true_base_tf), and
NAOqi's own odometry on hardware (/pepper_robot/odom) until the Task 6 pose rig exists. Over a
few seconds of driving NAOqi odometry is accurate to a few percent, but it is still odometry --
state that in the write-up.

REAL ROBOT SAFETY: every profile waits for Enter before moving. Clear space needed: about 1.5 m
ahead of Pepper for the straight runs, 0.6 m around her for spins and arcs. Ctrl-C stops the base.
"""
import argparse
import json
import math
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# (name, v m/s, w rad/s, seconds, sim start pose). Starts are checked-clear in room_standard
# (arms included, ~0.3 m radius); on the real robot the operator positions her instead.
PROFILES = [
    ("straight_0.3", 0.30, 0.0, 4.0, (1.0, 1.4)),
    ("slow_0.1", 0.10, 0.0, 5.0, (1.0, 1.4)),
    ("reverse_-0.1", -0.10, 0.0, 5.0, (2.0, 1.4)),
    ("spin_left", 0.0, 0.90, 5.0, (3.1, 1.9)),
    ("spin_right", 0.0, -0.90, 5.0, (3.1, 1.9)),
    ("arc_left", 0.10, 0.50, 6.0, (3.1, 1.9)),
    ("arc_right", 0.10, -0.50, 6.0, (3.1, 1.9)),
]
REPEATS = 2
STEADY_AFTER_S = 1.5


def analyse(samples, v, w):
    """samples: list of (t, x, y, yaw) from command onset. Returns the summary dict."""
    import numpy as np
    s = np.array(samples)
    t = s[:, 0] - s[0, 0]
    yaw = np.unwrap(s[:, 3])
    seg = np.hypot(np.diff(s[:, 1]), np.diff(s[:, 2]))
    arclen = np.concatenate([[0.0], np.cumsum(seg)])
    m = t >= STEADY_AFTER_S
    out = {"n": int(len(s)), "duration_s": float(t[-1])}
    if m.sum() >= 3:
        speed = float(np.polyfit(t[m], arclen[m], 1)[0])
        rate = float(np.polyfit(t[m], yaw[m], 1)[0])
        out["speed_mps"] = speed
        out["turn_rps"] = rate
        out["speed_ratio"] = speed / abs(v) if v else None
        out["turn_ratio"] = rate / w if w else None
        out["heading_drift_rps"] = rate if not w else None
    # step response on the dominant channel: speed (path) or turn rate (yaw)
    dt = np.gradient(t)
    sig = np.gradient(arclen) / np.maximum(dt, 1e-6) if v else np.gradient(yaw) / np.maximum(dt, 1e-6) * math.copysign(1, w)
    sig = np.convolve(sig, np.ones(5) / 5, mode="same")
    target = out.get("speed_mps") if v else abs(out.get("turn_rps", 0.0))
    if target:
        def first(frac):
            idx = np.where(sig >= frac * target)[0]
            return float(t[idx[0]]) if len(idx) else None
        t10, t90 = first(0.1), first(0.9)
        out["latency_s"] = t10
        out["rise_10_90_s"] = (t90 - t10) if (t10 is not None and t90 is not None) else None
    return out


def run(args):
    import rclpy
    from geometry_msgs.msg import Twist

    rclpy.init()
    node = rclpy.create_node("base_sysid")
    lock = threading.Lock()
    latest = {"pose": None}

    def q_to_yaw(q):
        return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))

    now = lambda: node.get_clock().now().nanoseconds * 1e-9

    def stamp(h):
        """Message's own timestamp (sim time in sim, robot clock on hardware); arrival time
        only as a fallback -- arrival jitter distorts short, slow runs."""
        s = h.stamp.sec + h.stamp.nanosec * 1e-9
        return s if s > 0 else now()
    if args.source == "sim":
        from tf2_msgs.msg import TFMessage
        from rl.sim_process import POSE_TOPIC

        def on_tf(msg):
            for tr in msg.transforms:
                with lock:
                    latest["pose"] = (stamp(tr.header), tr.transform.translation.x, tr.transform.translation.y,
                                      q_to_yaw(tr.transform.rotation))
        node.create_subscription(TFMessage, POSE_TOPIC, on_tf, 50)
        from rl.env.ign_control import IgnitionControl
        from rl.env.floor_height import FloorHeight
        ign = IgnitionControl(args.world, timeout_ms=10000)
        floor = FloorHeight(args.world)
    else:
        from nav_msgs.msg import Odometry

        def on_odom(msg):
            p = msg.pose.pose
            with lock:
                latest["pose"] = (stamp(msg.header), p.position.x, p.position.y, q_to_yaw(p.orientation))
        node.create_subscription(Odometry, "/pepper_robot/odom", on_odom, 50)

    pub = node.create_publisher(Twist, "/pepper_robot/cmd_vel", 10)
    cmd = Twist()
    stop = threading.Event()

    def spin():
        while not stop.is_set():
            rclpy.spin_once(node, timeout_sec=0.01)

    def pump():
        while not stop.is_set():
            pub.publish(cmd)
            time.sleep(0.05)

    threading.Thread(target=spin, daemon=True).start()
    threading.Thread(target=pump, daemon=True).start()

    def halt():
        cmd.linear.x = cmd.angular.z = 0.0

    t0 = time.time()
    while latest["pose"] is None:
        if time.time() - t0 > 20:
            sys.exit("no pose received -- is the sim/bridge (or the real-robot bridge) running?")
        time.sleep(0.1)

    results = {"source": args.source, "world": args.world if args.source == "sim" else None,
               "generated": time.strftime("%Y-%m-%dT%H:%M:%S"), "profiles": {}}
    try:
        for name, v, w, secs, start in PROFILES:
            for rep in range(1, args.repeats + 1):
                halt(); time.sleep(2.0)
                if args.source == "sim":
                    ign.set_pose("pepper_robot", start[0], start[1],
                                 z=floor.spawn_z(*start), yaw=0.0)
                    time.sleep(3.0)
                else:
                    input(f"\n[{name} #{rep}] v={v:+.2f} m/s w={w:+.2f} rad/s for {secs:.0f} s. "
                          f"Position Pepper with clear space, then press Enter (Ctrl-C aborts)...")
                samples = []
                cmd.linear.x, cmd.angular.z = float(v), float(w)
                end = time.time() + secs
                last = None
                while time.time() < end:
                    with lock:
                        p = latest["pose"]
                    if p is not None and p is not last:
                        samples.append(p); last = p
                    time.sleep(0.01)
                halt()
                if len(samples) < 10:
                    print(f"{name} #{rep}: too few pose samples ({len(samples)})"); continue
                r = analyse(samples, v, w)
                results["profiles"].setdefault(name, {"v": v, "w": w, "runs": []})["runs"].append(r)
                sr = f"{100 * r['speed_ratio']:.0f}%" if r.get("speed_ratio") is not None else "-"
                tr = f"{100 * r['turn_ratio']:.0f}%" if r.get("turn_ratio") is not None else "-"
                lat = f"{r['latency_s']:.2f}s" if r.get("latency_s") is not None else "-"
                rise = f"{r['rise_10_90_s']:.2f}s" if r.get("rise_10_90_s") is not None else "-"
                hz = r["n"] / r["duration_s"] if r["duration_s"] > 0 else 0.0
                r["pose_rate_hz"] = hz
                print(f"{name:14s} #{rep}: speed {sr:>5s}  turn {tr:>5s}  latency {lat:>6s}  "
                      f"rise {rise:>6s}  (n={r['n']}, {hz:.0f} Hz)", flush=True)
                if hz < 30:
                    print(f"  WARNING: pose samples at {hz:.0f} Hz -- latency and rise time are "
                          f"quantised to {1000 / max(hz, 1):.0f} ms; rise times will read high", flush=True)
    except KeyboardInterrupt:
        print("\ninterrupted -- stopping the base")
    finally:
        halt(); time.sleep(0.5); stop.set()
        node.destroy_node(); rclpy.shutdown()
    Path(args.out).write_text(json.dumps(results, indent=1))
    print(f"\nwrote {args.out}")


def compare(real_path, sim_path):
    real, sim = (json.loads(Path(p).read_text()) for p in (real_path, sim_path))
    mean = lambda runs, k: (sum(r[k] for r in runs if r.get(k) is not None) /
                            max(1, sum(1 for r in runs if r.get(k) is not None))) \
        if any(r.get(k) is not None for r in runs) else None
    f = lambda x, pct=False: "-" if x is None else (f"{100 * x:.0f}%" if pct else f"{x:.2f}")
    print(f"{'profile':14s} | {'speed real':>10s} {'sim':>6s} | {'turn real':>9s} {'sim':>6s} | "
          f"{'latency real':>12s} {'sim':>5s} | {'rise real':>9s} {'sim':>5s}")
    for name, _, _, _, _ in PROFILES:
        if name not in real["profiles"] or name not in sim["profiles"]:
            continue
        rr, ss = real["profiles"][name]["runs"], sim["profiles"][name]["runs"]
        print(f"{name:14s} | {f(mean(rr, 'speed_ratio'), True):>10s} {f(mean(ss, 'speed_ratio'), True):>6s} | "
              f"{f(mean(rr, 'turn_ratio'), True):>9s} {f(mean(ss, 'turn_ratio'), True):>6s} | "
              f"{f(mean(rr, 'latency_s')):>12s} {f(mean(ss, 'latency_s')):>5s} | "
              f"{f(mean(rr, 'rise_10_90_s')):>9s} {f(mean(ss, 'rise_10_90_s')):>5s}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["sim", "real"])
    ap.add_argument("--world", default="room_standard")
    ap.add_argument("--out", default=None)
    ap.add_argument("--compare", nargs=2, metavar=("REAL_JSON", "SIM_JSON"))
    ap.add_argument("--repeats", type=int, default=REPEATS,
                    help=f"runs per profile (default {REPEATS}); --repeats 1 halves robot time, "
                         "at the cost of not seeing run-to-run spread")
    args = ap.parse_args()
    if args.compare:
        return compare(*args.compare)
    if not args.source:
        ap.error("--source sim|real (or --compare)")
    args.out = args.out or str(REPO / "docs" / f"base_sysid_{args.source}.json")
    run(args)


if __name__ == "__main__":
    main()
