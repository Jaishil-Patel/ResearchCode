#!/usr/bin/env python3
"""base_sysid's profiles, sent THROUGH PepperNavEnv.step() -- the path a policy's commands take
-- so the simulated latency (env.yaml action.sim_command_delay_steps) is included, exactly as
the real robot's measurement included the bridge + shim + NAOqi chain.

    python3 tools/env_sysid.py --world room_standard --out docs/calib_20261002_0948/base_sysid_sim_env.json
    python3 tools/base_sysid.py --compare docs/calib_.../base_sysid_real.json docs/calib_20261002_0948/base_sysid_sim_env.json

Launches its own world. No randomisation (the nominal sim), like evaluation."""
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "rl" / "scripts")); sys.path.insert(0, str(REPO / "tools"))
from base_sysid import PROFILES, analyse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", default="room_standard")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--out", default=str(REPO / "docs" / "calib_20261002_0948" / "base_sysid_sim_env.json"))
    a = ap.parse_args()
    from rl.sim_process import launch, shutdown, resolve_world, wait_for_sim
    from rl.env.floor_height import FloorHeight
    from rl.env.pepper_nav_env import PepperNavEnv
    from run import expected_controllers, wait_for_controllers

    path, name = resolve_world(a.world)
    fh = FloorHeight(name)
    procs = launch(path, spawn=(1.0, 1.4, fh.spawn_z(1.0, 1.4), 0.0))
    results = {"source": "sim-env", "world": a.world, "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "profiles": {}}
    try:
        assert wait_for_sim(name, 180), "world never started stepping"
        ok, _ = wait_for_controllers(expected_controllers(), 150)
        assert ok, "controllers not active"
        time.sleep(10)
        env = PepperNavEnv(world_name=name, branch="standard", source="sim", randomiser=None, seed=0,
                           room="final_scan")
        ros = env.unwrapped.ros if hasattr(env, "unwrapped") else env.ros
        print(f"sim command delay: {env.sim_delay_steps} steps = {env.sim_delay_steps * env.control_dt:.2f} s")
        to_a = lambda v, w: np.array([-1.0 + 2.0 * (v - env.v_min) / (env.v_max - env.v_min), w / env.w_max],
                                     dtype=np.float32)
        for pname, v, w, secs, start in PROFILES:
            for rep in range(1, a.repeats + 1):
                env.reset(options={"route": {"route_id": "sysid", "start": list(start), "start_yaw": 0.0,
                                             "goal": [3.75, 1.10], "optimal_length_m": 1.0}})
                for _ in range(3):                      # settle at rest, as the real runs did
                    env.step(to_a(0.0, 0.0))
                # Record the pose on every spin inside step(), not once per step: 5 Hz would
                # quantise latency to 0.2 s, coarser than the real-robot measurement.
                # Only NEW pose messages (the env replaces the tuple on each one), as base_sysid
                # does: spins run ~400 Hz, the pose ~50-100 Hz, and duplicate samples make the
                # differentiated speed spike through the 10%/90% thresholds instantly.
                samples = []
                spin0 = ros.spin
                last = [ros._pose]
                def spin_rec(sec, _s=spin0):
                    _s(sec)
                    p = ros._pose
                    if p is not None and p is not last[0]:
                        samples.append((ros.now(),) + tuple(p)); last[0] = p
                ros.spin = spin_rec
                t_cmd = ros.now()
                samples.append((t_cmd,) + tuple(ros._pose))
                steps = int(round(secs / env.control_dt))
                try:
                    for _ in range(steps):
                        _, _, term, _, _ = env.step(to_a(v, w))
                        if term:
                            break
                finally:
                    ros.spin = spin0
                r = analyse(samples, v, w)
                results["profiles"].setdefault(pname, {"v": v, "w": w, "runs": []})["runs"].append(r)
                f = lambda k, pct=False: "-" if r.get(k) is None else (f"{100 * r[k]:.0f}%" if pct else f"{r[k]:.2f}s")
                print(f"{pname:14s} #{rep}: speed {f('speed_ratio', True):>5s} turn {f('turn_ratio', True):>5s} "
                      f"latency {f('latency_s'):>6s} rise {f('rise_10_90_s'):>6s} (n={r['n']})", flush=True)
    finally:
        shutdown(procs)
    Path(a.out).write_text(json.dumps(results, indent=1))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
