#!/usr/bin/env python3
"""Measure how fast the simulator actually runs, before committing to a training budget.

    python3 rl/scripts/bench_throughput.py --world room_photogrammetry
    python3 rl/scripts/bench_throughput.py --world room_photogrammetry --no-depth
    python3 rl/scripts/bench_throughput.py --world room_photogrammetry --compare

Why this exists
---------------
Software rendering was an open question for Task 5: under
`LIBGL_ALWAYS_SOFTWARE=1` every rendered sensor is CPU-rasterised, and the cost is
dominated by the 320x240 depth camera rather than the 15-ray lidar, and was
worth measuring early because it sets the Task 5 budget. This is that measurement.

The number that matters is not frames per second, it is **how many wall-clock hours a
training run costs**. So the report ends with that, computed from the two things that
actually set it:

  * **RTF** -- how much simulated time passes per second of real time. Ignition publishes
    it on `/world/<world>/stats`, and it is the honest figure: it already accounts for
    physics, rendering, and every plugin in the world.
  * **The achievable control rate** -- an agent cannot act faster than its observations
    arrive. A depth camera *declared* at 15 Hz that only delivers 4 Hz caps the control
    rate at 4 Hz no matter what the env asks for, and the gap between declared and
    delivered is exactly what software rendering costs.

Both are measured, because either one alone is misleading. A healthy RTF with a starved
camera means the physics is fine and the policy is blind; a fast camera at RTF 0.1 means
the observations are fine and a 500k-step run takes a fortnight.

What it does NOT do
-------------------
It does not train, and it does not move the robot. A stationary robot is the optimistic
case for rendering cost (no motion blur work, maximal cache coherence), so treat the
result as an upper bound and expect training to be somewhat slower.
"""

import argparse
import json
import os
import re
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
REPORT_DIR = REPO / "rl" / "reports"

# Bringing the sim up and taking it down lives in one place, shared with rl/scripts/run.py
# -- see the module docstring there for the two failures its shape is built around.
from rl.sim_process import (  # noqa: E402
    die, launch, resolve_world, shutdown, wait_for_sim, sample_stats,
    DEPTH_TOPIC as ROS_DEPTH_TOPIC, SCAN_TOPIC as ROS_SCAN_TOPIC,
)

# `ign gazebo` names the world from the SDF's <world name="...">, not from the filename.
# gen_world.py sets them equal, but read it rather than assume -- a mismatch here shows up
# as a silent zero-sample benchmark, which looks like "the simulator is broken".

# `ros2 topic hz` output. Lost when 143cf45 moved the launch code into rl/sim_process.py,
# which left sample_topic_hz raising NameError at the end of every benchmark.
ROS_HZ_RE = re.compile(r"average rate:\s*([\d.]+)")


# --------------------------------------------------------------------------------------
# launching the stack
# --------------------------------------------------------------------------------------

def sample_topic_hz(ros_topic, window=40, timeout=90.0):
    """Delivered rate of a ROS 2 topic, in Hz, via `ros2 topic hz`.

    Measured on the ROS side, and with someone else's implementation, both deliberately.

    The ROS side because that is what the environment subscribes to: a rate taken on the
    Ignition side excludes ros_ign_bridge and DDS, which are in the path the agent
    actually observes through, and the agent cannot act faster than observations reach it.

    Someone else's implementation because two home-grown attempts gave confidently wrong
    answers here. Timing `ign topic -e -n N` measures the transport's delivery of a queued
    backlog, not the publication rate -- n=11 and n=21 both returned in ~0.18 s, implying
    52 Hz and 111 Hz for the same 10 Hz lidar. Reading the message stamps instead fixes
    that but runs into a second artefact: the first moments after subscribing are a drain
    burst with stamps 5 ms apart, which reports exactly 200.00 Hz -- the physics rate --
    for every sensor regardless of its update_rate. Discarding a warm-up window helped and
    did not converge: successive runs reported 200, 187, 157, 58 and 28 Hz for the same
    lidar. `ros2 topic hz` is well-tested and its window semantics are documented, which
    is worth more here than a bespoke measurement that is subtly wrong in a way that reads
    as a real number.
    """
    if ros_topic is None:
        return None
    try:
        proc = subprocess.Popen(
            ["ros2", "topic", "hz", ros_topic, "--window", str(window)],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
            start_new_session=True)
    except FileNotFoundError:
        return None
    rates, deadline = [], time.time() + timeout
    try:
        for line in proc.stdout:
            m = ROS_HZ_RE.search(line)
            if m:
                rates.append(float(m.group(1)))
                # The first report covers a partially-filled window. Take several and use
                # the last, by which point the window is full.
                if len(rates) >= 4:
                    break
            if time.time() > deadline:
                break
    finally:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            proc.kill()
        proc.wait()
    return rates[-1] if rates else None


# --------------------------------------------------------------------------------------
# the part that answers the actual question
# --------------------------------------------------------------------------------------

def budget(rtf, control_hz, timesteps, n_branches=2):
    """Wall-clock hours for a training run.

    An SAC timestep is one agent action, so it consumes 1/control_hz of *simulated*
    seconds, which at the measured RTF costs 1/(control_hz * rtf) seconds of real time.
    Doubled for the two branches, because the experiment is a comparison and a plan that
    only affords one branch affords none.
    """
    if rtf <= 0 or control_hz <= 0:
        return float("inf")
    return timesteps / (control_hz * rtf) / 3600.0 * n_branches


def configured_control_hz():
    """What rl/config/env.yaml actually asks for.

    Read rather than assumed, because the point of this benchmark is to check the
    configuration against reality. A budget computed at the sensor ceiling would flatter
    a config that asks for less, and one computed at a hard-coded rate would not notice a
    config that asks for more than the sensors can give.
    """
    import yaml
    cfg = REPO / "rl" / "config" / "env.yaml"
    try:
        return float(yaml.safe_load(cfg.read_text(encoding="utf-8"))["task"]["control_hz"])
    except Exception:
        return float("inf")


def report(res, args):
    w = 34
    def row(k, v):
        print(f"  {k:<{w}} {v}")

    print()
    print("=" * 74)
    print(f"  {res['label']}")
    print("=" * 74)
    row("world", res["world_name"])
    row("depth camera", "on" if res["depth"] else "OFF")
    row("front lidar", "on" if res["lidar"] else "OFF")
    print()
    row("real-time factor (RTF)", f"{res['rtf']:.3f}")
    row("physics steps / sec", f"{res['steps_per_sec']:.0f}")
    row("sampled over", f"{res['real_elapsed']:.1f} s real, {res['sim_elapsed']:.1f} s sim")
    print()
    for name, declared, measured in res["sensors"]:
        if measured is None:
            row(f"{name} rate", "not measured (topic absent)")
        else:
            frac = measured / declared if declared else 0.0
            flag = "  <-- STARVED" if frac < 0.75 else ""
            row(f"{name} rate", f"{measured:5.2f} Hz  of {declared} Hz declared "
                                f"({frac:.0%}){flag}")
    print()

    # The achievable control rate is the slowest thing the policy must wait for. If depth
    # is the observation modality, depth sets it -- there is no point asking for 10 Hz
    # control off a 3 Hz camera.
    limit = res["control_hz_limit"]
    row("achievable control rate", f"{limit:.2f} Hz  ({res['control_limited_by']})")

    configured = configured_control_hz()
    row("configured control rate", f"{configured:.2f} Hz  (rl/config/env.yaml)")
    if configured > limit:
        print(f"  {'':34} ^^ ASKS FOR MORE THAN THE SENSORS DELIVER: the env would wait "
              f"on observations that are not there. Lower task.control_hz to {limit:.1f} "
              f"or below, or raise the sensor's update_rate in pepper_robot.gazebo.")
    print()

    # Budget at the rate training will actually use, not at the ceiling. Quoting the
    # ceiling would understate the cost by whatever margin the config leaves.
    rate = min(configured, limit)
    print(f"  Wall-clock cost of training BOTH branches at {rate:.1f} Hz:")
    for steps in (100_000, 300_000, 500_000, 1_000_000):
        hours = budget(res["rtf"], rate, steps)
        days = hours / 24.0
        note = ""
        if args.deadline_days is not None:
            note = "  OK" if days <= args.deadline_days else f"  OVER by {days - args.deadline_days:.1f} d"
        print(f"    {steps:>9,} steps/branch   {hours:7.1f} h  ({days:5.1f} days){note}")
    print("=" * 74)


DEPTH_HZ_DECLARED = 15    # pepper_robot.gazebo, ign_depth_camera update_rate
LIDAR_HZ_DECLARED = 10    # pepper_robot.gazebo, ign_front_lidar update_rate

# The names the environment subscribes to. Kept in step with
# rl/env/pepper_nav_env.RosInterface -- if these drift apart, this benchmark measures
# topics the agent never reads.


def run_once(world_path, world_name, depth, lidar, args, label):
    spawn = None
    if args.spawn:
        # Same pose convention as run.py: z is the branch's own local floor plus a drop.
        from rl.env.floor_height import FloorHeight
        x, y, yaw = (float(v) for v in args.spawn.split())
        spawn = (x, y, FloorHeight(world_name).spawn_z(x, y), yaw)
    procs = launch(world_path, depth, lidar, args.ign_verbosity, spawn=spawn)
    try:
        print(f"[{label}] launching (this takes ~30-60 s: xacro -> SDF, then spawn)...")
        if not wait_for_sim(world_name, args.startup_timeout):
            die(f"world /{world_name} never started stepping within "
                f"{args.startup_timeout}s. "
                f"Run the same launch by hand and read the log. The usual cause is "
                f"gz_ros2_control blocking in Configure -- look for "
                f"'robot_state_publisher service not available, waiting again...' "
                f"repeating at 2 Hz. That node lives in move_group.launch.py, so "
                f"enable_moveit:=false must be paired with a robot_state_publisher of "
                f"its own (gz.launch.py launches one under UnlessCondition).")
        print(f"[{label}] up. settling for {args.settle}s, then sampling for "
              f"{args.duration}s...")
        time.sleep(args.settle)

        rtf, sps, d_sim, d_real = sample_stats(world_name, args.duration)

        sensors = []
        control_hz_limit = float("inf")
        limited_by = "physics"
        if depth:
            hz = sample_topic_hz(ROS_DEPTH_TOPIC, window=args.hz_window)
            sensors.append(("depth camera", DEPTH_HZ_DECLARED, hz))
            if hz and hz < control_hz_limit:
                control_hz_limit, limited_by = hz, "depth camera"
        if lidar:
            hz = sample_topic_hz(ROS_SCAN_TOPIC, window=args.hz_window)
            sensors.append(("front lidar", LIDAR_HZ_DECLARED, hz))
            if hz and hz < control_hz_limit:
                control_hz_limit, limited_by = hz, "front lidar"
        if control_hz_limit == float("inf"):
            # Either no rendered sensor, or none could be measured. Falling back to the
            # declared rate rather than to infinity: an unmeasurable sensor should not
            # silently produce an optimistic budget.
            control_hz_limit = float(DEPTH_HZ_DECLARED if depth else LIDAR_HZ_DECLARED)
            limited_by = "NOT MEASURED -- assuming the declared rate"

        return {
            "label": label, "world_name": world_name, "world_path": str(world_path),
            "depth": depth, "lidar": lidar,
            "rtf": rtf, "steps_per_sec": sps,
            "sim_elapsed": d_sim, "real_elapsed": d_real,
            "sensors": sensors,
            "control_hz_limit": control_hz_limit,
            "control_limited_by": limited_by,
            "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
    finally:
        print(f"[{label}] shutting down...")
        shutdown(procs)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--world", default="room_photogrammetry",
                    help="room name, world name, or path to an SDF world")
    ap.add_argument("--no-depth", action="store_true",
                    help="disable the depth camera (lidar-only). A deviation from the "
                         "depth-based modality of proposal S4.4.2 -- benchmark freely, "
                         "but declare it if you ever train on it.")
    ap.add_argument("--no-lidar", action="store_true", help="disable the front lidar")
    ap.add_argument("--compare", action="store_true",
                    help="run depth-on and depth-off back to back and print both")
    ap.add_argument("--duration", type=float, default=60.0,
                    help="seconds to sample RTF over (default 60)")
    ap.add_argument("--settle", type=float, default=15.0,
                    help="seconds to wait after startup before sampling; the first "
                         "seconds include controller spin-up and asset loading and are "
                         "not representative (default 15)")
    ap.add_argument("--hz-window", type=int, default=40,
                    help="`ros2 topic hz --window` size for the sensor rates (default 40)")
    ap.add_argument("--startup-timeout", type=float, default=180.0)
    ap.add_argument("--ign-verbosity", default="1")
    ap.add_argument("--spawn", default=None, metavar="'X Y YAW'",
                    help="where to stand Pepper, as for run.py. Pass the training pose: "
                         "without it she spawns at (0, 0, 0), which in the final_scan "
                         "rooms is the corner -- touching the walls, and the contact "
                         "solve slows the physics enough to make the RTF meaningless.")
    ap.add_argument("--deadline-days", type=float, default=None,
                    help="if given, flag which training budgets fit in this many days")
    ap.add_argument("--json", metavar="PATH", nargs="?", const="auto",
                    help="also write the raw numbers as JSON (default: rl/reports/)")
    args = ap.parse_args()

    if os.environ.get("LIBGL_ALWAYS_SOFTWARE") != "1":
        print("note: LIBGL_ALWAYS_SOFTWARE is not set in your shell; this script sets it "
              "for the simulator itself, which is what matters.", file=sys.stderr)

    world_path, world_name = resolve_world(args.world)
    print(f"world: {world_path}  (<world name=\"{world_name}\">)")

    runs = []
    if args.compare:
        runs.append(run_once(world_path, world_name, True, True, args, "depth ON  + lidar"))
        runs.append(run_once(world_path, world_name, False, True, args, "depth OFF + lidar"))
    else:
        runs.append(run_once(world_path, world_name, not args.no_depth, not args.no_lidar,
                             args, "throughput"))

    for r in runs:
        report(r, args)

    if len(runs) == 2:
        a, b = runs
        print()
        print(f"  Dropping the depth camera: RTF {a['rtf']:.3f} -> {b['rtf']:.3f} "
              f"({b['rtf'] / a['rtf']:.1f}x), control rate "
              f"{a['control_hz_limit']:.2f} -> {b['control_hz_limit']:.2f} Hz.")
        print("  That speed-up is the price of the depth-based modality fixed by S4.4.2.")
        print("  It is a real option, and it is a deviation. Do not take it silently.")
        print()

    if args.json:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        path = (REPORT_DIR / f"throughput_{time.strftime('%Y%m%d_%H%M%S')}.json"
                if args.json == "auto" else Path(args.json))
        path.write_text(json.dumps(runs, indent=2), encoding="utf-8")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
