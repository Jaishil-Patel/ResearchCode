#!/usr/bin/env python3
"""The one command: bring up the simulator, train, tear down. Both branches if asked.

    python3 rl/scripts/run.py --room lab_a --branch both --timesteps 300000
    python3 rl/scripts/run.py --room lab_a --branch photogrammetry --timesteps 5000 --smoke

This is what closes the gap between "the code exists" and "drop in a room and go":
before this, a run needed three terminals kept in sync by hand -- gz.launch.py, then
bridge.launch.py, then train.py, in that order, with the world argument typed correctly
in two of them and the topic remap correct in the third. That is exactly the kind of
manual bookkeeping that goes stale silently; get one of the three flags wrong and the env
does not error, it just hangs in `wait_for_data` looking like a frozen training run.

What this script actually does, per branch:
    1. `rl.sim_process.launch()` -- Ignition headless, no MoveIt, no RViz, plus the ROS
       bridge with the pose remap already correct (see rl/sim_process.py for why that
       remap exists and what it silently breaks if it drifts).
    2. `wait_for_sim()` -- waits for the world to be STEPPING, not merely advertised.
       Distinguishing those is what stopped an `enable_moveit:=false` regression from
       wedging the simulator invisibly during Task 4 development.
    3. `rl/scripts/train.py`, as a subprocess with the same Python interpreter (so it
       runs inside whatever venv this was launched from) -- not called in-process,
       because train.py is also a normal standalone entry point and duplicating its
       argument handling here would be a second copy of logic that could drift from
       the first.
    4. Teardown, whether training succeeded, failed, or was interrupted.

`--branch both` runs the two branches SEQUENTIALLY, each against its own simulator
instance, never simultaneously -- there is one GPU and one set of CPU cores here, and
running both branches at once would make the RTF (and therefore the training budget
`rl/scripts/bench_throughput.py` measured) meaningless for either.

`--smoke` is not a separate code path, it just defaults `--timesteps` low (5000) and
tags the run "smoke" -- exactly what the plan's verification section asks for: a fast
end-to-end proof that dropping in a room and running one command actually trains,
before spending the wall-clock budget `bench_throughput.py` computed on the real run.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from rl.sim_process import die, launch, resolve_world, shutdown, wait_for_sim  # noqa: E402
from rl.scripts.train import WORLD_FOR_BRANCH  # noqa: E402
from rl.env.floor_height import FloorHeight  # noqa: E402

LAUNCH_ATTEMPTS = 3
CONTROLLER_TIMEOUT_S = 150.0


def expected_controllers():
    """The controllers gz.launch.py spawns when MoveIt is off -- read from the same YAML."""
    import yaml
    from ament_index_python.packages import get_package_share_directory
    cfg = Path(get_package_share_directory("pepper_robot_moveit_config")) / "config" / \
        "moveit_controller_manager.yaml"
    return set(yaml.safe_load(cfg.read_text())["controller_names"]) | {"joint_state_broadcaster"}


def active_controllers():
    try:
        out = subprocess.run(["ros2", "control", "list_controllers"], capture_output=True,
                             text=True, timeout=20).stdout
    except subprocess.TimeoutExpired:
        return set()
    import re
    clean = re.sub(r"\x1b\[[0-9;]*m", "", out)
    return {ln.split()[0] for ln in clean.splitlines() if ln.strip().endswith("active")}


def wait_for_controllers(expected, timeout):
    """Block until every expected controller is active. Without them every joint is limp:
    found 30 Sep 2026, `ros2 control list_controllers` -> "No controllers are currently
    loaded", and Pepper sagged, fell and could not be reset; the controller manager also
    failed to come up once in a clean launch, so this is checked, not assumed."""
    t0 = time.time()
    got = set()
    while time.time() - t0 < timeout:
        got = active_controllers()
        if expected <= got:
            return True, got
        time.sleep(5.0)
    return False, got


def run_branch(branch, room, timesteps, seed, tag, no_randomisation, require_match,
               settle, startup_timeout, ign_verbosity, extra_train_args, spawn):
    world_path, world_name = resolve_world(WORLD_FOR_BRANCH[branch])
    print(f"\n{'=' * 74}\n  {branch}  (world: {world_name})\n{'=' * 74}")

    # Never the default (0, 0, 0): with the corner origin that is inside two wall boxes
    # in the standard world, which drops it to RTF ~0.002 and it never starts stepping
    # (found by the final_scan smoke test, 27 Sep 2026). Same x, y, yaw for both
    # branches; z is each branch's own local floor + a drop, as at every reset.
    x, y, yaw = spawn
    z = FloorHeight(world_name).spawn_z(x, y)
    print(f"spawning Pepper at ({x:.2f}, {y:.2f}, {z:.3f}), yaw {yaw:.2f}")
    expected = expected_controllers()
    for attempt in range(1, LAUNCH_ATTEMPTS + 1):
        procs = launch(world_path, depth=True, lidar=True, verbosity=ign_verbosity,
                       spawn=(x, y, z, yaw))
        print(f"launch attempt {attempt}/{LAUNCH_ATTEMPTS}: waiting for the world to start stepping...")
        if not wait_for_sim(world_name, startup_timeout):
            print(f"  {world_name} never started stepping within {startup_timeout}s")
            shutdown(procs)
            continue
        print(f"  stepping. waiting for {len(expected)} controllers to be active...")
        ok, got = wait_for_controllers(expected, CONTROLLER_TIMEOUT_S)
        if ok:
            print(f"  all controllers active: {sorted(got)}")
            break
        print(f"  controllers missing after {CONTROLLER_TIMEOUT_S:.0f}s: {sorted(expected - got)}; relaunching")
        shutdown(procs)
    else:
        die(f"{world_name}: no clean launch in {LAUNCH_ATTEMPTS} attempts (world never stepped, or "
            f"the ros2_control controllers never became active). Run the launch by hand and read "
            f"the log.")
    try:
        print(f"settling {settle}s for the bridge to come up...")
        time.sleep(settle)

        cmd = [sys.executable, str(REPO / "rl" / "scripts" / "train.py"),
               "--branch", branch, "--room", room, "--seed", str(seed)]
        if timesteps is not None:
            cmd += ["--timesteps", str(timesteps)]
        if tag:
            cmd += ["--tag", tag]
        if no_randomisation:
            cmd.append("--no-randomisation")
        if not require_match:
            cmd.append("--no-require-match")
        cmd += extra_train_args

        print(f"  $ {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=str(REPO))
        if result.returncode != 0:
            die(f"train.py exited {result.returncode} for branch {branch!r}. The "
                f"simulator is still being torn down; nothing else runs after this.")
    finally:
        print(f"tearing down {branch}...")
        shutdown(procs)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--room", required=True,
                    help="physical-room slug -- matches "
                         "simulation/photogrammetry/ground_truth/<room>_spec.yaml and "
                         "tools/gen_world.py --room. The same slug is used for both "
                         "branches, since S4.4.3 scores them on one shared route set.")
    ap.add_argument("--branch", default="both",
                    choices=["both", "photogrammetry", "standard"])
    ap.add_argument("--timesteps", type=int, default=None,
                    help="per branch. Defaults to sac.yaml's training.total_timesteps, "
                         "or 5000 under --smoke.")
    ap.add_argument("--smoke", action="store_true",
                    help="fast end-to-end check: --timesteps 5000 and tag 'smoke', "
                         "unless --timesteps is given explicitly. Run this before "
                         "committing to the real wall-clock budget "
                         "rl/scripts/bench_throughput.py computed.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--no-randomisation", action="store_true",
                    help="an ablation, not a convenience -- see train.py's own warning")
    ap.add_argument("--no-require-match", dest="require_match", action="store_false")
    ap.add_argument("--settle", type=float, default=15.0,
                    help="seconds to wait after the world starts stepping before "
                         "starting train.py, so the bridge has finished subscribing")
    ap.add_argument("--startup-timeout", type=float, default=180.0)
    ap.add_argument("--ign-verbosity", default="1")
    ap.add_argument("--spawn", default=None, metavar="'X Y YAW'",
                    help="initial pose for Pepper, identical in both branches (z is "
                         "each branch's local floor). Default: the start of the first "
                         "route in the fixed route set -- never (0,0,0), the corner.")
    args, extra = ap.parse_known_args()

    if not Path(REPO / "simulation" / "photogrammetry" / "ground_truth"
               / f"{args.room}_spec.yaml").is_file():
        die(f"no spec for room {args.room!r}. This is required even to train the "
            f"photogrammetry branch alone: the shared evaluation route set (S4.4.3) is "
            f"always planned on the standard branch's spec -- see the decision recorded "
            f"in rl/env/routes.py. Copy simulation/photogrammetry/ground_truth/"
            f"TEMPLATE_spec.yaml to {args.room}_spec.yaml and fill it in from the tape "
            f"measurements (docs/photogrammetry_runbook.md step 9), then build the "
            f"route set once:\n"
            f"    python3 -m rl.env.routes --room {args.room} --build")

    timesteps = args.timesteps if args.timesteps is not None else (5000 if args.smoke
                                                                    else None)
    tag = args.tag or ("smoke" if args.smoke else None)

    if args.spawn:
        spawn = tuple(float(v) for v in args.spawn.split())
        if len(spawn) != 3:
            die("--spawn wants three numbers: 'X Y YAW'")
    else:
        from rl.env.routes import load_routes
        r0 = load_routes(args.room)[0]
        spawn = (r0["start"][0], r0["start"][1], r0.get("start_yaw", 0.0))

    branches = ["photogrammetry", "standard"] if args.branch == "both" else [args.branch]
    print(f"room={args.room}  branches={branches}  "
          f"timesteps={'sac.yaml default' if timesteps is None else timesteps}"
          f"{'  [SMOKE]' if args.smoke else ''}")

    for branch in branches:
        # timesteps=None is passed straight through to run_branch/train.py, meaning
        # "let train.py read sac.yaml's own default" -- kept as None rather than reading
        # the config a second time here, so there is exactly one place that default lives.
        run_branch(branch, args.room, timesteps=timesteps,
                  seed=args.seed, tag=tag, no_randomisation=args.no_randomisation,
                  require_match=args.require_match, settle=args.settle,
                  startup_timeout=args.startup_timeout,
                  ign_verbosity=args.ign_verbosity, extra_train_args=extra,
                  spawn=spawn)

    print(f"\ndone: {branches}")


if __name__ == "__main__":
    main()
