#!/usr/bin/env python3
"""Train SAC on one branch. The world file is the only thing that differs between them.

    python3 rl/scripts/train.py --branch photogrammetry --room lab_a --timesteps 5000 --seed 0
    python3 rl/scripts/train.py --branch standard       --room lab_a --timesteps 300000 --seed 0

`--room` is the physical-room slug, not the branch -- both branches of the same room
pass the same `--room`, so they score against the same evaluation route set (S4.4.3).

Start the simulator first, in another terminal (or use rl/scripts/run.py, which does
this for you):

    export LIBGL_ALWAYS_SOFTWARE=1
    ros2 launch pepper_robot gz.launch.py world:=<the branch's world> \
         headless:=true enable_moveit:=false enable_rviz:=false
    ros2 launch pepper_robot_ign bridge.launch.py \
         ros_true_base_tf:=/pepper_robot/true_base_tf

What this script is careful about
---------------------------------
Proposal S4.4.2 makes the geometry source the ONLY independent variable. Everything else
-- the SAC configuration, the reward weights, the randomisation ranges, the route set --
must be identical across the two branches, and "must be identical" is not a thing to
remember, it is a thing to check. So every run directory records the SHA-256 of each
config file it used, plus the seed, the world, and the branch, and `--require-match`
refuses to start a second branch whose configs differ from the first's.
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
CONFIG_DIR = REPO / "rl" / "config"
RUNS = REPO / "rl" / "runs"

WORLD_FOR_BRANCH = {
    "photogrammetry": "room_photogrammetry",
    "standard": "room_standard",
}


def config_hashes():
    """SHA-256 of every config that shapes a run. The evidence for 'identical'."""
    out = {}
    for name in ("env.yaml", "sac.yaml", "randomisation.yaml"):
        p = CONFIG_DIR / name
        out[name] = hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None
    return out


def assert_configs_match(run_dir, hashes, require):
    """Compare this run's configs against every earlier run of the other branch.

    A mid-experiment tweak to the reward is the quiet failure this exists for: it is
    invisible in the logs, entirely reasonable-looking at the time, and it turns the
    branch comparison into a comparison of two different agents.
    """
    others = []
    for meta_path in RUNS.glob("*/meta.json"):
        try:
            m = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if m.get("config_hashes") and m["config_hashes"] != hashes:
            others.append((meta_path.parent.name, m.get("branch"), m["config_hashes"]))
    if not others:
        return
    msg = ["configs differ from earlier runs:"]
    for name, branch, h in others[:5]:
        diff = [k for k in hashes if hashes[k] != h.get(k)]
        msg.append(f"    {name} (branch={branch}) differs in: {', '.join(diff)}")
    msg.append("  S4.4.2 makes the geometry source the only independent variable, so a "
               "config change invalidates the comparison with those runs.")
    if require:
        msg.append("  Re-run with --no-require-match only if you intend to start the "
                   "experiment over.")
        print("ERROR: " + "\n".join(msg), file=sys.stderr)
        sys.exit(2)
    print("WARNING: " + "\n".join(msg), file=sys.stderr)


def save_resume_point(model, run_dir, done):
    """Checkpoint, replay buffer, then resume.json -- in that order, each written atomically.

    Needed because the full run outlives what one machine session can be trusted with (a
    cluster job's wall-time limit, a pre-emption, a Windows reboot). The checkpoint alone
    is not enough: SAC restarted on an empty replay buffer re-enters `learning_starts`
    and forgets everything it was bootstrapping from, which is a different experiment for
    the branch that happened to be interrupted. resume.json is written last, so it only
    ever names a checkpoint whose buffer is complete on disk.
    """
    import os
    ckpt = f"checkpoint_{done:08d}"
    model.save(str(run_dir / ckpt))
    tmp = run_dir / "replay_buffer.tmp.pkl"
    model.save_replay_buffer(str(tmp))
    os.replace(tmp, run_dir / "replay_buffer.pkl")
    (run_dir / "resume.tmp.json").write_text(json.dumps({"timesteps": done, "checkpoint": ckpt}),
                                             encoding="utf-8")
    os.replace(run_dir / "resume.tmp.json", run_dir / "resume.json")


def find_resumable(branch, seed, tag):
    """Newest run of this branch, seed and tag that has a resume point and is unfinished.

    The seed has to match: seeds of one branch run as parallel jobs, and a resubmitted
    seed-1 job must never continue seed 0's run.
    """
    want = f"_{branch}_seed{seed}" + (f"_{tag}" if tag else "")
    for meta_path in sorted(RUNS.glob(f"*{want}/meta.json"), reverse=True):
        d = meta_path.parent
        if not (d / "resume.json").is_file():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        point = json.loads((d / "resume.json").read_text(encoding="utf-8"))
        if point["timesteps"] < int(meta["timesteps"]):
            return d
    return None


def train_in_segments(model, run_dir, total_timesteps, segment_steps, checkpoint_every, done=0):
    """Train in segments, so a resume point can be written at a clean `learn()` boundary.

    `reset_num_timesteps=False` keeps the step counter and the TensorBoard x-axis continuous
    across segments, and leaves the in-flight episode running into the next one.
    """
    while done < total_timesteps:
        segment = min(segment_steps, total_timesteps - done)
        model.learn(total_timesteps=segment, reset_num_timesteps=False,
                    progress_bar=False, tb_log_name="SAC")
        done += segment
        print(f"  [{done:>8}/{total_timesteps}]")

        if checkpoint_every and done % checkpoint_every == 0:
            save_resume_point(model, run_dir, done)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--branch", required=True, choices=sorted(WORLD_FOR_BRANCH))
    ap.add_argument("--room", required=True,
                    help="physical-room slug, e.g. lab_a -- matches "
                         "simulation/photogrammetry/ground_truth/<room>_spec.yaml and "
                         "tools/gen_world.py --room. Selects the evaluation route set "
                         "(shared by both branches, per S4.4.3 -- see rl/env/routes.py) "
                         "and, on the standard branch, the geometry training samples "
                         "start/goal pairs from.")
    ap.add_argument("--world", default=None,
                    help="override the branch's world model name (for smoke tests)")
    ap.add_argument("--timesteps", type=int, default=None,
                    help="overrides sac.yaml training.total_timesteps")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--no-randomisation", action="store_true",
                    help="an ablation, not a convenience. S4.4.2 applies the same "
                         "randomisation to both branches; turning it off for one of them "
                         "destroys the comparison.")
    ap.add_argument("--no-require-match", dest="require_match", action="store_false",
                    help="allow configs that differ from earlier runs (starts over)")
    ap.add_argument("--tag", default="", help="suffix for the run directory name")
    ap.add_argument("--resume", default=None, metavar="RUN_DIR|auto",
                    help="continue an interrupted run from its last resume point "
                         "(checkpoint + replay buffer, written every "
                         "checkpoint_every_steps). 'auto' picks the newest unfinished run "
                         "of this --branch and --tag, and starts a fresh run if there is "
                         "none -- what a resubmitted cluster job wants.")
    args = ap.parse_args()

    import yaml
    from stable_baselines3 import SAC
    from stable_baselines3.common.monitor import Monitor

    from rl.env.pepper_nav_env import PepperNavEnv
    from rl.env.randomisation import Randomiser
    from rl.env.routes import load_routes

    sac_cfg = yaml.safe_load((CONFIG_DIR / "sac.yaml").read_text(encoding="utf-8"))
    tr = sac_cfg["training"]
    timesteps = args.timesteps or int(tr["total_timesteps"])
    seed = args.seed if args.seed is not None else int(tr["seed"])
    world = args.world or WORLD_FOR_BRANCH[args.branch]

    hashes = config_hashes()
    RUNS.mkdir(parents=True, exist_ok=True)
    randomiser = Randomiser(enabled=not args.no_randomisation)

    resume_dir = None
    if args.resume == "auto":
        resume_dir = find_resumable(args.branch, seed, args.tag)
        print(f"--resume auto: {resume_dir.name if resume_dir else 'nothing unfinished; fresh run'}")
    elif args.resume:
        resume_dir = Path(args.resume).resolve()
        if not (resume_dir / "resume.json").is_file():
            print(f"ERROR: {resume_dir} has no resume.json -- it never reached a checkpoint",
                  file=sys.stderr)
            sys.exit(2)

    if resume_dir is not None:
        # Everything that defines the experiment comes from the run being resumed, never
        # from this invocation's flags -- and anything that would have changed it is refused.
        meta = json.loads((resume_dir / "meta.json").read_text(encoding="utf-8"))
        problems = [f"{k}: run has {meta.get(k)!r}, now {v!r}" for k, v in (
            ("branch", args.branch), ("room", args.room), ("config_hashes", hashes),
            ("randomisation", randomiser.fingerprint())) if meta.get(k) != v]
        if problems:
            print("ERROR: cannot resume, the experiment would change:\n    "
                  + "\n    ".join(problems), file=sys.stderr)
            sys.exit(2)
        run_dir = resume_dir
        point = json.loads((run_dir / "resume.json").read_text(encoding="utf-8"))
        seed, world, timesteps = int(meta["seed"]), meta["world"], int(meta["timesteps"])
        meta.setdefault("resumed", []).append(
            {"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "from_timesteps": point["timesteps"]})
    else:
        name = f"{time.strftime('%Y%m%d_%H%M%S')}_{args.branch}_seed{seed}" + (
            f"_{args.tag}" if args.tag else "")
        run_dir = RUNS / name
        run_dir.mkdir()
        assert_configs_match(run_dir, hashes, args.require_match)

    env = PepperNavEnv(world_name=world, branch=args.branch, source="sim",
                       randomiser=randomiser, seed=seed, room=args.room)
    env = Monitor(env, str(run_dir / "monitor.csv"), info_keywords=("is_success",),
                  override_existing=resume_dir is None)
    # Keyed by --room, not by `world` (the model directory name). S4.4.3 evaluates both
    # branches "on the same fixed set of N start/goal routes" -- one set, and it is
    # planned on the standard branch's spec regardless of which branch is training here.
    # See the decision recorded in rl/env/routes.py.
    routes = load_routes(args.room)

    if resume_dir is None:
        meta = {
            "branch": args.branch,
            "room": args.room,
            "world": world,
            "seed": seed,
            "timesteps": timesteps,
            "config_hashes": hashes,
            "randomisation": randomiser.fingerprint(),
            "n_routes": len(routes),
            "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"run: {run_dir.relative_to(REPO)}")
    print(f"  branch={args.branch} room={args.room} world={world} seed={seed} "
          f"timesteps={timesteps:,}")
    print(f"  randomisation={randomiser.fingerprint()}")

    hp = dict(sac_cfg["hyperparameters"])
    net = sac_cfg["network"]
    done = 0
    if resume_dir is not None:
        done = int(point["timesteps"])
        # num_timesteps, the entropy coefficient and the optimiser states come back with
        # the checkpoint; the replay buffer does not, hence the second file.
        model = SAC.load(str(run_dir / point["checkpoint"]), env=env,
                         tensorboard_log=str(run_dir / "tb"))
        model.load_replay_buffer(str(run_dir / "replay_buffer.pkl"))
        # Reseeded with an offset, so the resumed segment does not replay the action noise
        # and route draws of the run's first segment.
        model.set_random_seed(seed + done)
        print(f"  resumed at {done:,} steps; replay buffer {model.replay_buffer.size():,} "
              f"transitions")
    else:
        model = SAC(sac_cfg["policy"], env, seed=seed, verbose=1,
                    tensorboard_log=str(run_dir / "tb"),
                    policy_kwargs={
                        "net_arch": list(net["net_arch"]),
                        # Width of NatureCNN's output inside SB3's CombinedExtractor,
                        # before it is concatenated with the goal vector. Reached only
                        # because the depth channel is uint8 -- with float32 SB3 would not
                        # see an image and this key would be ignored along with the CNN.
                        "features_extractor_kwargs": {
                            "cnn_output_dim": int(net["cnn_output_dim"])},
                    },
                    **hp)

    # model.policy.features_extractor is None here, not the CombinedExtractor: SACPolicy
    # defaults to share_features_extractor=False, so it never sets a top-level extractor
    # of its own -- actor and critic each build their own instance. The actor's is the
    # one that actually shapes what the trained policy sees, so it is the one to inspect.
    extractor = type(model.policy.actor.features_extractor).__name__
    print(f"  features extractor: {extractor}")
    if "depth" in model.policy.actor.features_extractor.extractors:
        sub = type(model.policy.actor.features_extractor.extractors["depth"]).__name__
        print(f"  depth sub-extractor: {sub}")
        if sub != "NatureCNN":
            print("  WARNING: the depth channel is not going through a CNN. The "
                  "observation dtype is probably not uint8 -- see rl/env/"
                  "pepper_nav_env.py, and note that SB3 reports nothing when this "
                  "happens.", file=sys.stderr)

    try:
        train_in_segments(
            model, run_dir,
            total_timesteps=timesteps,
            segment_steps=int(tr["eval_every_steps"]),
            checkpoint_every=int(tr["checkpoint_every_steps"]),
            done=done)
    finally:
        model.save(str(run_dir / "final_model"))
        env.close()
        print(f"saved {run_dir / 'final_model.zip'}")


if __name__ == "__main__":
    main()
