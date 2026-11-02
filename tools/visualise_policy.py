#!/usr/bin/env python3
"""Watch what a trained policy does: roll it out on the fixed route set and draw it.

    python3 tools/visualise_policy.py rl/runs/<run_dir> [--routes 8] [--gif r003]

Launches the run's own world exactly as training does (spawn, controller check, contact relay),
loads <run_dir>/final_model.zip and drives every requested route deterministically with
randomisation OFF. Writes into <run_dir>/viz/:

  paths.png      top-down room map: every route's path, start (o), goal (*), outcome colour,
                 plus the room as the policy's branch sees it (the tape-measured boxes; for the
                 photogrammetric branch also the scanned obstacles, in grey)
  <route>.gif    a route frame by frame: top-down position beside the 64x48 depth image the
                 policy actually received (default: the longest reached and the longest
                 collided route; --gif picks one)
  rollouts.json  the raw trajectories and outcomes; frames.npz the depth frames, so
                 `--render-only` can redraw everything without the simulator
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "rl" / "scripts"))
sys.path.insert(0, str(REPO / "tools"))


def room_geometry(room, world):
    import gen_world
    from rl.env.routes import spec_path_for
    spec = gen_world.load_spec(spec_path_for(room))
    boxes = gen_world.standard_room_obstacles(spec)
    L, W = float(spec["room"]["length"]), float(spec["room"]["width"])
    scan = None
    if world == "room_photogrammetry":
        import trimesh
        m = trimesh.load(str(REPO / "ros2_ws/src/pepper_room_description/models/room_photogrammetry"
                             "/meshes/visual/room.obj"), force="mesh", process=False)
        P, _ = trimesh.sample.sample_surface(m, 300000)
        P = P[(P[:, 2] > 0.25) & (P[:, 2] < 1.6)]
        r = 0.05
        H, xe, ye = np.histogram2d(P[:, 0], P[:, 1], bins=[np.arange(-0.2, L + 0.2, r), np.arange(-0.2, W + 0.2, r)])
        scan = (H.T > 4, xe, ye)
    return L, W, boxes, scan


def draw_room(ax, L, W, boxes, scan):
    import matplotlib.patches as mp
    if scan is not None:
        occ, xe, ye = scan
        ax.imshow(np.where(occ, 0.75, np.nan), extent=(xe[0], xe[-1], ye[0], ye[-1]), origin="lower",
                  cmap="gray", vmin=0, vmax=1, alpha=0.8, zorder=0)
    ax.add_patch(mp.Rectangle((0, 0), L, W, fill=False, lw=1.5, ec="#333"))
    for b in boxes:
        if b["name"].startswith(("x_", "y_")):
            continue  # walls are the room outline
        (cx, cy, _), (sx, sy, _) = b["centre"], b["size"]
        if b.get("shape") == "cylinder":
            ax.add_patch(mp.Circle((cx, cy), sx / 2, fc="#b85", ec="#743", alpha=0.7))
        else:
            ax.add_patch(mp.Rectangle((cx - sx / 2, cy - sy / 2), sx, sy, fc="#cba", ec="#865", alpha=0.6))
            ax.text(cx, cy, b["name"], ha="center", va="center", fontsize=6, color="#543")
    ax.set_xlim(-0.2, L + 0.2); ax.set_ylim(-0.2, W + 0.2); ax.set_aspect("equal")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")


def outcome(rr):
    if rr["reached_goal"]:
        return "reached", "#2a7"
    if rr["collisions"]:
        return "collided", "#c33"
    return ("timed out" if rr["timed_out"] else "stopped"), "#d90"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--routes", type=int, default=20, help="how many of the fixed routes (default all 20)")
    ap.add_argument("--gif", default=None, help="route id to animate (default: the longest rollout)")
    ap.add_argument("--spawn", default="0.6 2.105 0.0")
    ap.add_argument("--render-only", action="store_true", help="redraw from viz/rollouts.json + frames.npz")
    ap.add_argument("--world", default=None,
                    help="run the policy in this world instead of its own (cross-branch diagnostic, "
                         "e.g. room_standard); writes to viz_in_<world>/")
    ap.add_argument("--variant", default=None,
                    help="route set variant (the final evaluations use v2); "
                         "writes to viz_<variant>/ so the default set's drawings are kept")
    args = ap.parse_args()

    run = Path(args.run_dir)
    meta = json.loads((run / "meta.json").read_text())
    branch, room, world = meta["branch"], meta["room"], meta["world"]
    out = run / ("viz" if not args.variant else f"viz_{args.variant}")
    if args.world and args.world != world:
        world = args.world
        branch = "photogrammetry" if world == "room_photogrammetry" else "standard"
        out = run / f"viz_in_{world}" if not args.variant else run / f"viz_{args.variant}_in_{world}"
    out.mkdir(exist_ok=True)
    if args.render_only:
        recs = json.loads((out / "rollouts.json").read_text())
        fr = np.load(out / "frames.npz")
        render(out, run.name, branch, room, world, [(r, list(fr[r["route_id"]])) for r in recs], args.gif)
        return

    from stable_baselines3 import SAC
    from rl.sim_process import launch, shutdown, resolve_world, wait_for_sim
    from rl.env.floor_height import FloorHeight
    from rl.env.routes import load_routes
    from rl.env.pepper_nav_env import PepperNavEnv
    from run import expected_controllers, wait_for_controllers

    routes = load_routes(room, variant=args.variant)[: args.routes]
    path, name = resolve_world(world)
    x, y, yaw = (float(v) for v in args.spawn.split())
    procs = launch(path, depth=True, lidar=True, spawn=(x, y, FloorHeight(name).spawn_z(x, y), yaw))
    rollouts = []
    try:
        if not wait_for_sim(name, 180):
            sys.exit("world never started stepping")
        ok, got = wait_for_controllers(expected_controllers(), 150)
        if not ok:
            sys.exit(f"controllers not active: {sorted(expected_controllers() - got)}")
        time.sleep(15)
        env = PepperNavEnv(world_name=name, branch=branch, source="sim", randomiser=None, seed=0, room=room)
        model = SAC.load(str(run / "final_model.zip"), env=env)
        base = env.unwrapped if hasattr(env, "unwrapped") else env
        print(f"{run.name}: {branch} policy on {len(routes)} routes")
        for route in routes:
          try:
              obs, _ = env.reset(options={"route": route})
              traj, frames = [], []
              while True:
                  pose = base.ros._pose
                  traj.append([float(v) for v in pose])
                  frames.append(obs["depth"][0].copy())
                  action, _ = model.predict(obs, deterministic=True)
                  obs, _, term, trunc, info = env.step(action)
                  if term or trunc:
                      traj.append([float(v) for v in base.ros._pose])
                      break
              rr = base.route_result(info)
              rec = {"route_id": rr.route_id, "start": route["start"], "goal": route["goal"],
                     "reached_goal": bool(rr.reached_goal), "collisions": int(rr.collisions),
                     "timed_out": bool(rr.timed_out), "path_length_m": float(rr.path_length_m),
                     "optimal_length_m": float(rr.optimal_length_m), "trajectory": traj}
              rollouts.append((rec, frames))
              print(f"  {rr.route_id}: {outcome(rec)[0]:9s} path {rr.path_length_m:4.2f} m "
                    f"(l* {rr.optimal_length_m:4.2f}), {len(traj)} steps", flush=True)
          except Exception:
            # Keep what was recorded: log the failure and draw the routes that finished.
            import traceback
            (out / "error.log").write_text(f"route {route['route_id']}\n" + traceback.format_exc())
            print(f"  {route['route_id']}: ERROR (see {out / 'error.log'}); drawing the "
                  f"{len(rollouts)} finished routes", flush=True)
            break
    finally:
        shutdown(procs)
    if not rollouts:
        sys.exit("no routes finished")

    (out / "rollouts.json").write_text(json.dumps([r for r, _ in rollouts], indent=1))
    np.savez_compressed(out / "frames.npz", **{r["route_id"]: np.array(f) for r, f in rollouts})
    render(out, run.name, branch, room, world, rollouts, args.gif)


def render(out, run_name, branch, room, world, rollouts, gif_id):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    L, W, boxes, scan = room_geometry(room, world)
    # --- all paths
    fig, ax = plt.subplots(figsize=(9, 7))
    draw_room(ax, L, W, boxes, scan)
    counts = {}
    for rec, _ in rollouts:
        label, col = outcome(rec)
        counts[label] = counts.get(label, 0) + 1
        t = np.array(rec["trajectory"])
        ax.plot(t[:, 0], t[:, 1], "-", color=col, lw=1.4, alpha=0.85)
        ax.plot(*rec["start"], "o", color=col, ms=5)
        ax.plot(*rec["goal"], "*", color=col, ms=10, mec="k", mew=0.4)
        ax.text(rec["goal"][0] + 0.05, rec["goal"][1] + 0.05, rec["route_id"], fontsize=6)
    summary = ", ".join(f"{v} {k}" for k, v in sorted(counts.items()))
    ax.set_title(f"{run_name}\n{branch} branch, {len(rollouts)} routes: {summary}\n"
                 "o start   * goal   green reached / red collided / amber timed out", fontsize=9)
    fig.tight_layout(); fig.savefig(out / "paths.png", dpi=130); plt.close(fig)
    print(f"wrote {out / 'paths.png'}")

    # --- routes animated
    if gif_id:
        picks = [p for p in rollouts if p[0]["route_id"] == gif_id]
    else:
        picks = []
        for want in ("reached", "collided"):
            c = [p for p in rollouts if outcome(p[0])[0] == want]
            if c:
                picks.append(max(c, key=lambda p: len(p[0]["trajectory"])))
    for rec, frames in picks:
        animate(out, rec, frames, L, W, boxes, scan)


def animate(out, rec, frames, L, W, boxes, scan):
    import matplotlib.pyplot as plt
    from PIL import Image
    t = np.array(rec["trajectory"])
    images = []
    step = max(1, len(frames) // 120)  # cap the GIF at ~120 frames
    for i in range(0, len(frames), step):
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [1.3, 1]})
        draw_room(a1, L, W, boxes, scan)
        a1.plot(t[: i + 1, 0], t[: i + 1, 1], "-", color="#27c", lw=1.5)
        a1.plot(*rec["goal"], "*", color="#2a7", ms=12, mec="k")
        px, py, pyaw = t[i]
        a1.add_patch(plt.Circle((px, py), 0.24, fc="#27c", alpha=0.35))
        a1.arrow(px, py, 0.3 * math.cos(pyaw), 0.3 * math.sin(pyaw), width=0.02, color="#135")
        a1.set_title(f"{rec['route_id']}  step {i}/{len(frames) - 1}", fontsize=9)
        a2.imshow(frames[i], cmap="viridis_r", vmin=0, vmax=255)
        a2.set_title("depth the policy sees (64x48; bright = near)", fontsize=9); a2.axis("off")
        fig.tight_layout(); fig.canvas.draw()
        images.append(Image.frombytes("RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb()))
        plt.close(fig)
    gif = out / f"{rec['route_id']}.gif"
    images[0].save(gif, save_all=True, append_images=images[1:], duration=200 * step, loop=0)
    print(f"wrote {gif}  ({outcome(rec)[0]}, {len(frames)} steps)")


if __name__ == "__main__":
    main()
