#!/usr/bin/env python3
"""Prove the environment is sane before spending days of compute on it.

    python3 -m rl.scripts.check_env --branch photogrammetry            # needs a running sim
    python3 -m rl.scripts.check_env --offline                          # no sim, no ROS

Two modes, because they catch different things.

`--offline` replaces the ROS interface with a stub and exercises the MDP itself: space
conformance, observation dtypes and bounds, the reward's sign and shape, the
terminated/truncated distinction, and the episode bookkeeping the metrics consume. It
needs neither Ignition nor rclpy, runs in a second, and is the one to run after touching
the reward.

The online mode runs Gymnasium's own API conformance checker and then 100 random-action
episodes against the real simulator. That is where the failures that only appear under a
real robot show up: controller faults after a teleport, NaN observations, resets that
never settle, and /clock stalling.

A policy trained against a quietly broken env still produces a smooth reward curve. It is
the curve of the env's bug.
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


# --------------------------------------------------------------------------------------
# offline stub
# --------------------------------------------------------------------------------------

class StubRos:
    """A fake robot that integrates the commands it is given.

    Deliberately not a mock that records calls: the MDP's interesting behaviour is how
    reward and termination respond to the robot actually moving, and a stub that never
    moves would pass a reward test that a real robot fails.
    """

    def __init__(self, depth_shape=(240, 320), start=(0.0, 0.0, 0.0), dt=0.2):
        self.h, self.w = depth_shape
        self._pose = start
        self._depth = np.full(depth_shape, 3.0, dtype=np.float32)
        self._scan_min = 5.0
        self._contact = False
        self._t = 0.0
        self.dt = dt
        self.last_cmd = (0.0, 0.0)

    def spin(self, seconds):
        """Time flows while spinning, and the last command keeps being executed.

        This is the shape of the real system, and getting it wrong matters: an earlier
        version advanced the clock only when a command was published, so `step()` -- which
        publishes once and then waits out the control period -- waited forever. A stub
        whose clock does not flow does not exercise the control loop at all.
        """
        dt = max(float(seconds), 0.0)
        if dt <= 0.0:
            return
        v, w = self.last_cmd
        x, y, yaw = self._pose
        yaw = math.atan2(math.sin(yaw + w * dt), math.cos(yaw + w * dt))
        self._pose = (x + v * math.cos(yaw) * dt, y + v * math.sin(yaw) * dt, yaw)
        self._t += dt

    def now(self):
        return self._t

    def publish(self, v, w):
        # Recorded, not integrated. A Twist sets a persistent velocity; the motion happens
        # as time passes, which is `spin`'s job. tools/naoqi_shim.py's watchdog exists
        # because the real ALMotion.move() behaves exactly this way.
        self.last_cmd = (v, w)

    def stop(self):
        self.last_cmd = (0.0, 0.0)

    def wait_for_data(self, timeout=30.0):
        return True

    def close(self):
        pass


def make_offline_env(**kw):
    """Build a PepperNavEnv with the stub in place of ROS and no Ignition."""
    from rl.env.pepper_nav_env import PepperNavEnv
    from rl.env.randomisation import CommandDelay
    env = PepperNavEnv.__new__(PepperNavEnv)
    # Construct by hand rather than calling __init__: __init__ opens rclpy and an
    # Ignition connection, and the point of offline mode is to need neither.
    from rl.env.pepper_nav_env import load_config
    from gymnasium import spaces

    cfg = kw.pop("config", None) or load_config()
    env.cfg = cfg
    env.branch, env.source = "offline", "offline"
    env.world_name = "stub"
    env.routes = None
    env.randomiser = kw.pop("randomiser", None)
    env.spawn_z = 0.0
    env.room = kw.pop("room", None)
    env.rng = np.random.default_rng(kw.pop("seed", 0))

    t, a, o = cfg["task"], cfg["action"], cfg["observation"]
    env.goal_tol = float(t["goal_tolerance_m"])
    env.max_duration = float(t["max_duration_s"])
    env.control_dt = 1.0 / float(t["control_hz"])
    env.v_max, env.v_min = float(a["v_max_mps"]), float(a["v_min_mps"])
    env.w_max = float(a["omega_max_rps"])
    env.dw, env.dh = int(o["depth_width"]), int(o["depth_height"])
    env.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
    env.n_hist = int(o.get("action_history", 0))
    env._act_hist = np.zeros((env.n_hist, 2), dtype=np.float32)
    env.observation_space = env._make_observation_space()
    env.ros = StubRos(dt=env.control_dt)
    env._ign = None
    env._goal = None
    env._t0 = None
    env._prev_dist = None
    env._path_length = 0.0
    env._collisions = 0
    env._last_xy = None
    env._route_id = None
    env._optimal_length = None
    env._jitter = []
    env._episode_count = 0
    env._last_body_params = None
    # Added to __init__ after this stub was written (latency 2 Oct, stale frames 7 Oct);
    # source "offline" is not "sim", so both are inert here, as on hardware.
    env.sim_delay_steps = 0
    env._delay = CommandDelay(0)
    env._held_depth = None
    return env


# --------------------------------------------------------------------------------------
# offline checks: the MDP itself
# --------------------------------------------------------------------------------------

def offline_checks():
    fails = []

    def check(label, ok, detail=""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}{('  ' + detail) if detail else ''}")
        if not ok:
            fails.append(label)

    env = make_offline_env(seed=0)
    route = {"route_id": "t000", "start": [0.0, 0.0], "start_yaw": 0.0,
             "goal": [3.0, 0.0], "optimal_length_m": 3.0}

    obs, info = env.reset(options={"route": route})

    # -- spaces -------------------------------------------------------------------------
    check("reset() returns an observation inside observation_space",
          env.observation_space.contains(obs))
    # uint8 is what makes SB3 treat the depth channel as an image and reach for a CNN.
    # A float32 Box is silently flattened into an MLP input instead, with no warning
    # anywhere, so this check is guarding against an invisible regression.
    check("depth is uint8, so SB3 sees an image and uses a CNN",
          obs["depth"].dtype == np.uint8,
          f"dtype {obs['depth'].dtype}, range [{obs['depth'].min()}, {obs['depth'].max()}]")

    from stable_baselines3.common.preprocessing import is_image_space
    check("SB3 agrees the depth space is an image",
          is_image_space(env.observation_space["depth"]))

    # 3.0 m in a 0.4-8.0 m window is (3.0-0.4)/7.6 = 0.342 -> 87 after quantisation.
    check("metres quantise to the expected code value",
          int(obs["depth"].max()) == 87, f"got {int(obs['depth'].max())}, expected 87")
    check("depth is downsampled to the configured shape",
          obs["depth"].shape == (1, env.dh, env.dw), str(obs["depth"].shape))
    check("goal vector is (range, sin bearing, cos bearing)",
          abs(obs["goal"][0] - 3.0) < 1e-4 and abs(obs["goal"][1]) < 1e-4
          and abs(obs["goal"][2] - 1.0) < 1e-4, str(obs["goal"]))

    # -- action history: the commands still in flight under the 0.8 s latency ------------
    check("action history starts empty (zeros) at reset",
          obs["action_hist"].shape == (2 * env.n_hist,) and not obs["action_hist"].any(),
          str(obs["action_hist"]))
    # A far goal, so the episode cannot end inside the window being checked.
    env.reset(options={"route": {**route, "goal": [40.0, 0.0]}})
    n = env.n_hist + 2
    for k in range(n):
        obs_h, _r, term, trunc, _i = env.step(
            np.array([0.1 * (k + 1) - 0.5, -0.05 * (k + 1)], dtype=np.float32))
    sent = [[0.1 * (k + 1) - 0.5, -0.05 * (k + 1)] for k in range(n)][-env.n_hist:]
    check("action history holds exactly the last n actions, oldest first",
          not (term or trunc)
          and np.allclose(obs_h["action_hist"].reshape(-1, 2), sent, atol=1e-6),
          f"{np.round(obs_h['action_hist'].reshape(-1, 2), 2).tolist()}")
    obs_r, _ = env.reset(options={"route": route})
    check("reset clears the action history", not obs_r["action_hist"].any())

    # -- bearing has no discontinuity at +-pi ------------------------------------------
    env2 = make_offline_env(seed=0)
    env2.reset(options={"route": {**route, "goal": [-3.0, 0.0]}})
    env2.ros._pose = (0.0, 0.0, math.pi - 1e-6)
    a = env2._goal_obs(env2.ros._pose)
    env2.ros._pose = (0.0, 0.0, -math.pi + 1e-6)
    b = env2._goal_obs(env2.ros._pose)
    check("sin/cos bearing is continuous across +-pi",
          np.allclose(a, b, atol=1e-3), f"{a} vs {b}")

    # -- action scaling ------------------------------------------------------------------
    env.reset(options={"route": route})
    env.step(np.array([1.0, 0.0], dtype=np.float32))
    check("action +1 maps to v_max", abs(env.ros.last_cmd[0] - env.v_max) < 1e-6,
          f"{env.ros.last_cmd[0]:.3f} vs {env.v_max}")
    env.step(np.array([-1.0, -1.0], dtype=np.float32))
    check("action -1 maps to v_min and -omega_max",
          abs(env.ros.last_cmd[0] - env.v_min) < 1e-6
          and abs(env.ros.last_cmd[1] + env.w_max) < 1e-6, str(env.ros.last_cmd))

    # -- reward ----------------------------------------------------------------------------
    env.reset(options={"route": route})
    _, r_fwd, _, _, _ = env.step(np.array([1.0, 0.0], dtype=np.float32))
    env.reset(options={"route": route})
    _, r_back, _, _, _ = env.step(np.array([-1.0, 0.0], dtype=np.float32))
    check("driving toward the goal beats driving away from it", r_fwd > r_back,
          f"forward {r_fwd:.2f} vs reverse {r_back:.2f}")

    env.reset(options={"route": route})
    _, r_straight, _, _, _ = env.step(np.array([0.0, 0.0], dtype=np.float32))
    env.reset(options={"route": route})
    _, r_spin, _, _, _ = env.step(np.array([0.0, 1.0], dtype=np.float32))
    check("spinning on the spot is penalised relative to not spinning",
          r_spin < r_straight, f"spin {r_spin:.3f} vs still {r_straight:.3f}")

    # -- termination vs truncation ---------------------------------------------------------
    env.reset(options={"route": {**route, "goal": [0.2, 0.0]}})
    _, r, term, trunc, info = env.step(np.array([0.0, 0.0], dtype=np.float32))
    check("reaching the goal terminates (not truncates)", term and not trunc)
    check("reaching the goal pays the terminal reward",
          r > env.cfg["reward"]["goal_reached"] * 0.5, f"reward {r:.1f}")
    check("a successful episode is flagged is_success", info.get("is_success") is True)

    env.reset(options={"route": route})
    env.max_duration = env.control_dt * 2.5
    for _ in range(5):
        _, _, term, trunc, info = env.step(np.array([0.0, 0.0], dtype=np.float32))
        if term or trunc:
            break
    check("running out of time truncates, and does NOT terminate", trunc and not term,
          f"terminated={term} truncated={trunc}")
    check("a timed-out episode is not flagged is_success",
          info.get("is_success") is False)

    # -- collisions ------------------------------------------------------------------------
    env.reset(options={"route": route})
    env.cfg["collision"]["source"] = "proximity"
    env.ros._scan_min = 0.05
    _, r_hit, term, _, info = env.step(np.array([1.0, 0.0], dtype=np.float32))
    check("a collision terminates the episode", term)
    check("a collision is penalised", r_hit < env.cfg["reward"]["collision"] * 0.5,
          f"reward {r_hit:.1f}")
    check("a collision is counted for CR", info["collisions"] == 1)
    check("a collided episode is not flagged is_success",
          info.get("is_success") is False)
    env.cfg["collision"]["source"] = "contact"
    env.ros._scan_min = 5.0

    # -- bookkeeping the metrics consume ------------------------------------------------
    env.reset(options={"route": route})
    for _ in range(6):
        env.step(np.array([1.0, 0.0], dtype=np.float32))
    travelled = env._path_length
    expected = 6 * env.v_max * env.control_dt
    check("path length accumulates what the robot actually travelled",
          abs(travelled - expected) < 0.05, f"{travelled:.3f} m vs {expected:.3f} m")

    _, _, term, trunc, info = env.step(np.array([1.0, 0.0], dtype=np.float32))
    rr = env.route_result(info)
    check("route_result carries l* through to the metrics",
          abs(rr.optimal_length_m - 3.0) < 1e-6)
    check("route_result keeps the route id", rr.route_id == "t000")

    # -- randomisation integration ---------------------------------------------------------
    from rl.env.randomisation import Randomiser
    envr = make_offline_env(seed=1, randomiser=Randomiser())
    obs, _ = envr.reset(options={"route": route})
    check("the env still produces a valid observation with randomisation on",
          envr.observation_space.contains(obs))
    p1 = dict(envr.randomiser.params)
    for _ in range(5):
        envr.step(np.array([0.5, 0.1], dtype=np.float32))
    check("randomisation parameters survive a whole episode unchanged",
          envr.randomiser.params == p1)
    envr.reset(options={"route": route})
    check("randomisation resamples on reset", envr.randomiser.params != p1)

    print()
    if fails:
        print(f"{len(fails)} FAILED: {fails}")
        return 1
    print("all offline environment checks passed")
    return 0


# --------------------------------------------------------------------------------------
# online checks: the real simulator
# --------------------------------------------------------------------------------------

def online_checks(branch, world, episodes, max_steps, room=None):
    """Gymnasium conformance, then random-action episodes against the running sim.

    Random actions rather than a policy on purpose: a policy avoids the states that break
    an env, and those are exactly the states worth visiting before a training run does.
    """
    from gymnasium.utils.env_checker import check_env as gym_check
    from rl.env.pepper_nav_env import PepperNavEnv
    from rl.env.randomisation import Randomiser

    if branch == "standard" and not room:
        sys.exit(
            "ERROR: --branch standard needs --room (the training-time free-space "
            "sampler has no mesh to fall back on; it reads simulation/photogrammetry/"
            "ground_truth/<room>_spec.yaml)")
    env = PepperNavEnv(world_name=world, branch=branch, source="sim",
                       randomiser=Randomiser(), seed=0, room=room)
    fails = []

    def check(label, ok, detail=""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}{('  ' + detail) if detail else ''}")
        if not ok:
            fails.append(label)

    try:
        gym_check(env, skip_render_check=True)
        check("Gymnasium API conformance", True)
    except Exception as e:
        check("Gymnasium API conformance", False, f"{type(e).__name__}: {e}")

    rng = np.random.default_rng(0)
    stuck_resets, nan_obs, jitters, lengths = 0, 0, [], []
    for ep in range(episodes):
        try:
            obs, _ = env.reset()
        except Exception as e:
            stuck_resets += 1
            print(f"    episode {ep}: reset failed: {type(e).__name__}: {e}")
            continue
        for _ in range(max_steps):
            obs, r, term, trunc, info = env.step(rng.uniform(-1, 1, size=2).astype(np.float32))
            if not np.all(np.isfinite(obs["depth"])) or not np.all(np.isfinite(obs["goal"])):
                nan_obs += 1
                break
            if term or trunc:
                break
        jitters.append(info["jitter_s"])
        lengths.append(info["path_length_m"])
        print(f"    episode {ep + 1}/{episodes}: {info['duration_s']:5.1f} s sim, "
              f"{info['path_length_m']:5.2f} m, {info['collisions']} collisions, "
              f"jitter {info['jitter_s'] * 1000:+.0f} ms")

    check(f"{episodes} random-action episodes ran without a stuck reset",
          stuck_resets == 0, f"{stuck_resets} failed")
    check("no non-finite values ever reached the observation", nan_obs == 0,
          f"{nan_obs} episodes produced them")
    check("the robot actually moved", bool(lengths) and max(lengths) > 0.1,
          f"furthest episode travelled {max(lengths):.2f} m" if lengths else "no episodes")
    if jitters:
        mj = float(np.mean(jitters))
        # Free-running control has jitter by construction. It only matters if it is a
        # large fraction of the control period -- at which point ign_control.step() is
        # the documented fallback.
        check("control-period jitter is a small fraction of the period",
              abs(mj) < 0.25 * env.control_dt,
              f"mean {mj * 1000:+.0f} ms against a {env.control_dt * 1000:.0f} ms period")
    env.close()

    print()
    if fails:
        print(f"{len(fails)} FAILED: {fails}")
        return 1
    print("all online environment checks passed")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--offline", action="store_true",
                    help="check the MDP with a stub robot; needs no sim and no ROS")
    ap.add_argument("--branch", default="photogrammetry",
                    choices=["photogrammetry", "standard"])
    ap.add_argument("--world", default="room_photogrammetry")
    ap.add_argument("--room", default=None,
                    help="physical-room slug; required for --branch standard")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--max-steps", type=int, default=200)
    args = ap.parse_args()

    if args.offline:
        sys.exit(offline_checks())
    sys.exit(online_checks(args.branch, args.world, args.episodes, args.max_steps,
                           room=args.room))


if __name__ == "__main__":
    main()
