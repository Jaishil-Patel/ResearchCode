#!/usr/bin/env python3
"""The one module that talks to Ignition Transport. Everything else uses ROS 2.

    python3 -m rl.env.ign_control --selftest --world room_photogrammetry

Why this module is quarantined
------------------------------
The environment's observations and actions go over ROS 2 topics -- `/pepper_robot/cmd_vel`,
`depth_camera/image_raw`, `/pepper_robot/odom` -- because `tools/pepper_bridge_node.py`
already publishes the *physical* robot onto those same names, with `_to_metres` chosen
specifically so the value domains match. Keep the env on ROS 2 topics and the same env
runs against hardware in Task 6 by changing which node is publishing.

Three things have no ROS 2 equivalent: teleporting a model, resetting the world, and
stepping physics. They are Ignition Transport services. They only ever make sense in
simulation -- you cannot teleport a real Pepper -- so they live here, in a module the
real-robot path never imports.

Why subprocess rather than bindings
-----------------------------------
Ignition Fortress ships no usable Python bindings for ign-transport, so the `ign service`
CLI is the interface. Each call costs roughly 30-60 ms of process spawn. That is why the
env is designed to run free and sample off `/clock` rather than pausing and stepping
every tick: per-step, this overhead would dominate; per-episode-reset, it is noise
against the settling time.
"""

import argparse
import math
import subprocess
import sys
import time
from pathlib import Path


class IgnServiceError(RuntimeError):
    pass


class IgnitionControl:
    """Ignition Transport services for one running world."""

    def __init__(self, world, timeout_ms=3000, verbose=False):
        self.world = world
        self.timeout_ms = timeout_ms
        self.verbose = verbose
        # base model name -> the name it currently lives under (see replace_model)
        self._live_names = {}

    # -- plumbing ----------------------------------------------------------------------

    def _service(self, service, reqtype, req, reptype="ignition.msgs.Boolean"):
        cmd = [
            "ign", "service", "-s", f"/world/{self.world}{service}",
            "--reqtype", reqtype,
            "--reptype", reptype,
            "--timeout", str(self.timeout_ms),
            "--req", req,
        ]
        if self.verbose:
            print("  $ " + " ".join(repr(c) if " " in c else c for c in cmd), file=sys.stderr)
        r = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=self.timeout_ms / 1000.0 + 10)
        out = (r.stdout + r.stderr).strip()
        if r.returncode != 0:
            raise IgnServiceError(f"{service} failed (rc={r.returncode}): {out}")
        # `ign service` exits 0 on a timeout and prints the timeout to stdout, so the
        # return code alone is not evidence that anything happened. This is the whole
        # reason for parsing the reply: a silently-ignored reset would show up much later
        # as an episode that starts in the previous episode's final pose.
        if "timed out" in out.lower():
            raise IgnServiceError(
                f"{service} timed out after {self.timeout_ms} ms. The world is probably "
                f"not named {self.world!r}, or the server is not running. "
                f"Check `ign service -l | grep /world/`."
            )
        if "data: false" in out.replace(" ", " ").lower():
            raise IgnServiceError(f"{service} was rejected by the server: {out}")
        return out

    # -- pose --------------------------------------------------------------------------

    def set_pose(self, name, x, y, z=0.0, yaw=0.0):
        """Teleport a model. Requires the UserCommands system in the world.

        `tools/gen_world.py`'s WORLD_TEMPLATE already includes it ("User Commands
        (transform control)"), so both branch worlds support this with no world change.

        Teleporting does not zero velocity: a model moving when it is moved keeps its
        velocity at the new pose. Episode reset must therefore settle afterwards --
        see `settle()` -- or the first observation of the episode is taken mid-slide.
        """
        qz, qw = math.sin(yaw / 2.0), math.cos(yaw / 2.0)
        req = (f'name: "{name}", '
               f'position: {{x: {x}, y: {y}, z: {z}}}, '
               f'orientation: {{x: 0, y: 0, z: {qz}, w: {qw}}}')
        # Retried on a timeout, with a longer wait each time. A teleport is idempotent, so a
        # repeat is harmless, and a single slow reply must not end a run: on 1 Oct 2026 the
        # photogrammetry smoke test died at its first evaluation reset on one 3 s timeout,
        # with the server alive and memory free -- training plus the software-rendered
        # camera had the server too busy to answer in time. A dead server still fails, after
        # ~3+6+12 s. (create/remove are NOT retried this way: a repeated create duplicates.)
        base = self.timeout_ms
        for attempt, mult in enumerate((1, 2, 4)):
            self.timeout_ms = base * mult
            try:
                return self._service("/set_pose", "ignition.msgs.Pose", req)
            except IgnServiceError as e:
                if "timed out" not in str(e) or mult == 4:
                    raise
                print(f"[ign_control] /set_pose timed out ({base * mult} ms); retrying "
                      f"with {base * mult * 2} ms", file=sys.stderr)
            finally:
                self.timeout_ms = base

    # -- world control -----------------------------------------------------------------

    def pause(self, paused=True):
        return self._service("/control", "ignition.msgs.WorldControl",
                             f"pause: {str(paused).lower()}")

    def unpause(self):
        return self.pause(False)

    def step(self, n=1):
        """Advance exactly n physics steps, leaving the world paused.

        The fallback control path, not the default one. The real robot cannot be stepped,
        so a free-running env trains the policy under the asynchrony it will actually meet
        on hardware. If control-period jitter turns out to be unacceptable, this is the
        documented alternative -- and whichever is used has to be stated in the write-up.
        """
        return self._service("/control", "ignition.msgs.WorldControl",
                             f"pause: true, multi_step: {int(n)}")

    def reset_world(self):
        """Reset simulated time and every model pose to the world file's initial state.

        Heavier than `set_pose` and used only where that is the point: Phase 3 resamples
        mass, mass distribution and friction at world reset, because Fortress has no clean
        runtime API for mutating them.
        """
        return self._service("/control", "ignition.msgs.WorldControl",
                             "reset: {all: true}")

    # -- model lifecycle: remove + create, for what has no runtime mutation API ---------
    #
    # Verified against a running world (see rl/env/ign_control.py --selftest and
    # rl/env/body_randomisation.py): /world/<w>/remove takes ignition.msgs.Entity, and
    # /world/<w>/create takes ignition.msgs.EntityFactory, both replying
    # ignition.msgs.Boolean. Fortress has no runtime service to mutate an existing
    # entity's collision friction, so the room's floor mu (rl/env/body_randomisation.py)
    # goes through here: remove the room model, then create it again from a rewritten
    # copy of its own model.sdf.

    def remove(self, name):
        """Delete a named model from the world. No-op-looking failure mode: removing a
        name that does not exist returns `data: false` rather than raising, which
        `_service` already treats as an error -- so a stale name here surfaces loudly
        instead of silently leaving the old model in place."""
        return self._service("/remove", "ignition.msgs.Entity",
                             f'name: "{name}", type: MODEL')

    def create(self, name, sdf_path, x=0.0, y=0.0, z=0.0, yaw=0.0):
        """Spawn a model from an SDF file at a given pose.

        `sdf_path` must be readable by the Ignition server process, not just by this
        script -- an absolute path is used throughout this project for exactly that
        reason. Relative `<uri>` mesh references inside the SDF resolve against the
        directory `sdf_path` lives in (that is how the checked-in room models already
        work), which is why `body_randomisation.py` rewrites those URIs to absolute paths
        before writing its modified copy anywhere else.
        """
        qz, qw = math.sin(yaw / 2.0), math.cos(yaw / 2.0)
        req = (f'sdf_filename: "{sdf_path}", name: "{name}", '
               f'pose: {{position: {{x: {x}, y: {y}, z: {z}}}, '
               f'orientation: {{x: 0, y: 0, z: {qz}, w: {qw}}}}}')
        return self._service("/create", "ignition.msgs.EntityFactory", req)

    def respawn(self, name, sdf_path, x=0.0, y=0.0, z=0.0, yaw=0.0):
        """Remove a model and create it again from a (presumably modified) SDF file.

        The only mechanism this project has for the world-reset half of dynamics
        randomisation (mass, mass distribution, friction): Fortress exposes no service
        to mutate an existing entity's inertial or surface properties, so changing them
        means respawning the entity from a rewritten description.
        """
        self.remove(name)
        # A removal takes a moment to actually clear the entity from the scene graph.
        # Creating too soon risks a name collision that /create rejects; a short, fixed
        # settle is simpler than polling for the entity's disappearance and this call is
        # already infrequent (world-reset cadence, not per-episode).
        time.sleep(0.5)
        return self.create(name, sdf_path, x=x, y=y, z=z, yaw=yaw)

    def live_name(self, name):
        """The name `name` currently lives under after any replace_model() calls."""
        return self._live_names.get(name, name)

    def replace_model(self, name, sdf_path, x=0.0, y=0.0, z=0.0, yaw=0.0, timeout=90.0):
        """Swap a model for a modified copy WITHOUT a gap: create the copy first, under an
        alternate name, and only then remove the old one.

        respawn() removes first and creates second, with physics running in between. For
        a ROOM model that is a gap with no floor -- the floor is part of the room in both
        branches, deliberately (no shared ground plane) -- and Pepper falls through it:
        measured 27 Sep 2026 on room_standard, z = -76.5 m three seconds after a
        friction respawn, and the next teleport kept that falling velocity. In training it
        also stalled the server until /set_pose timed out. Pausing is no fix: Fortress
        does not process /create while paused. Overlapping the two copies briefly is
        harmless -- their geometry is identical; only the friction differs.

        The two names alternate, `name` <-> `name__b`; live_name() says which is current.

        The /create REPLY is not trusted; the model appearing is. Loading the scanned room
        (a 10 MB visual mesh) under training load took longer than the 3 s service timeout on
        2 Oct 2026, and the IgnServiceError ended the run -- while the server was still
        creating it. So a timed-out reply is treated as "still loading" and the poll below
        decides; a create that never lands still raises, with the old model left in place.
        Not retried: a repeated /create would add a second copy.
        """
        cur = self.live_name(name)
        new = f"{name}__b" if cur == name else name
        base = self.timeout_ms
        self.timeout_ms = max(base, 30000)
        try:
            self.create(new, sdf_path, x=x, y=y, z=z, yaw=yaw)
        except IgnServiceError as e:
            if "timed out" not in str(e):
                raise
            print(f"[ign_control] /create {new}: no reply in {self.timeout_ms} ms; waiting for it "
                  f"to appear (up to {timeout:.0f} s)", file=sys.stderr)
        finally:
            self.timeout_ms = base
        deadline = time.time() + timeout
        while self.pose_of(new) is None:
            if time.time() > deadline:
                raise IgnServiceError(f"{new} did not appear within {timeout}s of /create; "
                                      f"{cur} left in place")
            time.sleep(0.2)
        # The new copy is live from here, whatever happens to the old one.
        self._live_names[name] = new
        # /remove gets the same treatment as /create: under training load it too can outlast
        # the 3 s reply timeout (7 Oct 2026, a cluster run died on exactly that), so a
        # timed-out reply means "poll for the old copy to vanish", not "failed". Not retried:
        # removing a name that is already gone returns false and would raise.
        self.timeout_ms = max(base, 30000)
        try:
            self.remove(cur)
        except IgnServiceError as e:
            if "timed out" not in str(e):
                raise
            print(f"[ign_control] /remove {cur}: no reply in {self.timeout_ms} ms; waiting for it "
                  f"to disappear (up to {timeout:.0f} s)", file=sys.stderr)
        finally:
            self.timeout_ms = base
        deadline = time.time() + timeout
        while self.pose_of(cur) is not None:
            if time.time() > deadline:
                raise IgnServiceError(f"{cur} still present {timeout}s after /remove")
            time.sleep(0.2)
        return new

    # -- observation of the sim's own state --------------------------------------------

    def pose_of(self, name, timeout=5.0):
        """Ground-truth pose of a model, as (x, y, yaw), read off /world/<w>/pose/info.

        This is the *simulator's* truth, used to verify that a teleport landed. It is not
        what simulated evaluation reads: that is `true_base_tf`.
        """
        try:
            r = subprocess.run(
                ["ign", "topic", "-e", "-t", f"/world/{self.world}/pose/info", "-n", "1"],
                capture_output=True, text=True, timeout=timeout + 10)
        except subprocess.TimeoutExpired:
            return None
        return _parse_pose_for(r.stdout, name)

    def height_of(self, name, timeout=5.0):
        """z of a model's origin. pose_of() drops z on purpose (navigation is planar), which
        is exactly why a robot falling out of the world went unnoticed by the self-tests."""
        try:
            r = subprocess.run(
                ["ign", "topic", "-e", "-t", f"/world/{self.world}/pose/info", "-n", "1"],
                capture_output=True, text=True, timeout=timeout + 10)
        except subprocess.TimeoutExpired:
            return None
        import re
        for b in r.stdout.split("pose {"):
            m = re.search(r'name:\s*"([^"]+)"', b)
            if m and m.group(1) == name:
                pos = re.search(r"position\s*\{([^}]*)\}", b)
                mz = re.search(r"z:\s*(-?[\d.eE+]+)", pos.group(1)) if pos else None
                return float(mz.group(1)) if mz else 0.0
        return None

    def settle(self, seconds=0.5):
        """Let the physics absorb a teleport before anything reads the state."""
        time.sleep(seconds)


# --------------------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------------------

def _parse_pose_for(text, name):
    """Find `name`'s pose in an ignition.msgs.Pose_V text dump.

    The dump is a flat sequence of `pose { name: ... position { } orientation { } }`
    blocks, one per model *and* per nested link, so a naive regex for the first `position`
    after the name can pick up a child link. Blocks are split on the top-level `pose {`
    marker instead.
    """
    import re
    blocks = text.split("pose {")
    for b in blocks:
        m = re.search(r'name:\s*"([^"]+)"', b)
        if not m or m.group(1) != name:
            continue
        pos = re.search(r"position\s*\{([^}]*)\}", b)
        ori = re.search(r"orientation\s*\{([^}]*)\}", b)
        if not pos or not ori:
            continue

        def field(blob, key):
            mm = re.search(rf"{key}:\s*(-?[\d.eE+]+)", blob)
            return float(mm.group(1)) if mm else 0.0

        x, y = field(pos.group(1), "x"), field(pos.group(1), "y")
        qz, qw = field(ori.group(1), "z"), field(ori.group(1), "w")
        return x, y, 2.0 * math.atan2(qz, qw)
    return None


# --------------------------------------------------------------------------------------
# self-test
# --------------------------------------------------------------------------------------

def selftest(world, model, verbose):
    """Prove a teleport actually lands, against a world that is already running.

    Start the sim first, in another terminal:

        export LIBGL_ALWAYS_SOFTWARE=1
        ros2 launch pepper_robot gz.launch.py world:=<path> headless:=true \
             enable_moveit:=false enable_rviz:=false

    A teleport that silently does nothing is the failure this is for. Every episode would
    then start from the previous episode's final pose, training would still run, the
    reward curve would still look plausible, and the routes would be wrong.
    """
    ign = IgnitionControl(world, verbose=verbose)
    failures = []

    def check(label, ok, detail=""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}{'  ' + detail if detail else ''}")
        if not ok:
            failures.append(label)

    print(f"self-test against world {world!r}, model {model!r}")

    start = ign.pose_of(model)
    check("model is present and publishing a pose", start is not None,
          f"at ({start[0]:.2f}, {start[1]:.2f}, yaw {start[2]:.2f})" if start else
          "-- is the robot spawned? is the world name right?")
    if start is None:
        return 1

    for (dx, dy, yaw) in [(1.0, 0.5, 0.0), (-0.5, 1.5, math.pi / 2), (0.0, 0.0, -math.pi / 2)]:
        tx, ty = start[0] + dx, start[1] + dy
        ign.set_pose(model, tx, ty, z=start_z((tx, ty), world), yaw=yaw)
        ign.settle(1.0)
        got = ign.pose_of(model)
        if got is None:
            check(f"teleport to ({tx:.2f}, {ty:.2f})", False, "-- pose vanished")
            continue
        derr = math.hypot(got[0] - tx, got[1] - ty)
        yerr = abs(math.atan2(math.sin(got[2] - yaw), math.cos(got[2] - yaw)))
        # 5 cm / 5 deg: generous enough for the settle to have nudged her on uneven
        # reconstructed floor, tight enough that "the teleport was ignored" cannot pass.
        check(f"teleport to ({tx:.2f}, {ty:.2f}, yaw {yaw:.2f})",
              derr < 0.05 and yerr < 0.09,
              f"landed {derr * 100:.1f} cm / {math.degrees(yerr):.1f} deg off")

    ign.set_pose(model, start[0], start[1], z=start_z(start, world), yaw=start[2])
    ign.settle(1.0)
    print()
    if failures:
        print(f"{len(failures)} check(s) FAILED: {failures}")
        return 1
    print("all checks passed -- teleport-based episode reset is usable")
    return 0


_FLOORS = {}


def start_z(pose, world=None):
    """Teleport height: the local floor under (x, y) plus a small drop -- the same rule
    PepperNavEnv._place uses (rl/env/floor_height.py). It used to return 0.0, on the
    assumption that a floor-aligned room has its floor at 0; a reconstructed floor does
    not, and a teleport to 0 embedded the base in it."""
    if world is None:
        return 0.0
    if world not in _FLOORS:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from rl.env.floor_height import FloorHeight
        _FLOORS[world] = FloorHeight(world)
    return _FLOORS[world].spawn_z(pose[0], pose[1])


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--world", default="room_photogrammetry")
    ap.add_argument("--model", default="pepper_robot")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    if not args.selftest:
        ap.error("nothing to do; pass --selftest (this module is a library otherwise)")
    sys.exit(selftest(args.world, args.model, args.verbose))


if __name__ == "__main__":
    main()
