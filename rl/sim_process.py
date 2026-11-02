#!/usr/bin/env python3
"""Bringing the simulator up, waiting for it, and taking it down again.

Shared by `rl/scripts/bench_throughput.py` and `rl/scripts/run.py`, because getting this
wrong is not obvious from the outside. Two details in particular were paid for:

  * A world that is merely *advertising* /world/<w>/stats is not a world that is
    *stepping*. gz_ros2_control blocks in Configure waiting for robot_state_publisher,
    and blocking there blocks the simulation loop, so the topic appears and nothing ever
    publishes on it. `wait_for_sim` therefore requires sim time to advance between two
    stats messages, not just for the topic to exist.
  * `ros2 launch` does not reliably reap the Ignition server on SIGINT, and a survivor
    holds the world name -- so the next run silently attaches to the stale server. Hence
    the process group handling and the pkill sweep in `shutdown`.
"""

import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WS = REPO / "ros2_ws"
WORLD_DIR = WS / "src" / "pepper_room_description" / "worlds"

# `ign gazebo` names the world from the SDF's <world name="...">, not from the filename.
# gen_world.py sets them equal, but read it rather than assume -- a mismatch shows up as a
# benchmark that samples nothing, which looks like a broken simulator.
WORLD_NAME_RE = re.compile(r'<world\s+name="([^"]+)"')


# The topic names the environment subscribes to. Kept here as the single source of truth
# so the launch arguments and rl/env/pepper_nav_env.RosInterface cannot drift apart -- if
# they do, the env waits forever for a topic nobody publishes.
POSE_TOPIC = "/pepper_robot/true_base_tf"
DEPTH_TOPIC = "/pepper_robot/depth_camera/image_raw"
SCAN_TOPIC = "/pepper_robot/scan"
BUMPER_TOPIC = "/pepper_robot/bumper"

# The IGNITION-side name of the base contact sensor is NOT what `<topic>bumper</topic>`
# in ign_base_contact (pepper_robot.gazebo) asks for. Verified against a running sim: the
# Contact system does not honour a custom <topic> the way the rendering sensors
# (depth/lidar/imu, via the Sensors system) do, and publishes on its own auto-generated,
# world-scoped default instead. `bridge_contacts` in bridge.launch.py has to be pointed
# at THIS, per world, or the bridge subscribes to a topic that is silently never
# published and CR (Eq. 4.5) reads zero collisions forever with no error anywhere.
CONTACT_LINK = "Tibia"
CONTACT_SENSOR = "base_contact"


def contact_ign_topic(world_name, robot_name="pepper_robot"):
    return (f"/world/{world_name}/model/{robot_name}/link/{CONTACT_LINK}"
           f"/sensor/{CONTACT_SENSOR}/contact")


def die(msg):
    print("ERROR: " + msg, file=sys.stderr)
    sys.exit(1)


def resolve_world(arg):
    """Accept a bare room name, a world name, or a path."""
    p = Path(arg)
    if p.is_file():
        path = p
    else:
        for cand in (WORLD_DIR / arg, WORLD_DIR / f"{arg}.sdf"):
            if cand.is_file():
                path = cand
                break
        else:
            avail = sorted(x.name for x in WORLD_DIR.glob("*.sdf"))
            die(f"no world {arg!r} in {WORLD_DIR}. Available: {avail or '(none -- run tools/gen_world.py)'}")
    m = WORLD_NAME_RE.search(path.read_text(encoding="utf-8", errors="replace"))
    if not m:
        die(f"{path} has no <world name=...>; it is not a world file")
    return path, m.group(1)


def launch(world_path, depth=True, lidar=True, verbosity="1",
           spawn=None, extra_world_args=(), bridge=True):
    """Start the sim stack headless, plus the ROS bridge. Returns the processes.

    MoveIt and RViz are off deliberately, not for convenience: they would compete for the
    same six cores the software-rendered depth camera needs, and an RTF measured with them
    attached is a property of the benchmark rather than of the simulator. gz.launch.py
    supplies its own robot_state_publisher when move_group is not launching one.
    """
    env = dict(os.environ)
    # Mandatory on WSL2. ogre2 on hardware GL aborts in GL3PlusTextureGpu::copyTo, and
    # ogre v1 is worse -- it runs and silently pins every lidar beam at <min>, which would
    # feed SAC constant, plausible-looking observations and never raise an error.
    env["LIBGL_ALWAYS_SOFTWARE"] = "1"

    cmd = [
        "ros2", "launch", "pepper_robot", "gz.launch.py",
        f"world:={world_path}",
        "headless:=true",
        "enable_moveit:=false",
        "enable_rviz:=false",
        f"sensor_depth_camera:={'true' if depth else 'false'}",
        f"sensor_front_lidar:={'true' if lidar else 'false'}",
        f"ign_verbosity:={verbosity}",
    ]
    if spawn is not None:
        x, y, z, yaw = spawn
        cmd += [f"spawn_x:={x}", f"spawn_y:={y}", f"spawn_z:={z}", f"spawn_yaw:={yaw}"]
    cmd += list(extra_world_args)

    proc = subprocess.Popen(
        cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True,  # so we can kill the whole process group, not just ros2
    )

    bridge_proc = None
    if bridge:
        # The environment consumes ROS 2 topics, not Ignition ones, so the bridge is part
        # of the stack rather than an extra. gz.launch.py bridges /clock only.
        #
        # ros_true_base_tf is remapped away from its `/tf` default on purpose. `/tf` also
        # carries odom_tf and every link transform, and ground truth needs a topic of its
        # own: rl/env/pepper_nav_env.RosInterface.POSE_TOPIC subscribes to exactly this
        # name.
        #
        # ign_contacts is remapped for the reason contact_ign_topic()'s comment gives:
        # the Contact system ignores the <topic> element in the SDF and always uses its
        # own world-scoped default, so the bridge has to be told that name explicitly,
        # per world, or it subscribes to something nobody publishes.
        world_name_match = WORLD_NAME_RE.search(
            Path(world_path).read_text(encoding="utf-8", errors="replace"))
        contacts_arg = []
        if world_name_match:
            contacts_arg = [f"ign_contacts:={contact_ign_topic(world_name_match.group(1))}"]
        bridge_proc = subprocess.Popen(
            ["ros2", "launch", "pepper_robot_ign", "bridge.launch.py",
             "use_sim_time:=true",
             f"ros_true_base_tf:={POSE_TOPIC}"] + contacts_arg,
            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True,
        )

    # Contact sensors -> /pepper_robot/bumper. The system parameter_bridge cannot bridge
    # ros_ign_interfaces/msg/Contacts, so tools/contact_relay.py reads the Ignition topics
    # directly (and drops floor contacts). Own process group, so shutdown() stops it and its
    # `ign topic` readers together.
    relay_proc = None
    if bridge and world_name_match:
        relay_proc = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve().parents[1] / "tools" / "contact_relay.py"),
             "--world", world_name_match.group(1)],
            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    return proc, bridge_proc, relay_proc


def shutdown(procs):
    for proc in procs:
        if proc is None or proc.poll() is not None:
            continue
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGINT)
            proc.wait(timeout=20)
        except Exception:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                pass
    # ros2 launch does not always reap the Ignition server on SIGINT, and a survivor holds
    # the world name -- so the *next* run silently attaches to the stale server and
    # benchmarks it instead. Leaving that to chance would make the comparison meaningless.
    subprocess.run(["pkill", "-9", "-f", "ign gazebo"], capture_output=True)
    subprocess.run(["pkill", "-9", "-f", "parameter_bridge"], capture_output=True)
    subprocess.run(["pkill", "-9", "-f", "ruby.*ign"], capture_output=True)
    time.sleep(2.0)


def ign_topic_exists(topic, timeout=1.0):
    try:
        out = subprocess.run(["ign", "topic", "-l"], capture_output=True, text=True,
                             timeout=timeout + 5).stdout
    except subprocess.TimeoutExpired:
        return False
    return topic in out.split()


def wait_for_sim(world_name, timeout):
    """Block until the world is actually STEPPING, not merely advertising.

    The distinction is the whole point. The server advertises /world/<w>/stats as soon as
    it starts loading, long before it steps, and it can advertise and then never step at
    all: gz_ros2_control blocks in Configure waiting for robot_state_publisher, and
    blocking there blocks the simulation loop. Waiting on `ign topic -l` alone therefore
    returns success against a world that is wedged, and the benchmark then hangs in a
    `ign topic -e` that will never receive anything.

    So this waits for a stats message to arrive and for sim time to have advanced between
    two of them. Both conditions are needed: one message proves the loop has started,
    two with different sim times prove it is still running.
    """
    stats = f"/world/{world_name}/stats"
    deadline = time.time() + timeout
    previous = None
    while time.time() < deadline:
        if not ign_topic_exists(stats):
            time.sleep(2.0)
            continue
        try:
            r = subprocess.run(["ign", "topic", "-e", "-t", stats, "-n", "1"],
                               capture_output=True, text=True, timeout=8)
        except subprocess.TimeoutExpired:
            # Advertised but silent: the loop is wedged, or still loading assets.
            continue
        parsed = parse_stats(r.stdout)
        if parsed is None:
            continue
        if previous is not None and parsed[0] > previous[0]:
            return True
        previous = parsed
        time.sleep(1.0)
    return False


# --------------------------------------------------------------------------------------
# measurement
# --------------------------------------------------------------------------------------


SEC_RE = re.compile(r"^\s*sec:\s*(-?\d+)", re.M)
NSEC_RE = re.compile(r"^\s*nsec:\s*(-?\d+)", re.M)
ITER_RE = re.compile(r"^\s*iterations:\s*(\d+)", re.M)

def parse_stats(text):
    """Pull (sim_time, real_time, iterations) out of one WorldStatistics message.

    The message carries sim_time and real_time as consecutive nested Time blocks, so the
    first two sec/nsec pairs are sim then real. Deriving RTF from these rather than
    reading the `real_time_factor` field is deliberate: that field is an instantaneous
    estimate and jitters hard under software rendering, whereas a ratio taken over the
    whole sampling window is the number a training budget actually depends on.
    """
    secs = SEC_RE.findall(text)
    nsecs = NSEC_RE.findall(text)
    iters = ITER_RE.findall(text)
    if len(secs) < 2 or len(nsecs) < 2 or not iters:
        return None
    sim = int(secs[0]) + int(nsecs[0]) / 1e9
    real = int(secs[1]) + int(nsecs[1]) / 1e9
    return sim, real, int(iters[0])


def sample_stats(world_name, seconds):
    """Take a stats sample at the start and end of the window and difference them.

    Returns (rtf, steps_per_sec, sim_elapsed, real_elapsed).
    """
    topic = f"/world/{world_name}/stats"

    def one():
        r = subprocess.run(["ign", "topic", "-e", "-t", topic, "-n", "1"],
                           capture_output=True, text=True, timeout=30)
        return parse_stats(r.stdout)

    first = one()
    if first is None:
        die(f"could not read {topic} -- is the world actually stepping? "
            f"(gz.launch.py passes `-r`, so it should be)")
    time.sleep(seconds)
    last = one()
    if last is None:
        die(f"{topic} stopped publishing mid-measurement; the server probably died")

    d_sim = last[0] - first[0]
    d_real = last[1] - first[1]
    d_iter = last[2] - first[2]
    if d_real <= 0:
        die("real time did not advance between samples")
    return d_sim / d_real, d_iter / d_real, d_sim, d_real


def ros_topic_exists(topic, timeout=20.0):
    try:
        out = subprocess.run(["ros2", "topic", "list"], capture_output=True, text=True,
                             timeout=timeout).stdout
    except subprocess.TimeoutExpired:
        return False
    return topic in out.split()
