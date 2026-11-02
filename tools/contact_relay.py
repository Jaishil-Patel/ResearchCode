#!/usr/bin/env python3
"""Relay Pepper's Ignition contact sensors to ROS 2, as /pepper_robot/bumper.

Why this exists: the system parameter_bridge cannot bridge ros_ign_interfaces/msg/Contacts
("Failed to create a bridge ... with ROS2 type [ros_ign_interfaces/msg/Contacts]"), so until
now `collision.source: contact` was unusable and collisions came from a front-lidar proxy that
cannot see side or rear contact. The sensors themselves work; this reads
their Ignition topics directly (`ign topic -e`, one reader per sensor) and republishes each
contact on the ROS topic PepperNavEnv already subscribes to.

Floor contacts are dropped, because the floor is not an obstacle:
  * by name, where the floor is its own collision (the standard room's `floor_collision`);
  * by position everywhere else -- a reconstructed room is ONE mesh, walls and floor together,
    and the Contact system reports positions but no normals. A contact counts as floor when
    every contact point is within FLOOR_BAND_M of Pepper's own base height (her base origin
    sits on the floor). NOT the floor ray-cast under the contact point: on a scanned wall,
    which is not perfectly vertical, that ray hits the wall itself just below the contact and
    every wall touch looked like floor (found 30 Sep 2026). The lowest part of the robot that
    can touch a wall, the rim of the base shell, meets it about 4.5 cm up.

The `ign topic -e` readers run under `stdbuf -oL`: piped, ign block-buffers its output, so the
last messages of a contact sat in the buffer until the next contact -- a stale bump that would
have ended the NEXT episode.

usage: contact_relay.py --world room_photogrammetry [--verbose]
"""
import argparse
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# A contact is floor when it is within this height of the floor under Pepper's own base.
# Her base origin sits on the floor; the lowest wall contact (base-shell rim) is ~4.5 cm up;
# 3 cm allows for a contact point 0.24 m out on the steepest (~6 deg) part of the scan.
FLOOR_BAND_M = 0.03
TOPIC_RE = re.compile(r"^/world/[^/]+/model/pepper_robot/link/[^/]+/sensor/[^/]+/contact$")
NUM = r"(-?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?)"
# ign_base_contact in pepper_robot.gazebo: base, torso, both biceps/forearms/hands (8), plus
# hip, pelvis, neck, head, both shoulders and elbows (8 more, 1 Oct 2026). Waiting for all of
# them matters: they are advertised one by one as the model loads, and returning at the first
# eight would silently leave the rest unrelayed.
EXPECTED_SENSORS = 16


def discover(world, timeout=120.0):
    """The sensor topics, once the robot has spawned. All of them, or whatever appeared."""
    t0, seen = time.time(), []
    while time.time() - t0 < timeout:
        out = subprocess.run(["ign", "topic", "-l"], capture_output=True, text=True).stdout
        seen = sorted(t for t in out.split() if TOPIC_RE.match(t) and f"/world/{world}/" in t)
        if len(seen) >= EXPECTED_SENSORS:
            return seen
        time.sleep(2.0)
    return seen


def parse_contacts(block):
    """-> list of (collision1, collision2, [(x, y, z), ...]) from one ign text message."""
    out = []
    for c in block.split("contact {")[1:]:
        n1 = re.search(r'collision1\s*\{[^}]*?name:\s*"([^"]+)"', c, re.S)
        n2 = re.search(r'collision2\s*\{[^}]*?name:\s*"([^"]+)"', c, re.S)
        pts = []
        for p in re.findall(r"position\s*\{([^}]*)\}", c):
            g = lambda k: float((re.search(rf"\b{k}:\s*{NUM}", p) or [None, "0"])[1])
            pts.append((g("x"), g("y"), g("z")))
        if n1 and n2:
            out.append((n1.group(1), n2.group(1), pts))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--world", required=True)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    import rclpy
    # ros_gz_interfaces, NOT ros_ign_interfaces: on Humble the ros_ign_interfaces Contacts is a
    # deprecated shim wrapping ros_gz_interfaces/Contact, and that mixed message publishes but
    # no subscriber can deserialise it ("Unable to convert call argument to Python object",
    # even for `ros2 topic echo`). Found 30 Sep 2026. PepperNavEnv subscribes with the same type.
    from ros_gz_interfaces.msg import Contact, Contacts
    from rl.sim_process import BUMPER_TOPIC

    from tf2_msgs.msg import TFMessage
    from rl.sim_process import POSE_TOPIC
    rclpy.init()
    node = rclpy.create_node("contact_relay")
    pub = node.create_publisher(Contacts, BUMPER_TOPIC, 10)
    log = node.get_logger()

    topics = discover(args.world)
    if not topics:
        log.error(f"no Pepper contact sensor topics in world {args.world!r}; nothing to relay")
        sys.exit(1)
    log.info(f"relaying {len(topics)} contact sensors -> {BUMPER_TOPIC}")
    counts = {"relayed": 0, "floor": 0}
    base = {"z": None}

    def on_pose(msg):
        for tr in msg.transforms:
            base["z"] = tr.transform.translation.z
    node.create_subscription(TFMessage, POSE_TOPIC, on_pose, 10)
    lock = threading.Lock()

    def is_floor(c2, pts):
        if "floor_collision" in c2:
            return True
        z0 = base["z"]
        if z0 is None or not pts:
            return False          # no pose yet: never silently drop a possible bump
        return all(z - z0 < FLOOR_BAND_M for _, _, z in pts)

    def reader(topic):
        proc = subprocess.Popen(["stdbuf", "-oL", "ign", "topic", "-e", "-t", topic], stdout=subprocess.PIPE,
                                text=True, bufsize=1)
        buf = []
        for line in proc.stdout:
            if line.strip() == "" and buf:        # messages are separated by a blank line
                block, buf = "".join(buf), []
                hits = [(a, b, p) for a, b, p in parse_contacts(block) if not is_floor(b, p)]
                with lock:
                    counts["floor"] += len(parse_contacts(block)) - len(hits)
                if not hits:
                    continue
                msg = Contacts()
                msg.header.stamp = node.get_clock().now().to_msg()
                for a, b, _ in hits:
                    c = Contact()
                    c.collision1.name, c.collision2.name = a, b
                    msg.contacts.append(c)
                try:
                    pub.publish(msg)
                except Exception as e:  # never let a reader thread die silently
                    log.error(f"publish failed: {type(e).__name__}: {e}")
                    continue
                with lock:
                    counts["relayed"] += 1
                if args.verbose:
                    log.info(f"contact: {hits[0][0]} <-> {hits[0][1]}")
            else:
                buf.append(line)

    for t in topics:
        threading.Thread(target=reader, args=(t,), daemon=True).start()
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=1.0)
    except KeyboardInterrupt:
        pass
    finally:
        log.info(f"relayed {counts['relayed']} contact messages; dropped {counts['floor']} floor contacts")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
