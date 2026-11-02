#!/usr/bin/env python3
"""Make the physical Pepper dance on the spot -- arms, head and torso, no wheels.

A presentation demo, like demo_wave_real.py and demo_shake_real.py, and not part of the
experiment.

**Deliberately no base motion.** Adding `/cmd_vel` would make it look better and would
also mean a moving robot on a floor that has not been cleared, plus the shim's 0.75 s
watchdog to feed. The torso joints give the sway without any of that, so the robot stays
on one spot and the demo is safe to run in a cluttered room.

Torso amplitudes are small on purpose. `HipRoll`, `HipPitch` and `KneePitch` swing a heavy
torso above a wheeled base, and the URDF allows far more travel than is sensible to use
while she is standing unattended:

    HeadYaw    -2.08567  .. 2.08567     HipPitch    -1.03847  .. 1.03847
    HeadPitch  -0.706858 .. 0.637045    KneePitch   -0.514872 .. 0.514872
    HipRoll    -0.514872 .. 0.514872

The head returns to SIM_HEAD_YAW / SIM_HEAD_PITCH (0.0, 0.45) at the end, not to a neutral
zero. Head pitch is the depth camera's extrinsics -- step 4 of verify_pepper.sh asserts it
-- so a demo that left the head somewhere else would quietly invalidate the next depth
measurement taken after it.

Requires naoqi_shim.py in another terminal, and Pepper awake:

    tools/run_shim.sh --robot-ip 192.168.1.9      # terminal 1
    python3 tools/demo_dance_real.py --bars 4     # terminal 2

Joints are left stiff at the end; `tools/pepper_stop.sh` puts her back to rest.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pepper_bridge_node import DEFAULT_HOST, DEFAULT_PORT, ShimClient  # noqa: E402

JOINTS = [
    "HeadYaw", "HeadPitch",
    "HipRoll", "HipPitch", "KneePitch",
    "LShoulderPitch", "LShoulderRoll", "LElbowYaw", "LElbowRoll", "LWristYaw",
    "RShoulderPitch", "RShoulderRoll", "RElbowYaw", "RElbowRoll", "RWristYaw",
]

# The simulated head pose, from naoqi_shim.SIM_HEAD_YAW / SIM_HEAD_PITCH.
SIM_HEAD_YAW = 0.0
SIM_HEAD_PITCH = 0.45

# Neutral stand, from pepper_robot_description/config/initial_joint_positions.yaml.
HOME = [
    SIM_HEAD_YAW, SIM_HEAD_PITCH,
    0.0, 0.0, 0.0,
    1.77, 0.17, -1.22, -0.52, 0.0,
    1.77, -0.17, 1.22, 0.52, 0.0,
]

# Sway. Small, because the torso is heavy and the base is wheeled.
HIP_ROLL = 0.16
HIP_PITCH = 0.10
KNEE_DIP = 0.09
HEAD_TURN = 0.40
HEAD_BOB = 0.12

# Arm poses. "up" is raised and open, "down" is low and tucked -- alternating them across
# the beat is what reads as dancing rather than as two arms doing the same thing.
UP_PITCH, UP_ROLL, UP_ELBOW_YAW, UP_ELBOW_ROLL = -0.55, 0.85, -0.90, -0.95
DOWN_PITCH, DOWN_ROLL, DOWN_ELBOW_YAW, DOWN_ELBOW_ROLL = 0.95, 0.25, -1.30, -0.45

BEAT = 0.50       # seconds per beat
INTRO_TIME = 1.5  # rest -> the first beat pose
OUTRO_TIME = 2.0  # last beat -> rest


def _arm(up, right):
    """One arm's five angles. `right` mirrors the roll and elbow-roll signs."""
    pitch = UP_PITCH if up else DOWN_PITCH
    roll = UP_ROLL if up else DOWN_ROLL
    elbow_yaw = UP_ELBOW_YAW if up else DOWN_ELBOW_YAW
    elbow_roll = UP_ELBOW_ROLL if up else DOWN_ELBOW_ROLL
    if right:
        return [pitch, -roll, -elbow_yaw, -elbow_roll, 0.0]
    return [pitch, roll, elbow_yaw, elbow_roll, 0.0]


def _beat(lean, left_up, dip):
    """One beat: `lean` is -1 or +1, `left_up` picks which arm is raised."""
    return (
        [SIM_HEAD_YAW + HEAD_TURN * lean, SIM_HEAD_PITCH + (HEAD_BOB if dip else -HEAD_BOB)]
        + [HIP_ROLL * lean, HIP_PITCH if dip else -HIP_PITCH, KNEE_DIP if dip else 0.0]
        + _arm(left_up, right=False)
        + _arm(not left_up, right=True)
    )


def build_waypoints(bars=4):
    """Intro, `bars` x four beats, outro back to the rest pose.

    Returns [(positions, duration_seconds), ...] in the order of JOINTS.
    """
    beats = []
    for bar in range(bars):
        # Lean alternates every beat; the raised arm alternates every beat too but out of
        # phase with the dip, so the four beats of a bar never repeat exactly.
        beats.append(_beat(lean=+1, left_up=True, dip=False))
        beats.append(_beat(lean=-1, left_up=False, dip=True))
        beats.append(_beat(lean=-1, left_up=True, dip=False))
        beats.append(_beat(lean=+1, left_up=False, dip=True))

    waypoints = [(beats[0], INTRO_TIME)]
    waypoints += [(pose, BEAT) for pose in beats[1:]]
    waypoints.append((HOME, OUTRO_TIME))
    return waypoints


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--wake", action="store_true",
                        help="send WAKEUP first (takes several seconds)")
    parser.add_argument("--bars", type=int, default=4,
                        help="four beats each (default 4)")
    args = parser.parse_args()

    client = ShimClient(args.host, args.port)
    try:
        if client.request("PING") != "PONG":
            print("ERR shim did not answer PING", file=sys.stderr)
            return 1

        if args.wake:
            print("waking up (this takes a few seconds) ...")
            reply = client.request("WAKEUP")
            if reply != "OK":
                print("ERR WAKEUP failed: %s" % reply, file=sys.stderr)
                return 1

        waypoints = build_waypoints(max(1, args.bars))
        names = ",".join(JOINTS)
        print("dancing: %d bars, %d waypoints" % (max(1, args.bars), len(waypoints)))

        for index, (positions, duration) in enumerate(waypoints, start=1):
            angles = ",".join("%.4f" % p for p in positions)
            reply = client.request("ANGLES %s %s %.3f" % (names, angles, duration))
            if reply != "OK":
                print("ERR waypoint %d rejected: %s" % (index, reply), file=sys.stderr)
                return 1
            print("  %2d/%d  %.2fs" % (index, len(waypoints), duration))

        print("Done. Head is back at the simulated pose; joints left stiff.")
        return 0
    except TimeoutError as exc:
        print("ERR %s" % exc, file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
    finally:
        client.close()


if __name__ == "__main__":
    sys.exit(main())
