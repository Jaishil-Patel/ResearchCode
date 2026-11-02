#!/usr/bin/env python3
"""Replay tools/demo_wave.py's wave on the *physical* Pepper, through NAOqi.

The counterpart to demo_wave.py, and like it a presentation demo rather than part of the
experiment -- the navigation work uses the wheels, not the arms. Its value is that it puts
the same motion side by side in simulation and on hardware, which is the clearest way to
show on video that the two branches really are the same robot.

It imports `JOINTS` and `build_waypoints()` from demo_wave.py rather than restating them.
That is possible because NAOqi and the URDF use identical joint names -- `LShoulderPitch`,
`LElbowRoll` and the rest -- so a waypoint written for the simulated controller is already
a valid NAOqi target. If the demo is ever retimed or repose, both robots change together.

Requires naoqi_shim.py in another terminal. No ROS needed:

    python3 tools/naoqi_shim.py --fake                      # rehearsal
    python2 tools/naoqi_shim.py --robot-ip 192.168.1.42     # the real thing
    python3 tools/demo_wave_real.py

Pepper must already be awake (`verify_pepper.sh` does this, or --wake below). The arm is
stiffened by the shim on each command and left stiff at the end; `tools/pepper_stop.sh`
puts her back to rest.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from demo_wave import JOINTS, build_waypoints  # noqa: E402
from pepper_bridge_node import DEFAULT_HOST, DEFAULT_PORT, ShimClient  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--wake", action="store_true",
                        help="send WAKEUP first (takes several seconds)")
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

        waypoints = build_waypoints()
        names = ",".join(JOINTS)
        print("replaying %d waypoints from demo_wave.py" % len(waypoints))

        # build_waypoints() gives cumulative times, but the shim interpolates over a
        # duration, so each leg is the gap since the previous waypoint. ANGLES blocks
        # until the motion finishes, so no local sleeping is needed and the timing stays
        # driven by the robot rather than by this process.
        previous = 0.0
        for index, (positions, elapsed) in enumerate(waypoints, start=1):
            duration = elapsed - previous
            previous = elapsed
            angles = ",".join("%.4f" % p for p in positions)
            reply = client.request("ANGLES %s %s %.3f" % (names, angles, duration))
            if reply != "OK":
                print("ERR waypoint %d rejected: %s" % (index, reply), file=sys.stderr)
                return 1
            print("  %2d/%d  %.2fs" % (index, len(waypoints), duration))

        print("Done. Arm is left stiff -- run tools/pepper_stop.sh to rest her.")
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
