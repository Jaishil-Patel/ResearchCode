"""Drive the real Pepper to the next evaluation start, so the operator does not carry her.

Between scored routes only. The operator still squares her up on the taped cross by hand
and presses Enter; that Enter re-anchors the odometry pose (RosInterface.anchor_odometry),
so nothing here touches what is measured. This only saves lifting a 28 kg robot 40 times.

How it moves, and why that way:
  * The path is planned on the CURRENT layout's room spec (normal or robustness), inflated
    to 0.32 m, and simplified to a few straight legs (OccupancyGrid.shortest_path). Her
    position comes from the odometry pose, which by now has run on from the last anchor,
    so it is a few cm to ~0.2 m off -- fine for getting near a cross, never for scoring.
  * Each leg is turn-to-face, then drive straight, sent as NAOqi `moveTo` through the shim
    (MOVETO). moveTo closes the loop on the robot; steering from the laptop through the
    0.8 s command latency would overshoot every turn.
  * NAOqi's external collision protection is switched ON for the drive and back OFF before
    returning, so the scored routes keep the protection-off setting of the protocol
    (docs/real_eval_session.md, B). A bumper press stops her and gives control back.
"""

import math
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


class Repositioner:
    def __init__(self, layout_room, ros, shim=None, clearance_radius=0.32):
        from rl.env.routes import build_grid
        if shim is None:
            sys.path.insert(0, str(REPO / "tools"))
            from pepper_bridge_node import ShimClient
            shim = ShimClient()
        self.ros, self.shim = ros, shim
        self.layout_room = layout_room
        self.grid = build_grid(layout_room, radius=clearance_radius, verbose=False)[0]

    def _cmd(self, line):
        reply = self.shim.request(line)
        if reply.startswith("ERR"):
            raise RuntimeError(f"shim refused {line!r}: {reply}")
        return reply

    def _pose(self, settle=0.6):
        end = time.time() + settle            # let odometry catch up after a move
        while time.time() < end:
            self.ros.spin(0.05)
        return self.ros._pose

    def _run(self, dx, dy, dth, timeout):
        """One MOVETO, waited out. False if a bumper or the timeout stopped it."""
        self.ros._contact = False
        self._cmd(f"MOVETO {dx:.4f} {dy:.4f} {dth:.4f}")
        end = time.time() + timeout
        time.sleep(0.3)
        while time.time() < end:
            self.ros.spin(0.1)
            if self.ros._contact:
                self._cmd("STOP")
                print("    bumper pressed -- stopped")
                return False
            if self._cmd("MOVING") == "MOVING 0":
                return True
        self._cmd("STOP")
        print("    move timed out -- stopped")
        return False

    def plan(self, x, y):
        here = self.ros._pose
        return self.grid.shortest_path((here[0], here[1]), (x, y))

    def drive_to(self, x, y, yaw):
        """Drive to roughly (x, y, yaw). Returns True if every leg completed."""
        if self.ros._pose is None:
            print("    position unknown (no route run yet) -- place her by hand this time")
            return False
        path = self.plan(x, y)
        if not path:
            print("    no free path from where she is -- place her by hand")
            return False
        legs = " -> ".join(f"({px:.2f}, {py:.2f})" for px, py in path[1:])
        print(f"    driving via {legs}; collision protection ON while moving")
        self._cmd("SAFETY on")
        try:
            for px, py in path[1:]:
                cx, cy, cyaw = self._pose()
                d = math.hypot(px - cx, py - cy)
                if d < 0.05:
                    continue
                turn = _wrap(math.atan2(py - cy, px - cx) - cyaw)
                if abs(turn) > 0.05 and not self._run(0.0, 0.0, turn, 10 + abs(turn) * 8):
                    return False
                if not self._run(d, 0.0, 0.0, 10 + d * 10):
                    return False
            # Final heading, then one corrective turn: an in-place turn on a real floor
            # under- or over-shoots a few degrees, and heading is the part of the
            # placement that matters most (3 deg is ~13 cm off course after 2.5 m).
            for _ in range(2):
                turn = _wrap(yaw - self._pose()[2])
                if abs(turn) <= 0.05:
                    break
                if not self._run(0.0, 0.0, turn, 10 + abs(turn) * 8):
                    return False
            fx, fy, fyaw = self._pose()
            print(f"    arrived about {math.hypot(fx - x, fy - y):.2f} m / "
                  f"{abs(math.degrees(_wrap(fyaw - yaw))):.0f} deg from the cross "
                  f"(odometry estimate)")
            return True
        finally:
            # Back to the protocol's setting before a scored route can start. NAOqi refuses
            # "off" unless safety-reflex deactivation is enabled in the robot's settings; then
            # protection simply stays on, which is what such a session runs with throughout
            # (15 Oct 2026). A refusal must not crash the evaluation mid-run.
            try:
                self._cmd("SAFETY off")
            except RuntimeError:
                pass
