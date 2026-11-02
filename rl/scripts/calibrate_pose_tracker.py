#!/usr/bin/env python3
"""Calibrate the overhead pose tracker: pixel-to-room homography plus tag yaw offset.

    # Live, walking through each point interactively:
    python3 rl/scripts/calibrate_pose_tracker.py --live --camera-source 0 --tag-id 7 \\
        --room-point 0.0 0.0 --room-point 4.0 0.0 --room-point 4.0 3.0 \\
        --room-point 0.0 3.0 --room-point 2.0 1.5 \\
        --yaw-room-point 2.0 1.5 --yaw-heading-deg 0

    # From already-recorded pixel coordinates (e.g. re-deriving a calibration from a
    # session's own log, or for testing without a camera at all):
    python3 rl/scripts/calibrate_pose_tracker.py --points-file calib_points.csv

See rl/env/pose_tracker.py's module docstring for the whole mechanism this calibrates:
an overhead camera, one AprilTag on the robot, and a homography from pixel coordinates to
the room frame every other piece of this project already shares.

What "the room frame" means here, precisely
---------------------------------------------
Interior at x in [0, length], y in [0, width], floor at z = 0, origin at the SAME
physical corner `tools/mesh_import.py --origin-mode floor-min-corner` puts the
reconstruction's origin at. Measure every `--room-point` with a tape measure from that
corner. Get the corner wrong here and the pose tracker will report positions in a frame
that disagrees with the route set, the room spec, and the sim ground truth it is meant to
match -- silently, since a wrong-but-consistent frame looks exactly like a right one until
something is compared against it.

Why >= 5 points are asked for, not the mathematical minimum of 4
-------------------------------------------------------------------
4 points fit a homography exactly, with zero residual, REGARDLESS of how good the
measurements were -- a fat-fingered tape reading on point 3 is invisible with only 4
points. A 5th (or more) lets `rl.env.pose_tracker.calibration_residuals` report a
meaningful number: near-zero if the calibration is good, and large exactly on the point
that was measured or captured wrong.

The yaw offset step, and why it is separate from the position points
------------------------------------------------------------------------
Position calibration only needs the tag to sit still at known room coordinates -- its
heading does not matter for that. Getting the YAW offset needs the opposite: one point
where the robot's actual heading is known (place her facing along a wall, or square her up
against a known reference direction) so `tag_yaw_offset` can be solved as the one free
parameter linking "which way the tag's printed top edge happens to point" to "which way
Pepper is actually facing". Skipping this step defaults the offset to 0, which is only
correct if the tag happens to be stuck on facing exactly forward -- worth checking with
`--live-preview` in rl/env/pose_tracker.py rather than assuming.
"""

import argparse
import csv
import math
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from rl.env.pose_tracker import (  # noqa: E402
    detect_tag, tag_reference_points, solve_homography, calibration_residuals,
    apply_homography, save_calibration, DEFAULT_CALIBRATION, die, PoseTrackerNode,
)


def capture_tag_pixel(cap, tag_id, settle_frames=5):
    """Grab a frame, find the tag, return (centre_px, forward_edge_px) or None.

    Reads and discards a few frames first: a USB camera's auto-exposure/auto-white-
    balance needs a moment after `read()` starts returning frames at all, and the very
    first frame is disproportionately likely to be stale or under/over-exposed.
    """
    corners = None
    for _ in range(settle_frames):
        ok, frame = cap.read()
        if not ok:
            return None
        corners = detect_tag(frame, tag_id)
    if corners is None:
        return None
    return tag_reference_points(corners)


def run_live(args):
    from rl.env.pose_tracker import _load_cv2
    _load_cv2()
    cap = PoseTrackerNode._open_camera(args.camera_source)
    image_size = (int(cap.get(3)), int(cap.get(4)))  # CAP_PROP_FRAME_WIDTH/HEIGHT
    print(f"camera open, reported frame size {image_size}")

    pixel_points, room_points = [], []
    for (rx, ry) in args.room_point:
        input(f"\nPlace the tag at room ({rx:.2f}, {ry:.2f}) m, hold it flat and still, "
             f"then press Enter...")
        found = capture_tag_pixel(cap, args.tag_id)
        if found is None:
            die(f"tag {args.tag_id} not detected at room point ({rx}, {ry}). Check "
                f"lighting, focus, and that the tag isn't tilted -- an overhead camera "
                f"needs it flat.")
        centre_px, _ = found
        print(f"  detected at pixel {tuple(round(v, 1) for v in centre_px)}")
        pixel_points.append(tuple(centre_px))
        room_points.append((rx, ry))

    H = solve_homography(pixel_points, room_points)
    residuals = calibration_residuals(H, pixel_points, room_points)
    report_residuals(room_points, residuals)

    tag_yaw_offset = 0.0
    if args.yaw_room_point is not None:
        rx, ry = args.yaw_room_point
        heading = math.radians(args.yaw_heading_deg)
        input(f"\nPlace the tag at ({rx:.2f}, {ry:.2f}) m, facing {args.yaw_heading_deg} "
             f"deg (room frame), then press Enter...")
        found = capture_tag_pixel(cap, args.tag_id)
        if found is None:
            die("tag not detected for the yaw calibration step")
        centre_px, forward_px = found
        mapped = apply_homography(H, [centre_px, forward_px])
        raw_heading = math.atan2(mapped[1][1] - mapped[0][1], mapped[1][0] - mapped[0][0])
        tag_yaw_offset = math.atan2(math.sin(heading - raw_heading),
                                    math.cos(heading - raw_heading))
        print(f"  raw heading {math.degrees(raw_heading):.1f} deg -> "
             f"tag_yaw_offset = {math.degrees(tag_yaw_offset):.1f} deg")
    else:
        print("\nNo --yaw-room-point given: tag_yaw_offset defaults to 0. This is only "
             "correct if the tag happens to be stuck on facing exactly Pepper's front. "
             "Check with rl/env/pose_tracker.py --live-preview before trusting it.")

    cap.release()
    save_calibration(args.out, H, pixel_points, room_points, args.tag_id, tag_yaw_offset,
                     args.camera_source, image_size, residuals)
    print(f"\nwrote {args.out}")


def run_from_file(args):
    rows = list(csv.reader(Path(args.points_file).read_text().splitlines()))
    rows = [r for r in rows if r and not r[0].strip().startswith("#")]
    pixel_points = [(float(r[0]), float(r[1])) for r in rows]
    room_points = [(float(r[2]), float(r[3])) for r in rows]
    H = solve_homography(pixel_points, room_points)
    residuals = calibration_residuals(H, pixel_points, room_points)
    report_residuals(room_points, residuals)
    save_calibration(args.out, H, pixel_points, room_points, args.tag_id,
                     args.tag_yaw_offset_deg and math.radians(args.tag_yaw_offset_deg) or 0.0,
                     args.camera_source or "unknown", args.image_size or [0, 0], residuals)
    print(f"\nwrote {args.out}")


def report_residuals(room_points, residuals):
    print("\nresiduals (should be small relative to how precisely you can place the "
         "tag by hand -- a few cm is typical, not a few mm):")
    for (rx, ry), r in zip(room_points, residuals):
        flag = "  <-- check this one" if r > 0.15 else ""
        print(f"  ({rx:6.2f}, {ry:6.2f}) m : {r * 100:5.1f} cm{flag}")
    worst = max(residuals)
    if worst > 0.30:
        print(f"\nWARNING: worst residual is {worst * 100:.0f} cm. That is large enough "
             f"to matter against the goal tolerance in rl/config/env.yaml -- recheck the "
             f"corner the room frame is measured from, and that no point was recorded "
             f"against the wrong room coordinate.")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(DEFAULT_CALIBRATION))
    ap.add_argument("--tag-id", type=int, default=0)
    ap.add_argument("--camera-source", default=0,
                    help="device index (e.g. 0) or a path/URL cv2.VideoCapture accepts")

    ap.add_argument("--live", action="store_true",
                    help="walk through each --room-point interactively against a real "
                         "camera")
    ap.add_argument("--room-point", nargs=2, type=float, action="append", default=[],
                    metavar=("X", "Y"),
                    help="a room-frame calibration point, in metres from the room "
                         "origin corner. Repeat at least 5 times. Only used with --live.")
    ap.add_argument("--yaw-room-point", nargs=2, type=float, default=None,
                    metavar=("X", "Y"))
    ap.add_argument("--yaw-heading-deg", type=float, default=0.0,
                    help="the known room-frame heading (degrees, 0 = +x axis) the tag "
                         "will be facing at --yaw-room-point")

    ap.add_argument("--points-file", default=None,
                    help="CSV of pixel_x,pixel_y,room_x,room_y rows, one per "
                         "calibration point, as an alternative to --live")
    ap.add_argument("--tag-yaw-offset-deg", type=float, default=0.0,
                    help="only used with --points-file, where there is no live yaw "
                         "calibration step")
    ap.add_argument("--image-size", nargs=2, type=int, default=None, metavar=("W", "H"),
                    help="only used with --points-file, for the calibration record")
    args = ap.parse_args()

    if args.live and args.points_file:
        ap.error("--live and --points-file are alternatives, not both")
    if not args.live and not args.points_file:
        ap.error("pass --live (with a real camera) or --points-file")
    if args.live and len(args.room_point) < 4:
        ap.error(f"need at least 4 --room-point (5+ recommended so residuals mean "
                 f"something), got {len(args.room_point)}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    if args.live:
        run_live(args)
    else:
        run_from_file(args)


if __name__ == "__main__":
    main()
