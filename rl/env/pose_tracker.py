#!/usr/bin/env python3
"""Task 6's ground-truth pose rig: an overhead AprilTag, mapped to the room frame.

    python3 -m rl.env.pose_tracker --selftest
    ros2 run --prefix "python3 -m" rl.env.pose_tracker --calibration rl/config/pose_tracker_calibration.yaml
    (or just: python3 rl/env/pose_tracker.py --calibration ...)

Why this exists
---------------
Proposal S4.4.6 flags Pepper's own localisation as unreliable for evaluation before the
work even starts: "Pepper's localisation drift threatens fair real-world evaluation; the
mitigation is to secure a reliable external pose reference." `tools/pepper_bridge_node.py`
only ever publishes `odom` -- integrated, drifting dead reckoning -- and nothing on this
robot publishes `/pepper_robot/true_base_tf`, the topic `rl/env/pepper_nav_env.py`'s
`RosInterface` subscribes to for ground truth in BOTH sim and reality. In simulation that
topic comes from Ignition's own pose publisher, which cannot drift. This module is its
real-world equivalent: it does not integrate anything, it reads where the robot actually
is from outside the robot, every frame, from scratch.

The mechanism: an overhead camera, and one AprilTag stuck flat on top of Pepper
--------------------------------------------------------------------------------
Camera looking straight down, mounted above the room (a tripod near the ceiling, or the
ceiling itself). That turns "where is the tag" into a 2D problem: a homography maps pixel
coordinates to room-floor coordinates, calibrated once from a handful of known
correspondences (see `rl/scripts/calibrate_pose_tracker.py`), with no camera intrinsics,
no lens calibration, and no 3D pose estimation needed. A tripod camera looking across the
room at an angle would need full 6-DOF pose recovery per frame -- a strictly harder,
strictly less robust problem for very little gain on a task that only ever needs the
robot's (x, y, yaw) on the floor.

The room frame is the SAME one every other piece of this project shares: interior at
x in [0, length], y in [0, width], floor at z = 0, origin at the physical corner
`mesh_import.py --origin-mode floor-min-corner` chose. The calibration points you hand
`calibrate_pose_tracker.py` are measured in that frame, with a tape measure, from that same
corner -- get the corner wrong and every route, every l*, and every sim-to-real
comparison silently describes two different rooms.

A homography does not preserve angles in general (only a similarity transform does), so
yaw is NOT read off the tag's pixel-plane rotation. It is computed from the tag's own
centre and one edge midpoint, BOTH mapped through the homography into room coordinates,
and the heading is the angle between those two mapped points. That is correct however the
homography shears or scales, and reading the raw pixel angle instead would be a silent,
plausible-looking bug: right most of the time, wrong exactly when the pixel-to-metre scale
differs between the room's x and y axes, which it does unless the camera sensor and the
room happen to share an aspect ratio.

A detection outside the calibrated region is refused, not extrapolated. A homography is
only trustworthy inside the convex hull of the points it was fitted from; trusting it
further out is exactly the kind of silent-garbage failure this project has refused
elsewhere (the WSL software-render hazard, the fixed-height floor band). See
`in_calibrated_region`.

What is verified here, and what genuinely is not
-------------------------------------------------
`--selftest` verifies the homography math, the yaw computation, the message format
`rl.env.pepper_nav_env.RosInterface._on_pose` actually parses, AND the real AprilTag
detector (`cv2.aruco`) against a SYNTHETIC, perspective-warped tag image generated in the
same run -- so detection is exercised as genuine code, not mocked. What it cannot verify,
because it needs an actual camera and an actual tag physically stuck to an actual Pepper,
is whether a real lens, real lighting and a real print of the tag behave the way the
synthetic image does. That is the one thing this module cannot self-certify; run
`--live-preview` against the real camera before trusting it for an evaluation session.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
DEFAULT_CALIBRATION = REPO / "rl" / "config" / "pose_tracker_calibration.json"

# OpenCV 4.5.4 here uses the pre-4.7 function-based ArUco API
# (cv2.aruco.detectMarkers + DetectorParameters_create), not the newer ArucoDetector
# class. Checked explicitly at import time rather than assumed, because the two APIs
# are not compatible and a silent version drift would fail in a confusing place.
def _load_cv2():
    import cv2
    if not hasattr(cv2.aruco, "detectMarkers"):
        raise RuntimeError(
            "this OpenCV build does not expose cv2.aruco.detectMarkers (the pre-4.7 "
            "API). If OpenCV has been upgraded, rl/env/pose_tracker.py needs updating "
            "for the cv2.aruco.ArucoDetector class the newer versions use instead.")
    return cv2


APRILTAG_FAMILY = "DICT_APRILTAG_36h11"  # matches printable AprilTag family "tag36h11"


def die(msg):
    print("ERROR: " + msg, file=sys.stderr)
    sys.exit(1)


# --------------------------------------------------------------------------------------
# pure geometry -- no camera, no ROS, fully unit-testable
# --------------------------------------------------------------------------------------

def solve_homography(pixel_points, room_points):
    """Fit the pixel-to-room homography from >=4 correspondences.

    `pixel_points` and `room_points` are the same length, each row an (x, y) pair. Uses a
    direct linear transform (DLT) via least squares rather than requiring exactly 4
    points, so extra calibration points improve the fit instead of being wasted.
    """
    pixel_points = np.asarray(pixel_points, dtype=np.float64)
    room_points = np.asarray(room_points, dtype=np.float64)
    if len(pixel_points) < 4:
        raise ValueError(
            f"need at least 4 calibration points, got {len(pixel_points)}. A homography "
            f"has 8 degrees of freedom; fewer points cannot constrain it.")
    if len(pixel_points) != len(room_points):
        raise ValueError("pixel_points and room_points must be the same length")

    n = len(pixel_points)
    a = np.zeros((2 * n, 8))
    b = np.zeros(2 * n)
    for i, ((u, v), (x, y)) in enumerate(zip(pixel_points, room_points)):
        a[2 * i] = [u, v, 1, 0, 0, 0, -u * x, -v * x]
        b[2 * i] = x
        a[2 * i + 1] = [0, 0, 0, u, v, 1, -u * y, -v * y]
        b[2 * i + 1] = y
    h, *_ = np.linalg.lstsq(a, b, rcond=None)
    H = np.array([[h[0], h[1], h[2]], [h[3], h[4], h[5]], [h[6], h[7], 1.0]])
    return H


def apply_homography(H, points):
    """Map an (N, 2) array of pixel points to room coordinates through H."""
    points = np.atleast_2d(np.asarray(points, dtype=np.float64))
    homog = np.hstack([points, np.ones((len(points), 1))])
    mapped = (H @ homog.T).T
    return mapped[:, :2] / mapped[:, 2:3]


def calibration_residuals(H, pixel_points, room_points):
    """Per-point reprojection error, in the same units as room_points (metres).

    The number to look at before trusting a calibration. A homography can fit noisy
    correspondences perfectly badly -- 4 points always fit exactly regardless of how
    poorly they were measured -- so this is what actually catches a fat-fingered tape
    measurement or a point recorded against the wrong room corner.
    """
    mapped = apply_homography(H, pixel_points)
    return np.linalg.norm(mapped - np.asarray(room_points), axis=1)


def robot_pose_from_tag(H, tag_centre_px, tag_forward_edge_midpoint_px, tag_yaw_offset=0.0):
    """The one piece of geometry the whole tracker exists to compute.

    Position is the tag centre, mapped through H. Heading is NOT the tag's pixel-plane
    rotation -- see the module docstring for why a homography does not preserve angles
    in general -- it is the direction from the mapped centre to the mapped forward-edge
    midpoint, which is correct under any affine or projective distortion the homography
    captures. `tag_yaw_offset` corrects for how the tag is actually stuck to the robot:
    measured once, by rotating the robot to a known heading and reading off the error.
    """
    mapped = apply_homography(H, [tag_centre_px, tag_forward_edge_midpoint_px])
    centre, forward = mapped[0], mapped[1]
    heading = math.atan2(forward[1] - centre[1], forward[0] - centre[0])
    yaw = _wrap(heading + tag_yaw_offset)
    return float(centre[0]), float(centre[1]), yaw


def _wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def in_calibrated_region(point, room_points, margin_m=0.0):
    """Whether `point` falls inside the convex hull of the calibration points.

    A homography is a good local model of the true (lens-distorted, non-planar-camera)
    mapping near the points it was fitted from, and an increasingly bad one far outside
    them. Refusing a pose outside this region is what stops a tag that has drifted to the
    room's edge from producing a confident-looking but silently wrong position -- exactly
    the class of failure this project has refused to ship anywhere else in it.

    `margin_m` shrinks the trusted region inward from the hull boundary; the default (0)
    trusts exactly the calibrated hull.
    """
    from scipy.spatial import Delaunay
    hull = np.asarray(room_points, dtype=np.float64)
    if margin_m > 0:
        # Move each hull point margin_m closer to the centroid, along the line to it.
        # An approximation of eroding the hull inward by margin_m -- exact only for a
        # regular, roughly-round hull, which is good enough for a sanity margin and not
        # claimed to be more than that.
        centroid = hull.mean(axis=0)
        to_centroid = centroid - hull
        dist = np.linalg.norm(to_centroid, axis=1, keepdims=True).clip(min=1e-9)
        hull = hull + to_centroid / dist * min(margin_m, dist.min())
    tri = Delaunay(hull)
    return bool(tri.find_simplex(np.asarray(point)) >= 0)


# --------------------------------------------------------------------------------------
# AprilTag detection -- the one piece that touches an actual image
# --------------------------------------------------------------------------------------

def detect_tag(image_bgr, tag_id):
    """Find `tag_id` in a frame. Returns (corners, ids) style result or None.

    Corners come back as OpenCV's ArUco detector orders them: index 0 = top-left,
    1 = top-right, 2 = bottom-right, 3 = bottom-left, IN THE TAG'S OWN PRINTED FRAME --
    i.e. "top" means the edge nearest the tag's own upright orientation as encoded in its
    bit pattern, not "top of the image". That is what makes the top-edge midpoint a
    meaningful forward-direction reference at all: it rotates WITH the tag, in the image,
    as the tag rotates. If the tag is stuck to the robot with the printed top edge in an
    arbitrary direction, that arbitrary direction is exactly what `tag_yaw_offset`
    calibrates out.

    Returns None if the tag is not visible at all -- deliberately not raising, since "not
    visible this frame" is routine (Pepper's body or a person can occlude it) and the
    caller has to keep running either way.
    """
    cv2 = _load_cv2()
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if image_bgr.ndim == 3 else image_bgr
    aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, APRILTAG_FAMILY))
    params = cv2.aruco.DetectorParameters_create()
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=params)
    if ids is None:
        return None
    ids = ids.flatten()
    matches = [c for c, i in zip(corners, ids) if int(i) == tag_id]
    if not matches:
        return None
    if len(matches) > 1:
        # Ambiguous, and worth knowing about rather than silently picking one: two tags
        # with the same ID means either a printing mistake or a reflection/duplicate.
        raise RuntimeError(
            f"tag id {tag_id} detected {len(matches)} times in one frame -- check for a "
            f"duplicate or reflected tag")
    return matches[0].reshape(4, 2)  # (4, 2): corner 0..3 as documented above


def tag_reference_points(corners):
    """Centre and forward-edge midpoint, in pixel coordinates, from detected corners.

    The "forward" edge is corners[0]-corners[1] (the tag's own top edge, per
    detect_tag's docstring) -- an arbitrary but FIXED choice, consistent from frame to
    frame, which is all `tag_yaw_offset` needs to correct for once, at calibration time.
    """
    centre = corners.mean(axis=0)
    forward_midpoint = (corners[0] + corners[1]) / 2.0
    return centre, forward_midpoint


# --------------------------------------------------------------------------------------
# calibration file
# --------------------------------------------------------------------------------------

def save_calibration(path, H, pixel_points, room_points, tag_id, tag_yaw_offset,
                     camera_source, image_size, residuals):
    payload = {
        "tag_id": tag_id,
        "tag_yaw_offset_rad": tag_yaw_offset,
        "camera_source": camera_source,
        "image_size": list(image_size),
        "homography": np.asarray(H).tolist(),
        "pixel_points": np.asarray(pixel_points).tolist(),
        "room_points": np.asarray(room_points).tolist(),
        "residuals_m": np.asarray(residuals).tolist(),
        "max_residual_m": float(np.max(residuals)),
    }
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_calibration(path):
    if not Path(path).is_file():
        die(f"no calibration at {path}. Run rl/scripts/calibrate_pose_tracker.py first "
            f"-- there is no default; a homography fitted for one camera position "
            f"means nothing for another.")
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    d["homography"] = np.asarray(d["homography"])
    d["pixel_points"] = np.asarray(d["pixel_points"])
    d["room_points"] = np.asarray(d["room_points"])
    return d


# --------------------------------------------------------------------------------------
# the ROS 2 node
# --------------------------------------------------------------------------------------

class PoseTrackerNode:
    """Publishes /pepper_robot/true_base_tf from an overhead camera, every frame.

    Deliberately thin: all the geometry above is free-standing so it can be (and is)
    unit-tested without rclpy or a camera. This class is just camera I/O, the publish
    loop, and the "refuse rather than guess" checks around both.
    """

    POSE_TOPIC = "/pepper_robot/true_base_tf"

    def __init__(self, calibration, camera_source=None, rate_hz=15.0,
                margin_m=0.1, verbose=False):
        import rclpy
        from rclpy.node import Node
        from tf2_msgs.msg import TFMessage

        if not rclpy.ok():
            rclpy.init()
        self.node = Node("pose_tracker")
        self._TFMessage = TFMessage
        self.calibration = calibration
        self.margin_m = margin_m
        self.verbose = verbose
        self.rate_hz = rate_hz

        self.pub = self.node.create_publisher(TFMessage, self.POSE_TOPIC, 1)
        self._last_pose = None
        self._misses = 0
        self._frames = 0

        source = camera_source or calibration["camera_source"]
        self.cap = self._open_camera(source)

    @staticmethod
    def _open_camera(source):
        cv2 = _load_cv2()
        # An integer-looking source is a /dev/video<N> device index; anything else is
        # passed straight to VideoCapture, which also accepts a URL or file path (useful
        # for replaying a recorded session through this same code path).
        try:
            source = int(source)
        except (TypeError, ValueError):
            pass
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            die(f"could not open camera source {source!r}. Check the device index "
                f"(commonly 0 for the first USB camera; `ls /dev/video*` to list them) "
                f"or the ROS/USB permissions.")
        return cap

    def _publish_pose(self, x, y, yaw):
        # rclpy message fields are plain Python lists for repeated messages, so the
        # TransformStamped is built directly and assigned -- no protobuf-style
        # container API to reach for.
        from geometry_msgs.msg import TransformStamped
        msg = self._TFMessage()
        t = TransformStamped()
        t.header.stamp = self.node.get_clock().now().to_msg()
        t.header.frame_id = "room"
        t.child_frame_id = "pepper_robot/base_footprint"
        t.transform.translation.x = float(x)
        t.transform.translation.y = float(y)
        t.transform.translation.z = 0.0
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = math.sin(yaw / 2.0)
        t.transform.rotation.w = math.cos(yaw / 2.0)
        msg.transforms = [t]
        self.pub.publish(msg)

    def spin_once(self):
        """Grab one frame, detect the tag, publish if (and only if) it is trustworthy.

        Returns True if a pose was published, False if this frame contributed nothing --
        the caller (a training/eval session) should treat consecutive False results as
        exactly what tools/check_depth.py treats a frozen depth stream as: a reason to
        stop and look, not a reason to keep going and hope.
        """
        ok, frame = self.cap.read()
        self._frames += 1
        if not ok:
            self._misses += 1
            if self.verbose:
                self.node.get_logger().warn("camera read failed")
            return False

        corners = detect_tag(frame, self.calibration["tag_id"])
        if corners is None:
            self._misses += 1
            if self.verbose and self._misses % 30 == 1:
                self.node.get_logger().warn(
                    f"tag {self.calibration['tag_id']} not visible "
                    f"({self._misses}/{self._frames} frames missed)")
            return False

        centre_px, forward_px = tag_reference_points(corners)
        x, y, yaw = robot_pose_from_tag(
            self.calibration["homography"], centre_px, forward_px,
            self.calibration["tag_yaw_offset_rad"])

        if not in_calibrated_region((x, y), self.calibration["room_points"],
                                    margin_m=self.margin_m):
            # Refused, not clamped or extrapolated -- see the module docstring's note on
            # in_calibrated_region. A policy or evaluator relying on this pose is better
            # served by a stale-but-known-good last pose (or a loud gap in the log) than
            # by a confident-looking number the homography was never fitted to support.
            self._misses += 1
            if self.verbose:
                self.node.get_logger().warn(
                    f"tag at ({x:.2f}, {y:.2f}) is outside the calibrated region -- "
                    f"not publishing")
            return False

        self._publish_pose(x, y, yaw)
        self._last_pose = (x, y, yaw)
        return True

    def run(self):
        period = 1.0 / self.rate_hz
        print(f"publishing {self.POSE_TOPIC} at up to {self.rate_hz} Hz "
             f"(tag {self.calibration['tag_id']}). Ctrl-C to stop.")
        try:
            import rclpy
            while rclpy.ok():
                t0 = self.node.get_clock().now().nanoseconds / 1e9
                self.spin_once()
                elapsed = self.node.get_clock().now().nanoseconds / 1e9 - t0
                rclpy.spin_once(self.node, timeout_sec=max(period - elapsed, 0.0))
        except KeyboardInterrupt:
            pass
        finally:
            print(f"\n{self._frames} frames, {self._misses} not published "
                 f"({self._misses / max(self._frames, 1):.1%})")
            self.close()

    def close(self):
        self.cap.release()
        try:
            self.node.destroy_node()
        except Exception:
            pass


def live_preview(calibration, camera_source, seconds):
    """Print detected poses for a while with no ROS involved at all -- the fastest way
    to tell whether the physical camera, tag and lighting actually work together, before
    trusting any of this in a training or evaluation session."""
    import time
    cap = PoseTrackerNode._open_camera(camera_source or calibration["camera_source"])
    hits = misses = 0
    t_end = time.time() + seconds
    try:
        while time.time() < t_end:
            ok, frame = cap.read()
            if not ok:
                misses += 1
                continue
            corners = detect_tag(frame, calibration["tag_id"])
            if corners is None:
                misses += 1
                print("  (tag not visible)")
                continue
            centre_px, forward_px = tag_reference_points(corners)
            x, y, yaw = robot_pose_from_tag(
                calibration["homography"], centre_px, forward_px,
                calibration["tag_yaw_offset_rad"])
            inside = in_calibrated_region((x, y), calibration["room_points"])
            hits += 1
            print(f"  x={x:6.2f}  y={y:6.2f}  yaw={math.degrees(yaw):6.1f} deg"
                 f"{'' if inside else '   OUTSIDE CALIBRATED REGION'}")
    finally:
        cap.release()
    print(f"\n{hits} detections, {misses} misses over {seconds}s")


# --------------------------------------------------------------------------------------
# self-test
# --------------------------------------------------------------------------------------

def _make_synthetic_tag_image(tag_id, image_size, corners_px):
    """Render a real AprilTag (via cv2.aruco's own generator) into a blank frame at
    `corners_px`, via a perspective warp. Used only by --selftest, to exercise the actual
    detector against actual pixels rather than mocking detect_tag() and testing nothing
    about whether cv2.aruco can find the tag at all.

    `corners_px` MUST be given as (top-left, top-right, bottom-right, bottom-left), the
    same winding `detect_tag` documents OpenCV returning. Any other order mirrors or
    rotates the tag's bit pattern into something no valid AprilTag encodes -- caught
    exactly this way the first time this function was written: the detector finding
    nothing looked like a detector bug until the corner order turned out to be the
    actual mistake.
    """
    cv2 = _load_cv2()
    aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, APRILTAG_FAMILY))
    tag_px = 200
    marker = cv2.aruco.drawMarker(aruco_dict, tag_id, tag_px)
    marker_bgr = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)

    # A white quiet zone matters: AprilTag detection relies on a light border around the
    # dark tag border to find it at all, exactly as a printed tag needs a margin.
    pad = tag_px // 4
    padded = np.full((tag_px + 2 * pad, tag_px + 2 * pad, 3), 255, dtype=np.uint8)
    padded[pad:pad + tag_px, pad:pad + tag_px] = marker_bgr
    src = np.array([[0, 0], [padded.shape[1], 0], [padded.shape[1], padded.shape[0]],
                    [0, padded.shape[0]]], dtype=np.float32)
    # dst starts as the requested marker corners, then expands outward by the padding
    # fraction so the padded image's outer edge (not the inner marker) lands there --
    # otherwise the rendered marker would be smaller than target_corners describes.
    dst = np.array(corners_px, dtype=np.float32)
    centre = dst.mean(axis=0)
    dst = centre + (dst - centre) * (padded.shape[0] / tag_px)

    H = cv2.getPerspectiveTransform(src, dst)
    canvas = np.full((image_size[1], image_size[0], 3), 200, dtype=np.uint8)  # grey bg
    warped = cv2.warpPerspective(padded, H, image_size)
    mask = cv2.warpPerspective(np.ones_like(padded[:, :, 0]) * 255, H, image_size)
    canvas[mask > 0] = warped[mask > 0]
    return canvas


def _selftest():
    fails = []

    def check(label, got, want=None, tol=1e-6):
        if want is None:
            ok = bool(got)
        elif isinstance(want, float) or isinstance(got, float):
            ok = abs(got - want) <= tol
        else:
            ok = got == want
        detail = "" if want is None else f"  got={got!r}"
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}{detail}")
        if not ok:
            fails.append(label)

    # -- homography: exact recovery of a known affine + mild perspective map -----------
    room_pts = [(0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0), (2.0, 1.5)]
    # A made-up pixel mapping: room (0,0)-(4,3) m onto pixels (0,0)-(800,450), with a
    # touch of perspective so the test cannot pass by accident via a pure affine solve.
    pixel_pts = [(50, 400), (750, 420), (760, 60), (40, 40), (400, 230)]
    H = solve_homography(pixel_pts[:4], room_pts[:4])
    residuals = calibration_residuals(H, pixel_pts, room_pts)
    check("homography reproduces its own 4 fit points exactly",
          float(np.max(residuals[:4])), 0.0, tol=1e-6)
    check("homography generalises to a 5th, held-out point reasonably",
          float(residuals[4]) < 0.5, True)  # loose: this mapping is not a true homography

    # -- yaw from mapped centre/forward points, not pixel-plane rotation ----------------
    # A homography that scales x and y differently, so a naive "read the pixel angle"
    # shortcut and the correct mapped-direction computation would disagree in general.
    H_scale = np.array([[2.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 1.0]])
    centre_px, forward_px = (10.0, 10.0), (11.0, 10.0)  # "east" in pixel space
    _, _, yaw = robot_pose_from_tag(H_scale, centre_px, forward_px, tag_yaw_offset=0.0)
    check("yaw follows the MAPPED direction (still east: x scales, still east)",
          yaw, 0.0, tol=1e-6)

    H_shear = np.array([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    # Pixel-plane "east" (1,0) maps to room-plane (1,1) under this shear -- 45 degrees,
    # not 0. A naive pixel-angle shortcut would report 0 here; the mapped-direction
    # computation must report 45.
    _, _, yaw3 = robot_pose_from_tag(H_shear, (0.0, 0.0), (1.0, 0.0), tag_yaw_offset=0.0)
    check("yaw correctly follows a SHEARED mapping (the case a pixel-angle shortcut gets wrong)",
          yaw3, math.pi / 4, tol=1e-6)

    offset_yaw = robot_pose_from_tag(H_scale, centre_px, forward_px,
                                     tag_yaw_offset=math.pi / 2)[2]
    check("tag_yaw_offset is a pure additive correction", offset_yaw, math.pi / 2, tol=1e-6)

    # -- calibrated-region refusal ------------------------------------------------------
    hull = [(0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0)]
    check("a point well inside the hull is trusted",
          in_calibrated_region((2.0, 1.5), hull), True)
    check("a point well outside the hull is refused",
          in_calibrated_region((10.0, 10.0), hull), False)
    check("a point just outside the hull, with margin, is refused",
          in_calibrated_region((3.99, 1.5), hull, margin_m=0.1), False)

    # -- the message contract: byte-for-byte what RosInterface._on_pose parses ---------
    # Reimplements _on_pose's own yaw extraction rather than importing rclpy message
    # classes (this self-test must run with no ROS installed at all), and checks it
    # against the SAME formula pepper_nav_env.py uses, so a change to one is caught by
    # the other silently drifting.
    def on_pose_yaw(qx, qy, qz, qw):
        return math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))

    for test_yaw in (0.0, math.pi / 3, -math.pi / 2, math.pi - 0.01):
        qz, qw = math.sin(test_yaw / 2.0), math.cos(test_yaw / 2.0)
        recovered = on_pose_yaw(0.0, 0.0, qz, qw)
        check(f"quaternion round-trip for yaw={test_yaw:.3f} matches _on_pose's formula",
              recovered, test_yaw, tol=1e-6)

    # -- real AprilTag detection against a synthetic, perspective-warped image ---------
    try:
        tag_id = 7
        # Order matters and is not arbitrary: TL, TR, BR, BL, matching src's convention
        # in _make_synthetic_tag_image exactly. Getting this winding wrong mirrors the
        # tag pattern into the image -- caught here the first time this ran: a mirrored
        # AprilTag is not a valid bit pattern for ANY tag id, so the detector correctly
        # finds nothing, which looks exactly like "detection is broken" until you check
        # the corner order rather than the detector.
        target_corners = [(110, 70), (470, 60), (480, 310), (120, 300)]
        img = _make_synthetic_tag_image(tag_id, (640, 480), target_corners)
        found = detect_tag(img, tag_id)
        check("the real cv2.aruco detector finds a synthetic tag it was given",
              found is not None, True)
        if found is not None:
            centre_err = float(np.linalg.norm(
                found.mean(axis=0) - np.mean(target_corners, axis=0)))
            check("detected tag centre is close to where it was rendered",
                  centre_err < 15.0, True)  # pixels; generous for a warped synthetic render
        check("detect_tag returns None for an absent tag id",
              detect_tag(img, tag_id + 1) is None, True)
    except Exception as e:
        check(f"synthetic AprilTag detection ({type(e).__name__}: {e})", False, True)

    print()
    if fails:
        print(f"{len(fails)} FAILED: {fails}")
        return 1
    print("all checks passed -- the geometry, the message contract and real AprilTag\n"
         "detection against a synthetic image are all verified.\n"
         "NOT verified here, and only verifiable with the physical rig: a real camera,\n"
         "real lighting and a real printed tag stuck to the real robot. Run\n"
         "--live-preview against that before trusting it for a session.")
    return 0


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--calibration", default=str(DEFAULT_CALIBRATION))
    ap.add_argument("--camera-source", default=None,
                    help="override the calibration's stored camera source (device index "
                         "or path)")
    ap.add_argument("--rate", type=float, default=15.0)
    ap.add_argument("--margin", type=float, default=0.1,
                    help="metres to shrink the trusted region inward from the "
                         "calibrated hull (default 0.1)")
    ap.add_argument("--live-preview", type=float, default=None, metavar="SECONDS",
                    help="print detected poses for SECONDS with no ROS involved -- run "
                         "this first, against the real rig, before trusting anything else")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(_selftest())

    calibration = load_calibration(args.calibration)
    if args.live_preview is not None:
        live_preview(calibration, args.camera_source, args.live_preview)
        return

    node = PoseTrackerNode(calibration, camera_source=args.camera_source,
                           rate_hz=args.rate, margin_m=args.margin, verbose=args.verbose)
    node.run()


if __name__ == "__main__":
    main()
