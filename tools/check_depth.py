#!/usr/bin/env python3
"""Check the physical depth sensor hard enough to trust it as an observation source.

Steps 7 and 8 of verify_pepper.sh already grab a depth frame and measure the stream
rate, but getting to them means running the whole eight-step session including step 6,
which drives the robot around a cleared floor. This is the depth half on its own, and it
asks more of the sensor than one frame can answer:

  * **Shape.** 320x240, and a buffer of exactly w*h*2 bytes. The observation shape has
    to be identical across sim and real or the policy sees a different input space.
  * **Coverage.** What fraction of pixels return anything. Pepper's depth sensor is
    structured light, so glossy, dark and backlit surfaces come back as zero (S4.4.4) --
    and the number is a property of the *room*, which is why it is worth measuring
    before committing to one.
  * **Liveness.** Consecutive frames that are byte-identical mean a stuck buffer, not a
    still scene: real sensor noise guarantees some pixels differ every frame. A frozen
    stream otherwise looks perfectly healthy, which is what makes it worth testing for.
  * **Temporal noise.** The spread of the centre-patch distance across frames on a
    static scene. This is the number that says how much observation noise a sim2real
    transfer has to tolerate, and it cannot be read off one frame.
  * **Throughput.** Reuses pepper_bridge_node.py --probe-depth, the same measurement
    step 8 makes.

Keep the scene still for the few seconds this takes -- the noise and liveness numbers
both assume a static scene, and someone walking through the frame will inflate them.

**By default this moves nothing.** `ALVideoDevice` is independent of `ALMotion`, so the
depth camera works with the motors relaxed -- a rested robot still returns frames. That
matters when the floor around her is not clear: there is no reason to stand a robot up
in a cluttered room to find out whether its camera works.

The cost of not moving is that the head is wherever it happens to be, and the head *is*
the depth camera's extrinsics. That splits the results in two:

  * shape, coverage, liveness, noise and throughput are properties of the sensor and
    are valid at any head pose -- everything this script is really for;
  * "aimed along the same ray as the training data" is a property of the pose, and is
    reported as a note rather than a failure when the head has not been placed.

If you do want the head at the simulated pose, `--head-pose` stiffens the Head chain and
moves the head *only* -- not the base, not the arms. A head rotating on the spot cannot
reach anything the robot is not already touching, so it is safe in a room the base and
arms are not. There is deliberately no whole-body wake-up option here; use
`tools/run_shim.sh` and a WAKEUP by hand if you want one, on a floor you have cleared.

    tools/run_shim.sh --robot-ip 192.168.1.12     # terminal 1
    python3 tools/check_depth.py                  # terminal 2
"""

import argparse
import hashlib
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pepper_bridge_node import DEFAULT_HOST, DEFAULT_PORT, ShimClient  # noqa: E402

# The simulated camera, from naoqi_shim: pepper_robot.gazebo's camera_depth.
SIM_WIDTH = 320
SIM_HEIGHT = 240
SIM_CLIP_NEAR_M = 0.4
SIM_CLIP_FAR_M = 8.0

# The pose the simulated depth camera is fixed at, from naoqi_shim.SIM_HEAD_YAW /
# SIM_HEAD_PITCH. The tolerance is loose enough not to fire on ALMotion's steady-state
# error, tight enough to catch a head someone left pointing somewhere else.
SIM_HEAD_YAW = 0.0
SIM_HEAD_PITCH = 0.45
HEAD_TOLERANCE = 0.05

# The centre patch used for the distance and noise numbers. Small enough to sit on one
# surface at typical standing distance, big enough that its median is not itself noise.
PATCH_W = 64
PATCH_H = 48


def grab(client, raw_path, retries=1):
    """One frame. Returns (width, height, millimetre array, sha1 of the raw buffer).

    A reply that is not a DEPTH line is worth one retry rather than an abort: ShimClient
    drains stale datagrams before each request, but a reply crossing in flight with the
    one we want can still land out of order, and losing a ten-frame run to that would be
    a poor trade.
    """
    for attempt in range(retries + 1):
        reply = client.request("DEPTH %s" % raw_path)
        parts = reply.split()
        if parts and parts[0] == "DEPTH":
            break
        if attempt == retries:
            raise RuntimeError("expected a DEPTH reply, got %r" % reply)
    width, height = int(parts[1]), int(parts[2])
    with open(raw_path, "rb") as handle:
        buf = handle.read()
    expected = width * height * 2
    if len(buf) != expected:
        raise RuntimeError("buffer is %d bytes, expected %d (%dx%d, 16-bit)"
                           % (len(buf), expected, width, height))
    # "<u2" not "=u2": the wire format is little-endian 16-bit millimetres regardless of
    # what the host happens to be.
    frame = np.frombuffer(buf, dtype="<u2").reshape(height, width)
    return width, height, frame, hashlib.sha1(buf).hexdigest()


def centre_patch(frame):
    """Median distance in metres over the valid pixels of the centre patch, or None."""
    height, width = frame.shape
    top = (height - PATCH_H) // 2
    left = (width - PATCH_W) // 2
    patch = frame[top:top + PATCH_H, left:left + PATCH_W]
    valid = patch[patch != 0]
    if valid.size == 0:
        return None
    return float(np.median(valid)) / 1000.0


def save_png(frame, path):
    """Turbo preview on the simulator's fixed clip range, black where there is no return.

    Fixed scale rather than per-frame autoscale, matching pepper_live_view.py: an
    autoscaled depth image looks better and makes it impossible to compare two of them.
    """
    try:
        from matplotlib import cm
        from PIL import Image
    except ImportError as exc:
        return "preview skipped (%s)" % exc
    metres = frame.astype(np.float32) / 1000.0
    norm = np.clip((metres - SIM_CLIP_NEAR_M) / (SIM_CLIP_FAR_M - SIM_CLIP_NEAR_M), 0, 1)
    rgb = (cm.turbo(norm)[:, :, :3] * 255).astype(np.uint8)
    rgb[frame == 0] = 0
    Image.fromarray(rgb, "RGB").save(path)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--frames", type=int, default=10,
                        help="frames to sample for coverage and noise (default 10)")
    parser.add_argument("--interval", type=float, default=0.15,
                        help="seconds between grabs (default 0.15). The sensor runs at "
                             "15 fps, so polling faster than ~0.07 s just re-reads the "
                             "same frame and looks like a freeze")
    parser.add_argument("--head-pose", action="store_true",
                        help="stiffen the Head chain and move the head -- and only the "
                             "head -- to the simulated pose first. The base and arms do "
                             "not move. Without this the script moves nothing at all")
    parser.add_argument("--stream", type=float, default=5.0,
                        help="seconds to measure stream throughput; 0 skips (default 5)")
    parser.add_argument("--png", default=None,
                        help="where to write the preview (default docs/pepper_depth_"
                             "<timestamp>.png; '' to skip)")
    parser.add_argument("--still", action="store_true",
                        help="disable BasicAwareness and BackgroundMovement first, so "
                             "she stops repointing her own head mid-run. Changes no "
                             "posture -- she stays exactly where she is stood")
    parser.add_argument("--raw-path", default="/tmp/pepper_check_depth.raw")
    args = parser.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if args.png is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        args.png = os.path.join(root, "docs", "pepper_depth_%s.png" % stamp)

    client = ShimClient(args.host, args.port)
    problems = []
    notes = []
    try:
        if client.request("PING") != "PONG":
            print("ERR shim did not answer PING -- is run_shim.sh up?", file=sys.stderr)
            return 1

        if args.head_pose:
            # Head chain only. setStiffnesses("Head", ...) touches HeadYaw and HeadPitch
            # and nothing else, so the arms stay limp and the base stays put.
            print("stiffening the Head chain and posing the head (nothing else moves) ...")
            for command in ("STIFFNESS Head 1.0",
                            "HEAD %.4f %.4f 1.5" % (SIM_HEAD_YAW, SIM_HEAD_PITCH)):
                reply = client.request(command)
                if reply != "OK":
                    print("ERR %s -> %s" % (command, reply), file=sys.stderr)
                    return 1

        head = client.request("GETANGLES HeadYaw,HeadPitch").split()[-1]
        frames = max(1, args.frames)
        print("sampling %d frames ..." % frames)

        shapes = set()
        hashes = []
        coverage = []
        centres = []
        lo_mm, hi_mm = None, None
        last = None

        for index in range(frames):
            if index and args.interval > 0:
                time.sleep(args.interval)
            width, height, frame, digest = grab(client, args.raw_path)
            shapes.add((width, height))
            hashes.append(digest)
            valid = frame != 0
            coverage.append(float(valid.mean()))
            if valid.any():
                low, high = int(frame[valid].min()), int(frame[valid].max())
                lo_mm = low if lo_mm is None else min(lo_mm, low)
                hi_mm = high if hi_mm is None else max(hi_mm, high)
            patch = centre_patch(frame)
            if patch is not None:
                centres.append(patch)
            last = frame
            print("  %2d/%d  valid %5.1f%%  centre %s"
                  % (index + 1, frames, coverage[-1] * 100,
                     "%.3f m" % patch if patch is not None else "no return"))

        print()
        print("-- Depth sensor check --")

        # Head pose --------------------------------------------------------------
        # The depth camera is bolted to the Head link, so the head angles *are* the
        # camera extrinsics. A frame taken at the wrong head pose looks perfectly
        # normal and is aimed along a different ray than the training data -- which is
        # exactly the failure that never announces itself. Step 4 of verify_pepper.sh
        # asserts this too.
        print("Head pose:    %s   (simulated is %.4f,%.4f)"
              % (head, SIM_HEAD_YAW, SIM_HEAD_PITCH))
        pose_matches = None
        try:
            yaw, pitch = (float(v) for v in head.split(","))
            pose_matches = (abs(yaw - SIM_HEAD_YAW) <= HEAD_TOLERANCE
                            and abs(pitch - SIM_HEAD_PITCH) <= HEAD_TOLERANCE)
            offset = "%+.3f rad in yaw, %+.3f in pitch" % (yaw - SIM_HEAD_YAW,
                                                           pitch - SIM_HEAD_PITCH)
            if pose_matches:
                pass
            elif args.head_pose:
                # We asked for the pose and did not get it: that is a real fault, not a
                # consequence of choosing not to move her.
                problems.append(
                    "the head was commanded to the simulated pose but is still %s off. "
                    "The Head chain may not have taken stiffness" % offset)
            else:
                notes.append(
                    "head is %s off the simulated pose, because this run moved nothing. "
                    "Every sensor result below is still valid -- shape, coverage, "
                    "liveness, noise and throughput do not depend on where she is "
                    "looking. What is not valid is treating these frames as aimed like "
                    "the training data. Re-run with --head-pose when you want that; it "
                    "moves the head only" % offset)
        except ValueError:
            notes.append("could not parse the head angles (%r) -- pose not checked"
                         % head)

        # Shape ------------------------------------------------------------------
        width, height = last.shape[1], last.shape[0]
        if len(shapes) > 1:
            problems.append("resolution changed mid-run: %s" % sorted(shapes))
        print("Resolution:   %dx%d   (simulated camera_depth is %dx%d)"
              % (width, height, SIM_WIDTH, SIM_HEIGHT))
        if (width, height) != (SIM_WIDTH, SIM_HEIGHT):
            problems.append("resolution does not match the simulator -- the observation "
                            "shape must be identical across sim and real")

        # Range ------------------------------------------------------------------
        if lo_mm is None:
            problems.append("not one valid pixel in any frame -- the sensor returned "
                            "nothing at all")
            print("Range seen:   nothing")
        else:
            print("Range seen:   %d mm to %d mm   (simulated clip is %d-%d mm)"
                  % (lo_mm, hi_mm, SIM_CLIP_NEAR_M * 1000, SIM_CLIP_FAR_M * 1000))
            if hi_mm > SIM_CLIP_FAR_M * 1000:
                notes.append("returns beyond the simulated far clip (%d mm) will be "
                             "clipped when the observation is built -- expected indoors, "
                             "worth knowing" % (SIM_CLIP_FAR_M * 1000))

        # Coverage ---------------------------------------------------------------
        mean_cov = sum(coverage) / len(coverage)
        print("Valid pixels: %.1f%% mean, %.1f%% worst, %.1f%% best"
              % (mean_cov * 100, min(coverage) * 100, max(coverage) * 100))
        if mean_cov < 0.5:
            problems.append("over half the frame has no depth return. Glossy, dark or "
                            "backlit surfaces (S4.4.4) -- this is a property of the room, "
                            "so consider that when choosing one")
        elif mean_cov < 0.75:
            notes.append("coverage is moderate; a room with fewer glossy surfaces would "
                         "give the policy more to work with")

        # Liveness ---------------------------------------------------------------
        repeats = sum(1 for a, b in zip(hashes, hashes[1:]) if a == b)
        print("Distinct frames: %d of %d" % (len(set(hashes)), len(hashes)))
        # A freeze is only worth calling on enough evidence. Two identical frames in a
        # row is normal when the poll happens to beat the sensor's 15 Hz; four or more
        # with a 0.15 s gap between them is not.
        if frames >= 4 and repeats == frames - 1:
            problems.append("every frame is byte-identical across %d samples -- the "
                            "sensor is frozen or a cached buffer is being returned. Real "
                            "noise makes some pixels differ every frame even on a still "
                            "scene" % frames)
        elif frames < 4 and repeats == frames - 1:
            notes.append("all %d frames identical, but that is too few to distinguish a "
                         "frozen sensor from fast polling -- re-run with --frames 10"
                         % frames)
        elif repeats:
            notes.append("%d consecutive frame pair(s) identical -- frames are being "
                         "polled faster than the sensor produces them" % repeats)

        # Noise ------------------------------------------------------------------
        if len(centres) >= 2:
            array = np.array(centres)
            print("Centre patch: %.3f m mean, %.1f mm std over %d frames"
                  % (array.mean(), array.std() * 1000, array.size))
            if array.std() * 1000 > 50:
                notes.append("centre-patch noise above 50 mm. If the scene really was "
                             "static, that is the observation noise a transfer has to "
                             "tolerate -- worth stating in the write-up")
        elif centres:
            print("Centre patch: %.3f m (one frame only)" % centres[0])
        else:
            problems.append("the centre of the frame never returned depth -- she may be "
                            "facing a window, a mirror, or nothing within range")

        # Preview ----------------------------------------------------------------
        if args.png:
            png_dir = os.path.dirname(args.png)
            if png_dir and not os.path.isdir(png_dir):
                os.makedirs(png_dir)
            print("Preview:      %s" % save_png(last, args.png))

        # Throughput -------------------------------------------------------------
        if args.stream > 0:
            print()
            print("measuring stream throughput for %.0f s ..." % args.stream)
            bridge = os.path.join(root, "tools", "pepper_bridge_node.py")
            line = subprocess.run(
                [sys.executable, bridge, "--host", args.host,
                 "--port", str(args.port), "--probe-depth", str(args.stream)],
                capture_output=True, text=True).stdout.strip()
            parts = line.split()
            if len(parts) == 5 and parts[0] == "STREAM":
                sw, sh, sbytes, sfps = int(parts[1]), int(parts[2]), int(parts[3]), float(parts[4])
                print("Stream:       %dx%d, %d bytes/frame, %.2f fps  (target 15)"
                      % (sw, sh, sbytes, sfps))
                print("Throughput:   %.2f MB/s" % (sfps * sbytes / 1048576.0))
                if sfps < 5:
                    problems.append("depth stream below 5 fps. Prefer ethernet over "
                                    "wifi, and state the real rate in the write-up")
                elif sfps < 12:
                    notes.append("stream is short of 15 fps -- set the control rate from "
                                 "this measured number, not from the target")
            else:
                problems.append("no frames from the depth stream in %.0f s -- single "
                                "frames work but the stream on 45456 does not"
                                % args.stream)

        # Verdict ----------------------------------------------------------------
        print()
        for note in notes:
            print("NOTE:    %s" % note)
        for problem in problems:
            print("PROBLEM: %s" % problem)
        if problems:
            print()
            print("VERDICT: depth sensor is NOT ready to use as an observation source.")
            return 1
        print()
        if pose_matches:
            print("VERDICT: depth sensor is working and matches the simulated camera.")
        else:
            print("VERDICT: depth sensor is working. Frame shape matches the simulated "
                  "camera;")
            print("         the head pose does not, so these frames are not aimed like "
                  "training data.")
        return 0

    except (RuntimeError, TimeoutError) as exc:
        print("ERR %s" % exc, file=sys.stderr)
        if "module destroyed" in str(exc):
            # Worth spelling out: this reads like the robot broke, and it is almost
            # always the network. The shim holds ALProxy objects for its whole life, and
            # they are invalidated by any interruption to the qi session -- most often
            # the laptop silently roaming onto another wifi network mid-session. The
            # proxies do not recover on their own, so the shim has to be restarted; the
            # robot itself needs nothing done to it.
            print("", file=sys.stderr)
            print("This is a stale-proxy error, not a sensor fault. In order:",
                  file=sys.stderr)
            print("  1. check the laptop is still on her network "
                  "(ip -4 addr | grep 192.168.1)", file=sys.stderr)
            print("  2. restart the shim:  pkill -f '[n]aoqi_shim.py'  then run "
                  "tools/run_shim.sh again", file=sys.stderr)
            print("  3. re-run this script", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
    finally:
        client.close()


if __name__ == "__main__":
    sys.exit(main())
