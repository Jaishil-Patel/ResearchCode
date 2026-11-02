#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""Side-car that owns the NAOqi connection to the physical Pepper.

The NAOqi Python SDK is Python 2 only; `rclpy` on Humble is Python 3.10. The two cannot
share a process, so all NAOqi contact is quarantined here and exposed over sockets.
`pepper_bridge_node.py` (Python 3) is the only intended client. This is the same
py2/py3 socket split the CSSR4Africa team fell back on in D3.3 S3.2.2 after the official
`naoqi_driver` proved unreliable -- minus ROS 1, which this project does not use.

Run on the workstation, not on the robot:

    python2 tools/naoqi_shim.py --robot-ip 192.168.1.42

The file is deliberately written to run under **both** Python 2 and Python 3. `naoqi` is
imported lazily, inside PepperLink, so `--fake` works on a machine with no SDK and no
python2 at all:

    python3 tools/naoqi_shim.py --fake

`--fake` synthesises odometry and depth frames. It exists so the entire pipeline -- this
shim, the bridge node, verify_pepper.sh, demo_wave_real.py -- can be rehearsed before the
robot is in front of you. Robot access is the scarce resource; debugging is not something
to do while a robot stands waiting.

Two sockets:

  * UDP `--listen-port` (45455) carries the request/response command protocol below.
  * TCP `--depth-port` (45456) carries the depth video stream. A 320x240 16-bit frame is
    153,600 bytes, which is over the 65,507-byte UDP datagram limit, so this cannot ride
    on the command channel. Frames are length-prefixed:

        struct.pack("!IIHHd", MAGIC, payload_len, width, height, timestamp) + payload

    TCP rather than fragmented UDP because handling reordering and loss by hand would be
    reimplementing TCP, badly.

WSL2 note: the default NAT networking is fine. This process opens *outbound* TCP to
Pepper:9559 and the qi protocol multiplexes replies back over that same socket, so the
robot never has to initiate a connection into WSL. Both sockets above are loopback-only.

Wire protocol -- one ASCII line per datagram, request/response:

    PING                        -> PONG
    VERSION                     -> VERSION <naoqi_version> <battery_percent>
    WAKEUP                      -> OK       (motors on, autonomous life off, head posed)
    REST                        -> OK
    MOVE <vx> <vy> <wz>         -> OK       (m/s, m/s, rad/s -- persistent velocity)
    STOP                        -> OK
    ODOM                        -> ODOM <x> <y> <th> <vx> <vy> <vth>
    HEAD <yaw> <pitch> [secs]   -> OK
    ANGLES <names> <angles> <secs>   -> OK  (comma-separated, radians)
    GETANGLES <names>           -> GETANGLES <a1,a2,...>
    STIFFNESS <chain> <0..1>    -> OK
    DEPTH <path>                -> DEPTH <w> <h> <min_mm> <max_mm> <valid_frac>
    STREAM <on|off>             -> OK
    AWARENESS <on|off>          -> AWARENESS <state> <abilities>  (self-directed motion)
    FPS                         -> FPS <achieved_fps> <frames_sent> <clients>
    SAFETY <on|off>             -> OK       (external collision protection)
    MOVETO <dx> <dy> <dth>      -> OK       (robot frame, non-blocking; NAOqi closes the loop.
                                             For repositioning between routes only)
    MOVING                      -> MOVING <0|1>   (is a MOVETO still running)
    BUMPERS                     -> BUMPERS <n_fr> <n_fl> <n_back> <p_fr> <p_fl> <p_back> <age_s>
                                   (cumulative press counts since start, pressed now 0/1,
                                    seconds since the last successful poll)
    FAKEBUMP <fr|fl|back>       -> OK       (--fake only: a 0.3 s press, for rehearsal)
    QUIT                        -> OK

Any failure returns `ERR <message>` rather than dropping the datagram, so the caller
always learns something.

Safety: `ALMotion.move()` sets a *persistent* velocity -- Pepper keeps driving until she
is told otherwise. If the bridge process dies mid-motion nothing would stop her, so a
watchdog thread halts the base if no MOVE arrives within `--watchdog` seconds. The bridge
re-sends at 10 Hz to keep it fed.
"""

from __future__ import print_function

import argparse
import array
import socket
import struct
import sys
import threading
import time

NAOQI_PORT = 9559

# ALVideoDevice constants. Pepper exposes three cameras; index 2 is the head depth
# sensor. kQVGA is 320x240 and kDepthColorSpace is 16-bit millimetres -- both chosen to
# match the simulated `camera_depth` in pepper_robot.gazebo so the observation the policy
# sees is the same shape in both branches (proposal S4.4.2).
DEPTH_CAMERA_INDEX = 2
RESOLUTION_QVGA = 1
COLORSPACE_DEPTH = 17
DEPTH_FPS = 15
DEPTH_WIDTH = 320
DEPTH_HEIGHT = 240

# The head RGB camera. Index 0 is the top camera; kVGA is 640x480 and kRGBColorSpace is
# three bytes per pixel. Not part of the experiment -- the policy observes depth only
# (proposal S4.4.2) -- but a depth frame is nearly impossible for a human to sanity-check
# on its own, and the RGB view is how you confirm the head is aimed where you think it is.
RGB_CAMERA_INDEX = 0
RESOLUTION_QQVGA = 0
RESOLUTION_VGA = 2
COLORSPACE_RGB = 11
RGB_FPS = 5

# Wi-Fi, not the sensor, sets the achievable frame rate: a VGA frame is 921600 raw
# bytes and 2.4 GHz gives roughly 1.2 MB/s, i.e. about 1.3 fps. QVGA is a quarter of
# that and is what makes a live view feel live. Callers pick per request.
RGB_RESOLUTIONS = (RESOLUTION_QQVGA, RESOLUTION_QVGA, RESOLUTION_VGA)

# The simulated depth camera is rigidly attached to the `Head` link at rpy 0 0 0
# (pepper_robot.gazebo, `ign_depth_camera`), and the simulated head sits at the rest pose
# declared in pepper_robot_description/config/initial_joint_positions.yaml. On the real
# robot head pitch is a live joint that both goToPosture and Autonomous Life will move.
# If it is not driven here and held stiff, the real depth image is taken along a
# different ray than every image the policy trained on -- a zero-shot failure (S4.4.2)
# that produces no error message and no obviously wrong data.
#
# Pitched DOWN for navigation (positive HeadPitch looks down), not StandInit's -0.2, which
# looks ~12 deg UP for facing people. Found 7 Oct 2026: at -0.2 the bottom of the 45-deg
# depth image is only ~11 deg below horizontal, so the floor never appears and a 0.5 m
# object is invisible nearer than ~3.3 m -- in the 2 Oct smoke run Pepper hit the bag while
# her nearest depth return was 1.5-1.9 m away. At 0.45 the floor is in view from ~1 m.
# Must equal HeadPitch in pepper_robot_description/config/initial_joint_positions.yaml.
SIM_HEAD_YAW = 0.0
SIM_HEAD_PITCH = 0.45

# Pepper's three base bumpers, in ALMemory. These are the collision signal on hardware
# (rl/config/env.yaml `collision.source: contact`): the bridge republishes them on
# /pepper_robot/bumper, the topic the simulator's contact sensors publish on.
# The coverage differs and must be declared: in sim, contact sensors cover the base,
# torso, arms and hands; on the robot only the base bumpers exist, so an arm grazing a
# shelf is a collision in sim and invisible here unless the operator logs it by hand.
BUMPER_NAMES = ("fr", "fl", "back")
BUMPER_KEYS = [
    "Device/SubDeviceList/Platform/FrontRight/Bumper/Sensor/Value",
    "Device/SubDeviceList/Platform/FrontLeft/Bumper/Sensor/Value",
    "Device/SubDeviceList/Platform/Back/Bumper/Sensor/Value",
]
BUMPER_POLL_HZ = 20.0

# Frame header for the depth stream: magic, payload length, width, height, timestamp.
FRAME_MAGIC = 0x50455044  # "PEPD"
FRAME_HEADER = "!IIHHd"
FRAME_HEADER_SIZE = struct.calcsize(FRAME_HEADER)


def say(message):
    """Print and flush.

    Python buffers stdout whenever it is not a terminal, so a plain print() vanishes when
    the shim is run as `naoqi_shim.py > shim.log &` -- which is exactly how it is run
    during a session, and exactly when the watchdog and depth messages matter most.
    """
    print(message)
    sys.stdout.flush()


def _bytes_to_u16(payload):
    """Unpack a little-endian uint16 buffer, on either Python 2 or Python 3."""
    pixels = array.array("H")
    if hasattr(pixels, "frombytes"):
        pixels.frombytes(payload)  # Python 3
    else:
        pixels.fromstring(payload)  # Python 2
    if sys.byteorder == "big":
        pixels.byteswap()
    return pixels


def _clamp(value, limit):
    return max(-limit, min(limit, float(value)))


class PepperLink(object):
    """Holds the NAOqi proxies and serialises access to them."""

    def __init__(self, robot_ip, max_lin, max_ang, watchdog):
        # Imported here, not at module scope, so --fake needs no SDK.
        try:
            from naoqi import ALProxy
        except ImportError:
            # Two variables, not one. The SDK's native modules carry no RPATH, so
            # PYTHONPATH alone finds naoqi.py and then fails to load libinaoqi.so --
            # which is why nearly every install guide's one-line PYTHONPATH export is
            # not sufficient on its own.
            sdk = "$HOME/naoqi-sdk/pynaoqi-python2.7-2.5.5.5-linux64"
            sys.stderr.write(
                "ERROR: cannot import `naoqi`.\n"
                "The pynaoqi SDK needs BOTH of these:\n"
                "  export PYTHONPATH=$PYTHONPATH:%s/lib/python2.7/site-packages\n"
                "  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:%s/lib\n"
                "and the system libpython2.7 (sudo apt install python2 libpython2.7).\n"
                "Run tools/pepper_preflight.sh to check. To rehearse without a robot or "
                "an SDK, use --fake.\n" % (sdk, sdk)
            )
            raise

        self.robot_ip = robot_ip
        self.max_lin = max_lin
        self.max_ang = max_ang
        self.watchdog = watchdog

        self._lock = threading.Lock()
        self._video_lock = threading.Lock()
        self._last_move = 0.0
        self._moving = False
        self._depth_handle = None
        self._rgb_handle = None
        self._rgb_resolution = None
        self._head_target = None      # (yaw, pitch) re-asserted while driving; see move()
        self._head_sent = 0.0
        self._goto = False            # a MOVETO is running; see move()

        say("connecting to NAOqi at %s:%d ..." % (robot_ip, NAOQI_PORT))
        # NAOqi's own failure for an unreachable robot is a bare RuntimeError under a wall
        # of qimessaging log lines. On the day, "is she on, and can I reach her" needs to
        # be answerable at a glance.
        try:
            self.motion = ALProxy("ALMotion", robot_ip, NAOQI_PORT)
            self.posture = ALProxy("ALRobotPosture", robot_ip, NAOQI_PORT)
            self.system = ALProxy("ALSystem", robot_ip, NAOQI_PORT)
            self.battery = ALProxy("ALBattery", robot_ip, NAOQI_PORT)
            self.video = ALProxy("ALVideoDevice", robot_ip, NAOQI_PORT)
            # Its own proxy, used only by BumperMonitor's thread, so a bumper poll never
            # waits behind a blocking motion call on self._lock.
            self.memory = ALProxy("ALMemory", robot_ip, NAOQI_PORT)
        except RuntimeError as exc:
            sys.stderr.write(
                "\nCannot reach NAOqi at %s:%d\n"
                "  %s\n\n"
                "Check, in order:\n"
                "  - the robot is on and has finished booting\n"
                "  - the IP is right (press her chest button once; she says it)\n"
                "  - this machine can reach her:  tools/pepper_preflight.sh %s\n"
                % (robot_ip, NAOQI_PORT, exc, robot_ip)
            )
            raise SystemExit(1)
        try:
            self.life = ALProxy("ALAutonomousLife", robot_ip, NAOQI_PORT)
        except Exception:
            self.life = None  # not present on every NAOqi build
        say("connected: NAOqi %s" % self.system.systemVersion())

    # -- identity ----------------------------------------------------------------

    def version(self):
        return self.system.systemVersion(), int(self.battery.getBatteryCharge())

    # -- lifecycle ---------------------------------------------------------------

    def wake_up(self):
        """Enable motors and put Pepper in a known state ready to drive.

        Autonomous Life is disabled first. Left on, Pepper reacts to faces and sound and
        issues her own motion commands, which would fight the policy and silently corrupt
        every measurement taken through this shim.

        The head is posed last, and deliberately after goToPosture -- StandInit sets its
        own head angles, so doing this earlier would be undone. See SIM_HEAD_PITCH.
        """
        if self.life is not None:
            try:
                if self.life.getState() != "disabled":
                    self.life.setState("disabled")
            except Exception as exc:
                say("warning: could not disable Autonomous Life: %s" % exc)
        self.motion.wakeUp()
        self.posture.goToPosture("StandInit", 0.5)
        self.motion.moveInit()
        # Smart stiffness lowers the stiffness of joints that are not being commanded.
        # Measured 15 Oct 2026: with it on, the head posed at (0, 0.45) swung to yaw -0.12
        # rad as the base accelerated and stayed 0.06 off after stopping -- the depth camera
        # looking 7 deg sideways of where the policy thinks it is. Off, plus the re-assert
        # in move(), keeps the camera where the simulated one is.
        try:
            self.motion.setSmartStiffnessEnabled(False)
        except Exception as exc:
            say("warning: could not disable smart stiffness: %s" % exc)
        self.head(SIM_HEAD_YAW, SIM_HEAD_PITCH)
        self._head_target = (SIM_HEAD_YAW, SIM_HEAD_PITCH)

    def rest(self):
        self._head_target = None
        self.stop()
        self.motion.rest()

    def awareness(self, enabled):
        """Turn Pepper's self-directed head and body motion on or off without moving her.

        Deliberately not `setState("disabled")`, which is what wake_up() uses: that stops
        Autonomous Life wholesale and takes her to a rest posture on the way, a
        whole-body motion that is no use in a room nobody has cleared. The two abilities
        below are the ones that actually make the head wander -- BasicAwareness turns her
        toward faces and sound, BackgroundMovement adds idle sway -- and disabling them
        changes no posture at all. She simply stops moving herself, where she stands.

        Worth doing before any depth measurement taken without a wake-up. The head is the
        depth camera's extrinsics, so a head that repoints itself mid-run silently
        changes what is being measured, and the frames look perfectly fine afterwards.
        """
        if self.life is None:
            raise RuntimeError("ALAutonomousLife is not available on this NAOqi build")
        changed = []
        for ability in ("BasicAwareness", "BackgroundMovement"):
            try:
                self.life.setAutonomousAbilityEnabled(ability, bool(enabled))
                changed.append(ability)
            except Exception as exc:
                say("warning: could not set %s: %s" % (ability, exc))
        if not changed:
            raise RuntimeError("no autonomous ability could be set")
        return changed

    def shutdown(self):
        """Power the robot off at the OS level.

        Deliberately stops the base and rests the motors first: ALSystem.shutdown() cuts
        power without putting the joints anywhere safe, and a Pepper that loses power with
        her arms out drops them under gravity.

        There is no remote undo. Once this returns, bringing her back needs a physical
        press of the chest button.
        """
        try:
            self.stop()
            self.motion.rest()
        except Exception as exc:
            say("shutdown: could not rest before powering off: %s" % exc)
        self.system.shutdown()

    # -- base control ------------------------------------------------------------

    def move(self, vx, vy, wz):
        """Command a persistent base velocity, clamped to the configured limits.

        Clamping is duplicated here and in the bridge on purpose: the bridge protects the
        experiment, this protects the robot and whoever is stood next to it
        (proposal S4.4.5, conservative velocity limits).
        """
        vx = _clamp(vx, self.max_lin)
        vy = _clamp(vy, self.max_lin)
        wz = _clamp(wz, self.max_ang)
        with self._lock:
            # The bridge re-sends the last cmd_vel at 10 Hz, which between routes is zero.
            # move(0, 0, 0) would cancel a running MOVETO, so zeros are dropped while one
            # runs; any real command (or STOP) still takes over immediately.
            if (self._goto and vx == 0.0 and vy == 0.0 and wz == 0.0
                    and self.motion.moveIsActive()):
                return
            self._goto = False
            self.motion.move(vx, vy, wz)
            now = time.time()
            self._last_move = now
            self._moving = not (vx == 0.0 and vy == 0.0 and wz == 0.0)
            # Hold the head at the camera pose while driving: MOVE arrives at 10 Hz, so
            # this re-sends the head target at ~2 Hz. setAngles is non-blocking, unlike
            # angleInterpolation, so it never delays the base command.
            if self._head_target is not None and now - self._head_sent > 0.5:
                self.motion.setAngles(["HeadYaw", "HeadPitch"], list(self._head_target), 0.15)
                self._head_sent = now

    def stop(self):
        with self._lock:
            self.motion.stopMove()
            self._moving = False
            self._goto = False

    def move_to(self, dx, dy, dth):
        """Start a relative move in the robot frame and return at once.

        ALMotion.moveTo closes the loop on the robot, so the 0.8 s command latency of the
        laptop->shim->NAOqi chain does not turn into overshoot the way it would if the
        laptop steered. Only for repositioning between routes -- never during a scored
        route, where the policy drives through MOVE exactly as in simulation.
        """
        with self._lock:
            self.motion.post.moveTo(float(dx), float(dy), float(dth))
            self._goto = True
            self._moving = False          # MOVETO is not a persistent velocity

    def goto_active(self):
        with self._lock:
            active = bool(self.motion.moveIsActive())
            if not active:
                self._goto = False
            return active

    def odometry(self):
        """Pose and velocity in NAOqi's own odometry frame.

        `getRobotPosition(True)` uses sensor feedback rather than the commanded value.
        This frame is set at wakeUp and drifts -- quantifying that drift is the point of
        the loop-closure step in verify_pepper.sh, and it is why proposal S4.4.6 flags an
        external pose reference as needing early confirmation.
        """
        with self._lock:
            x, y, theta = self.motion.getRobotPosition(True)
            vx, vy, vth = self.motion.getRobotVelocity()
        return x, y, theta, vx, vy, vth

    def read_bumpers(self):
        """Pressed state of (front-right, front-left, back), one ALMemory round trip."""
        return tuple(float(v) > 0.5 for v in self.memory.getListData(BUMPER_KEYS))

    def set_safety(self, enabled):
        """Toggle NAOqi's built-in external collision protection for the base.

        Leave this ON for ordinary testing. It matters for the experiment: with
        protection enabled NAOqi refuses motion near obstacles, so the robot cannot
        collide -- which makes the collision rate of Eq. 4.5 measure NAOqi's safety
        layer rather than the policy. Whichever way it is set for the real-world trials,
        it has to be set the same way for both branches and stated in the write-up.
        """
        self.motion.setExternalCollisionProtectionEnabled("Move", bool(enabled))

    # -- joints ------------------------------------------------------------------

    def head(self, yaw, pitch, duration=1.0):
        self.angles(["HeadYaw", "HeadPitch"], [yaw, pitch], duration)

    def angles(self, names, values, duration=1.0):
        """Drive named joints to absolute angles over `duration` seconds.

        A duration rather than NAOqi's `angleInterpolationWithSpeed` fraction, because
        the waypoints this replays come from tools/demo_wave.py, where they are expressed
        as `time_from_start` on a JointTrajectory. Keeping the same units means the same
        motion happens in simulation and on the robot.

        Joints are stiffened first. A joint with zero stiffness accepts the command and
        then sags under gravity -- for the head, that means the depth camera quietly
        drifts off the pose the simulated camera is fixed at.

        This blocks until the motion completes, which is why ShimClient gives joint
        commands a long timeout and no retries.
        """
        duration = max(0.1, float(duration))
        with self._lock:
            self.motion.setStiffnesses(names, [1.0] * len(names))
            self.motion.angleInterpolation(
                list(names),
                [[float(v)] for v in values],
                [[duration]] * len(names),
                True,  # absolute angles
            )

    def stiffness(self, chain, value):
        with self._lock:
            self.motion.setStiffnesses(chain, float(value))

    def joint_angles(self, names):
        with self._lock:
            return self.motion.getAngles(names, True)

    # -- perception --------------------------------------------------------------

    def grab_depth(self):
        """Return one depth frame as (width, height, raw 16-bit millimetre buffer).

        The subscription is opened once and kept. Subscribing per frame is slow, and
        NAOqi has a finite number of subscriber slots which leak when a client dies --
        the classic symptom is subscribeCamera starting to fail after a few crashes,
        fixed only by restarting NAOqi on the robot.
        """
        with self._video_lock:
            if self._depth_handle is None:
                self._depth_handle = self.video.subscribeCamera(
                    "pepper_shim_depth",
                    DEPTH_CAMERA_INDEX,
                    getattr(self, "depth_resolution", RESOLUTION_QVGA),
                    COLORSPACE_DEPTH,
                    DEPTH_FPS,
                )
            frame = self.video.getImageRemote(self._depth_handle)
            if frame is None:
                raise RuntimeError("getImageRemote returned None")
            try:
                return frame[0], frame[1], frame[6]
            finally:
                try:
                    self.video.releaseImage(self._depth_handle)
                except Exception:
                    pass

    def grab_rgb(self, resolution=RESOLUTION_VGA):
        """Return one RGB frame as (width, height, layers, raw byte buffer).

        Subscribed once and kept, for the same subscriber-slot reason as grab_depth: the
        slots leak when a client dies, and the symptom is subscribeCamera failing later
        for reasons that look nothing like the cause. Changing resolution has to tear the
        subscription down and build a new one -- ALVideoDevice fixes the frame size at
        subscribe time -- so switching every frame would be pathological. Callers pick a
        resolution and stay there.
        """
        with self._video_lock:
            if self._rgb_handle is not None and self._rgb_resolution != resolution:
                try:
                    self.video.unsubscribe(self._rgb_handle)
                except Exception:
                    pass
                self._rgb_handle = None
            if self._rgb_handle is None:
                self._rgb_handle = self.video.subscribeCamera(
                    "pepper_shim_rgb",
                    RGB_CAMERA_INDEX,
                    resolution,
                    COLORSPACE_RGB,
                    RGB_FPS,
                )
                self._rgb_resolution = resolution
            frame = self.video.getImageRemote(self._rgb_handle)
            if frame is None:
                raise RuntimeError("getImageRemote returned None")
            try:
                return frame[0], frame[1], frame[2], frame[6]
            finally:
                try:
                    self.video.releaseImage(self._rgb_handle)
                except Exception:
                    pass

    def release_depth(self):
        with self._video_lock:
            if self._depth_handle is not None:
                try:
                    self.video.unsubscribe(self._depth_handle)
                except Exception:
                    pass
                self._depth_handle = None
            if self._rgb_handle is not None:
                try:
                    self.video.unsubscribe(self._rgb_handle)
                except Exception:
                    pass
                self._rgb_handle = None

    # -- watchdog ----------------------------------------------------------------

    def watchdog_loop(self, stop_event):
        while not stop_event.is_set():
            if self._moving and (time.time() - self._last_move) > self.watchdog:
                say("watchdog: no MOVE for %.2fs -- stopping base" % self.watchdog)
                try:
                    self.stop()
                except Exception as exc:
                    say("watchdog: stopMove failed: %s" % exc)
            stop_event.wait(0.1)


class FakeLink(object):
    """Stand-in for PepperLink with no robot and no SDK.

    Not a mock in the testing sense -- it is a rehearsal rig. The numbers it returns are
    deliberately imperfect so that the arithmetic downstream is exercised on realistic
    input: odometry under-reports slightly and accumulates a yaw bias, and the depth frame
    contains invalid pixels. A fake that returned exact values would let a broken
    verify_pepper.sh calculation pass unnoticed.
    """

    # Achieved/commanded ratio and per-second yaw bias. Roughly what a wheeled base on a
    # hard floor does; the real values are what step 4 and step 5 of verify_pepper.sh go
    # to the robot to measure.
    SLIP = 0.93
    YAW_BIAS = 0.01

    def __init__(self, max_lin, max_ang, watchdog):
        self.max_lin = max_lin
        self.max_ang = max_ang
        self.watchdog = watchdog

        self._lock = threading.Lock()
        self._last_move = 0.0
        self._moving = False
        self._target = (0.0, 0.0, 0.0)
        self._pose = [0.0, 0.0, 0.0]
        self._updated = time.time()
        self._awake = False
        self._joints = {"HeadYaw": 0.0, "HeadPitch": 0.0}
        self._safety = True
        self._bump_until = [0.0, 0.0, 0.0]
        say("FAKE MODE: no robot, no SDK. Synthesising odometry and depth.")

    def _integrate(self):
        """Advance the pose to now. Called by every reader, so odometry is continuous."""
        now = time.time()
        dt = now - self._updated
        self._updated = now
        if not self._moving:
            return
        vx, vy, wz = self._target
        theta = self._pose[2]
        import math

        self._pose[0] += self.SLIP * (vx * math.cos(theta) - vy * math.sin(theta)) * dt
        self._pose[1] += self.SLIP * (vx * math.sin(theta) + vy * math.cos(theta)) * dt
        self._pose[2] += (self.SLIP * wz + self.YAW_BIAS) * dt

    def version(self):
        return "2.5.10.7-FAKE", 87

    def wake_up(self):
        self._awake = True
        self.head(SIM_HEAD_YAW, SIM_HEAD_PITCH)

    def rest(self):
        self.stop()
        self._awake = False

    def awareness(self, enabled):
        # FakeLink never moves its own head, so there is nothing to disable.
        return ["BasicAwareness", "BackgroundMovement"]

    def shutdown(self):
        say("FAKE MODE: SHUTDOWN would power the robot off here.")
        self._awake = False

    def move(self, vx, vy, wz):
        with self._lock:
            self._integrate()
            self._target = (
                _clamp(vx, self.max_lin),
                _clamp(vy, self.max_lin),
                _clamp(wz, self.max_ang),
            )
            self._last_move = time.time()
            self._moving = self._target != (0.0, 0.0, 0.0)

    def stop(self):
        with self._lock:
            self._integrate()
            self._target = (0.0, 0.0, 0.0)
            self._moving = False
            self._goto_until = 0.0

    def move_to(self, dx, dy, dth):
        # The fake moves the pose at once (with its usual slip) and reports busy for as
        # long as the real move would take at 0.2 m/s and 0.5 rad/s.
        import math
        with self._lock:
            self._integrate()
            th = self._pose[2]
            self._pose[0] += self.SLIP * (dx * math.cos(th) - dy * math.sin(th))
            self._pose[1] += self.SLIP * (dx * math.sin(th) + dy * math.cos(th))
            self._pose[2] += self.SLIP * dth
            self._goto_until = time.time() + math.hypot(dx, dy) / 0.2 + abs(dth) / 0.5

    def goto_active(self):
        return time.time() < getattr(self, "_goto_until", 0.0)

    def odometry(self):
        with self._lock:
            self._integrate()
            vx, vy, wz = self._target if self._moving else (0.0, 0.0, 0.0)
            return (
                self._pose[0],
                self._pose[1],
                self._pose[2],
                self.SLIP * vx,
                self.SLIP * vy,
                self.SLIP * wz,
            )

    def set_safety(self, enabled):
        self._safety = bool(enabled)

    def read_bumpers(self):
        now = time.time()
        return tuple(now < t for t in self._bump_until)

    def fake_bump(self, which, seconds=0.3):
        """Press one bumper briefly: long enough for a BumperMonitor poll to see it, short
        enough to exercise "pressed and released between two bridge reads"."""
        self._bump_until[BUMPER_NAMES.index(which)] = time.time() + seconds

    def head(self, yaw, pitch, duration=1.0):
        self.angles(["HeadYaw", "HeadPitch"], [yaw, pitch], duration)

    def angles(self, names, values, duration=1.0):
        # Sleeps like the real interpolation does, so demo_wave_real.py's pacing and the
        # client's long joint timeout are both exercised in fake mode.
        time.sleep(min(float(duration), 3.0))
        with self._lock:
            for name, value in zip(names, values):
                self._joints[name] = float(value)

    def stiffness(self, chain, value):
        pass

    def joint_angles(self, names):
        with self._lock:
            return [self._joints.get(n, 0.0) for n in names]

    def grab_depth(self):
        """A horizontal distance ramp with two dead patches.

        The ramp spans the simulated clip range so the bridge's millimetre-to-metre
        conversion and far-clip masking are both exercised; the dead patches make the
        valid-pixel fraction genuinely less than 1, which is what a real depth sensor
        returns off glossy or dark surfaces (proposal S4.4.4).
        """
        if getattr(self, "depth_resolution", RESOLUTION_QVGA) == RESOLUTION_QQVGA:
            # Same scene at half size, as the real camera would deliver it.
            w, h, full = 160, 120, _bytes_to_u16(self._qvga_frame())
            small = array.array("H", [full[(2 * r) * DEPTH_WIDTH + 2 * c]
                                      for r in range(h) for c in range(w)])
            if sys.byteorder == "big":
                small.byteswap()
            return w, h, small.tobytes() if hasattr(small, "tobytes") else small.tostring()
        return DEPTH_WIDTH, DEPTH_HEIGHT, self._qvga_frame()

    def _qvga_frame(self):
        pixels = array.array("H", [0]) * (DEPTH_WIDTH * DEPTH_HEIGHT)
        phase = int(time.time() * 30) % DEPTH_WIDTH
        for row in range(DEPTH_HEIGHT):
            base = row * DEPTH_WIDTH
            for col in range(DEPTH_WIDTH):
                column = (col + phase) % DEPTH_WIDTH
                # 300..8300 mm, i.e. deliberately outside the 400-8000 clip at both ends.
                pixels[base + col] = 300 + (column * 8000) // DEPTH_WIDTH
        for row in range(60, 100):  # dead patch: no return
            for col in range(40, 120):
                pixels[row * DEPTH_WIDTH + col] = 0
        for row in range(150, 180):
            for col in range(200, 260):
                pixels[row * DEPTH_WIDTH + col] = 0
        if sys.byteorder == "big":
            pixels.byteswap()
        return pixels.tobytes() if hasattr(pixels, "tobytes") else pixels.tostring()

    def grab_rgb(self, resolution=RESOLUTION_VGA):
        """A colour gradient, so --fake exercises the same write path as the robot."""
        sizes = {RESOLUTION_QQVGA: (160, 120), RESOLUTION_QVGA: (320, 240),
                 RESOLUTION_VGA: (640, 480)}
        width, height = sizes.get(resolution, (320, 240))
        layers = 3
        buf = array.array("B", [0]) * (width * height * layers)
        phase = int(time.time() * 30) % width
        for row in range(height):
            for col in range(width):
                index = (row * width + col) * layers
                buf[index] = (col + phase) % 256
                buf[index + 1] = (row * 255) // height
                buf[index + 2] = 128
        payload = buf.tobytes() if hasattr(buf, "tobytes") else buf.tostring()
        return width, height, layers, payload

    def release_depth(self):
        pass

    def watchdog_loop(self, stop_event):
        while not stop_event.is_set():
            if self._moving and (time.time() - self._last_move) > self.watchdog:
                say("watchdog: no MOVE for %.2fs -- stopping base" % self.watchdog)
                self.stop()
            stop_event.wait(0.1)


class BumperMonitor(object):
    """Polls the bumpers on the shim side of the link and counts presses.

    Counts, not a latch cleared on read. A cleared-on-read latch loses a collision
    whenever a BUMPERS reply is lost or retried, or when someone types
    `pepper_bridge_node.py --raw BUMPERS` mid-session -- each read would steal the event
    from the bridge. A cumulative count has no reader to steal from: the bridge compares
    against the last count it saw, and any number of readers can look.

    Polled here at BUMPER_POLL_HZ rather than per bridge request, so a press shorter than
    the bridge's period is still counted. One shorter than a poll interval (50 ms) can
    still be missed; that is the remaining limit of polling over the network.
    """

    def __init__(self, link, hz=BUMPER_POLL_HZ):
        self.link = link
        self.period = 1.0 / hz
        self._lock = threading.Lock()
        self._counts = [0, 0, 0]
        self._pressed = (False, False, False)
        self._last_ok = None
        self._stop = threading.Event()
        self._warned = False

    def start(self):
        thread = threading.Thread(target=self._run)
        thread.daemon = True
        thread.start()

    def stop(self):
        self._stop.set()

    def snapshot(self):
        """(counts, pressed, seconds since the last good poll -- None if never)."""
        with self._lock:
            age = None if self._last_ok is None else time.time() - self._last_ok
            return list(self._counts), self._pressed, age

    def _run(self):
        while not self._stop.is_set():
            try:
                now = self.link.read_bumpers()
            except Exception as exc:
                if not self._warned:
                    say("bumpers: poll failed: %s (repeats suppressed)" % exc)
                    self._warned = True
                self._stop.wait(0.5)
                continue
            with self._lock:
                for i, (was, is_) in enumerate(zip(self._pressed, now)):
                    if is_ and not was:
                        self._counts[i] += 1
                        say("bumper %s pressed (count %d)" % (BUMPER_NAMES[i], self._counts[i]))
                self._pressed = tuple(now)
                self._last_ok = time.time()
                self._warned = False
            self._stop.wait(self.period)


class DepthStream(object):
    """TCP fan-out of depth frames, plus the one-shot grab and its statistics.

    Kept separate from the link so PepperLink and FakeLink only have to provide
    `grab_depth()`.
    """

    def __init__(self, link, host, port, fps):
        self.link = link
        self.address = (host, port)
        self.period = 1.0 / float(fps)

        self._clients = []
        self._clients_lock = threading.Lock()
        self._streaming = threading.Event()
        self._stop = threading.Event()
        self._frames = 0
        self._achieved_fps = 0.0

        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind(self.address)
        self.server.listen(4)

    def start(self):
        for target in (self._accept_loop, self._produce_loop):
            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()

    def set_streaming(self, enabled):
        if enabled:
            self._streaming.set()
        else:
            self._streaming.clear()

    def stats(self):
        with self._clients_lock:
            clients = len(self._clients)
        return self._achieved_fps, self._frames, clients

    def snapshot(self, path):
        """Grab one frame, write the raw 16-bit buffer, return simple statistics.

        Raw rather than PNG so this stays dependency-free on the Python 2 side. Values
        are millimetres; 0 means "no return", which is normal for glossy or dark
        surfaces and is exactly the artefact class proposal S4.4.4 anticipates.
        """
        width, height, payload = self.link.grab_depth()
        with open(path, "wb") as handle:
            handle.write(payload)

        pixels = _bytes_to_u16(payload)
        valid = [p for p in pixels if p > 0]
        low, high = (min(valid), max(valid)) if valid else (0, 0)
        fraction = float(len(valid)) / float(len(pixels)) if len(pixels) else 0.0
        return width, height, low, high, fraction

    def _accept_loop(self):
        while not self._stop.is_set():
            try:
                client, peer = self.server.accept()
            except Exception:
                if self._stop.is_set():
                    return
                continue
            client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            say("depth: client connected from %s:%d" % peer)
            with self._clients_lock:
                self._clients.append(client)

    def _produce_loop(self):
        window_start = time.time()
        window_frames = 0
        while not self._stop.is_set():
            if not self._streaming.wait(0.2):
                window_start, window_frames = time.time(), 0
                continue
            with self._clients_lock:
                idle = not self._clients
            if idle:
                # Nobody is listening; do not pull 2.3 MB/s off the robot for nothing.
                time.sleep(0.2)
                continue

            started = time.time()
            try:
                width, height, payload = self.link.grab_depth()
            except Exception as exc:
                say("depth: grab failed: %s" % exc)
                time.sleep(0.5)
                continue

            header = struct.pack(
                FRAME_HEADER, FRAME_MAGIC, len(payload), width, height, started
            )
            self._broadcast(header + payload)

            self._frames += 1
            window_frames += 1
            elapsed = time.time() - window_start
            if elapsed >= 1.0:
                self._achieved_fps = window_frames / elapsed
                window_start, window_frames = time.time(), 0

            # Pace to the requested rate. If getImageRemote is already slower than the
            # period -- likely over wifi, where 320x240x2 at 15 Hz is 2.3 MB/s -- this
            # sleeps zero and the achieved rate is whatever the link sustains.
            remaining = self.period - (time.time() - started)
            if remaining > 0:
                time.sleep(remaining)

    def _broadcast(self, blob):
        with self._clients_lock:
            clients = list(self._clients)
        for client in clients:
            try:
                client.sendall(blob)
            except Exception:
                say("depth: client disconnected")
                try:
                    client.close()
                except Exception:
                    pass
                with self._clients_lock:
                    if client in self._clients:
                        self._clients.remove(client)

    def close(self):
        self._stop.set()
        self._streaming.clear()
        with self._clients_lock:
            for client in self._clients:
                try:
                    client.close()
                except Exception:
                    pass
            self._clients = []
        try:
            self.server.close()
        except Exception:
            pass
        self.link.release_depth()


def handle(link, depth, line, bumpers=None):
    """Turn one request line into one response line."""
    parts = line.strip().split()
    if not parts:
        return "ERR empty"
    verb = parts[0].upper()

    if verb == "PING":
        return "PONG"

    if verb == "VERSION":
        version, battery = link.version()
        return "VERSION %s %d" % (version, battery)

    if verb == "WAKEUP":
        link.wake_up()
        return "OK"

    if verb == "REST":
        link.rest()
        return "OK"

    if verb == "MOVE":
        if len(parts) != 4:
            return "ERR usage: MOVE <vx> <vy> <wz>"
        link.move(float(parts[1]), float(parts[2]), float(parts[3]))
        return "OK"

    if verb == "STOP":
        link.stop()
        return "OK"

    if verb == "ODOM":
        x, y, theta, vx, vy, vth = link.odometry()
        return "ODOM %.6f %.6f %.6f %.6f %.6f %.6f" % (x, y, theta, vx, vy, vth)

    if verb == "HEAD":
        if len(parts) not in (3, 4):
            return "ERR usage: HEAD <yaw> <pitch> [seconds]"
        duration = float(parts[3]) if len(parts) == 4 else 1.0
        link.head(float(parts[1]), float(parts[2]), duration)
        return "OK"

    if verb == "ANGLES":
        if len(parts) != 4:
            return "ERR usage: ANGLES <names_csv> <angles_csv> <seconds>"
        names = parts[1].split(",")
        values = [float(v) for v in parts[2].split(",")]
        if len(names) != len(values):
            return "ERR ANGLES: %d names but %d angles" % (len(names), len(values))
        link.angles(names, values, float(parts[3]))
        return "OK"

    if verb == "GETANGLES":
        if len(parts) != 2:
            return "ERR usage: GETANGLES <names_csv>"
        values = link.joint_angles(parts[1].split(","))
        return "GETANGLES %s" % ",".join("%.4f" % v for v in values)

    if verb == "STIFFNESS":
        if len(parts) != 3:
            return "ERR usage: STIFFNESS <chain> <0..1>"
        link.stiffness(parts[1], float(parts[2]))
        return "OK"

    if verb == "DEPTH":
        if len(parts) != 2:
            return "ERR usage: DEPTH <output_path>"
        w, h, low, high, frac = depth.snapshot(parts[1])
        return "DEPTH %d %d %d %d %.4f" % (w, h, low, high, frac)

    if verb == "RGB":
        if len(parts) not in (2, 3):
            return "ERR usage: RGB <output_path> [resolution 0|1|2]"
        resolution = int(parts[2]) if len(parts) == 3 else RESOLUTION_VGA
        if resolution not in RGB_RESOLUTIONS:
            return "ERR RGB: resolution must be one of %s" % (RGB_RESOLUTIONS,)
        width, height, layers, payload = link.grab_rgb(resolution)
        with open(parts[1], "wb") as out:
            out.write(payload)
        return "RGB %d %d %d %d" % (width, height, layers, len(payload))

    if verb == "AWARENESS":
        if len(parts) != 2 or parts[1].lower() not in ("on", "off"):
            return "ERR usage: AWARENESS <on|off>"
        state = parts[1].lower()
        changed = link.awareness(state == "on")
        return "AWARENESS %s %s" % (state, ",".join(changed))

    if verb == "STREAM":
        if len(parts) != 2 or parts[1].lower() not in ("on", "off"):
            return "ERR usage: STREAM <on|off>"
        depth.set_streaming(parts[1].lower() == "on")
        return "OK"

    if verb == "FPS":
        achieved, frames, clients = depth.stats()
        return "FPS %.2f %d %d" % (achieved, frames, clients)

    if verb == "SAFETY":
        if len(parts) != 2 or parts[1].lower() not in ("on", "off"):
            return "ERR usage: SAFETY <on|off>"
        link.set_safety(parts[1].lower() == "on")
        return "OK"

    if verb == "MOVETO":
        if len(parts) != 4:
            return "ERR usage: MOVETO <dx> <dy> <dth>"
        link.move_to(float(parts[1]), float(parts[2]), float(parts[3]))
        return "OK"

    if verb == "MOVING":
        return "MOVING %d" % int(link.goto_active())

    if verb == "BUMPERS":
        if bumpers is None:
            return "ERR bumper monitor not running"
        counts, pressed, age = bumpers.snapshot()
        if age is None:
            return "ERR bumpers never read -- ALMemory unreachable?"
        return "BUMPERS %d %d %d %d %d %d %.2f" % (
            tuple(counts) + tuple(int(x) for x in pressed) + (age,))

    if verb == "FAKEBUMP":
        if not hasattr(link, "fake_bump"):
            return "ERR FAKEBUMP is --fake only"
        if len(parts) != 2 or parts[1].lower() not in BUMPER_NAMES:
            return "ERR usage: FAKEBUMP <fr|fl|back>"
        link.fake_bump(parts[1].lower())
        return "OK"

    if verb == "SHUTDOWN":
        # No remote undo -- she comes back only via the chest button.
        link.shutdown()
        return "OK"

    if verb == "QUIT":
        return "OK"

    return "ERR unknown verb %s" % verb


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--robot-ip",
                        help="Pepper's IP -- press the chest button once and she says it")
    source.add_argument("--fake", action="store_true",
                        help="no robot: synthesise odometry and depth, for rehearsal")
    parser.add_argument("--listen-host", default="127.0.0.1")
    parser.add_argument("--listen-port", type=int, default=45455)
    parser.add_argument("--depth-port", type=int, default=45456)
    parser.add_argument("--depth-fps", type=float, default=DEPTH_FPS,
                        help="target depth stream rate; the achieved rate may be lower")
    # Hard backstop at the limits the policy is TRAINED with (rl/config/env.yaml `action`:
    # v_max_mps 0.30, omega_max_rps 0.90). Keep the two in step by hand -- this side is
    # Python 2 with no YAML reader. A lower clamp here silently slows every fast command
    # the policy learned (it was 0.25 / 0.6 until 1 Oct 2026); pepper_bridge_node.py reads
    # env.yaml itself and clamps first. NAOqi's own limit is 0.55 m/s.
    parser.add_argument("--max-lin", type=float, default=0.30,
                        help="hard clamp on linear velocity, m/s (= env.yaml v_max_mps)")
    parser.add_argument("--max-ang", type=float, default=0.90,
                        help="hard clamp on angular velocity, rad/s (= env.yaml omega_max_rps)")
    parser.add_argument("--depth-res", choices=["qvga", "qqvga"], default="qvga",
                        help="depth camera resolution. qqvga (160x120) is a quarter of the bytes, "
                             "for links too slow for 320x240 (15 Oct 2026); the bridge then "
                             "doubles each pixel back to 320x240, so the env sees the same frame "
                             "size and the full field of view")
    parser.add_argument("--bumper-hz", type=float, default=BUMPER_POLL_HZ,
                        help="bumper polling rate. Each poll is one NAOqi round trip; on a slow "
                             "link (15 Oct 2026, D-Link 2.4 GHz) 20 Hz helped starve MOVE, so 10")
    parser.add_argument("--watchdog", type=float, default=0.75,
                        help="stop the base if no MOVE arrives within this many seconds")
    args = parser.parse_args()

    if args.fake:
        link = FakeLink(args.max_lin, args.max_ang, args.watchdog)
    else:
        link = PepperLink(args.robot_ip, args.max_lin, args.max_ang, args.watchdog)
    link.depth_resolution = RESOLUTION_QQVGA if args.depth_res == "qqvga" else RESOLUTION_QVGA

    stop_event = threading.Event()
    dog = threading.Thread(target=link.watchdog_loop, args=(stop_event,))
    dog.daemon = True
    dog.start()

    depth = DepthStream(link, args.listen_host, args.depth_port, args.depth_fps)
    depth.start()

    bumpers = BumperMonitor(link, hz=args.bumper_hz)
    bumpers.start()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.listen_host, args.listen_port))
    say("commands on %s:%d, depth on %s:%d -- Ctrl-C to exit" % (
        args.listen_host, args.listen_port, args.listen_host, args.depth_port))

    try:
        while True:
            data, peer = sock.recvfrom(4096)
            # Optional request tag "#<n> ", echoed on the reply so the client can match
            # replies to requests (see ShimClient.request in pepper_bridge_node.py).
            # Untagged requests work exactly as before.
            tag = ""
            stripped = data.strip()
            if stripped.startswith(b"#") and b" " in stripped:
                tag, data = stripped.split(b" ", 1)
                tag = tag.decode("utf-8") + " "
            try:
                text = data.decode("utf-8")
                if str is bytes:
                    # Python 2: NAOqi's bindings reject unicode and want native
                    # str. Joint names arrive off the wire, so ANGLES, GETANGLES
                    # and STIFFNESS all fail with "collection contains invalid or
                    # non-string items" without this. Only reproducible on the
                    # robot -- FakeLink never calls into NAOqi.
                    text = text.encode("utf-8")
                reply = handle(link, depth, text, bumpers)
            except Exception as exc:
                reply = "ERR %s" % str(exc).replace("\n", " ")
            sock.sendto((tag + reply).encode("utf-8"), peer)
            if data.strip().upper().startswith(b"QUIT"):
                break
    except KeyboardInterrupt:
        say("\ninterrupted")
    finally:
        stop_event.set()
        bumpers.stop()
        try:
            link.stop()
        except Exception:
            pass
        depth.close()
        sock.close()
        say("base stopped, sockets closed")


if __name__ == "__main__":
    main()
