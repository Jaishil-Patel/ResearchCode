#!/usr/bin/env python3
"""ROS 2 node that makes the physical Pepper answer to the same interface as the sim.

This is the file the experiment actually depends on. A policy trained in Ignition
publishes `geometry_msgs/Twist` on `/pepper_robot/cmd_vel` and reads depth from
`/pepper_robot/depth_camera/image_raw`; this node subscribes and publishes those exact
topics against the real robot, so the policy cannot tell which one is on the other end.
That is what proposal S2.6 means by keeping "the same ROS interface in simulation and
reality", and it is the precondition for the zero-shot deployment of S4.4.2 -- no
real-world fine-tuning, so the input/output contract has to match exactly.

The topic names and the message encodings below are not arbitrary; they are taken from
`pepper_robot_ign/launch/bridge.launch.py` and `pepper_robot_description/urdf/
pepper_robot.gazebo`. If either changes, this file has to change with it, and the check
is `ros2 topic list` against a running simulator -- not reading the launch file.

It talks to `naoqi_shim.py` over sockets because the NAOqi SDK is Python 2 and cannot be
imported here. Start the shim first:

    python2 tools/naoqi_shim.py --robot-ip 192.168.1.42     # terminal 1
    python3 tools/pepper_bridge_node.py                     # terminal 2

Or, with no robot and no SDK, rehearse the whole thing:

    python3 tools/naoqi_shim.py --fake                      # terminal 1
    python3 tools/pepper_bridge_node.py                     # terminal 2

A `--raw` mode is also provided for talking to the shim without ROS, which
verify_pepper.sh uses and which is the quickest way to triage a bad connection:

    python3 tools/pepper_bridge_node.py --raw PING
    python3 tools/pepper_bridge_node.py --raw "MOVE 0.1 0 0"
"""

import argparse
import math
import socket
import struct
import sys
import threading
import time

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 45455
DEFAULT_DEPTH_PORT = 45456

# Must match naoqi_shim.py.
FRAME_MAGIC = 0x50455044
FRAME_HEADER = "!IIHHd"
FRAME_HEADER_SIZE = struct.calcsize(FRAME_HEADER)

# From `ign_depth_camera` in pepper_robot_description/urdf/pepper_robot.gazebo. The
# simulated camera emits R_FLOAT32 metres and clips outside this range; the real camera
# emits 16-bit millimetres with 0 meaning "no return". Reconciling the two is an
# experimental decision, not an implementation detail -- see _to_metres.
SIM_NEAR_CLIP = 0.4
SIM_FAR_CLIP = 8.0
SIM_HFOV = 1.0123


def trained_velocity_limits():
    """(max |linear| m/s, max |angular| rad/s) the policy is trained with: rl/config/env.yaml
    `action`. The real robot must clamp at exactly these, not at a separate number. Until
    1 Oct 2026 this node and naoqi_shim.py both clamped at 0.25 m/s / 0.6 rad/s while the
    simulator allowed 0.30 / 0.90 -- every fast command the policy learned would have been
    cut by 17% (linear) and 33% (angular) on hardware, and the base calibration would have
    measured the clamp instead of the robot. Reverse is limited further by the policy's own
    action mapping (v_min), so the symmetric clamp here never binds on it."""
    from pathlib import Path
    try:
        import yaml
        a = yaml.safe_load((Path(__file__).resolve().parents[1] / "rl" / "config" /
                            "env.yaml").read_text())["action"]
        return float(max(a["v_max_mps"], -a["v_min_mps"])), float(a["omega_max_rps"])
    except Exception as exc:  # never start with a silently different limit
        raise RuntimeError(f"cannot read the trained velocity limits from rl/config/env.yaml: {exc}")


# Some commands block the shim while the robot physically moves. Retrying those is
# actively harmful -- a retried WAKEUP re-runs goToPosture, a retried ANGLES replays the
# motion -- so they get a generous timeout and no retries instead. Everything not listed
# is a fast query and keeps the tight timeout, because a policy stalling for seconds on a
# lost MOVE datagram would be worse than the datagram loss.
SLOW_COMMANDS = {
    "WAKEUP": 45.0,    # disable Autonomous Life + wakeUp + goToPosture + pose the head
    "REST": 30.0,
    "SHUTDOWN": 30.0,  # rests the joints first, then powers off; the reply may never arrive
    "ANGLES": 20.0,
    "HEAD": 20.0,
    "STIFFNESS": 10.0,
    "DEPTH": 10.0,     # one remote frame grab, plus a first-time camera subscription
    "RGB": 15.0,       # a VGA colour frame is ~920 kB off the robot, slower than depth
    "VERSION": 5.0,
    "STREAM": 5.0,
    "AWARENESS": 10.0,
}
FAST_TIMEOUT = 1.0
FAST_RETRIES = 2


class ShimClient:
    """Request/response UDP client for naoqi_shim.py.

    UDP over loopback effectively never drops, but a lost datagram here would hang the
    node, so fast requests time out and retry. See SLOW_COMMANDS for why the blocking
    ones do neither.
    """

    def __init__(self, host=DEFAULT_HOST, port=DEFAULT_PORT):
        self.address = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._seq = 0

    def _drain(self):
        """Discard replies left over from an earlier request.

        request() retries by re-sending and returns the first datagram to arrive. When
        the original reply then turns up as well it stays in the receive buffer, and the
        *next* request is handed it as if it were its own answer -- which is how a DEPTH
        call ends up being told "GETANGLES -1.3622,0.0936". Anything already queued
        before we send is stale by definition, so drop it.

        Only at the start of a request, never between retries: between retries a late
        reply is a perfectly good answer to the request being retried.
        """
        self.sock.setblocking(False)
        dropped = 0
        try:
            while True:
                try:
                    self.sock.recvfrom(4096)
                    dropped += 1
                except OSError:       # BlockingIOError: the buffer is empty
                    break
        finally:
            # Not setblocking(True): that clears the timeout to None and would make a
            # lost datagram hang the caller forever. request() sets it again below.
            self.sock.settimeout(None)
        return dropped

    def request(self, line, timeout=None):
        """Send one request and return ITS reply.

        Every request is tagged `#<n> ` and the shim echoes the tag, so a late reply to an
        earlier request can never be mistaken for this one's. Measured on the real robot
        2 Oct 2026: a Wi-Fi stall held the shim inside a NAOqi call for seconds, three
        retried MOVEs queued up behind it, and when they were answered the bridge took those
        "OK"s as the replies to its next ODOM requests ("ODOM rejected: OK") and never
        recovered -- the base stopped responding for the rest of the session. _drain() alone
        cannot prevent that: the stale replies arrive after it has run.

        MOVE is never retried: the bridge re-sends the CURRENT command at command_rate
        anyway, so a retry could only replay a stale velocity after the link recovers.
        """
        verb = line.strip().split(" ")[0].upper() if line.strip() else ""
        if timeout is None:
            timeout = SLOW_COMMANDS.get(verb, FAST_TIMEOUT)
        attempts = 1 if (verb in SLOW_COMMANDS or verb == "MOVE") else FAST_RETRIES + 1

        self._seq = (self._seq + 1) % 1000000
        tag = f"#{self._seq}"
        wire = f"{tag} {line}".encode("utf-8")
        self._drain()
        last_error = None
        for _ in range(attempts):
            self.sock.sendto(wire, self.address)
            deadline = time.time() + timeout
            while True:
                remaining = deadline - time.time()
                if remaining <= 0:
                    last_error = socket.timeout("timed out")
                    break
                self.sock.settimeout(remaining)
                try:
                    data, _peer = self.sock.recvfrom(4096)
                except socket.timeout as exc:
                    last_error = exc
                    break
                text = data.decode("utf-8").strip()
                if text.startswith(tag + " "):
                    return text[len(tag) + 1:]
                if text.startswith("#"):
                    continue            # a late reply to an earlier request: discard it
                return text             # an untagged shim (older version): accept as before
        raise TimeoutError(
            f"no reply to {verb!r} from shim at {self.address[0]}:{self.address[1]} "
            f"after {attempts} attempt(s) of {timeout:.0f}s -- is naoqi_shim.py running?"
        ) from last_error

    def close(self):
        self.sock.close()


class DepthClient:
    """Reads the shim's TCP depth stream, keeping only the newest frame.

    Latest-frame-wins rather than queueing: a policy acting on a stale depth image is
    worse than one acting on a slightly older timestamp, and a queue would grow without
    bound if the ROS side ever fell behind the robot.
    """

    def __init__(self, host, port, logger):
        self.address = (host, port)
        self.logger = logger
        self._lock = threading.Lock()
        self._frame = None  # (width, height, payload, timestamp)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def latest(self):
        with self._lock:
            return self._frame

    def stop(self):
        self._stop.set()

    def _run(self):
        while not self._stop.is_set():
            try:
                sock = socket.create_connection(self.address, timeout=5.0)
            except OSError as exc:
                self.logger.warn(f"depth stream not available ({exc}); retrying")
                if self._stop.wait(2.0):
                    return
                continue

            self.logger.info(f"depth stream connected to {self.address[0]}:{self.address[1]}")
            try:
                while not self._stop.is_set():
                    header = self._read_exactly(sock, FRAME_HEADER_SIZE)
                    magic, length, width, height, stamp = struct.unpack(FRAME_HEADER, header)
                    if magic != FRAME_MAGIC:
                        raise ValueError(f"bad frame magic {magic:#x} -- stream desynchronised")
                    payload = self._read_exactly(sock, length)
                    with self._lock:
                        self._frame = (width, height, payload, stamp)
            except (OSError, ValueError) as exc:
                if not self._stop.is_set():
                    self.logger.warn(f"depth stream dropped ({exc}); reconnecting")
            finally:
                try:
                    sock.close()
                except OSError:
                    pass

    @staticmethod
    def _read_exactly(sock, count):
        chunks = []
        remaining = count
        while remaining:
            chunk = sock.recv(remaining)
            if not chunk:
                raise OSError("depth stream closed by peer")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)


class _PrintLogger:
    """Minimal stand-in for a rclpy logger, for the no-ROS modes below.

    Everything goes to stderr. The no-ROS modes are parsed by verify_pepper.sh, which
    reads their single result line off stdout -- a stray status message there silently
    corrupts a measurement rather than failing.
    """

    def info(self, message):
        print(message, file=sys.stderr)

    def warn(self, message):
        print(f"warn: {message}", file=sys.stderr)


def drive(host, port, vx, vy, wz, seconds, rate=10.0):
    """Hold a base velocity for a fixed duration, then stop.

    This exists because of the shim's watchdog. A single MOVE is a *persistent* velocity
    command, but the watchdog halts the base if it hears nothing for ~0.75 s -- so
    "MOVE then sleep 5" travels for 0.75 s, not 5 s. Anything that drives the robot for a
    measured duration has to re-send at a steady rate, exactly as the ROS node does.

    Prints `DRIVE <elapsed>` so the caller can compute achieved velocity against the time
    that actually elapsed rather than the time it asked for.
    """
    client = ShimClient(host, port)
    period = 1.0 / rate
    started = time.time()
    try:
        while time.time() - started < seconds:
            reply = client.request(f"MOVE {vx} {vy} {wz}")
            if reply.startswith("ERR"):
                print(reply, file=sys.stderr)
                return 1
            time.sleep(period)
        elapsed = time.time() - started
        client.request("STOP")
        print("DRIVE %.3f" % elapsed)
        return 0
    except TimeoutError as exc:
        print(f"ERR {exc}", file=sys.stderr)
        return 1
    finally:
        try:
            client.request("STOP")
        except TimeoutError:
            pass
        client.close()


def probe_depth(host, port, command_port, seconds):
    """Measure the achieved depth frame rate, without ROS.

    verify_pepper.sh calls this. It matters because 320x240x2 bytes at 15 Hz is 2.3 MB/s
    pulled off the robot by getImageRemote, which over Pepper's wifi is marginal -- the
    rate the pipeline actually sustains is a number to measure and report, not assume.
    """
    control = ShimClient(host, command_port)
    client = DepthClient(host, port, _PrintLogger())
    try:
        control.request("STREAM ON")
        client.start()

        seen = 0
        last_stamp = None
        started = time.time()
        shape = None
        while time.time() - started < seconds:
            frame = client.latest()
            if frame is not None and frame[3] != last_stamp:
                last_stamp = frame[3]
                shape = (frame[0], frame[1], len(frame[2]))
                seen += 1
            time.sleep(0.005)

        elapsed = time.time() - started
        if not seen:
            print("STREAM 0 0 0 0.00", file=sys.stderr)
            return 1
        print("STREAM %d %d %d %.2f" % (shape[0], shape[1], shape[2], seen / elapsed))
        return 0
    finally:
        client.stop()
        try:
            control.request("STREAM OFF")
        except TimeoutError:
            pass
        control.close()


def yaw_to_quaternion(yaw):
    """Minimal yaw-only quaternion, to avoid a tf_transformations dependency."""
    return (0.0, 0.0, math.sin(yaw * 0.5), math.cos(yaw * 0.5))


def _to_metres(payload, numpy):
    """Convert NAOqi's 16-bit millimetres into the simulator's float32 metres.

    The simulated camera is declared `R_FLOAT32` with `<clip>0.4 8.0</clip>`, so it
    reports metres and returns non-finite values outside that range. The real camera
    reports millimetres and uses 0 for "no return" -- a distinct third case that the
    simulator has no equivalent for.

    Mapping all three of no-return, too-near and too-far to `inf` is a deliberate choice:
    it keeps the observation's *value domain* identical across branches, which is what
    zero-shot transfer needs. The cost is that the policy cannot distinguish "nothing
    there" from "sensor gave up on this pixel". If the depth artefacts of S4.4.4 turn out
    to matter, this is the line to revisit -- and whatever it does must be identical in
    both branches and stated in the write-up.
    """
    millimetres = numpy.frombuffer(payload, dtype="<u2")
    metres = millimetres.astype(numpy.float32) / 1000.0
    invalid = (millimetres == 0) | (metres < SIM_NEAR_CLIP) | (metres > SIM_FAR_CLIP)
    metres[invalid] = numpy.inf
    return metres


def run_node(args, ros_argv):
    import numpy
    import rclpy
    from geometry_msgs.msg import Twist
    from nav_msgs.msg import Odometry
    from rclpy.executors import ExternalShutdownException
    from rclpy.node import Node
    from sensor_msgs.msg import CameraInfo, Image
    # ros_gz_interfaces, the type the env subscribes with (see tools/contact_relay.py).
    from ros_gz_interfaces.msg import Contact, Contacts

    class PepperBridge(Node):
        def __init__(self):
            super().__init__("pepper_bridge")
            self._closing = False

            self.declare_parameter("shim_host", args.host)
            self.declare_parameter("shim_port", args.port)
            self.declare_parameter("depth_port", args.depth_port)
            lin_lim, ang_lim = trained_velocity_limits()
            self.declare_parameter("max_linear", lin_lim)
            self.declare_parameter("max_angular", ang_lim)
            self.declare_parameter("command_rate", 10.0)
            self.declare_parameter("odom_rate", 20.0)
            self.declare_parameter("depth_rate", 15.0)
            self.declare_parameter("publish_depth", True)
            self.declare_parameter("bumper_rate", 10.0)
            self.declare_parameter("publish_bumpers", True)
            self.declare_parameter("odom_frame", "odom")
            self.declare_parameter("base_frame", "base_link")
            # Read this off the running simulator rather than trusting the default:
            #   ros2 topic echo /pepper_robot/depth_camera/image_raw --once --field header
            self.declare_parameter("depth_frame", "pepper_robot/Head/camera_depth")

            get = lambda name: self.get_parameter(name).value  # noqa: E731
            self.max_linear = get("max_linear")
            self.max_angular = get("max_angular")
            self.odom_frame = get("odom_frame")
            self.base_frame = get("base_frame")
            self.depth_frame_id = get("depth_frame")
            self.numpy = numpy

            self.shim = ShimClient(get("shim_host"), get("shim_port"))
            reply = self.shim.request("PING")
            if reply != "PONG":
                raise RuntimeError(f"shim did not answer PING (got {reply!r})")

            version = self.shim.request("VERSION")
            self.get_logger().info(f"shim reports: {version}")
            self.get_logger().info(
                f"velocity clamp {self.max_linear:.2f} m/s, {self.max_angular:.2f} rad/s "
                f"(trained limits: env.yaml action); odometry at {get('odom_rate'):.0f} Hz")

            self.target = (0.0, 0.0, 0.0)

            # Relative names. The node runs in the `pepper_robot` namespace (see main),
            # so these resolve to exactly the topics bridge.launch.py gives the simulator.
            self.create_subscription(Twist, "cmd_vel", self.on_cmd_vel, 10)
            self.odom_pub = self.create_publisher(Odometry, "odom", 10)

            # The shim's watchdog stops the base if it hears nothing, so the last
            # command is re-sent continuously rather than only on receipt. A policy
            # publishing at its own control rate therefore cannot leave Pepper coasting.
            self.create_timer(1.0 / get("command_rate"), self.push_command)
            self.create_timer(1.0 / get("odom_rate"), self.publish_odometry)

            self.depth = None
            if get("publish_depth"):
                self.depth_pub = self.create_publisher(
                    Image, "depth_camera/image_raw", 10)
                self.info_pub = self.create_publisher(
                    CameraInfo, "depth_camera/camera_info", 10)
                self.shim.request("STREAM ON")
                self.depth = DepthClient(
                    get("shim_host"), get("depth_port"), self.get_logger())
                self.depth.start()
                self.create_timer(1.0 / get("depth_rate"), self.publish_depth)

            # The collision signal on hardware: env.yaml `collision.source: contact`
            # makes the env refuse to start unless this topic exists. Published on every
            # tick, empty when nothing is pressed, so `ros2 topic hz` shows it is alive.
            self._bump_counts = None
            if get("publish_bumpers"):
                self.bumper_pub = self.create_publisher(Contacts, "bumper", 10)
                self.create_timer(1.0 / get("bumper_rate"), self.publish_bumpers)

            self.get_logger().info(
                f"bridge up in namespace '{self.get_namespace()}': "
                "cmd_vel -> Pepper, Pepper -> odom + depth_camera/image_raw + bumper")

        def on_cmd_vel(self, msg):
            """Clamp and store. Clamping protects the trial, not just the robot --
            proposal S4.4.5 requires conservative velocity limits with the experimenter
            present, and an untrained policy early in evaluation can emit anything."""
            self.target = (
                max(-self.max_linear, min(self.max_linear, msg.linear.x)),
                max(-self.max_linear, min(self.max_linear, msg.linear.y)),
                max(-self.max_angular, min(self.max_angular, msg.angular.z)),
            )

        def _live(self):
            """False once rclpy is tearing down.

            Timers can fire between the context being invalidated and the executor
            stopping, and publishing then raises RCLError. That traceback is not just
            noise: it lands in the same terminal as the shutdown path that sends STOP,
            and during a session anything that obscures whether the robot was told to
            stop is worth suppressing at the source.
            """
            return rclpy.ok() and not self._closing

        def _safe_publish(self, publisher, msg):
            """Publish, tolerating the shutdown race that _live() cannot close.

            rclpy handles SIGTERM by invalidating the context from a signal handler,
            which can land between the _live() check and this call. A genuine publish
            failure is still raised.
            """
            try:
                publisher.publish(msg)
            except Exception:
                if self._live():
                    raise

        def push_command(self):
            if not self._live():
                return
            vx, vy, wz = self.target
            try:
                reply = self.shim.request(f"MOVE {vx:.4f} {vy:.4f} {wz:.4f}")
                if reply.startswith("ERR"):
                    self.get_logger().warn(f"MOVE rejected: {reply}")
            except TimeoutError as exc:
                self.get_logger().error(str(exc))

        def publish_odometry(self):
            if not self._live():
                return
            try:
                reply = self.shim.request("ODOM")
            except TimeoutError as exc:
                self.get_logger().error(str(exc))
                return
            if not reply.startswith("ODOM"):
                self.get_logger().warn(f"ODOM rejected: {reply}")
                return

            _, x, y, theta, vx, vy, vth = reply.split()
            msg = Odometry()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = self.odom_frame
            msg.child_frame_id = self.base_frame
            msg.pose.pose.position.x = float(x)
            msg.pose.pose.position.y = float(y)
            (
                msg.pose.pose.orientation.x,
                msg.pose.pose.orientation.y,
                msg.pose.pose.orientation.z,
                msg.pose.pose.orientation.w,
            ) = yaw_to_quaternion(float(theta))
            msg.twist.twist.linear.x = float(vx)
            msg.twist.twist.linear.y = float(vy)
            msg.twist.twist.angular.z = float(vth)
            self._safe_publish(self.odom_pub, msg)

        def publish_bumpers(self):
            """Republish the base bumpers as the simulator's contact message.

            A bumper counts if its press count rose since the last tick (a press that came
            and went between two ticks) or it is pressed now. Presses from before the bridge
            started are not reported: the first reply only sets the baseline counts.

            Stale or failed reads are logged, not hidden. A bumper topic that stays alive
            while nobody is reading the bumpers would report CR = 0 for the whole session
            -- the failure the env's collision-source check exists to prevent.
            """
            if not self._live():
                return
            try:
                reply = self.shim.request("BUMPERS")
            except TimeoutError as exc:
                self.get_logger().error(str(exc))
                return
            if not reply.startswith("BUMPERS"):
                self.get_logger().error(f"BUMPERS rejected: {reply}",
                                        throttle_duration_sec=5.0)
                return
            fields = reply.split()
            counts = [int(v) for v in fields[1:4]]
            pressed = [v == "1" for v in fields[4:7]]
            age = float(fields[7])
            if age > 1.0:
                self.get_logger().error(
                    f"bumper data is {age:.1f} s old -- collisions are not being seen",
                    throttle_duration_sec=5.0)
            rose = ([c > b for c, b in zip(counts, self._bump_counts)]
                    if self._bump_counts is not None else [False] * 3)
            self._bump_counts = counts

            msg = Contacts()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = self.base_frame
            for name, hit in zip(("front_right", "front_left", "back"),
                                 (r or n for r, n in zip(rose, pressed))):
                if hit:
                    c = Contact()
                    c.collision1.name = f"pepper_robot::base::bumper_{name}"
                    c.collision2.name = "real_world"
                    msg.contacts.append(c)
            self._safe_publish(self.bumper_pub, msg)

        def publish_depth(self):
            if not self._live():
                return
            frame = self.depth.latest()
            if frame is None:
                return
            width, height, payload, _stamp = frame
            if len(payload) != width * height * 2:
                self.get_logger().warn(
                    f"depth frame is {len(payload)} bytes, expected {width * height * 2}")
                return

            metres = _to_metres(payload, self.numpy)
            if (width, height) == (160, 120):
                # naoqi_shim.py --depth-res qqvga (slow links): repeat each pixel 2x2 so the
                # published frame is 320x240 like the simulated camera. The env's 5x5 block-min
                # to 64x48 assumes 320x240; fed 160x120 it would keep only the top-left
                # 128x96 and silently crop the field of view.
                metres = self.numpy.repeat(self.numpy.repeat(
                    metres.reshape(height, width), 2, axis=0), 2, axis=1).ravel()
                width, height = 320, 240
            now = self.get_clock().now().to_msg()

            msg = Image()
            msg.header.stamp = now
            msg.header.frame_id = self.depth_frame_id
            msg.height = height
            msg.width = width
            msg.encoding = "32FC1"
            msg.is_bigendian = 0
            msg.step = width * 4
            msg.data = metres.tobytes()
            self._safe_publish(self.depth_pub, msg)

            # Pinhole intrinsics from the simulated camera's horizontal FOV, so anything
            # that deprojects depth (point clouds, obstacle distances) agrees across
            # branches. Square pixels, principal point at the centre.
            focal = (width / 2.0) / math.tan(SIM_HFOV / 2.0)
            info = CameraInfo()
            info.header.stamp = now
            info.header.frame_id = self.depth_frame_id
            info.height = height
            info.width = width
            info.distortion_model = "plumb_bob"
            info.d = [0.0, 0.0, 0.0, 0.0, 0.0]
            info.k = [focal, 0.0, width / 2.0,
                      0.0, focal, height / 2.0,
                      0.0, 0.0, 1.0]
            info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
            info.p = [focal, 0.0, width / 2.0, 0.0,
                      0.0, focal, height / 2.0, 0.0,
                      0.0, 0.0, 1.0, 0.0]
            self._safe_publish(self.info_pub, info)

        def shutdown(self):
            # Set first, so any timer still in flight stops touching dead publishers.
            self._closing = True
            if self.depth is not None:
                self.depth.stop()
            # STOP goes first and its result is reported. Everything else is best-effort:
            # the one thing that must not fail silently on the way out is halting a robot
            # that is physically moving.
            # print(), not get_logger(): by this point rosout's publisher is being torn
            # down too, and the one message you most want to see on the way out should
            # not depend on ROS still being alive.
            try:
                reply = self.shim.request("STOP")
                print(f"STOP -> {reply}")
            except Exception as exc:
                print(f"WARNING: could not stop the base on shutdown: {exc}\n"
                      f"The shim watchdog should halt her within a second. If the shim is "
                      f"also gone, use the button on the robot.", file=sys.stderr)
            try:
                self.shim.request("STREAM OFF")
            except Exception:
                pass
            self.shim.close()

    rclpy.init(args=ros_argv)
    node = PepperBridge()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default=DEFAULT_HOST, help="naoqi_shim.py host")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help="naoqi_shim.py command port")
    parser.add_argument("--depth-port", type=int, default=DEFAULT_DEPTH_PORT,
                        help="naoqi_shim.py depth stream port")
    parser.add_argument("--namespace", default="pepper_robot",
                        help="ROS namespace; the default matches the simulator's topics")
    parser.add_argument("--raw", metavar="COMMAND",
                        help="send one command to the shim, print the reply, exit "
                             "(no ROS required)")
    parser.add_argument("--probe-depth", metavar="SECONDS", type=float,
                        help="measure the achieved depth frame rate for SECONDS and "
                             "print 'STREAM <w> <h> <bytes> <fps>' (no ROS required)")
    parser.add_argument("--drive", nargs=4, type=float,
                        metavar=("VX", "VY", "WZ", "SECONDS"),
                        help="hold a base velocity for SECONDS, feeding the shim "
                             "watchdog, then stop (no ROS required)")
    args, unknown = parser.parse_known_args()

    if args.probe_depth:
        return probe_depth(args.host, args.depth_port, args.port, args.probe_depth)

    if args.drive:
        return drive(args.host, args.port, *args.drive)

    if args.raw:
        client = ShimClient(args.host, args.port)
        try:
            print(client.request(args.raw))
        except TimeoutError as exc:
            print(f"ERR {exc}", file=sys.stderr)
            return 1
        finally:
            client.close()
        return 0

    # Push the namespace in as a ROS argument unless the caller already set one, so the
    # topics match the simulator without every launch site having to remember to do it.
    # Anything unrecognised is passed straight through, so `--ros-args -p max_linear:=0.1`
    # still works; ROS 2 accepts more than one `--ros-args` group.
    ros_argv = [sys.argv[0]] + unknown
    if args.namespace and "__ns:=" not in " ".join(unknown):
        ros_argv += ["--ros-args", "-r", f"__ns:=/{args.namespace.strip('/')}"]

    run_node(args, ros_argv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
