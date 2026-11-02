#!/usr/bin/env python3
"""The navigation MDP of proposal S4.4.2, as a Gymnasium environment.

    python3 -m rl.scripts.check_env --branch photogrammetry

State, action and reward follow Chaffre et al. as the proposal specifies: the state is a
depth image plus the goal pose relative to the robot frame, the action is a continuous
(v, omega), and the reward is a terminal goal reward, a collision penalty and progress
shaping.

Everything here speaks ROS 2
---------------------------
Observations arrive on `depth_camera/image_raw`, actions leave on `/pepper_robot/cmd_vel`,
and pose comes from `true_base_tf`. Those are the same topic names
`tools/pepper_bridge_node.py` uses for the *physical* robot, and `_to_metres` there was
written specifically so the depth value domains match. So this class runs against hardware
in Task 6 by pointing it at a different publisher. The only simulator-specific calls are
episode resets, and they are isolated in `rl.env.ign_control` behind `self._ign`, which is
None when `source="real"`.

Free-running, not stepped
-------------------------
The world runs at its own pace and the env samples at `control_hz` off `/clock`. Pausing
and stepping Ignition each tick would be more reproducible, and it is available in
`ign_control.step()` -- but the real robot cannot be stepped, so a free-running env trains
the policy under the same asynchrony it will meet on hardware. The cost is control-period
jitter, which is measured and exposed in `info["jitter_s"]` rather than assumed away.
"""

import math
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import yaml
from gymnasium import spaces

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "rl" / "config" / "env.yaml"

# The depth value domain is fixed once, in the bridge, and imported rather than restated.
# If these two ever disagree, the policy trains on one distribution and is deployed on
# another, and nothing about either image looks wrong.
import sys
sys.path.insert(0, str(REPO / "tools"))
from pepper_bridge_node import SIM_NEAR_CLIP, SIM_FAR_CLIP  # noqa: E402

from rl import sim_process  # noqa: E402
from rl.env.floor_height import FloorHeight  # noqa: E402
from rl.env.randomisation import CommandDelay  # noqa: E402


def load_config(path=CONFIG):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def wrap_angle(a):
    return math.atan2(math.sin(a), math.cos(a))


# --------------------------------------------------------------------------------------
# ROS 2 plumbing
# --------------------------------------------------------------------------------------

class RosInterface:
    """Everything that touches rclpy. Identical against sim and against hardware.

    Kept separate from the Gymnasium class so the MDP -- reward, termination, observation
    assembly -- can be unit-tested with a stub in place of ROS, and so that the real-robot
    path has one obvious seam to attach to.
    """

    # Imported, not restated. rl/sim_process.py passes POSE_TOPIC to the bridge as a
    # remap, so a second copy here would silently become a topic nobody publishes -- and
    # the symptom is the env waiting forever in wait_for_data with everything else
    # apparently healthy.
    DEPTH_TOPIC = sim_process.DEPTH_TOPIC
    CMD_VEL_TOPIC = "/pepper_robot/cmd_vel"
    POSE_TOPIC = sim_process.POSE_TOPIC
    SCAN_TOPIC = sim_process.SCAN_TOPIC
    CONTACT_TOPIC = sim_process.BUMPER_TOPIC

    def __init__(self, node_name="pepper_nav_env", use_sim_time=True):
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
        from geometry_msgs.msg import Twist
        from sensor_msgs.msg import Image, LaserScan
        from tf2_msgs.msg import TFMessage
        # ros_gz_interfaces: must match tools/contact_relay.py (see the note there on why the
        # ros_ign_interfaces shim type cannot be deserialised).
        from ros_gz_interfaces.msg import Contacts

        if not rclpy.ok():
            rclpy.init()
        self._rclpy = rclpy
        self.node = Node(node_name)
        self.node.set_parameters([rclpy.parameter.Parameter(
            "use_sim_time", rclpy.Parameter.Type.BOOL, use_sim_time)])

        # Best-effort, depth 1. Sensor data is the textbook case for it, and it matters
        # here: a reliable queue would replay a backlog of stale depth frames after any
        # hiccup, and the policy would act on the room as it was seconds ago.
        sensor_qos = QoSProfile(depth=1,
                                reliability=ReliabilityPolicy.BEST_EFFORT,
                                history=HistoryPolicy.KEEP_LAST)

        self._depth = None
        self._depth_stamp = None
        self._pose = None            # (x, y, yaw)
        self._tilt_deg = None        # angle between the base's up axis and world up
        self._scan_min = None
        self._contact = False

        self.node.create_subscription(Image, self.DEPTH_TOPIC, self._on_depth, sensor_qos)
        self.node.create_subscription(LaserScan, self.SCAN_TOPIC, self._on_scan, sensor_qos)
        self.node.create_subscription(TFMessage, self.POSE_TOPIC, self._on_pose, sensor_qos)
        self.node.create_subscription(Contacts, self.CONTACT_TOPIC, self._on_contact,
                                      sensor_qos)
        self.cmd_pub = self.node.create_publisher(Twist, self.CMD_VEL_TOPIC, 1)
        self._Twist = Twist

    # -- callbacks ---------------------------------------------------------------------

    def _on_depth(self, msg):
        # 32FC1 metres in simulation; the bridge converts the real camera's 16-bit
        # millimetres into the same encoding, so one branch handles both.
        if msg.encoding not in ("32FC1", "32FC"):
            raise RuntimeError(
                f"depth encoding is {msg.encoding!r}, expected 32FC1 metres. The "
                f"simulated camera is declared R_FLOAT32 and pepper_bridge_node._to_metres "
                f"converts the real one to match; something upstream has changed.")
        a = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
        self._depth = a
        self._depth_stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

    def _on_scan(self, msg):
        r = np.asarray(msg.ranges, dtype=np.float32)
        r = r[np.isfinite(r) & (r >= msg.range_min)]
        self._scan_min = float(r.min()) if r.size else None

    def _on_contact(self, msg):
        # Latched, not overwritten: a contact can occur and clear again between two
        # control-period polls, and env._collided() only reads this at the control rate.
        # Overwriting with `bool(msg.contacts)` on every message would let a real but
        # brief contact land between reads and vanish unnoticed -- exactly the case that
        # matters for CR (Eq. 4.5), since a fast graze is still a collision.
        if msg.contacts:
            self._contact = True

    # -- odometry as the pose source (real robot, no overhead tracker) -----------------

    ODOM_TOPIC = "/pepper_robot/odom"

    def use_odometry_for_pose(self):
        """Take the room-frame pose from Pepper's wheel odometry instead of true_base_tf.

        For hardware sessions with no overhead camera to mount (proposal S4.4.6 asks for an
        external reference; this is the declared fallback). Odometry has its own arbitrary
        origin and drifts, so it is re-anchored at every route start: when the operator has
        placed her on the taped start pose and pressed Enter, that odometry reading IS the
        route's start pose, and every later reading is mapped through the same rigid
        transform. Error therefore accumulates over one 2-3 m route only, plus however far
        off the tape the operator placed her. No tilt: tip-over is the operator's call.
        """
        from nav_msgs.msg import Odometry
        self._pose_from_odom = True
        self._odom = None
        self._anchor = None
        self.node.create_subscription(Odometry, self.ODOM_TOPIC, self._on_odom, 10)

    def anchor_odometry(self, x, y, yaw, timeout=5.0):
        """Declare the current odometry reading to be room pose (x, y, yaw)."""
        deadline = time.time() + timeout
        self._odom = None                   # use a reading from AFTER the placement
        while self._odom is None and time.time() < deadline:
            self.spin(0.05)
        if self._odom is None:
            raise TimeoutError(f"no odometry on {self.ODOM_TOPIC} within {timeout}s -- "
                               f"is tools/pepper_bridge_node.py running?")
        self._anchor = ((x, y, yaw), self._odom)
        self._pose = (x, y, yaw)

    def _on_odom(self, msg):
        p = msg.pose.pose
        q = p.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self._odom = (p.position.x, p.position.y, yaw)
        if self._anchor is None:
            return
        (rx, ry, ryaw), (ax, ay, ayaw) = self._anchor
        # Displacement since the anchor, expressed in the robot's frame AT the anchor,
        # then re-expressed in the room frame from the room start pose.
        dx, dy = self._odom[0] - ax, self._odom[1] - ay
        bx = math.cos(ayaw) * dx + math.sin(ayaw) * dy
        by = -math.sin(ayaw) * dx + math.cos(ayaw) * dy
        dyaw = self._odom[2] - ayaw
        self._pose = (rx + math.cos(ryaw) * bx - math.sin(ryaw) * by,
                      ry + math.sin(ryaw) * bx + math.cos(ryaw) * by,
                      math.atan2(math.sin(ryaw + dyaw), math.cos(ryaw + dyaw)))

    def _on_pose(self, msg):
        if getattr(self, "_pose_from_odom", False):
            return                          # one pose source per session, never two
        for t in msg.transforms:
            q = t.transform.rotation
            yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                             1.0 - 2.0 * (q.y * q.y + q.z * q.z))
            self._pose = (t.transform.translation.x, t.transform.translation.y, yaw)
            # Tilt from vertical: acos of the z-component of the body's up axis. Only the
            # tip-over check reads it; navigation stays planar.
            up_z = 1.0 - 2.0 * (q.x * q.x + q.y * q.y)
            self._tilt_deg = math.degrees(math.acos(max(-1.0, min(1.0, up_z))))

    # -- accessors ---------------------------------------------------------------------

    def spin(self, seconds):
        self._rclpy.spin_once(self.node, timeout_sec=max(seconds, 0.0))

    def now(self):
        return self.node.get_clock().now().nanoseconds * 1e-9

    def publish(self, v, omega):
        m = self._Twist()
        m.linear.x = float(v)
        m.angular.z = float(omega)
        self.cmd_pub.publish(m)

    def stop(self):
        self.publish(0.0, 0.0)

    def wait_for_data(self, timeout=30.0):
        """Block until depth and pose have both arrived at least once."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            self.spin(0.1)
            if self._depth is not None and self._pose is not None:
                return True
        missing = [n for n, v in (("depth", self._depth), ("pose", self._pose)) if v is None]
        raise TimeoutError(
            f"no {' or '.join(missing)} within {timeout}s. Is the bridge running? "
            f"gz.launch.py bridges only /clock -- the sensor topics come from "
            f"pepper_robot_ign/launch/bridge.launch.py. And on WSL2, check "
            f"LIBGL_ALWAYS_SOFTWARE=1: under ogre v1 the sensors publish happily and "
            f"return garbage.")

    def close(self):
        try:
            self.stop()
            self.node.destroy_node()
        except Exception:
            pass


# --------------------------------------------------------------------------------------
# the environment
# --------------------------------------------------------------------------------------

class PepperNavEnv(gym.Env):
    """Point-to-point navigation. Gymnasium API; `gymnasium.Env` is the declared base.

    Construct with `source="sim"` for training and simulated evaluation, and
    `source="real"` for Task 6 -- the difference is only that the real one cannot
    teleport, so `reset()` asks a human to reposition the robot instead.
    """

    metadata = {"render_modes": []}

    def __init__(self, world_name="room_photogrammetry", branch="photogrammetry",
                 source="sim", config=None, routes=None, seed=None,
                 randomiser=None, spawn_z=0.0, room=None, pose_source="tracker"):
        self.cfg = config or load_config()
        # Hardware only: "tracker" (overhead camera + AprilTag on true_base_tf, S4.4.6) or
        # "odometry" (wheel odometry re-anchored at each taped start; see
        # RosInterface.use_odometry_for_pose). A run argument, not an env.yaml key, so the
        # config hashes recorded with the trained models are unchanged.
        if pose_source not in ("tracker", "odometry"):
            raise ValueError(f"pose_source must be 'tracker' or 'odometry', got {pose_source!r}")
        if pose_source == "odometry" and source != "real":
            raise ValueError("pose_source='odometry' is for source='real' only")
        self.pose_source = pose_source
        self.repositioner = None    # set by the caller on hardware; see rl/env/reposition.py
        self.branch = branch
        self.source = source
        self.world_name = world_name
        self.routes = routes
        self.randomiser = randomiser
        self.spawn_z = spawn_z
        self._floor = None  # FloorHeight, built on first teleport (sim only)
        # The physical-room slug (e.g. "lab_a"), matching
        # simulation/photogrammetry/ground_truth/<room>_spec.yaml. Needed only for
        # TRAINING-time free-space sampling on the standard branch, which has no
        # collision mesh to sample from -- see `_sample_start_goal`. The evaluation
        # route set is loaded separately by the caller (e.g. train.py), and is
        # always planned on this same spec regardless of branch (see rl/env/routes.py).
        self.room = room
        self.rng = np.random.default_rng(seed)

        t, a, o = self.cfg["task"], self.cfg["action"], self.cfg["observation"]
        self.goal_tol = float(t["goal_tolerance_m"])
        self.max_duration = float(t["max_duration_s"])
        self.control_dt = 1.0 / float(t["control_hz"])
        self.v_max, self.v_min = float(a["v_max_mps"]), float(a["v_min_mps"])
        self.w_max = float(a["omega_max_rps"])
        self.dw, self.dh = int(o["depth_width"]), int(o["depth_height"])
        # The real robot's command latency, reproduced in SIMULATION ONLY (the robot supplies
        # its own), in training and evaluation alike. See env.yaml and CommandDelay.
        self.sim_delay_steps = int(a.get("sim_command_delay_steps", 0)) if source == "sim" else 0
        self._delay = CommandDelay(self.sim_delay_steps)
        self._held_depth = None     # last depth observation, repeated while no new frame

        # Actions are [-1, 1] in both channels and scaled here rather than exposed in
        # engineering units. SAC's squashed Gaussian is built for a symmetric unit box,
        # and its entropy target is set from the action dimension assuming one.
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.n_hist = int(o.get("action_history", 0))
        self._act_hist = np.zeros((self.n_hist, 2), dtype=np.float32)
        self.observation_space = self._make_observation_space()

        self.ros = RosInterface(use_sim_time=(source == "sim"))
        if pose_source == "odometry":
            self.ros.use_odometry_for_pose()
        self._assert_collision_source_exists()
        self._ign = None
        if source == "sim":
            from rl.env.ign_control import IgnitionControl
            self._ign = IgnitionControl(world_name)

        self._goal = None
        self._t0 = None
        self._prev_dist = None
        self._path_length = 0.0
        self._collisions = 0
        self._last_xy = None
        self._route_id = None
        self._jitter = []
        # World-reset half of dynamics randomisation (floor friction; see
        # rl/env/body_randomisation.py). Counted from 0 so the very first episode gets a
        # freshly-sampled surface rather than whatever the checked-in model shipped with.
        self._episode_count = 0
        self._last_body_params = None

    def _make_observation_space(self):
        spaces_ = {
            # uint8, NOT float32, and the dtype is load-bearing rather than an
            # optimisation. stable_baselines3.common.preprocessing.is_image_space
            # requires uint8; with a float32 Box, CombinedExtractor silently flattens
            # the depth image into a 3072-long vector and feeds it to an MLP, so the
            # policy never sees a convolution at all and nothing anywhere reports a
            # problem. As uint8 SB3 recognises an image and uses NatureCNN.
            #
            # It also decides whether training fits in memory. SB3's DictReplayBuffer
            # stores obs and next_obs separately, so a transition costs ~6.2 kB here
            # against ~24.6 kB as float32 -- 620 MB versus 2.5 GB at 100k transitions,
            # on a machine with 7 GB total and Gazebo and torch in it as well.
            #
            # The cost is quantisation: 0.4-8.0 m across 256 levels is ~3 cm. Well under
            # the real sensor's own noise, which tools/check_depth.py measures.
            "depth": spaces.Box(0, 255, shape=(1, self.dh, self.dw), dtype=np.uint8),
            # (range, sin bearing, cos bearing). sin/cos avoids the discontinuity at +-pi.
            "goal": spaces.Box(np.array([0.0, -1.0, -1.0], dtype=np.float32),
                               np.array([np.inf, 1.0, 1.0], dtype=np.float32),
                               dtype=np.float32),
        }
        if self.n_hist:
            # Last n_hist actions, oldest first, flattened: (v0, w0, v1, w1, ...).
            spaces_["action_hist"] = spaces.Box(-1.0, 1.0, shape=(2 * self.n_hist,),
                                                dtype=np.float32)
        return spaces.Dict(spaces_)

    def _assert_collision_source_exists(self):
        """Refuse to start if the configured collision source publishes nothing.

        This is the silent failure worth spending a startup check on. If
        `self.ros._contact` never gets set, training still runs, the reward curve still
        looks healthy, C_i in Eq. 4.1 is 0 on every route, and CR (Eq. 4.5) is exactly
        zero for both branches -- a publishable-looking number, which is exactly why it
        must not be possible to produce one by accident.

        `collision.source: contact` currently fails this check, and the reason is
        recorded rather than mysterious: the sensor and its Ignition-side topic are
        real (`ign_base_contact` in pepper_robot.gazebo, confirmed by reading the raw
        Ignition topic against a live sim), but `ros2 launch` resolves `parameter_bridge`
        to the system apt package, whose factory refuses to bridge
        ros_ign_interfaces/msg/Contacts. The vendored ros2_ws/src/ros_ign DOES support
        it (see its mappings.py) but carries a COLCON_IGNORE and has never been built in
        this workspace. See the long comment on `collision.source` in rl/config/env.yaml
        for the fix options; `proximity` is the current default for exactly this reason.
        """
        src = self.cfg["collision"]["source"]
        if src != "contact":
            return
        names = [n for n, _ in self.ros.node.get_topic_names_and_types()]
        if self.ros.CONTACT_TOPIC in names:
            return
        raise RuntimeError(
            f"collision.source is 'contact' but nothing publishes "
            f"{self.ros.CONTACT_TOPIC}. Every episode would report zero collisions: "
            f"C_i in Eq. 4.1 would always be 0 and CR in Eq. 4.5 exactly 0.0, in both "
            f"branches, with no error anywhere.\n"
            f"  The sensor is real; the ROS bridge for it is not, in this workspace --\n"
            f"  see the comment on rl/config/env.yaml's collision.source for the full\n"
            f"  diagnosis and the two ways to fix it. Until then, set\n"
            f"  collision.source: proximity, which is a PROXY (the front lidar spans\n"
            f"  60 degrees and cannot see a side or rear collision at all) but does\n"
            f"  work today. Whichever source you use, it must be the same in both\n"
            f"  branches and stated in the write-up."
        )

    # -- observation --------------------------------------------------------------------

    def _depth_obs(self):
        d = self.ros._depth
        if d is None:
            raise RuntimeError("no depth frame")
        # Downsample by block-min, not by averaging or nearest-neighbour. Minimum is the
        # safety-preserving choice: averaging a thin chair leg against the far wall behind
        # it produces a mid-range pixel showing an obstacle that is not there and hiding
        # one that is. Min keeps the nearest thing in each block, which is the thing the
        # robot would hit.
        h, w = d.shape
        bh, bw = h // self.dh, w // self.dw
        if bh >= 1 and bw >= 1:
            d = d[:bh * self.dh, :bw * self.dw].reshape(self.dh, bh, self.dw, bw)
            d = np.nanmin(np.where(np.isfinite(d), d, np.inf), axis=(1, 3))
        # Observation randomisation goes here, while the image is still in metres and
        # before the invalid-value mapping -- so a synthetic dropout takes exactly the
        # same path through the code that a genuine no-return does. Applied after
        # downsampling rather than before so the noise is on the pixels the policy
        # actually sees; noising 320x240 and then taking a block-min would mostly select
        # the most negative noise sample in each block and bias the image toward the robot.
        if self.randomiser is not None:
            d = self.randomiser.perturb_depth(d, self.rng)

        # inf covers no-return, too-near and too-far alike -- one value domain across sim
        # and real, as fixed in pepper_bridge_node._to_metres. Mapped to FAR here; see the
        # `invalid_depth_as` note in env.yaml for why that direction and not the other.
        far = self.cfg["observation"]["invalid_depth_as"] == "far"
        d = np.where(np.isfinite(d), d, SIM_FAR_CLIP if far else SIM_NEAR_CLIP)
        d = np.clip(d, SIM_NEAR_CLIP, SIM_FAR_CLIP)
        d = (d - SIM_NEAR_CLIP) / (SIM_FAR_CLIP - SIM_NEAR_CLIP)
        # Quantise to 0-255. SB3's preprocessing divides by 255 again before the CNN, so
        # the network still sees [0, 1]; the uint8 is what the buffer stores and what
        # makes SB3 treat this as an image. Rounding rather than truncating, so the
        # quantisation error is symmetric instead of biasing every pixel toward the robot.
        return np.rint(d * 255.0).astype(np.uint8)[None, :, :]

    def _goal_obs(self, pose):
        x, y, yaw = pose
        gx, gy = self._goal
        dx, dy = gx - x, gy - y
        rng = math.hypot(dx, dy)
        bearing = wrap_angle(math.atan2(dy, dx) - yaw)
        return np.array([rng, math.sin(bearing), math.cos(bearing)], dtype=np.float32)

    def _observe(self):
        pose = self.ros._pose
        if pose is None:
            raise RuntimeError("no pose")
        # A slower camera than the control loop, in sim: when no new frame would have
        # arrived at this episode's depth_rate_hz, repeat the previous observation exactly,
        # as the real robot does (2-3 fps measured over Wi-Fi). The goal vector stays live:
        # on hardware it comes from the overhead pose tracker, not the depth stream.
        if (self.source == "sim" and self.randomiser is not None
                and self._held_depth is not None
                and not self.randomiser.depth_frame_due(self.ros.now(), self.rng)):
            depth = self._held_depth
        else:
            depth = self._depth_obs()
        self._held_depth = depth
        obs = {"depth": depth, "goal": self._goal_obs(pose)}
        if self.n_hist:
            obs["action_hist"] = self._act_hist.reshape(-1).copy()
        return obs, pose

    # -- collision ----------------------------------------------------------------------

    def _collided(self):
        """Whether a collision happened since the last check.

        `collision.source` in env.yaml decides which signal counts, and the choice is an
        experimental one rather than an implementation detail: it defines C_i in Eq. 4.1
        and the numerator of CR in Eq. 4.5. Whichever is used must be the same in both
        branches, and it has to be stated in the write-up.

        `proximity` is explicitly a proxy. The front lidar spans 60 degrees, so it cannot
        see a collision with the side or rear of the base at all -- a policy that clips a
        doorframe while turning registers nothing. Use `contact` once the base contact
        sensor is in the model.
        """
        # Tipping over is a collision whichever source is configured. Measured on
        # final_scan, 28 Sep 2026: reversing or sidestepping into a wall is invisible to
        # the 60-degree front lidar, so Pepper pushed into the wall until she toppled
        # (tilt 120 deg) and lay there until the episode timed out -- uncounted in CR and
        # free of penalty. A toppled Pepper is a failed episode on hardware too.
        tip = self.cfg["collision"].get("tip_over_deg")
        tilt = getattr(self.ros, "_tilt_deg", None)
        if tip is not None and tilt is not None and tilt > float(tip):
            return True
        src = self.cfg["collision"]["source"]
        if src == "contact":
            hit = self.ros._contact
            self.ros._contact = False
            return bool(hit)
        if src == "proximity":
            m = self.ros._scan_min
            return m is not None and m < float(self.cfg["collision"]["proximity_threshold_m"])
        raise ValueError(f"unknown collision.source {src!r}; expected 'contact' or 'proximity'")

    # -- Gymnasium API --------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        # Gymnasium's contract (checked by gymnasium.utils.env_checker.check_env, and
        # caught by rl/scripts/check_env.py's online mode the first time it was actually
        # run): reset() must call super().reset(seed=seed), which is what sets
        # self.np_random. This env keeps its own `self.rng` rather than np_random for
        # every draw it actually makes -- np_random exists here only so the base class's
        # own bookkeeping is satisfied and third-party wrappers that expect it (some SB3
        # utilities included) do not silently misbehave.
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        options = options or {}

        self.ros.stop()
        self._held_depth = None     # never carry a frame across episodes
        self._act_hist[:] = 0.0     # nothing sent yet: matches the empty CommandDelay
        self._maybe_resample_body()
        if self.randomiser is not None:
            # Per-episode sampling held fixed for the episode (Peng et al., via S4.4.2).
            self.randomiser.new_episode(self.rng)
        # Fresh delay line every episode, so no command from the last one is executed after
        # the reset. Sim only; the randomiser's latency is an offset around the fixed delay.
        offset = (self.randomiser.params.get("latency_steps", 0)
                  if self.randomiser is not None and self.randomiser.enabled else 0)
        self._delay.reset(self.sim_delay_steps + offset if self.source == "sim" else 0)

        route = options.get("route")
        if route is not None:
            sx, sy = route["start"]
            syaw = route.get("start_yaw", 0.0)
            self._goal = tuple(route["goal"])
            self._route_id = route["route_id"]
            self._optimal_length = route.get("optimal_length_m")
        else:
            (sx, sy), (gx, gy) = self._sample_start_goal()
            syaw = float(self.rng.uniform(-math.pi, math.pi))
            self._goal = (gx, gy)
            self._route_id = None
            self._optimal_length = None

        self._place(sx, sy, syaw)

        for _ in range(20):
            self.ros.spin(0.05)
        self.ros.wait_for_data()

        obs, pose = self._observe()
        self._t0 = self.ros.now()
        self._prev_dist = math.hypot(self._goal[0] - pose[0], self._goal[1] - pose[1])
        self._path_length = 0.0
        self._collisions = 0
        self._last_xy = (pose[0], pose[1])
        self._jitter = []
        self.ros._contact = False
        return obs, {"route_id": self._route_id}

    def _sample_start_goal(self):
        """A start uniform over free space, and the room's goal object as the goal.

        The goal never moves: the task is "get to the red pole" (env.yaml
        task.goal_object), in training exactly as in evaluation. The start is uniform
        during training so the policy is not fitted to the 20 start poses of the
        evaluation route set -- which it would otherwise see thousands of times each, and
        which are also the poses it is scored on.

        Each branch samples its OWN geometry here, unlike the fixed evaluation route
        set, which is always planned on the standard branch's spec (see the decision
        recorded in rl/env/routes.py). During training that distinction is the point:
        the photogrammetric branch should explore the mesh it will actually be scored
        in, holes and all, and the standard branch has no mesh to sample from at all.
        """
        from rl.env.routes import build_grid, goal_zone
        if not hasattr(self, "_grid"):
            if not self.room:
                raise RuntimeError(
                    "PepperNavEnv(...) needs room= to sample training episodes -- the "
                    "goal object's position comes from the room spec, and the standard "
                    "branch has no collision mesh to sample starts from either.")
            if self.branch == "standard":
                self._grid, _, _ = build_grid(self.room, source="spec", verbose=False)
            else:
                self._grid, _, _ = build_grid(self.world_name, source="mesh",
                                              verbose=False)
            self._goal_xy, _ = goal_zone(self.room, self.cfg)
        for _ in range(200):
            (sx, sy), = self._grid.sample_free(self.rng, 1)
            try:
                ell = self._grid.shortest_path_to_zone((sx, sy), self._goal_xy,
                                                       self.goal_tol)
            except ValueError:
                continue
            if ell is not None and ell >= 1.5:
                return (sx, sy), self._goal_xy
        raise RuntimeError("could not sample a start connected to the goal in 200 tries")

    def _maybe_resample_body(self):
        """World-reset half of dynamics randomisation: floor friction, every N episodes.

        Gated on `self.randomiser is not None` -- exactly the same guard `reset()` uses
        for the per-episode command/observation terms -- so evaluation (which constructs
        the env with `randomiser=None`) automatically
        never respawns the floor mid-route-set. Evaluation measures the policy, not
        variation in the surface it is standing on.

        Also gated on `self._ign is not None`: there is no floor to respawn on hardware,
        and this must never run there.
        """
        self._episode_count += 1
        if self.randomiser is None or not self.randomiser.enabled or self._ign is None:
            return
        every = int(self.randomiser.cfg["body"]["resample_every_episodes"])
        if self._episode_count == 1 or self._episode_count % every == 0:
            from rl.env.randomisation import sample_body_params
            from rl.env.body_randomisation import apply_friction
            from rl.env.ign_control import IgnServiceError
            params = sample_body_params(self.randomiser.cfg, self.rng)
            try:
                apply_friction(self._ign, self.world_name, params["mu"], params["mu2"])
            except IgnServiceError as e:
                # One slow Ignition service call must not end a 30-hour run (a /remove
                # timeout did, 7 Oct 2026). The room keeps its current friction until the
                # next resample; skipped resamples are logged, so they can be counted.
                self.ros.node.get_logger().warning(
                    f"body randomisation @ episode {self._episode_count} SKIPPED, floor "
                    f"friction unchanged: {e}")
                return
            self._last_body_params = params
            self.ros.node.get_logger().info(
                f"body randomisation @ episode {self._episode_count}: "
                f"mu={params['mu']:.3f}  "
                f"(mass_scale={params['mass_scale']:.3f}, "
                f"com_offset_m={params['com_offset_m']:.3f} sampled but NOT applied -- "
                f"see rl/env/body_randomisation.py)")

    def _place(self, x, y, yaw):
        """Put the robot at the episode's start pose.

        Three cases, and they are genuinely different rather than three spellings of one
        thing: simulation teleports, hardware asks a person, and the offline stub just
        assigns. Dispatching on `source` keeps that honest -- an offline test that
        silently used the hardware path would block forever on an input() prompt nobody
        is watching.
        """
        if self._ign is not None:
            # Local floor + a small drop, not a fixed z: a reconstructed floor is not at
            # z = 0 everywhere, and a fixed z embeds the base in it. See floor_height.py.
            if self._floor is None:
                self._floor = FloorHeight(self.world_name)
            z = max(self.spawn_z, self._floor.spawn_z(x, y))
            self._ign.set_pose("pepper_robot", x, y, z=z, yaw=yaw)
            # Teleporting does not zero velocity, so a moving robot keeps moving at the
            # new pose. Without settling, the episode's first observation is taken
            # mid-slide and its first progress term is a gift from the previous episode.
            time.sleep(float(self.cfg["reset"]["settle_s"]))
            # The teleport stands her upright; drop any tilt reading from before it, so a
            # stale "fallen" value cannot end the new episode at its first step.
            if hasattr(self.ros, "_tilt_deg"):
                self.ros._tilt_deg = None
            # Likewise any contact latched before or during the teleport (the robot has just
            # been moved away from whatever it touched).
            if hasattr(self.ros, "_contact"):
                self.ros._contact = False
        elif self.source == "real":
            self._await_manual_placement(x, y, yaw)
            # The operator has just handled her, and placing a robot by hand presses its
            # bumpers. Without this, that press (or a contact latched as the previous
            # route ended) ends the new route at its first step as a "collision". Drain
            # what is queued, then forget it: nothing before Enter belongs to this route.
            deadline = time.time() + 0.5
            while time.time() < deadline:
                self.ros.spin(0.05)
            self.ros._contact = False
            if self.pose_source == "odometry":
                # She is on the tape now: this odometry reading is the route's start pose.
                self.ros.anchor_odometry(x, y, yaw)
        else:
            self.ros._pose = (x, y, yaw)

    def _await_manual_placement(self, x, y, yaw):
        """On hardware there is no teleport. The operator places the robot, and the
        ground-truth pose rig of S4.4.6 confirms where it actually ended up -- the
        recorded start pose must be the measured one, not the intended one, or the real
        l* no longer matches the simulated l* for the same route."""
        where = f"x={x:.2f} y={y:.2f} yaw={math.degrees(yaw):.0f} deg"
        if self.repositioner is None or self.ros._pose is None:
            print(f"\n  place Pepper at {where}, then press Enter...")
            input()
            return
        # rl/env/reposition.py: she can drive most of the way herself; the final squaring
        # up on the tape, and the Enter that anchors the pose, stay with the operator.
        print(f"\n  next start: {where}")
        while True:
            reply = input("  type d + Enter to drive her there, or just Enter once she is "
                          "placed: ").strip().lower()
            if reply != "d":
                return
            self.repositioner.drive_to(x, y, yaw)
            print("  square her up on the cross and heading line by hand.")

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        if self.n_hist:
            # The policy's own action, before scaling or randomisation: what it can know.
            self._act_hist = np.roll(self._act_hist, -1, axis=0)
            self._act_hist[-1] = a
        # [-1, 1] -> [v_min, v_max]. Asymmetric, because reverse is limited far more than
        # forward: Pepper has no rear sensing, so fast reverse is blind.
        v = self.v_min + (a[0] + 1.0) * 0.5 * (self.v_max - self.v_min)
        w = float(a[1]) * self.w_max

        if self.randomiser is not None:
            v, w = self.randomiser.perturb_command(v, w, self.rng)

        # Delayed by the robot's measured latency in sim (identity on hardware). The reward
        # below still prices the command the policy chose (w), not the delayed one.
        self.ros.publish(*self._delay.push(v, w))

        # Hold the command for one control period of SIMULATED time. Sim time, not wall
        # time: under software rendering the RTF is well below 1, so a wall-clock sleep
        # would make the agent's effective control rate a function of machine load.
        target = self.ros.now() + self.control_dt
        t_start_wall = time.time()
        while self.ros.now() < target:
            self.ros.spin(0.01)
            if time.time() - t_start_wall > 10.0 * self.control_dt + 5.0:
                # /clock has stopped. Better to say so than to spin forever inside what
                # looks like a hung training run.
                raise TimeoutError(
                    "simulated time stopped advancing. The Ignition server has probably "
                    "died, or the world is paused (gz.launch.py passes `-r`, so it "
                    "should not be).")
        self._jitter.append(self.ros.now() - target)

        obs, pose = self._observe()
        x, y, _ = pose

        # -- bookkeeping the metrics need, recorded as it happens ----------------------
        step_dist = math.hypot(x - self._last_xy[0], y - self._last_xy[1])
        self._path_length += step_dist          # l_i, Eq. 4.3 and the CR denominator
        self._last_xy = (x, y)

        dist = math.hypot(self._goal[0] - x, self._goal[1] - y)
        collided = self._collided()
        if collided:
            self._collisions += 1               # c_i, Eq. 4.5 numerator

        elapsed = self.ros.now() - self._t0
        reached = dist <= self.goal_tol
        timed_out = elapsed >= self.max_duration

        # -- reward (Chaffre et al., weights in env.yaml) --------------------------------
        r = self.cfg["reward"]
        reward = float(r["step_cost"])
        reward += float(r["progress_per_m"]) * (self._prev_dist - dist)
        reward += float(r["angular_cost_per_rps"]) * abs(w)
        if collided:
            reward += float(r["collision"])
        if reached:
            reward += float(r["goal_reached"])
        self._prev_dist = dist

        # Gymnasium separates the two deliberately, and the distinction is load-bearing
        # for value bootstrapping: `terminated` means the MDP ended (goal or collision)
        # and the value beyond is genuinely zero; `truncated` means the clock ran out and
        # SAC should still bootstrap from the final state. Collapsing them into one flag
        # teaches the policy that time running out is as bad as hitting a wall.
        terminated = bool(reached or collided)
        truncated = bool(timed_out and not terminated)
        if terminated or truncated:
            self.ros.stop()

        info = {
            "route_id": self._route_id,
            "reached_goal": reached,
            "collided": collided,
            "collisions": self._collisions,
            "distance_to_goal_m": dist,
            "path_length_m": self._path_length,
            "optimal_length_m": self._optimal_length,
            "duration_s": elapsed,
            "timed_out": timed_out,
            "jitter_s": float(np.mean(self._jitter)) if self._jitter else 0.0,
            "commanded": (v, w),
        }
        if terminated or truncated:
            # SB3 reads this to log episode statistics, and Task 8 reads the logs rather
            # than re-running anything.
            info["is_success"] = bool(reached and self._collisions == 0 and not timed_out)
        return obs, reward, terminated, truncated, info

    def route_result(self, info):
        """Turn a finished episode into the record rl/metrics.py scores.

        Built here rather than in the evaluator because these are the env's own
        quantities.
        """
        from rl.metrics import RouteResult
        return RouteResult(
            route_id=info["route_id"] or "unnamed",
            reached_goal=bool(info["reached_goal"]),
            final_distance_m=float(info["distance_to_goal_m"]),
            collisions=int(info["collisions"]),
            path_length_m=float(info["path_length_m"]),
            optimal_length_m=float(info["optimal_length_m"] or 0.0),
            duration_s=float(info["duration_s"]),
            timed_out=bool(info["timed_out"]),
        )

    def close(self):
        self.ros.close()
