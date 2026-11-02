#!/usr/bin/env python3
"""Make the simulated Pepper wave with her left arm.

A presentation demo, not part of the experiment. It exists to show on video that the
ported stack actually actuates: the trajectory controllers accept a goal and the model
moves in Ignition. The navigation work uses the wheels and `/cmd_vel`, not the arms.

Run against a live simulation:

    ros2 launch pepper_robot gz.launch.py sensors:=false
    python3 tools/demo_wave.py

The left arm is used because it is the group `ex_follow_target.py` already exercises.
Joint order matches `left_arm_controller` in
`pepper_robot_moveit_config/config/controllers_position.yaml`; every waypoint below sits
inside the limits declared in `pepper_robot_description/urdf/pepper_robot.urdf`.

`build_waypoints()` returns plain numbers, deliberately: `demo_wave_real.py` replays the
same motion on the physical robot through NAOqi, where JointTrajectory does not exist.
The waypoints are the shared thing; the message type is only transport. ROS imports are
made lazily below for the same reason -- the real-robot script imports this module for
its waypoints without needing ROS sourced.
"""

import sys

TOPIC = "/left_arm_controller/joint_trajectory"

JOINTS = [
    "LShoulderPitch",
    "LShoulderRoll",
    "LElbowYaw",
    "LElbowRoll",
    "LWristYaw",
]

# Rest pose, from pepper_robot_description/config/initial_joint_positions.yaml.
HOME = [1.77, 0.17, -1.22, -0.52, 0.0]

# Arm raised and turned out, elbow bent — the pose the wave oscillates around.
# LShoulderPitch decreases to lift the arm (1.77 hangs down, 0.0 is horizontal).
RAISED_PITCH = -0.30
RAISED_ROLL = 0.45
RAISED_ELBOW_YAW = -1.00
WRIST = 0.0

# LElbowRoll is limited to [-1.56207, -0.00872665]; the forearm swings between these.
ELBOW_OUT = -0.35
ELBOW_IN = -1.10

RAISE_TIME = 2.0  # seconds to lift the arm from rest
SWING_TIME = 0.45  # seconds per half-cycle of the wave
CYCLES = 3
RETURN_TIME = 2.0  # seconds to lower the arm back to rest


def build_waypoints():
    """Raise the arm, wave, then return to rest.

    Returns [(positions, time_from_start), ...] in the order of JOINTS. No ROS types, so
    demo_wave_real.py can replay the identical motion through NAOqi.
    """
    def raised(elbow_roll):
        return [RAISED_PITCH, RAISED_ROLL, RAISED_ELBOW_YAW, elbow_roll, WRIST]

    waypoints = []
    elapsed = RAISE_TIME
    waypoints.append((raised(ELBOW_IN), elapsed))

    for _ in range(CYCLES):
        elapsed += SWING_TIME
        waypoints.append((raised(ELBOW_OUT), elapsed))
        elapsed += SWING_TIME
        waypoints.append((raised(ELBOW_IN), elapsed))

    elapsed += RETURN_TIME
    waypoints.append((HOME, elapsed))

    return waypoints


def build_trajectory():
    """Wrap build_waypoints() in the JointTrajectory the simulated controller expects."""
    from builtin_interfaces.msg import Duration
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

    trajectory = JointTrajectory()
    trajectory.joint_names = JOINTS

    waypoints = build_waypoints()
    for positions, time_from_start in waypoints:
        point = JointTrajectoryPoint()
        point.positions = [float(p) for p in positions]
        point.velocities = [0.0] * len(positions)
        point.time_from_start = Duration(
            sec=int(time_from_start),
            nanosec=int((time_from_start % 1.0) * 1e9),
        )
        trajectory.points.append(point)

    return trajectory, waypoints[-1][1]


def main():
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from trajectory_msgs.msg import JointTrajectory

    class WaveDemo(Node):
        def __init__(self):
            super().__init__("demo_wave")
            # JointTrajectoryController's topic interface expects a reliable, depth-1
            # queue.
            self._publisher = self.create_publisher(
                JointTrajectory,
                TOPIC,
                QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE),
            )

        def wait_for_controller(self, timeout=15.0):
            """Block until the controller subscribes, so the message is not dropped."""
            waited = 0.0
            while self._publisher.get_subscription_count() == 0:
                if waited >= timeout:
                    return False
                rclpy.spin_once(self, timeout_sec=0.1)
                waited += 0.1
            return True

        def wave(self):
            trajectory, duration = build_trajectory()
            self._publisher.publish(trajectory)
            self.get_logger().info(
                f"Waving: {len(trajectory.points)} waypoints over {duration:.1f} s."
            )
            # Hold the node open while the controller executes, plus a settling margin.
            end = self.get_clock().now().nanoseconds + int((duration + 1.0) * 1e9)
            while self.get_clock().now().nanoseconds < end:
                rclpy.spin_once(self, timeout_sec=0.1)

    rclpy.init()
    node = WaveDemo()
    try:
        if not node.wait_for_controller():
            node.get_logger().error(
                f"Nothing is subscribed to {TOPIC}. Is the simulation running, and did "
                "left_arm_controller load? Check `ros2 control list_controllers`."
            )
            return 1
        node.wave()
        node.get_logger().info("Done.")
        return 0
    except KeyboardInterrupt:
        return 130
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
