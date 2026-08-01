#!/usr/bin/env python3
"""
impedance_bridge_node.py

Bridges a MoveIt-planned trajectory (trajectory_msgs/JointTrajectory) into
franka_example_controllers/JointImpedanceController's expected input:
    topic: /joint_impedance/joints_desired
    msg:   std_msgs/msg/Float64MultiArray  (7 doubles, joint1..joint7 order)

WHY THIS NODE EXISTS:
JointImpedanceController has no FollowJointTrajectory action interface --
it only accepts single position setpoints on a topic. MoveIt's execution
pipeline expects to send a FollowJointTrajectory action goal to a controller.
This node sits in between: it exposes that action interface, and internally
converts each incoming trajectory into a steady stream of small setpoint
steps published to the impedance controller's topic.

THREADING NOTE (read this if you're new to rclpy):
The action callback (execute_callback) needs to run *at the same time* as
the timer callback that publishes each step -- not one blocking the other.
By default, rclpy uses a SINGLE-THREADED executor, meaning only one callback
runs at a time, ever. If execute_callback blocks (e.g. via time.sleep) while
waiting for the trajectory to finish, the timer callback that's supposed to
be publishing setpoints never gets a chance to run, and nothing moves.

The fix used below: a MultiThreadedExecutor with enough threads, combined
with a ReentrantCallbackGroup on every callback that needs to interleave.
This lets the action callback and the timer callback genuinely run
concurrently instead of fighting over one thread.
"""

import bisect

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory
from control_msgs.action import FollowJointTrajectory


DEFAULT_JOINT_ORDER = [
    'panda_joint1', 'panda_joint2', 'panda_joint3',
    'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7'
]


class ImpedanceBridgeNode(Node):
    def __init__(self):
        super().__init__('impedance_bridge_node')

        # --- Parameters: values you can override at launch time without
        # editing code, e.g. `ros2 run my_impedance_bridge impedance_bridge_node
        # --ros-args -p publish_rate_hz:=200.0` ---
        self.declare_parameter('publish_rate_hz', 500.0)
        self.declare_parameter('joint_order', DEFAULT_JOINT_ORDER)
        self.declare_parameter('setpoint_topic', '/joint_impedance/joints_desired')

        self.publish_rate = self.get_parameter('publish_rate_hz').value
        self.joint_order = list(self.get_parameter('joint_order').value)
        setpoint_topic = self.get_parameter('setpoint_topic').value

        # A single callback group shared by everything that needs to run
        # concurrently. Callbacks in the SAME reentrant group can preempt
        # each other freely; this is what makes the timer tick while the
        # action goal is still "in progress".
        self._cb_group = ReentrantCallbackGroup()

        # --- Publisher: sends setpoints to the impedance controller ---
        self.pub = self.create_publisher(Float64MultiArray, setpoint_topic, 10)

        # --- Action server: the thing that makes this node look, from
        # MoveIt's perspective, like a controller that can execute
        # trajectories ---
        self._action_server = ActionServer(
            self,
            FollowJointTrajectory,
            'joint_impedance_controller/follow_joint_trajectory',
            execute_callback=self.execute_callback,
            callback_group=self._cb_group,
        )

        # --- Internal trajectory-stepping state ---
        self._traj_points = []
        self._traj_joint_names = []
        self._times = []
        self._elapsed = 0.0
        self._period = 1.0 / self.publish_rate
        self._active = False

        # --- Timer: this is what actually publishes each interpolated
        # step, once a trajectory has been loaded ---
        self._timer = self.create_timer(
            self._period, self._tick, callback_group=self._cb_group)

        self.get_logger().info(
            f'Impedance bridge ready. Publishing to {setpoint_topic} '
            f'at {self.publish_rate} Hz. Action server: '
            f'joint_impedance_controller/follow_joint_trajectory'
        )

    # ---- Direct programmatic entry point (use this from your own script
    # that already called move_group.plan() -- no action goal needed) ----
    def send_trajectory_from_msg(self, joint_trajectory_msg: JointTrajectory):
        self._load_trajectory(joint_trajectory_msg)
        self._elapsed = 0.0
        self._active = True

    # ---- FollowJointTrajectory action callback ----
    def execute_callback(self, goal_handle):
        traj = goal_handle.request.trajectory
        self._load_trajectory(traj)
        self._elapsed = 0.0
        self._active = True

        # Poll until the timer callback finishes the trajectory. This loop
        # itself does NOT block the timer, because it's running in a
        # reentrant callback group under a multi-threaded executor -- the
        # timer callback can still fire concurrently on another thread.
        import time
        total_duration = self._times[-1] if self._times else 0.0
        deadline = time.time() + total_duration + (2.0 / self.publish_rate) + 0.5
        while self._active and time.time() < deadline:
            time.sleep(0.01)

        goal_handle.succeed()
        result = FollowJointTrajectory.Result()
        result.error_code = FollowJointTrajectory.Result.SUCCESSFUL
        return result

    def _load_trajectory(self, traj: JointTrajectory):
        self._traj_joint_names = list(traj.joint_names)
        self._traj_points = traj.points
        if not self._traj_points:
            self.get_logger().warn('Received empty trajectory, ignoring.')
            self._times = []
            return

        self._times = [
            p.time_from_start.sec + p.time_from_start.nanosec * 1e-9
            for p in self._traj_points
        ]
        self.get_logger().info(
            f'Loaded trajectory with {len(self._traj_points)} points, '
            f'duration {self._times[-1]:.3f}s'
        )

    def _reorder(self, positions):
        """Map an incoming point's positions (in the trajectory's own joint
        order) into self.joint_order (what the impedance controller expects:
        panda_joint1..panda_joint7, strictly in that order)."""
        name_to_pos = dict(zip(self._traj_joint_names, positions))
        return [name_to_pos[j] for j in self.joint_order]

    def _interpolate_at(self, t):
        times = self._times
        if t <= times[0]:
            return self._reorder(self._traj_points[0].positions)
        if t >= times[-1]:
            return self._reorder(self._traj_points[-1].positions)

        idx = bisect.bisect_right(times, t) - 1
        idx = max(0, min(idx, len(times) - 2))
        t0, t1 = times[idx], times[idx + 1]
        p0 = self._reorder(self._traj_points[idx].positions)
        p1 = self._reorder(self._traj_points[idx + 1].positions)
        alpha = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
        return [a + alpha * (b - a) for a, b in zip(p0, p1)]

    def _tick(self):
        """Runs every self._period seconds, always -- but only publishes
        while a trajectory is actively loaded (self._active)."""
        if not self._active or not self._traj_points:
            return

        positions = self._interpolate_at(self._elapsed)

        # WORKAROUND: JointImpedanceController::desiredJointCallback drops
        # the ENTIRE message if joint1's value is exactly 0.0 (a truthiness
        # bug in the controller's callback guard). Nudge it by a negligible
        # epsilon so the guard never trips. Prefer patching the controller
        # source instead -- this is a fallback only.
        if positions[0] == 0.0:
            positions[0] = 1e-6

        msg = Float64MultiArray()
        msg.data = positions
        self.pub.publish(msg)

        self._elapsed += self._period
        if self._elapsed > self._times[-1] + self._period:
            self._active = False


def main():
    rclpy.init()
    node = ImpedanceBridgeNode()

    # MultiThreadedExecutor: allows callbacks in the reentrant group (the
    # action callback and the timer callback) to run concurrently instead
    # of blocking one another on a single thread.
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
