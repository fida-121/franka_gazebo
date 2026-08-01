#!/usr/bin/env python3
"""
htm_ik_move.py

Takes a 4x4 homogeneous transformation matrix (HTM) as the target end-effector
pose, solves inverse kinematics via MoveIt's /compute_ik service, and sends
the result through the impedance bridge node's FollowJointTrajectory action
server -- NOT as a raw single setpoint -- so the motion is smoothly
interpolated instead of jumping straight to the IK solution.

WHY NOT publish the IK solution directly to /joint_impedance/joints_desired:
that reproduces the exact step-input problem that tripped a
power_limit_violation reflex fault earlier. Going through the bridge node's
action interface builds a proper 2-point trajectory (current -> target) that
gets interpolated over time, same as your validated ramp tests.

Usage:
    python3 htm_ik_move.py --htm \
        1 0 0 0.4 \
        0 1 0 0.0 \
        0 0 1 0.4 \
        0 0 0 1 \
        --duration 4.0

    (16 numbers, row-major 4x4 matrix)

Requirements: this assumes move_group is running (moveit_on_hw.launch.py)
and the impedance_bridge_node is running (ros2 run my_impedance_bridge
impedance_bridge_node).
"""

import argparse
import math
import sys

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from moveit_msgs.srv import GetPositionIK
from moveit_msgs.msg import PositionIKRequest, RobotState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory
from builtin_interfaces.msg import Duration as DurationMsg


JOINT_ORDER = [
    'panda_joint1', 'panda_joint2', 'panda_joint3',
    'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7'
]

# Change these to match your actual MoveIt config
PLANNING_GROUP = 'panda_arm'
IK_LINK_NAME = 'panda_link8'     # end-effector link the HTM is defined for
BASE_FRAME = 'panda_link0'       # planning frame / base frame


def rotation_matrix_to_quaternion(R):
    """Standard rotation-matrix-to-quaternion conversion (no external deps)."""
    trace = R[0][0] + R[1][1] + R[2][2]
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2][1] - R[1][2]) * s
        y = (R[0][2] - R[2][0]) * s
        z = (R[1][0] - R[0][1]) * s
    elif R[0][0] > R[1][1] and R[0][0] > R[2][2]:
        s = 2.0 * math.sqrt(1.0 + R[0][0] - R[1][1] - R[2][2])
        w = (R[2][1] - R[1][2]) / s
        x = 0.25 * s
        y = (R[0][1] + R[1][0]) / s
        z = (R[0][2] + R[2][0]) / s
    elif R[1][1] > R[2][2]:
        s = 2.0 * math.sqrt(1.0 + R[1][1] - R[0][0] - R[2][2])
        w = (R[0][2] - R[2][0]) / s
        x = (R[0][1] + R[1][0]) / s
        y = 0.25 * s
        z = (R[1][2] + R[2][1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2][2] - R[0][0] - R[1][1])
        w = (R[1][0] - R[0][1]) / s
        x = (R[0][2] + R[2][0]) / s
        y = (R[1][2] + R[2][1]) / s
        z = 0.25 * s
    return (x, y, z, w)


def htm_to_pose(htm):
    """htm: 4x4 list of lists -> geometry_msgs Pose fields (dict)"""
    R = [row[0:3] for row in htm[0:3]]
    px, py, pz = htm[0][3], htm[1][3], htm[2][3]
    qx, qy, qz, qw = rotation_matrix_to_quaternion(R)
    return {'position': (px, py, pz), 'orientation': (qx, qy, qz, qw)}


class HTMIKMove(Node):
    def __init__(self, htm, duration_s):
        super().__init__('htm_ik_move')
        self.htm = htm
        self.duration_s = duration_s

        self.current_positions = None
        self._sub = self.create_subscription(
            JointState, '/franka/joint_states', self._joint_state_cb, 10)

        self._ik_client = self.create_client(GetPositionIK, '/compute_ik')

        self._action_client = ActionClient(
            self, FollowJointTrajectory,
            'joint_impedance_controller/follow_joint_trajectory')

        self.get_logger().info('Waiting for current joint state...')

    def _joint_state_cb(self, msg: JointState):
        if self.current_positions is not None:
            return
        name_to_pos = dict(zip(msg.name, msg.position))
        try:
            self.current_positions = [name_to_pos[j] for j in JOINT_ORDER]
        except KeyError as e:
            self.get_logger().error(f'Missing joint in joint_states: {e}')
            return
        self.get_logger().info(f'Current joints: {self.current_positions}')
        self._solve_ik()

    def _solve_ik(self):
        if not self._ik_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('/compute_ik service not available. Is move_group running?')
            rclpy.shutdown()
            sys.exit(1)

        pose = htm_to_pose(self.htm)

        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = BASE_FRAME
        pose_stamped.pose.position.x = pose['position'][0]
        pose_stamped.pose.position.y = pose['position'][1]
        pose_stamped.pose.position.z = pose['position'][2]
        pose_stamped.pose.orientation.x = pose['orientation'][0]
        pose_stamped.pose.orientation.y = pose['orientation'][1]
        pose_stamped.pose.orientation.z = pose['orientation'][2]
        pose_stamped.pose.orientation.w = pose['orientation'][3]

        seed_state = RobotState()
        seed_state.joint_state.name = JOINT_ORDER
        seed_state.joint_state.position = self.current_positions

        req = GetPositionIK.Request()
        req.ik_request.group_name = PLANNING_GROUP
        req.ik_request.robot_state = seed_state
        req.ik_request.pose_stamped = pose_stamped
        req.ik_request.ik_link_name = IK_LINK_NAME
        req.ik_request.timeout = DurationMsg(sec=1, nanosec=0)
        req.ik_request.avoid_collisions = True

        future = self._ik_client.call_async(req)
        future.add_done_callback(self._ik_response_cb)

    def _ik_response_cb(self, future):
        result = future.result()
        if result.error_code.val != 1:  # moveit_msgs/MoveItErrorCodes.SUCCESS == 1
            self.get_logger().error(
                f'IK failed, error code {result.error_code.val}. '
                'Target pose may be unreachable or in collision.'
            )
            rclpy.shutdown()
            sys.exit(1)

        name_to_pos = dict(zip(
            result.solution.joint_state.name,
            result.solution.joint_state.position
        ))
        target_positions = [name_to_pos[j] for j in JOINT_ORDER]
        self.get_logger().info(f'IK solution: {target_positions}')
        self._send_trajectory(target_positions)

    def _send_trajectory(self, target_positions):
        if not self._action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error(
                'joint_impedance_controller/follow_joint_trajectory action '
                'server not found. Is impedance_bridge_node running?'
            )
            rclpy.shutdown()
            sys.exit(1)

        traj = JointTrajectory()
        traj.joint_names = JOINT_ORDER

        p0 = JointTrajectoryPoint()
        p0.positions = self.current_positions
        p0.time_from_start = DurationMsg(sec=0, nanosec=0)

        p1 = JointTrajectoryPoint()
        p1.positions = target_positions
        secs = int(self.duration_s)
        nsecs = int((self.duration_s - secs) * 1e9)
        p1.time_from_start = DurationMsg(sec=secs, nanosec=nsecs)

        traj.points = [p0, p1]

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj

        self.get_logger().info(
            f'Sending trajectory to impedance bridge, duration {self.duration_s}s...'
        )
        send_goal_future = self._action_client.send_goal_async(goal)
        send_goal_future.add_done_callback(self._goal_response_cb)

    def _goal_response_cb(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Goal rejected by bridge node.')
            rclpy.shutdown()
            return
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._result_cb)

    def _result_cb(self, future):
        self.get_logger().info('Motion complete.')
        rclpy.shutdown()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--htm', type=float, nargs=16, required=True,
                         help='16 numbers, row-major 4x4 homogeneous transform')
    parser.add_argument('--duration', type=float, default=4.0,
                         help='Motion duration in seconds (default 4.0, keep conservative)')
    args = parser.parse_args()

    htm = [args.htm[0:4], args.htm[4:8], args.htm[8:12], args.htm[12:16]]

    rclpy.init()
    node = HTMIKMove(htm, args.duration)
    rclpy.spin(node)


if __name__ == '__main__':
    main()
