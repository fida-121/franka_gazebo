#!/usr/bin/env python3
"""
pick_and_place.py

Full pick-and-place sequence, chaining together everything built so far:
    1. Move to HOME (standard Franka "ready" joint pose)
    2. Move to the object's HTM pose (via IK + impedance bridge)
    3. Grasp
    4. Move to the place HTM pose (via IK + impedance bridge)
    5. Release (open gripper)
    6. Return to HOME

Each step BLOCKS until the previous one is confirmed complete (using
rclpy.spin_until_future_complete), so the arm never starts moving to the
next target before the previous action has actually finished -- this is
the key requirement for a safe pick-and-place chain.

Requirements running before this script:
    - franka_bringup/franka.launch.py            (hardware)
    - franka_moveit_config/moveit_on_hw.launch.py (for /compute_ik)
    - joint_impedance_controller active (via your usual spawn + switch steps)
    - my_impedance_bridge's impedance_bridge_node running
    - franka_gripper running (gripper.launch.py or load_gripper:=true),
      and gripper already homed at least once this session

Usage:
    python3 pick_and_place.py \
        --object_htm  1 0 0 0.4  0 1 0 0.0  0 0 1 0.2  0 0 0 1 \
        --place_htm   1 0 0 0.4  0 1 0 0.3  0 0 1 0.2  0 0 0 1 \
        --grasp_width 0.03 --grasp_force 40 \
        --open_width 0.08 \
        --move_duration 5.0
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
from moveit_msgs.msg import RobotState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory
from builtin_interfaces.msg import Duration as DurationMsg
from franka_msgs.action import Homing, Move, Grasp


JOINT_ORDER = [
    'panda_joint1', 'panda_joint2', 'panda_joint3',
    'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7'
]

# Standard Franka "ready" pose (radians) -- widely used default home config.
HOME_JOINTS = [0.0, -0.785398, 0.0, -2.356194, 0.0, 1.570796, 0.785398]

PLANNING_GROUP = 'panda_arm'
IK_LINK_NAME = 'panda_link8'
BASE_FRAME = 'panda_link0'


def rotation_matrix_to_quaternion(R):
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
    R = [row[0:3] for row in htm[0:3]]
    px, py, pz = htm[0][3], htm[1][3], htm[2][3]
    qx, qy, qz, qw = rotation_matrix_to_quaternion(R)
    return {'position': (px, py, pz), 'orientation': (qx, qy, qz, qw)}


class PickAndPlace(Node):
    def __init__(self, args):
        super().__init__('pick_and_place')
        self.args = args
        self.current_positions = None

        self.create_subscription(
            JointState, '/franka/joint_states', self._joint_state_cb, 10)

        self.ik_client = self.create_client(GetPositionIK, '/compute_ik')
        self.traj_client = ActionClient(
            self, FollowJointTrajectory,
            'joint_impedance_controller/follow_joint_trajectory')

        self.homing_client = ActionClient(self, Homing, '/panda_gripper/homing')
        self.move_client = ActionClient(self, Move, '/panda_gripper/move')
        self.grasp_client = ActionClient(self, Grasp, '/panda_gripper/grasp')

    def _joint_state_cb(self, msg: JointState):
        name_to_pos = dict(zip(msg.name, msg.position))
        try:
            self.current_positions = [name_to_pos[j] for j in JOINT_ORDER]
        except KeyError:
            pass  # ignore malformed/partial messages

    def wait_for_joint_state(self):
        self.get_logger().info('Waiting for joint state...')
        while self.current_positions is None:
            rclpy.spin_once(self, timeout_sec=0.5)

    # ---- Arm motion (IK + trajectory through the impedance bridge) ----
    def move_to_joint_targets(self, target_positions, duration_s, label):
        if not self.traj_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('impedance bridge action server not found.')
            sys.exit(1)

        traj = JointTrajectory()
        traj.joint_names = JOINT_ORDER

        p0 = JointTrajectoryPoint()
        p0.positions = self.current_positions
        p0.time_from_start = DurationMsg(sec=0, nanosec=0)

        p1 = JointTrajectoryPoint()
        p1.positions = target_positions
        secs = int(duration_s)
        nsecs = int((duration_s - secs) * 1e9)
        p1.time_from_start = DurationMsg(sec=secs, nanosec=nsecs)

        traj.points = [p0, p1]
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj

        self.get_logger().info(f'[{label}] Sending arm move, duration {duration_s}s...')
        send_future = self.traj_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future)
        goal_handle = send_future.result()
        if not goal_handle.accepted:
            self.get_logger().error(f'[{label}] Goal rejected.')
            sys.exit(1)

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        self.get_logger().info(f'[{label}] Arm move complete.')
        self.current_positions = target_positions

    def move_to_htm(self, htm, duration_s, label):
        if not self.ik_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('/compute_ik service not available.')
            sys.exit(1)

        pose = htm_to_pose(htm)
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

        future = self.ik_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        result = future.result()
        if result.error_code.val != 1:
            self.get_logger().error(
                f'[{label}] IK failed, error code {result.error_code.val}.')
            sys.exit(1)

        name_to_pos = dict(zip(
            result.solution.joint_state.name,
            result.solution.joint_state.position))
        target_positions = [name_to_pos[j] for j in JOINT_ORDER]
        self.move_to_joint_targets(target_positions, duration_s, label)

    def move_home(self, duration_s):
        self.move_to_joint_targets(HOME_JOINTS, duration_s, 'HOME')

    # ---- Gripper actions ----
    def gripper_grasp(self, width, speed, force):
        if not self.grasp_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('Gripper grasp action server not found.')
            sys.exit(1)
        goal = Grasp.Goal()
        goal.width = width
        goal.speed = speed
        goal.force = force
        goal.epsilon.inner = 0.01
        goal.epsilon.outer = 0.01
        self.get_logger().info(f'Grasping: width={width}, force={force}...')
        future = self.grasp_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)
        goal_handle = future.result()
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        self.get_logger().info('Grasp complete.')

    def gripper_open(self, width, speed):
        if not self.move_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('Gripper move action server not found.')
            sys.exit(1)
        goal = Move.Goal()
        goal.width = width
        goal.speed = speed
        self.get_logger().info(f'Opening gripper to width={width}...')
        future = self.move_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)
        goal_handle = future.result()
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        self.get_logger().info('Release complete.')

    def run_sequence(self):
        self.wait_for_joint_state()

        self.move_home(self.args.move_duration)
        self.move_to_htm(self.args.object_htm, self.args.move_duration, 'OBJECT')
        self.gripper_grasp(self.args.grasp_width, self.args.grasp_speed, self.args.grasp_force)
        self.move_to_htm(self.args.place_htm, self.args.move_duration, 'PLACE')
        self.gripper_open(self.args.open_width, self.args.open_speed)
        self.move_home(self.args.move_duration)

        self.get_logger().info('Pick-and-place sequence complete.')


def parse_htm(values):
    return [values[0:4], values[4:8], values[8:12], values[12:16]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--object_htm', type=float, nargs=16, required=True)
    parser.add_argument('--place_htm', type=float, nargs=16, required=True)
    parser.add_argument('--move_duration', type=float, default=5.0)
    parser.add_argument('--grasp_width', type=float, default=0.03)
    parser.add_argument('--grasp_speed', type=float, default=0.03)
    parser.add_argument('--grasp_force', type=float, default=40.0)
    parser.add_argument('--open_width', type=float, default=0.08)
    parser.add_argument('--open_speed', type=float, default=0.03)
    args = parser.parse_args()

    args.object_htm = parse_htm(args.object_htm)
    args.place_htm = parse_htm(args.place_htm)

    rclpy.init()
    node = PickAndPlace(args)
    try:
        node.run_sequence()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
