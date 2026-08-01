#!/usr/bin/env python3
"""
htm_ik_server.py

Action server 'htm_ik'. Goal: a 4x4 homogeneous transform (row-major, 16 floats)
describing the desired end-effector pose. This node does NOT solve IK itself --
it hands the target pose to MoveIt (move_group) via the /compute_cartesian_path
service, so the robot moves in a STRAIGHT LINE from its current EE pose to the
target, instead of an arbitrary joint-space path. The resulting trajectory is
then sent to MoveIt's /execute_trajectory action.

Adjust the constants below (PLANNING_FRAME, GROUP_NAME, EE_LINK) to match your
SRDF / URDF if they differ.

Example goal send (matches the fixed-orientation matrix you gave, x=0.4 y=0.0 z=0.4):
  ros2 action send_goal /htm_ik htm_ik_interfaces/action/HTMIK \
    "{htm: [0.707, -0.707, 0.0, 0.4, \
             -0.707, -0.707, 0.0, 0.0, \
             0.0, 0.0, -1.0, 0.4, \
             0.0, 0.0, 0.0, 1.0]}"
"""

import math

import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from tf2_ros import Buffer, TransformListener
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException

from geometry_msgs.msg import Pose, PoseStamped
from moveit_msgs.srv import GetCartesianPath
from moveit_msgs.action import ExecuteTrajectory
from moveit_msgs.msg import RobotTrajectory

from htm_ik_interfaces.action import HTMIK

# ---- Adjust these to match your setup -------------------------------------
PLANNING_FRAME = 'panda_link0'      # MoveIt planning/model frame
GROUP_NAME = 'panda_manipulator'    # SRDF planning group name
EE_LINK = 'panda_hand_tcp'          # end-effector link used for the Cartesian path
MAX_STEP = 0.01                     # meters, Cartesian interpolation resolution
JUMP_THRESHOLD = 0.0                # 0.0 disables the jump-detection check
MIN_ACCEPTABLE_FRACTION = 0.95      # reject partial paths below this
# -----------------------------------------------------------------------------


def rotation_matrix_to_quaternion(r):
    """r is a 3x3 nested list. Returns (x, y, z, w). Shepperd's method."""
    m00, m01, m02 = r[0]
    m10, m11, m12 = r[1]
    m20, m21, m22 = r[2]

    trace = m00 + m11 + m22
    if trace > 0.0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m21 - m12) * s
        y = (m02 - m20) * s
        z = (m10 - m01) * s
    elif m00 > m11 and m00 > m22:
        s = 2.0 * math.sqrt(1.0 + m00 - m11 - m22)
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = 2.0 * math.sqrt(1.0 + m11 - m00 - m22)
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = 2.0 * math.sqrt(1.0 + m22 - m00 - m11)
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s

    return x, y, z, w


class HTMIKServer(Node):
    def __init__(self):
        super().__init__('htm_ik_server')

        self.cb_group = ReentrantCallbackGroup()

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.cartesian_client = self.create_client(
            GetCartesianPath, '/compute_cartesian_path',
            callback_group=self.cb_group)

        from rclpy.action import ActionClient
        self.execute_action_client = ActionClient(
            self, ExecuteTrajectory, '/execute_trajectory',
            callback_group=self.cb_group)

        self._action_server = ActionServer(
            self,
            HTMIK,
            'htm_ik',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=self.cb_group,
        )

        self.get_logger().info('htm_ik action server ready.')

    def goal_callback(self, goal_request):
        if len(goal_request.htm) != 16:
            self.get_logger().warn('Rejecting goal: htm must have exactly 16 elements.')
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        return CancelResponse.ACCEPT

    def htm_to_pose(self, htm):
        m = htm
        r = [
            [m[0], m[1], m[2]],
            [m[4], m[5], m[6]],
            [m[8], m[9], m[10]],
        ]
        x, y, z = m[3], m[7], m[11]
        qx, qy, qz, qw = rotation_matrix_to_quaternion(r)

        pose = Pose()
        pose.position.x = x
        pose.position.y = y
        pose.position.z = z
        pose.orientation.x = qx
        pose.orientation.y = qy
        pose.orientation.z = qz
        pose.orientation.w = qw
        return pose

    def get_current_ee_pose(self):
        try:
            tf = self.tf_buffer.lookup_transform(
                PLANNING_FRAME, EE_LINK, rclpy.time.Time())
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().error(f'TF lookup failed: {e}')
            return None

        pose = Pose()
        pose.position.x = tf.transform.translation.x
        pose.position.y = tf.transform.translation.y
        pose.position.z = tf.transform.translation.z
        pose.orientation = tf.transform.rotation
        return pose

    async def execute_callback(self, goal_handle):
        feedback = HTMIK.Feedback()
        result = HTMIK.Result()

        feedback.status = 'Reading current end-effector pose from TF...'
        goal_handle.publish_feedback(feedback)

        current_pose = self.get_current_ee_pose()
        if current_pose is None:
            result.success = False
            result.message = f'Could not look up TF {PLANNING_FRAME} -> {EE_LINK}'
            result.fraction = 0.0
            goal_handle.abort()
            return result

        target_pose = self.htm_to_pose(list(goal_handle.request.htm))

        feedback.status = 'Requesting Cartesian path from move_group...'
        goal_handle.publish_feedback(feedback)

        req = GetCartesianPath.Request()
        req.header = self._stamped_header()
        req.group_name = GROUP_NAME
        req.link_name = EE_LINK
        req.waypoints = [current_pose, target_pose]
        req.max_step = MAX_STEP
        req.jump_threshold = JUMP_THRESHOLD
        req.avoid_collisions = True

        if not self.cartesian_client.wait_for_service(timeout_sec=5.0):
            result.success = False
            result.message = '/compute_cartesian_path service not available'
            result.fraction = 0.0
            goal_handle.abort()
            return result

        self.get_logger().info(
            f'Current EE pose: pos=({current_pose.position.x:.3f}, '
            f'{current_pose.position.y:.3f}, {current_pose.position.z:.3f}) '
            f'orient=({current_pose.orientation.x:.3f}, {current_pose.orientation.y:.3f}, '
            f'{current_pose.orientation.z:.3f}, {current_pose.orientation.w:.3f})'
        )
        self.get_logger().info(
            f'Target EE pose: pos=({target_pose.position.x:.3f}, '
            f'{target_pose.position.y:.3f}, {target_pose.position.z:.3f}) '
            f'orient=({target_pose.orientation.x:.3f}, {target_pose.orientation.y:.3f}, '
            f'{target_pose.orientation.z:.3f}, {target_pose.orientation.w:.3f})'
        )

        cart_future = self.cartesian_client.call_async(req)
        cart_resp = await cart_future

        result.fraction = cart_resp.fraction
        self.get_logger().info(
            f'Cartesian path fraction achieved: {cart_resp.fraction:.3f}, '
            f'error_code={cart_resp.error_code.val}'
        )

        if cart_resp.fraction < MIN_ACCEPTABLE_FRACTION:
            result.success = False
            result.message = (
                f'Only {cart_resp.fraction*100:.1f}% of the straight-line path was '
                f'reachable/collision-free (need >= {MIN_ACCEPTABLE_FRACTION*100:.0f}%).'
            )
            goal_handle.abort()
            return result

        feedback.status = 'Executing trajectory...'
        goal_handle.publish_feedback(feedback)

        exec_goal = ExecuteTrajectory.Goal()
        exec_goal.trajectory = cart_resp.solution

        if not self.execute_action_client.wait_for_server(timeout_sec=5.0):
            result.success = False
            result.message = '/execute_trajectory action server not available'
            result.fraction = cart_resp.fraction
            goal_handle.abort()
            return result

        send_future = self.execute_action_client.send_goal_async(exec_goal)
        exec_goal_handle = await send_future

        if not exec_goal_handle.accepted:
            result.success = False
            result.message = 'Execution goal was rejected by move_group'
            result.fraction = cart_resp.fraction
            goal_handle.abort()
            return result

        exec_result_future = exec_goal_handle.get_result_async()
        exec_result = await exec_result_future

        error_code = exec_result.result.error_code.val
        if error_code == 1:  # moveit_msgs/MoveItErrorCodes.SUCCESS
            result.success = True
            result.message = 'Executed successfully.'
            goal_handle.succeed()
        else:
            result.success = False
            result.message = f'Execution failed, MoveItErrorCodes={error_code}'
            goal_handle.abort()

        return result

    def _stamped_header(self):
        h = PoseStamped().header
        h.frame_id = PLANNING_FRAME
        h.stamp = self.get_clock().now().to_msg()
        return h


def main():
    rclpy.init()
    node = HTMIKServer()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
