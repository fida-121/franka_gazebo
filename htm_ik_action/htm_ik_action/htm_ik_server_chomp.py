#!/usr/bin/env python3
"""
htm_ik_server_chomp.py

Action server 'htm_ik'. Goal: a 4x4 homogeneous transform (row-major, 16 floats)
describing the desired end-effector pose.

Unlike the Cartesian-path version, this node sends the target pose to
move_group's MoveGroup action as a set of goal Constraints (position +
orientation), with pipeline_id='chomp'. CHOMP optimizes a smooth joint-space
trajectory starting from an interpolation between the current and target
joint states, so it does NOT force a straight-line end-effector path, but it
also does NOT reshuffle the elbow the way independent per-waypoint KDL IK
solves (as used by /compute_cartesian_path) can.

If you need the elbow to *provably* stay up rather than just "probably stay
up because CHOMP prefers smoothness", add a joint constraint on panda_joint4
(or whichever joint controls elbow height) -- see ELBOW_JOINT_CONSTRAINT below.

Adjust the constants below (PLANNING_FRAME, GROUP_NAME, EE_LINK) to match your
SRDF / URDF if they differ.

Example goal send:
  ros2 action send_goal /htm_ik htm_ik_interfaces/action/HTMIK \
    "{htm: [0.707, -0.707, 0.0, 0.4, \
             -0.707, -0.707, 0.0, 0.0, \
             0.0, 0.0, -1.0, 0.4, \
             0.0, 0.0, 0.0, 1.0]}"
"""

import math

import rclpy
from rclpy.action import ActionServer, ActionClient, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from tf2_ros import Buffer, TransformListener
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException

from geometry_msgs.msg import Pose, PoseStamped
from shape_msgs.msg import SolidPrimitive
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    Constraints,
    PositionConstraint,
    OrientationConstraint,
    JointConstraint,
    BoundingVolume,
    MotionPlanRequest,
    PlanningOptions,
)

from htm_ik_interfaces.action import HTMIK

# ---- Adjust these to match your setup -------------------------------------
PLANNING_FRAME = 'panda_link0'      # MoveIt planning/model frame
GROUP_NAME = 'panda_manipulator'    # SRDF planning group name
EE_LINK = 'panda_hand_tcp'          # end-effector link the goal pose refers to
PLANNING_PIPELINE = 'chomp'         # 'ompl' or 'chomp' -- must match a pipeline
                                     # declared in planning_pipelines_config in
                                     # your move_group launch file
ALLOWED_PLANNING_TIME = 5.0         # seconds
POSITION_TOLERANCE = 0.001          # meters, radius of goal position sphere
ORIENTATION_TOLERANCE = 0.01        # radians, per-axis
VEL_SCALING = 0.3
ACC_SCALING = 0.3

# Optional: keep a specific joint close to its current value (e.g. to bias
# CHOMP toward keeping the elbow up). Set ELBOW_JOINT_NAME to None to disable.
ELBOW_JOINT_NAME = None             # e.g. 'panda_joint4'
ELBOW_JOINT_TOLERANCE = 0.3         # radians, allowed deviation from current value
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

        self.move_group_client = ActionClient(
            self, MoveGroup, '/move_action',
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

        self.get_logger().info(
            f'htm_ik action server ready (pipeline={PLANNING_PIPELINE}).')

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

    def build_goal_constraints(self, target_pose):
        constraints = Constraints()

        # ---- Position constraint: small sphere around target position ----
        pos_constraint = PositionConstraint()
        pos_constraint.header.frame_id = PLANNING_FRAME
        pos_constraint.link_name = EE_LINK
        pos_constraint.target_point_offset.x = 0.0
        pos_constraint.target_point_offset.y = 0.0
        pos_constraint.target_point_offset.z = 0.0

        sphere = SolidPrimitive()
        sphere.type = SolidPrimitive.SPHERE
        sphere.dimensions = [POSITION_TOLERANCE]

        bv = BoundingVolume()
        bv.primitives = [sphere]
        bv.primitive_poses = [target_pose]
        pos_constraint.constraint_region = bv
        pos_constraint.weight = 1.0
        constraints.position_constraints = [pos_constraint]

        # ---- Orientation constraint: target quaternion with small tolerance ----
        ori_constraint = OrientationConstraint()
        ori_constraint.header.frame_id = PLANNING_FRAME
        ori_constraint.link_name = EE_LINK
        ori_constraint.orientation = target_pose.orientation
        ori_constraint.absolute_x_axis_tolerance = ORIENTATION_TOLERANCE
        ori_constraint.absolute_y_axis_tolerance = ORIENTATION_TOLERANCE
        ori_constraint.absolute_z_axis_tolerance = ORIENTATION_TOLERANCE
        ori_constraint.weight = 1.0
        constraints.orientation_constraints = [ori_constraint]

        # ---- Optional joint constraint to bias elbow posture ----
        if ELBOW_JOINT_NAME is not None:
            current_val = self.get_current_joint_value(ELBOW_JOINT_NAME)
            if current_val is not None:
                jc = JointConstraint()
                jc.joint_name = ELBOW_JOINT_NAME
                jc.position = current_val
                jc.tolerance_above = ELBOW_JOINT_TOLERANCE
                jc.tolerance_below = ELBOW_JOINT_TOLERANCE
                jc.weight = 1.0
                constraints.joint_constraints = [jc]

        return constraints

    def get_current_joint_value(self, joint_name):
        # NOTE: reading live joint values without moveit_commander/PlanningSceneMonitor
        # access requires subscribing to /joint_states. Left as a stub -- wire this
        # up to a /joint_states subscriber if you enable ELBOW_JOINT_NAME.
        self.get_logger().warn(
            f'get_current_joint_value({joint_name}) not implemented -- '
            'skipping elbow joint constraint. Wire up a /joint_states '
            'subscriber if you need this.')
        return None

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

        feedback.status = 'Building goal constraints...'
        goal_handle.publish_feedback(feedback)

        target_pose = self.htm_to_pose(list(goal_handle.request.htm))
        constraints = self.build_goal_constraints(target_pose)

        motion_req = MotionPlanRequest()
        motion_req.group_name = GROUP_NAME
        motion_req.pipeline_id = PLANNING_PIPELINE
        motion_req.goal_constraints = [constraints]
        motion_req.allowed_planning_time = ALLOWED_PLANNING_TIME
        motion_req.num_planning_attempts = 1
        motion_req.max_velocity_scaling_factor = VEL_SCALING
        motion_req.max_acceleration_scaling_factor = ACC_SCALING

        planning_options = PlanningOptions()
        planning_options.plan_only = False          # plan AND execute
        planning_options.replan = False

        move_goal = MoveGroup.Goal()
        move_goal.request = motion_req
        move_goal.planning_options = planning_options

        feedback.status = f'Sending goal to move_group (pipeline={PLANNING_PIPELINE})...'
        goal_handle.publish_feedback(feedback)

        if not self.move_group_client.wait_for_server(timeout_sec=5.0):
            result.success = False
            result.message = '/move_action action server not available'
            result.fraction = 0.0
            goal_handle.abort()
            return result

        send_future = self.move_group_client.send_goal_async(move_goal)
        move_goal_handle = await send_future

        if not move_goal_handle.accepted:
            result.success = False
            result.message = 'Goal was rejected by move_group'
            result.fraction = 0.0
            goal_handle.abort()
            return result

        feedback.status = 'Planning and executing...'
        goal_handle.publish_feedback(feedback)

        move_result_future = move_goal_handle.get_result_async()
        move_result = await move_result_future

        error_code = move_result.result.error_code.val
        if error_code == 1:  # moveit_msgs/MoveItErrorCodes.SUCCESS
            result.success = True
            result.message = f'Executed successfully via {PLANNING_PIPELINE} pipeline.'
            result.fraction = 1.0
            goal_handle.succeed()
        else:
            result.success = False
            result.message = f'Planning/execution failed, MoveItErrorCodes={error_code}'
            result.fraction = 0.0
            goal_handle.abort()

        return result


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
