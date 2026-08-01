#!/usr/bin/env python3
"""
htm_ik_server_chomp_seeded.py

Action server 'htm_ik'. Goal: a 4x4 homogeneous transform (row-major, 16 floats)
describing the desired end-effector pose.

Why this version exists: sending a Cartesian pose goal (position + orientation
Constraints) straight to move_group lets it pick ANY joint solution that
satisfies the pose -- the goal-sampling IK call is unseeded, so the elbow can
land anywhere. CHOMP then smooths a path TO that arbitrary elbow config,
which looks like "smooth but still swings the elbow all over".

This version fixes that at the source:
  1. Subscribe to /joint_states, keep the latest reading.
  2. Call /compute_ik directly, passing the CURRENT joint state as the seed
     (robot_state in the request). KDL's IK is a local solver -- given a seed,
     it searches for a solution NEAR that seed, so the resulting joint config
     stays close to the current elbow posture (as long as one exists within
     reach without huge reconfiguration).
  3. Send that specific joint solution to move_group as JOINT constraints
     (not Cartesian constraints), with pipeline_id='chomp'. Since the goal is
     now a single nearby joint configuration, CHOMP just has to optimize a
     smooth path between two nearby joint states, which is a much better
     defined problem than "get to this pose somehow".

If /compute_ik itself returns a solution with a big elbow jump, that means
there ISN'T a nearby joint solution for that pose (e.g. it requires crossing
a singularity or joint limit) -- in that case no amount of trajectory
smoothing downstream will fix it; the target pose itself needs to change, or
you accept that a real reconfiguration is unavoidable to reach it.

Adjust JOINT_NAMES to match your URDF/SRDF group order if it differs.
"""

import math

import rclpy
from rclpy.action import ActionServer, ActionClient, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose, PoseStamped

from moveit_msgs.srv import GetPositionIK
from moveit_msgs.msg import (
    PositionIKRequest,
    RobotState,
    Constraints,
    JointConstraint,
    MotionPlanRequest,
    PlanningOptions,
)
from moveit_msgs.action import MoveGroup

from htm_ik_interfaces.action import HTMIK

# ---- Adjust these to match your setup -------------------------------------
PLANNING_FRAME = 'panda_link0'
GROUP_NAME = 'panda_manipulator'
EE_LINK = 'panda_hand_tcp'
PLANNING_PIPELINE = 'chomp'

# Order matters for building RobotState / JointConstraint lists cleanly, but
# lookups below are done by name so exact order in this list isn't critical.
JOINT_NAMES = [
    'panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4',
    'panda_joint5', 'panda_joint6', 'panda_joint7',
]

IK_TIMEOUT_SEC = 0.5
IK_ATTEMPTS = 5                     # retry compute_ik a few times if it fails
JOINT_GOAL_TOLERANCE = 0.01         # radians, tolerance on each joint goal
ALLOWED_PLANNING_TIME = 5.0
VEL_SCALING = 0.3
ACC_SCALING = 0.3
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

        self.latest_joint_state = None
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self._joint_state_cb, 10,
            callback_group=self.cb_group)

        self.ik_client = self.create_client(
            GetPositionIK, '/compute_ik', callback_group=self.cb_group)

        self.move_group_client = ActionClient(
            self, MoveGroup, '/move_action', callback_group=self.cb_group)

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
            f'htm_ik action server ready (seeded IK -> {PLANNING_PIPELINE}).')

    def _joint_state_cb(self, msg):
        self.latest_joint_state = msg

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

    def build_seed_robot_state(self):
        """Build a RobotState from the latest /joint_states, restricted to
        the arm's joints (drop gripper/finger joints etc.)."""
        if self.latest_joint_state is None:
            return None

        name_to_pos = dict(zip(
            self.latest_joint_state.name, self.latest_joint_state.position))

        js = JointState()
        js.name = []
        js.position = []
        for jn in JOINT_NAMES:
            if jn not in name_to_pos:
                self.get_logger().error(
                    f'Joint {jn} not found in /joint_states -- check JOINT_NAMES.')
                return None
            js.name.append(jn)
            js.position.append(name_to_pos[jn])

        rs = RobotState()
        rs.joint_state = js
        rs.is_diff = False
        return rs

    async def solve_seeded_ik(self, target_pose, seed_robot_state):
        req = GetPositionIK.Request()
        ik_req = PositionIKRequest()
        ik_req.group_name = GROUP_NAME
        ik_req.robot_state = seed_robot_state
        ik_req.avoid_collisions = True
        ik_req.ik_link_name = EE_LINK
        ik_req.pose_stamped = PoseStamped()
        ik_req.pose_stamped.header.frame_id = PLANNING_FRAME
        ik_req.pose_stamped.header.stamp = self.get_clock().now().to_msg()
        ik_req.pose_stamped.pose = target_pose
        ik_req.timeout.sec = int(IK_TIMEOUT_SEC)
        ik_req.timeout.nanosec = int((IK_TIMEOUT_SEC % 1.0) * 1e9)
        req.ik_request = ik_req

        for attempt in range(IK_ATTEMPTS):
            future = self.ik_client.call_async(req)
            resp = await future
            if resp.error_code.val == 1:  # SUCCESS
                return resp.solution
            self.get_logger().warn(
                f'compute_ik attempt {attempt + 1}/{IK_ATTEMPTS} failed, '
                f'error_code={resp.error_code.val}')
        return None

    async def execute_callback(self, goal_handle):
        feedback = HTMIK.Feedback()
        result = HTMIK.Result()

        feedback.status = 'Waiting for current joint state...'
        goal_handle.publish_feedback(feedback)

        seed_robot_state = self.build_seed_robot_state()
        if seed_robot_state is None:
            result.success = False
            result.message = 'No /joint_states received yet (or JOINT_NAMES mismatch)'
            result.fraction = 0.0
            goal_handle.abort()
            return result

        target_pose = self.htm_to_pose(list(goal_handle.request.htm))

        feedback.status = 'Solving IK seeded from current joint state...'
        goal_handle.publish_feedback(feedback)

        if not self.ik_client.wait_for_service(timeout_sec=5.0):
            result.success = False
            result.message = '/compute_ik service not available'
            result.fraction = 0.0
            goal_handle.abort()
            return result

        ik_solution = await self.solve_seeded_ik(target_pose, seed_robot_state)
        if ik_solution is None:
            result.success = False
            result.message = (
                'Seeded IK failed -- no nearby joint solution found for this '
                'pose (may require a large reconfiguration or be unreachable).'
            )
            result.fraction = 0.0
            goal_handle.abort()
            return result

        # Build joint-space goal constraints from the seeded IK solution.
        name_to_pos = dict(zip(
            ik_solution.joint_state.name, ik_solution.joint_state.position))

        constraints = Constraints()
        joint_constraints = []
        for jn in JOINT_NAMES:
            if jn not in name_to_pos:
                continue
            jc = JointConstraint()
            jc.joint_name = jn
            jc.position = name_to_pos[jn]
            jc.tolerance_above = JOINT_GOAL_TOLERANCE
            jc.tolerance_below = JOINT_GOAL_TOLERANCE
            jc.weight = 1.0
            joint_constraints.append(jc)
        constraints.joint_constraints = joint_constraints

        motion_req = MotionPlanRequest()
        motion_req.group_name = GROUP_NAME
        motion_req.pipeline_id = PLANNING_PIPELINE
        motion_req.goal_constraints = [constraints]
        motion_req.allowed_planning_time = ALLOWED_PLANNING_TIME
        motion_req.num_planning_attempts = 1
        motion_req.max_velocity_scaling_factor = VEL_SCALING
        motion_req.max_acceleration_scaling_factor = ACC_SCALING

        planning_options = PlanningOptions()
        planning_options.plan_only = False
        planning_options.replan = False

        move_goal = MoveGroup.Goal()
        move_goal.request = motion_req
        move_goal.planning_options = planning_options

        feedback.status = f'Planning+executing via {PLANNING_PIPELINE} to seeded joint goal...'
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

        move_result_future = move_goal_handle.get_result_async()
        move_result = await move_result_future

        error_code = move_result.result.error_code.val
        if error_code == 1:
            result.success = True
            result.message = f'Executed successfully via seeded {PLANNING_PIPELINE} plan.'
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
