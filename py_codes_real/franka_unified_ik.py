#!/usr/bin/env python3
"""
Franka Panda Unified IK — works on both MuJoCo sim and real robot.

Toggle USE_REAL_ROBOT at the top — that is the ONLY line to change.

Usage:
  python3 franka_unified_ik.py                          # default target
  python3 franka_unified_ik.py 0.4 0.1 0.4              # x y z
  python3 franka_unified_ik.py 0.4 0.1 0.4 0 0.924 0 0.383  # x y z qx qy qz qw
"""

import sys
import math
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import PoseStamped
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    MotionPlanRequest, WorkspaceParameters, Constraints,
    RobotState, DisplayTrajectory
)
from moveit_msgs.srv import GetPositionIK, GetPositionFK
from sensor_msgs.msg import JointState
from tf2_ros import Buffer, TransformListener
import rclpy.time

# ══════════════════════════════════════════════════════════════
#  THE ONLY LINE YOU CHANGE BETWEEN SIM AND REAL ROBOT
USE_REAL_ROBOT = True   # False = MuJoCo sim,  True = real robot
# ══════════════════════════════════════════════════════════════

EE_LINK            = 'panda_link8' if USE_REAL_ROBOT else 'panda_hand'
MAX_VELOCITY_SCALE = 0.1           if USE_REAL_ROBOT else 0.3
MAX_ACCEL_SCALE    = 0.1           if USE_REAL_ROBOT else 0.3

JOINT_NAMES = [
    'panda_joint1', 'panda_joint2', 'panda_joint3',
    'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7'
]

# ── Seed states ───────────────────────────────────────────────────────────────
SEED_ELBOW_UP = [
    0.0,    # joint1
    -0.3,   # joint2
    0.0,    # joint3
    -2.4,   # joint4  ← elbow high
    0.0,    # joint5
    2.1,    # joint6
    0.785   # joint7
]

SEED_ELBOW_DOWN = [
    0.0,    # joint1
    0.5,    # joint2
    0.0,    # joint3
    -0.8,   # joint4  ← elbow low
    0.0,    # joint5
    1.3,    # joint6
    0.785   # joint7
]


class PandaElbowPicker(Node):
    def __init__(self):
        super().__init__('panda_elbow_picker')

        self._action_client = ActionClient(self, MoveGroup, '/move_action')
        self._ik_client     = self.create_client(GetPositionIK, '/compute_ik')
        self._fk_client     = self.create_client(GetPositionFK, '/compute_fk')

        self._display_pub = self.create_publisher(
            DisplayTrajectory, '/display_planned_path', 10)

        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.joint_state = None
        self.create_subscription(JointState, '/joint_states', self._js_cb, 10)

    def _js_cb(self, msg):
        self.joint_state = msg

    def _wait_joint_state(self, timeout=5.0):
        start = self.get_clock().now()
        while self.joint_state is None:
            rclpy.spin_once(self, timeout_sec=0.1)
            if (self.get_clock().now() - start).nanoseconds / 1e9 > timeout:
                return False
        return True

    # ── FK ────────────────────────────────────────────────────────────────────
    def fk(self, positions):
        if not self._fk_client.wait_for_service(timeout_sec=3.0):
            return None
        req = GetPositionFK.Request()
        req.header.frame_id = 'panda_link0'
        req.fk_link_names   = [EE_LINK]                    # ← uses EE_LINK, not hardcoded
        req.robot_state.joint_state.name     = JOINT_NAMES
        req.robot_state.joint_state.position = list(positions)
        future = self._fk_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=3.0)
        if future.result() and future.result().error_code.val == 1:
            return future.result().pose_stamped[0].pose
        return None

    # ── IK with seed ──────────────────────────────────────────────────────────
    def ik_with_seed(self, target_pose: PoseStamped, seed_joints: list, label: str):
        if not self._ik_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('IK service not available')
            return None

        req = GetPositionIK.Request()
        req.ik_request.group_name       = 'panda_arm'
        req.ik_request.pose_stamped     = target_pose
        req.ik_request.ik_link_name     = EE_LINK          # ← uses EE_LINK, not hardcoded
        req.ik_request.timeout.sec      = 5
        req.ik_request.avoid_collisions = True

        seed_js = JointState()
        seed_js.name     = JOINT_NAMES
        seed_js.position = seed_joints
        req.ik_request.robot_state.joint_state = seed_js

        self.get_logger().info(f'  Computing IK ({label})...')
        future = self._ik_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=6.0)

        if not future.result():
            self.get_logger().warn(f'  IK ({label}): no response')
            return None

        error = future.result().error_code.val
        if error != 1:
            self.get_logger().warn(f'  IK ({label}): failed (code {error})')
            return None

        sol_js = future.result().solution.joint_state
        js_dict = dict(zip(sol_js.name, sol_js.position))
        try:
            positions = [js_dict[j] for j in JOINT_NAMES]
        except KeyError:
            self.get_logger().warn(f'  IK ({label}): incomplete joint solution')
            return None

        self.get_logger().info(f'  IK ({label}): OK')
        return positions

    # ── Plan ──────────────────────────────────────────────────────────────────
    def plan_to_joints(self, target_joints: list, label: str):
        if not self._wait_joint_state():
            return None

        from moveit_msgs.msg import JointConstraint
        joint_constraints = []
        for name, pos in zip(JOINT_NAMES, target_joints):
            jc = JointConstraint()
            jc.joint_name      = name
            jc.position        = pos
            jc.tolerance_above = 0.01
            jc.tolerance_below = 0.01
            jc.weight          = 1.0
            joint_constraints.append(jc)

        constraints = Constraints()
        constraints.joint_constraints = joint_constraints

        request = MotionPlanRequest()
        request.group_name                      = 'panda_arm'
        request.num_planning_attempts           = 10
        request.allowed_planning_time           = 5.0
        request.max_velocity_scaling_factor     = MAX_VELOCITY_SCALE  # ← uses flag
        request.max_acceleration_scaling_factor = MAX_ACCEL_SCALE     # ← uses flag
        request.start_state                     = RobotState()
        request.start_state.joint_state         = self.joint_state
        request.workspace_parameters            = WorkspaceParameters()
        request.workspace_parameters.header.frame_id = 'panda_link0'
        request.workspace_parameters.min_corner.x = -1.5
        request.workspace_parameters.min_corner.y = -1.5
        request.workspace_parameters.min_corner.z = -1.5
        request.workspace_parameters.max_corner.x =  1.5
        request.workspace_parameters.max_corner.y =  1.5
        request.workspace_parameters.max_corner.z =  1.5
        request.goal_constraints                = [constraints]

        goal = MoveGroup.Goal()
        goal.request                          = request
        goal.planning_options.plan_only       = True
        goal.planning_options.replan          = True
        goal.planning_options.replan_attempts = 3

        if not self._action_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error('move_group server not available')
            return None

        future = self._action_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)
        gh = future.result()
        if not gh.accepted:
            self.get_logger().warn(f'Plan ({label}): goal rejected')
            return None

        rf = gh.get_result_async()
        rclpy.spin_until_future_complete(self, rf)
        result = rf.result().result

        if result.error_code.val != 1:
            self.get_logger().warn(f'Plan ({label}): failed (code {result.error_code.val})')
            return None

        return result.planned_trajectory

    # ── Execute ───────────────────────────────────────────────────────────────
    def execute_to_joints(self, target_joints: list):
        if not self._wait_joint_state():
            return False

        from moveit_msgs.msg import JointConstraint
        joint_constraints = []
        for name, pos in zip(JOINT_NAMES, target_joints):
            jc = JointConstraint()
            jc.joint_name      = name
            jc.position        = pos
            jc.tolerance_above = 0.01
            jc.tolerance_below = 0.01
            jc.weight          = 1.0
            joint_constraints.append(jc)

        constraints = Constraints()
        constraints.joint_constraints = joint_constraints

        request = MotionPlanRequest()
        request.group_name                      = 'panda_arm'
        request.num_planning_attempts           = 10
        request.allowed_planning_time           = 5.0
        request.max_velocity_scaling_factor     = MAX_VELOCITY_SCALE  # ← uses flag
        request.max_acceleration_scaling_factor = MAX_ACCEL_SCALE     # ← uses flag
        request.start_state                     = RobotState()
        request.start_state.joint_state         = self.joint_state
        request.workspace_parameters            = WorkspaceParameters()
        request.workspace_parameters.header.frame_id = 'panda_link0'
        request.workspace_parameters.min_corner.x = -1.5
        request.workspace_parameters.min_corner.y = -1.5
        request.workspace_parameters.min_corner.z = -1.5
        request.workspace_parameters.max_corner.x =  1.5
        request.workspace_parameters.max_corner.y =  1.5
        request.workspace_parameters.max_corner.z =  1.5
        request.goal_constraints                = [constraints]

        goal = MoveGroup.Goal()
        goal.request                          = request
        goal.planning_options.plan_only       = False
        goal.planning_options.replan          = True
        goal.planning_options.replan_attempts = 3

        future = self._action_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)
        gh = future.result()
        if not gh.accepted:
            return False

        rf = gh.get_result_async()
        rclpy.spin_until_future_complete(self, rf)
        result_code = rf.result().result.error_code.val
        if result_code != 1:
            self.get_logger().error(f'Execution error code: {result_code}')
        return result_code == 1

    # ── Publish to RViz ───────────────────────────────────────────────────────
    def publish_to_rviz(self, trajectory, model_frame='panda_link0'):
        msg = DisplayTrajectory()
        msg.model_id = model_frame
        msg.trajectory.append(trajectory)
        msg.trajectory_start.joint_state = self.joint_state
        self._display_pub.publish(msg)


# ── Pretty print ──────────────────────────────────────────────────────────────
def print_solution(node, label, joints, trajectory=None):
    deg = [math.degrees(j) for j in joints]
    elbow_angle = math.degrees(joints[3])

    if joints[3] < -1.8:
        elbow_dir = 'UP   ▲'
    elif joints[3] < -1.2:
        elbow_dir = 'MID  ◆'
    else:
        elbow_dir = 'DOWN ▼'

    print(f'\n  {"─"*60}')
    print(f'  {label}  |  Elbow direction: {elbow_dir}  (joint4 = {elbow_angle:+.1f}°)')
    print(f'  {"─"*60}')

    headers = [f'j{i+1}' for i in range(7)]
    print(f'  Joints (°): ' + '  '.join(f'{h}={d:+6.1f}' for h, d in zip(headers, deg)))
    print(f'  Joints (r): ' + '  '.join(f'{h}={j:+.3f}'  for h, j in zip(headers, joints)))

    pose = node.fk(joints)
    if pose:
        p, o = pose.position, pose.orientation
        print(f'  EE Position:    x={p.x:+.4f}  y={p.y:+.4f}  z={p.z:+.4f}')
        print(f'  EE Orientation: qx={o.x:+.4f}  qy={o.y:+.4f}  qz={o.z:+.4f}  qw={o.w:+.4f}')
    else:
        print(f'  EE Position:    (FK unavailable)')

    if trajectory:
        pts = trajectory.joint_trajectory.points
        if pts:
            t_total = pts[-1].time_from_start.sec + pts[-1].time_from_start.nanosec / 1e9
            print(f'  Plan: {len(pts)} waypoints  |  duration: {t_total:.2f}s')


def main():
    rclpy.init()
    node = PandaElbowPicker()

    mode = 'REAL ROBOT' if USE_REAL_ROBOT else 'SIMULATION'
    node.get_logger().info(f'Mode: {mode}  |  EE link: {EE_LINK}  |  Speed: {MAX_VELOCITY_SCALE*100:.0f}%')
    node.get_logger().info('Collecting initial state...')
    for _ in range(40):
        rclpy.spin_once(node, timeout_sec=0.1)

    # ── Parse args ────────────────────────────────────────────────────────────
    args = sys.argv[1:]
    if len(args) >= 3:
        x, y, z = float(args[0]), float(args[1]), float(args[2])
        if len(args) == 7:
            qx, qy, qz, qw = float(args[3]), float(args[4]), float(args[5]), float(args[6])
        else:
            qx, qy, qz, qw = 0.0, 0.924, 0.0, 0.383
    else:
        x, y, z        = 0.4, 0.15, 0.4
        qx, qy, qz, qw = 0.0, 0.924, 0.0, 0.383

    print(f'\n  Target pose: x={x}  y={y}  z={z}')
    print(f'  Orientation: qx={qx}  qy={qy}  qz={qz}  qw={qw}')

    target = PoseStamped()
    target.header.frame_id    = 'panda_link0'
    target.header.stamp       = node.get_clock().now().to_msg()
    target.pose.position.x    = float(x)
    target.pose.position.y    = float(y)
    target.pose.position.z    = float(z)
    target.pose.orientation.x = float(qx)
    target.pose.orientation.y = float(qy)
    target.pose.orientation.z = float(qz)
    target.pose.orientation.w = float(qw)

    # ── Compute IK ────────────────────────────────────────────────────────────
    print('\n' + '═'*64)
    print('  COMPUTING IK SOLUTIONS')
    print('═'*64)

    joints_up   = node.ik_with_seed(target, SEED_ELBOW_UP,   'ELBOW UP')
    joints_down = node.ik_with_seed(target, SEED_ELBOW_DOWN, 'ELBOW DOWN')

    if joints_up is None and joints_down is None:
        print('\n  Both IK solutions failed — target may be out of reach.')
        print('  Try a point closer to: x=0.4  y=0.0  z=0.5')
        rclpy.shutdown()
        return

    # ── Plan ──────────────────────────────────────────────────────────────────
    print('\n' + '═'*64)
    print('  PLANNING TRAJECTORIES')
    print('═'*64)

    traj_up   = node.plan_to_joints(joints_up,   'ELBOW UP')   if joints_up   else None
    traj_down = node.plan_to_joints(joints_down, 'ELBOW DOWN') if joints_down else None

    # ── Display ───────────────────────────────────────────────────────────────
    print('\n' + '═'*64)
    print('  SOLUTIONS')
    print('═'*64)

    if joints_up:
        print_solution(node, '[1] ELBOW UP  ', joints_up,   traj_up)
    else:
        print('\n  [1] ELBOW UP   — No IK solution found for this target')

    if joints_down:
        print_solution(node, '[2] ELBOW DOWN', joints_down, traj_down)
    else:
        print('\n  [2] ELBOW DOWN — No IK solution found for this target')

    print('\n' + '═'*64)
    if traj_up:
        node.publish_to_rviz(traj_up)
        print('  RViz: showing ELBOW UP path (ghost robot)')
    print('  Switch between previews by entering a number below.')
    print('═'*64)

    # ── User selection ────────────────────────────────────────────────────────
    options = {}
    if joints_up   and traj_up:   options['1'] = ('ELBOW UP',   joints_up,   traj_up)
    if joints_down and traj_down: options['2'] = ('ELBOW DOWN', joints_down, traj_down)

    if not options:
        print('\n  No valid plans found. Cannot execute.')
        rclpy.shutdown()
        return

    available = '/'.join(options.keys())
    while True:
        choice = input(
            f'\n  Select solution [{available}] to preview in RViz, '
            f'or [e] to execute selected, [q] to quit: '
        ).strip().lower()

        if choice == 'q':
            print('  Cancelled.')
            break

        elif choice in options:
            label, joints, traj = options[choice]
            node.publish_to_rviz(traj)
            print(f'  RViz now showing: {label}')
            print_solution(node, label, joints, traj)

            confirm = input(f'  Execute {label}? [y/N]: ').strip().lower()
            if confirm == 'y':
                node.get_logger().info(f'Executing {label}...')
                success = node.execute_to_joints(joints)
                if success:
                    node.get_logger().info(f'Done — robot reached {label} configuration.')
                else:
                    node.get_logger().error('Execution failed.')
                break

        elif choice == 'e':
            print('  Select a solution number first.')
        else:
            print(f'  Invalid input. Enter {available}, e, or q')

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
