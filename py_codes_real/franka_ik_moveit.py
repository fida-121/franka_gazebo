#!/usr/bin/env python3
"""
Franka Panda — IK Solver with MoveIt Execution
===============================================
Workflow:
  1. User enters target Cartesian pose (x, y, z, roll, pitch, yaw)
  2. /compute_ik  → solves joint angles (KDL, seeded from current pose)
  3. MoveIt OMPL  → plans collision-free trajectory
  4. TOPP         → applies time-optimal velocity/acceleration profile
  5. panda_arm_controller → executes smoothly on real robot

Prerequisites:
  Terminal 1 → ros2 launch franka_bringup franka.launch.py robot_ip:=<IP>
  Terminal 2 → spawner panda_arm_controller  (position interface)
  Terminal 3 → ros2 launch franka_moveit_config moveit_on_hw.launch.py robot_ip:=<IP>

Usage:
  python3 franka_ik_moveit.py
"""

import math
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.action import ActionClient

from moveit_msgs.srv import GetPositionIK
from moveit_msgs.msg import (
    PositionIKRequest, RobotState,
    MotionPlanRequest, Constraints, JointConstraint,
    WorkspaceParameters, PlanningOptions,
)
from moveit_msgs.action import MoveGroup

from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped, Point, Quaternion, Vector3


# ═══════════════════════════════════════════════════════════════
#  ROBOT CONSTANTS
# ═══════════════════════════════════════════════════════════════

JOINT_NAMES = [f'panda_joint{i}' for i in range(1, 8)]

JOINT_LIMITS = {
    'panda_joint1': (-2.8973,  2.8973),   # ±166°
    'panda_joint2': (-1.7628,  1.7628),   # ±101°
    'panda_joint3': (-2.8973,  2.8973),   # ±166°
    'panda_joint4': (-3.0718, -0.0698),   # -176° to -4°
    'panda_joint5': (-2.8973,  2.8973),   # ±166°
    'panda_joint6': (-0.0175,  3.7525),   # -1° to 215°
    'panda_joint7': (-2.8973,  2.8973),   # ±166°
}

WORKSPACE = {
    'x':          (-0.855,  0.855),
    'y':          (-0.855,  0.855),
    'z':          (-0.360,  1.190),
    'min_radius':  0.10,
    'max_radius':  0.855,
}

MOVEIT_ERROR_CODES = {
     1: 'SUCCESS',
    -1: 'FAILURE',
    -2: 'INVALID_MOTION_PLAN',
    -4: 'CONTROL_FAILED',
    -6: 'TIMED_OUT',
    -7: 'PREEMPTED',
   -10: 'START_STATE_IN_COLLISION',
   -12: 'GOAL_IN_COLLISION',
   -13: 'GOAL_VIOLATES_PATH_CONSTRAINTS',
   -14: 'GOAL_CONSTRAINTS_VIOLATED',
   -15: 'INVALID_GROUP_NAME',
   -16: 'INVALID_GOAL_CONSTRAINTS',
   -17: 'INVALID_ROBOT_STATE',
   -18: 'INVALID_LINK_MODEL',
   -31: 'NO_IK_SOLUTION',
}


# ═══════════════════════════════════════════════════════════════
#  MATH HELPERS
# ═══════════════════════════════════════════════════════════════

def euler_to_quat(roll, pitch, yaw):
    """Roll/pitch/yaw (radians) → quaternion (x, y, z, w)."""
    cr, sr = math.cos(roll  / 2), math.sin(roll  / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw   / 2), math.sin(yaw   / 2)
    return (
        sr*cp*cy - cr*sp*sy,
        cr*sp*cy + sr*cp*sy,
        cr*cp*sy - sr*sp*cy,
        cr*cp*cy + sr*sp*sy,
    )


def quat_to_euler(x, y, z, w):
    """Quaternion → roll/pitch/yaw (radians)."""
    roll  = math.atan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    sinp  = 2*(w*y - z*x)
    pitch = (math.copysign(math.pi / 2, sinp)
             if abs(sinp) >= 1 else math.asin(sinp))
    yaw   = math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    return roll, pitch, yaw


# ═══════════════════════════════════════════════════════════════
#  DISPLAY HELPERS
# ═══════════════════════════════════════════════════════════════

def divider(char='─', width=64):
    print(char * width)


def header(title):
    divider('═')
    print(f'  {title}')
    divider('═')


def draw_bar(value, lo, hi, width=20):
    ratio  = max(0.0, min(1.0, (value - lo) / (hi - lo)))
    filled = int(ratio * width)
    return '#' * filled + '-' * (width - filled)


def get_float(prompt, default=None):
    while True:
        try:
            raw = input(prompt).strip()
            if raw == '' and default is not None:
                return default
            return float(raw)
        except ValueError:
            print('  ✗ Enter a valid number.')


def get_speed():
    print('\n  Velocity scaling (MoveIt applies TOPP at this fraction):')
    print('    [1]  10%  ← safest, recommended for first runs')
    print('    [2]  20%  ← normal everyday use')
    print('    [3]  40%  ← faster, only for known-safe targets')
    opts = {'1': 0.10, '2': 0.20, '3': 0.40}
    while True:
        c = input('  Choose [1/2/3, default=1]: ').strip() or '1'
        if c in opts:
            return opts[c]
        print('  Enter 1, 2, or 3.')


def print_joint_table(joints):
    """Print joint angle table with limit bars. Returns True if all within limits."""
    print(f'\n  {"Joint":<14} {"Degrees":>9}  {"Radians":>9}  '
          f'{"Min°":>7}  {"Max°":>7}  {"Range":^22}  Status')
    divider()
    all_ok = True
    for name, val in zip(JOINT_NAMES, joints):
        lo, hi = JOINT_LIMITS[name]
        deg    = math.degrees(val)
        bar    = draw_bar(val, lo, hi)
        ok     = lo <= val <= hi
        margin = min(abs(val - lo), abs(val - hi))
        status = f'✓ {math.degrees(margin):.1f}° left' if ok else '✗ LIMIT!'
        if not ok:
            all_ok = False
        print(f'  {name:<14} {deg:>9.2f}° {val:>9.4f}r '
              f'{math.degrees(lo):>7.1f}° {math.degrees(hi):>7.1f}°  '
              f'|{bar}|  {status}')
    return all_ok


# ═══════════════════════════════════════════════════════════════
#  FRANKA MOTION NODE
# ═══════════════════════════════════════════════════════════════

class FrankaMotionNode(Node):

    def __init__(self):
        super().__init__('franka_ik_moveit')
        self.cb_group = ReentrantCallbackGroup()

        # ── /compute_ik service ───────────────────────────────
        self._ik = self.create_client(
            GetPositionIK, '/compute_ik',
            callback_group=self.cb_group)
        print('  Connecting to /compute_ik ...')
        if not self._ik.wait_for_service(timeout_sec=8.0):
            raise RuntimeError(
                '/compute_ik not available.\n'
                '  Start: ros2 launch franka_moveit_config '
                'moveit_on_hw.launch.py robot_ip:=<IP>')
        print('  ✓ /compute_ik connected.')

        # ── MoveGroup action client ───────────────────────────
        self._mg = ActionClient(
            self, MoveGroup, '/move_action',
            callback_group=self.cb_group)
        print('  Connecting to /move_group ...')
        if not self._mg.wait_for_server(timeout_sec=10.0):
            raise RuntimeError(
                '/move_group not available.\n'
                '  Start: ros2 launch franka_moveit_config '
                'moveit_on_hw.launch.py robot_ip:=<IP>')
        print('  ✓ /move_group connected.')

        # ── Joint state subscriber ────────────────────────────
        self._lock           = threading.Lock()
        self._current_joints = None
        self.create_subscription(
            JointState, '/joint_states',
            self._joint_cb, 10,
            callback_group=self.cb_group)

        self.get_logger().info('FrankaMotionNode ready.')

    # ── Callbacks ──────────────────────────────────────────────

    def _joint_cb(self, msg: JointState):
        pos = {n: p for n, p in zip(msg.name, msg.position)}
        if all(j in pos for j in JOINT_NAMES):
            with self._lock:
                self._current_joints = [pos[j] for j in JOINT_NAMES]

    # ── Public helpers ─────────────────────────────────────────

    def wait_for_robot(self, timeout=8.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            with self._lock:
                if self._current_joints is not None:
                    return True
            time.sleep(0.05)
        return False

    def get_current_joints(self):
        with self._lock:
            return list(self._current_joints) if self._current_joints else None

    def validate_workspace(self, x, y, z):
        errors = []
        r = math.sqrt(x**2 + y**2 + z**2)
        if r < WORKSPACE['min_radius']:
            errors.append(f'Too close to base: r={r:.3f} m '
                          f'(min {WORKSPACE["min_radius"]} m)')
        if r > WORKSPACE['max_radius']:
            errors.append(f'Out of reach: r={r:.3f} m '
                          f'(max {WORKSPACE["max_radius"]} m)')
        for axis, (lo, hi), v in [('x', WORKSPACE['x'], x),
                                   ('y', WORKSPACE['y'], y),
                                   ('z', WORKSPACE['z'], z)]:
            if not (lo <= v <= hi):
                errors.append(
                    f'{axis}={v:.4f} m  out of range [{lo}, {hi}] m')
        return errors

    # ── Step 1: Solve IK ───────────────────────────────────────

    def solve_ik(self, x, y, z, roll_deg, pitch_deg, yaw_deg,
                 timeout=1.5):
        """
        Call /compute_ik (KDL solver via MoveIt).
        Seeded with current robot state → finds nearest configuration.
        Returns (joint_positions_list, error_string).
        """
        qx, qy, qz, qw = euler_to_quat(
            math.radians(roll_deg),
            math.radians(pitch_deg),
            math.radians(yaw_deg))

        ik_req                  = PositionIKRequest()
        ik_req.group_name       = 'panda_arm'
        ik_req.avoid_collisions = True
        ik_req.timeout.sec      = int(timeout)
        ik_req.timeout.nanosec  = int((timeout % 1) * 1e9)

        # Seed IK with current robot state
        ik_req.robot_state = RobotState()
        current = self.get_current_joints()
        if current:
            ik_req.robot_state.joint_state.name     = JOINT_NAMES
            ik_req.robot_state.joint_state.position = current

        target                  = PoseStamped()
        target.header.frame_id  = 'panda_link0'
        target.pose.position    = Point(x=x, y=y, z=z)
        target.pose.orientation = Quaternion(x=qx, y=qy, z=qz, w=qw)
        ik_req.pose_stamped     = target

        req            = GetPositionIK.Request()
        req.ik_request = ik_req

        future = self._ik.call_async(req)
        while not future.done():
            time.sleep(0.01)

        result = future.result()
        code   = result.error_code.val
        if code != 1:
            return None, MOVEIT_ERROR_CODES.get(code, f'code {code}')

        js      = result.solution.joint_state
        pos_map = {n: p for n, p in zip(js.name, js.position)}
        joints  = [pos_map.get(j, 0.0) for j in JOINT_NAMES]

        if len([v for v in joints if v != 0.0]) < 5:
            return None, 'Incomplete IK solution returned'

        return joints, None

    # ── Step 2: MoveIt plan + execute ──────────────────────────

    def moveit_execute(self, target_joints,
                       velocity_scale=0.1,
                       accel_scale=0.05,
                       planning_time=5.0,
                       replan_attempts=3):
        """
        Pass IK joint angles to MoveIt move_group for planning + execution.

        MoveIt internally:
          - Runs OMPL (RRTConnect) to find a collision-free joint path
          - Applies Time Optimal Path Parameterisation (TOPP)
            to generate smooth velocity/acceleration profiles
          - Sends the resulting trajectory to panda_arm_controller
            via the FollowJointTrajectory action

        Args:
            target_joints   : 7 joint positions from IK (rad)
            velocity_scale  : fraction of Franka max velocity [0.05–1.0]
            accel_scale     : fraction of Franka max accel    [0.05–1.0]
            planning_time   : OMPL time budget (seconds)
            replan_attempts : number of replan retries on failure
        """

        # ── MotionPlanRequest ─────────────────────────────────
        plan_req            = MotionPlanRequest()
        plan_req.group_name = 'panda_arm'

        # Workspace bounds (generous — used by OMPL for sampling)
        ws                 = WorkspaceParameters()
        ws.header.frame_id = 'panda_link0'
        ws.min_corner      = Vector3(x=-1.0, y=-1.0, z=-0.5)
        ws.max_corner      = Vector3(x= 1.0, y= 1.0, z= 1.5)
        plan_req.workspace_parameters = ws

        # Start state from live joint states
        plan_req.start_state = RobotState()
        current = self.get_current_joints()
        if current:
            plan_req.start_state.joint_state.name     = JOINT_NAMES
            plan_req.start_state.joint_state.position = current

        # Goal: joint constraints for all 7 joints
        goal_constraints      = Constraints()
        goal_constraints.name = 'ik_goal'
        for name, pos in zip(JOINT_NAMES, target_joints):
            jc                 = JointConstraint()
            jc.joint_name      = name
            jc.position        = pos
            jc.tolerance_above = 0.005   # ≈ 0.3°
            jc.tolerance_below = 0.005
            jc.weight          = 1.0
            goal_constraints.joint_constraints.append(jc)
        plan_req.goal_constraints.append(goal_constraints)

        # Planner and timing
        plan_req.planner_id                        = ''  # RRTConnect default
        plan_req.num_planning_attempts             = 5
        plan_req.allowed_planning_time             = planning_time
        plan_req.max_velocity_scaling_factor       = velocity_scale
        plan_req.max_acceleration_scaling_factor   = accel_scale

        # ── MoveGroup goal ────────────────────────────────────
        mg_goal                              = MoveGroup.Goal()
        mg_goal.request                      = plan_req
        opts                                 = PlanningOptions()
        opts.plan_only                       = False  # execute on robot
        opts.replan                          = True
        opts.replan_attempts                 = replan_attempts
        opts.replan_delay                    = 2.0
        mg_goal.planning_options             = opts

        # ── Send and wait ─────────────────────────────────────
        print('\n  Sending to MoveIt move_group ...')
        send_future = self._mg.send_goal_async(mg_goal)
        while not send_future.done():
            time.sleep(0.05)

        goal_handle = send_future.result()
        if not goal_handle.accepted:
            return False, 'Goal rejected by move_group'

        print('  ✓ Goal accepted.')
        print('  Planning ', end='', flush=True)

        result_future = goal_handle.get_result_async()
        phase         = 'Planning'
        dots          = 0
        t_start       = time.time()

        while not result_future.done():
            elapsed = time.time() - t_start
            # After planning_time seconds, switch label to Executing
            if elapsed > planning_time * 0.5 and phase == 'Planning':
                phase = 'Executing'
                print(f'\n  {phase} ', end='', flush=True)
            print('.', end='', flush=True)
            dots += 1
            time.sleep(0.3)
        print()

        result = result_future.result().result
        code   = result.error_code.val
        label  = MOVEIT_ERROR_CODES.get(code, f'code {code}')

        return (code == 1), label


# ═══════════════════════════════════════════════════════════════
#  MAIN INTERACTIVE LOOP
# ═══════════════════════════════════════════════════════════════

def main():
    rclpy.init()

    print('\n')
    header('FRANKA PANDA  —  IK + MOVEIT MOTION PLANNER')
    print("""
  Pipeline:
    /compute_ik  →  MoveIt OMPL  →  TOPP  →  panda_arm_controller

  Step 1  /compute_ik   : KDL solver finds joint angles for your target
  Step 2  MoveIt OMPL   : RRTConnect plans collision-free joint path
  Step 3  TOPP          : Time-optimal smooth velocity/accel profiling
  Step 4  Controller    : panda_arm_controller executes on real robot

  Frame   : panda_link0  (robot base centre)
  Input   : x, y, z (metres)  +  roll, pitch, yaw (degrees)
  EE link : panda_link8  (flange, no gripper)
""")

    # ── Initialise ───────────────────────────────────────────
    try:
        node = FrankaMotionNode()
    except RuntimeError as e:
        print(f'\n  ✗ {e}')
        rclpy.try_shutdown()
        return

    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    print('\n  Waiting for robot joint states ...')
    if not node.wait_for_robot(timeout=10.0):
        print('  ✗ No joint states — is franka_bringup running?')
        node.destroy_node()
        rclpy.try_shutdown()
        return

    current = node.get_current_joints()
    print('  ✓ Robot connected.\n')
    print('  Current joint angles:')
    for name, val in zip(JOINT_NAMES, current):
        print(f'    {name}: {math.degrees(val):>8.2f}°  ({val:.4f} rad)')

    # ════════════════════════════════════════════════════════════
    while True:
        try:
            print()
            divider()
            print('  Enter target Cartesian pose:')
            print('  (Press Enter for orientation defaults: '
                  'roll=180°, pitch=0°, yaw=0°)')
            divider()

            x     = get_float('  x     (m)         : ')
            y     = get_float('  y     (m)         : ')
            z     = get_float('  z     (m)         : ')
            roll  = get_float('  roll  (deg) [180] : ', default=180.0)
            pitch = get_float('  pitch (deg) [0]   : ', default=0.0)
            yaw   = get_float('  yaw   (deg) [0]   : ', default=0.0)

            r = math.sqrt(x**2 + y**2 + z**2)

            # ── 1. Workspace ──────────────────────────────────
            print()
            divider()
            print('  [1/4] Workspace Validation')
            divider()
            print(f'  Target : x={x:.4f}  y={y:.4f}  z={z:.4f}  '
                  f'(r={r:.4f} m)')
            print(f'  Orient : roll={roll:.1f}°  pitch={pitch:.1f}°  '
                  f'yaw={yaw:.1f}°')

            ws_errors = node.validate_workspace(x, y, z)
            if ws_errors:
                print('\n  ✗ Outside reachable workspace:')
                for e in ws_errors:
                    print(f'    · {e}')
                print()
                continue
            print('  ✓ Within workspace.')

            # ── 2. IK ─────────────────────────────────────────
            print()
            divider()
            print('  [2/4] Inverse Kinematics  (/compute_ik  KDL)')
            divider()
            print('  Solving IK ...')

            joints, err = node.solve_ik(x, y, z, roll, pitch, yaw)
            if err:
                print(f'\n  ✗ IK failed: {err}')
                print('  Tips:')
                print('    · Adjust roll / pitch / yaw orientation')
                print('    · Target may be near a singularity')
                print('    · Try a slightly different position')
                print()
                continue
            print('  ✓ IK solution found.')

            # ── 3. Joint limits ───────────────────────────────
            print()
            divider()
            print('  [3/4] Joint Limit Verification')
            divider()

            all_ok = print_joint_table(joints)
            if not all_ok:
                print('\n  ✗ Joint limit violation — cannot execute safely.')
                print('    Try a different orientation or target.')
                print()
                continue
            print('\n  ✓ All joint limits satisfied.')

            current = node.get_current_joints()
            max_move = max(abs(t - c)
                          for c, t in zip(current, joints))
            print(f'  Largest joint movement : {math.degrees(max_move):.1f}°')

            # ── 4. MoveIt plan + execute ──────────────────────
            print()
            divider()
            print('  [4/4] MoveIt Planning + Execution')
            divider()

            speed = get_speed()
            accel = round(speed * 0.5, 3)

            print(f'\n  MoveIt plan settings:')
            print(f'    Planner          : RRTConnect (OMPL)')
            print(f'    Velocity scale   : {int(speed * 100)}%  '
                  f'of Franka max speed')
            print(f'    Accel scale      : {int(accel * 100)}%  '
                  f'of Franka max accel')
            print(f'    Planning budget  : 5.0 s')
            print(f'    Goal tolerance   : 0.005 rad (~0.3°) per joint')
            print(f'    Replan attempts  : 3')
            print(f'\n  ⚠  Ensure the robot workspace is completely clear!')

            confirm = input(
                '\n  Execute on real robot? [y/N] : ').strip().lower()

            if confirm != 'y':
                print('  Execution skipped.')
            else:
                ok, msg = node.moveit_execute(
                    joints,
                    velocity_scale=speed,
                    accel_scale=accel,
                    planning_time=5.0,
                    replan_attempts=3)

                if ok:
                    print(f'\n  ✓ Motion complete!')
                    time.sleep(0.5)
                    current = node.get_current_joints() or joints
                    print('\n  Final joint angles:')
                    for name, val in zip(JOINT_NAMES, current):
                        lo, hi = JOINT_LIMITS[name]
                        margin = min(abs(val-lo), abs(val-hi))
                        print(f'    {name}: {math.degrees(val):>8.2f}°  '
                              f'({val:.4f} rad)  '
                              f'[margin: {math.degrees(margin):.1f}°]')
                else:
                    print(f'\n  ✗ Motion failed: {msg}')
                    if 'COLLISION' in msg:
                        print('    Path passes through a collision object.')
                        print('    Try a different target or orientation.')
                    elif 'TIMED_OUT' in msg:
                        print('    OMPL could not find a plan in 5 s.')
                        print('    Try a simpler/closer target.')
                    elif 'CONTROL_FAILED' in msg:
                        print('    Controller failed during execution.')
                        print('    Check robot status and error lights.')
                    else:
                        print(f'    Raw error: {msg}')

        except KeyboardInterrupt:
            print('\n\n  Interrupted. Goodbye.')
            break

        print()
        if input('  Solve another target? [Y/n] : ').strip().lower() == 'n':
            print('  Goodbye.')
            break

    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
