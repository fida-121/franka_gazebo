#!/usr/bin/env python3
"""
Franka Panda — IK Solver via panda_arm_controller FollowJointTrajectory
========================================================================
Workflow:
  1. User enters target Cartesian pose (x, y, z, roll, pitch, yaw)
  2. /compute_ik solves joint angles (KDL, seeded from current pose)
  3. Goal sent to /panda_arm_controller/follow_joint_trajectory action
  4. JTC handles all interpolation and execution internally

The JointTrajectoryController receives start + end positions with zero
velocities and uses cubic spline interpolation — smooth, no custom math.

Prerequisites:
  Terminal 1 → ros2 launch franka_bringup franka.launch.py robot_ip:=<IP>
  Terminal 2 → spawner panda_arm_controller  (position interface)
  Terminal 3 → ros2 launch franka_moveit_config moveit_on_hw.launch.py robot_ip:=<IP>

Usage:
  python3 franka_ik_jtc.py
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
from moveit_msgs.msg import PositionIKRequest, RobotState

from control_msgs.action import FollowJointTrajectory
from control_msgs.msg import JointTolerance

from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration


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

# Franka max joint speeds (rad/s) — official spec
JOINT_MAX_VEL = [2.175, 2.175, 2.175, 2.175, 2.610, 2.610, 2.610]

WORKSPACE = {
    'x':          (-0.855,  0.855),
    'y':          (-0.855,  0.855),
    'z':          (-0.360,  1.190),
    'min_radius':  0.10,
    'max_radius':  0.855,
}

MOVEIT_ERRORS = {
     1: 'SUCCESS',        -1: 'FAILURE',
    -6: 'TIMED_OUT',     -10: 'START_STATE_IN_COLLISION',
   -12: 'GOAL_IN_COLLISION',
   -15: 'INVALID_GROUP_NAME',
   -17: 'INVALID_ROBOT_STATE',
   -18: 'INVALID_LINK_MODEL',
   -31: 'NO_IK_SOLUTION',
}

FJT_ERRORS = {
    FollowJointTrajectory.Result.SUCCESSFUL:             'SUCCESSFUL',
    FollowJointTrajectory.Result.INVALID_GOAL:           'INVALID_GOAL',
    FollowJointTrajectory.Result.INVALID_JOINTS:         'INVALID_JOINTS',
    FollowJointTrajectory.Result.OLD_HEADER_TIMESTAMP:   'OLD_HEADER_TIMESTAMP',
    FollowJointTrajectory.Result.PATH_TOLERANCE_VIOLATED:'PATH_TOLERANCE_VIOLATED',
    FollowJointTrajectory.Result.GOAL_TOLERANCE_VIOLATED:'GOAL_TOLERANCE_VIOLATED',
}


# ═══════════════════════════════════════════════════════════════
#  MATH HELPERS
# ═══════════════════════════════════════════════════════════════

def euler_to_quat(roll, pitch, yaw):
    """Roll/pitch/yaw (rad) → quaternion (x, y, z, w)."""
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
    """Quaternion → roll/pitch/yaw (rad)."""
    roll  = math.atan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    sinp  = 2*(w*y - z*x)
    pitch = (math.copysign(math.pi / 2, sinp)
             if abs(sinp) >= 1 else math.asin(sinp))
    yaw   = math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    return roll, pitch, yaw


def estimate_duration(current, target, speed_factor):
    """
    Estimate motion time so no joint exceeds its speed limit.
    Each joint's required time = displacement / (max_vel * scale).
    Take the maximum across all joints — the slowest joint drives timing.
    Minimum 3 seconds for safety.
    """
    times = [
        abs(t - c) / (vmax * speed_factor)
        for c, t, vmax in zip(current, target, JOINT_MAX_VEL)
        if abs(t - c) > 1e-4
    ]
    return max(3.0, max(times) if times else 3.0)


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
    print('\n  Speed (fraction of Franka max joint velocity):')
    print('    [1]  10%  ← safest, recommended for first runs')
    print('    [2]  20%  ← normal use')
    print('    [3]  30%  ← faster, for small nearby movements')
    opts = {'1': 0.10, '2': 0.20, '3': 0.30}
    while True:
        c = input('  Choose [1/2/3, default=1]: ').strip() or '1'
        if c in opts:
            return opts[c]
        print('  Enter 1, 2, or 3.')


def print_joint_table(joints):
    """Print joint angle table with bars. Returns True if all within limits."""
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
        super().__init__('franka_ik_jtc')
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

        # ── FollowJointTrajectory action client ───────────────
        self._fjt = ActionClient(
            self,
            FollowJointTrajectory,
            '/panda_arm_controller/follow_joint_trajectory',
            callback_group=self.cb_group)
        print('  Connecting to /panda_arm_controller/follow_joint_trajectory ...')
        if not self._fjt.wait_for_server(timeout_sec=8.0):
            raise RuntimeError(
                'follow_joint_trajectory action not available.\n'
                '  Run: ros2 run controller_manager spawner panda_arm_controller')
        print('  ✓ follow_joint_trajectory connected.')

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

    # ── Helpers ────────────────────────────────────────────────

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
        Call /compute_ik seeded with current robot state.
        Returns (joint_positions, error_string).
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
            return None, MOVEIT_ERRORS.get(code, f'code {code}')

        js      = result.solution.joint_state
        pos_map = {n: p for n, p in zip(js.name, js.position)}
        joints  = [pos_map.get(j, 0.0) for j in JOINT_NAMES]

        if len([v for v in joints if v != 0.0]) < 5:
            return None, 'Incomplete IK solution'

        return joints, None

    # ── Step 2: Execute via JTC action ─────────────────────────

    def execute(self, target_joints, speed_factor=0.1):
        """
        Send IK joint angles to panda_arm_controller via
        FollowJointTrajectory action.

        The JTC receives:
          - Point 0: current joint positions at t=0  (zero vel)
          - Point 1: target joint positions at t=T   (zero vel)

        The JTC's built-in cubic spline interpolator generates a
        smooth trajectory between these two points automatically.
        No manual trajectory math needed.

        Args:
            target_joints : list of 7 joint positions (rad) from IK
            speed_factor  : fraction of per-joint max velocity
        """
        current = self.get_current_joints()
        if current is None:
            return False, 'No joint state available'

        duration = estimate_duration(current, target_joints, speed_factor)
        secs     = int(duration)
        nsec     = int(round((duration - secs) * 1e9))

        # ── Build 2-point trajectory ──────────────────────────
        traj             = JointTrajectory()
        traj.joint_names = JOINT_NAMES
        # header.stamp = 0 → controller starts immediately

        # Start point — current position, zero velocity
        p0               = JointTrajectoryPoint()
        p0.positions     = current
        p0.velocities    = [0.0] * 7
        p0.accelerations = [0.0] * 7
        p0.time_from_start = Duration(sec=0, nanosec=0)

        # End point — IK target, zero velocity
        p1               = JointTrajectoryPoint()
        p1.positions     = target_joints
        p1.velocities    = [0.0] * 7
        p1.accelerations = [0.0] * 7
        p1.time_from_start = Duration(sec=secs, nanosec=nsec)

        traj.points = [p0, p1]

        # ── Build FollowJointTrajectory goal ──────────────────
        goal          = FollowJointTrajectory.Goal()
        goal.trajectory = traj

        # Path tolerance — how far each joint can deviate mid-motion
        for name in JOINT_NAMES:
            pt          = JointTolerance()
            pt.name     = name
            pt.position = 0.2     # rad — generous, JTC handles it
            pt.velocity = 0.5
            goal.path_tolerance.append(pt)

        # Goal tolerance — accuracy required at final position
        for name in JOINT_NAMES:
            gt          = JointTolerance()
            gt.name     = name
            gt.position = 0.01    # rad ≈ 0.6°
            gt.velocity = 0.05
            goal.goal_tolerance.append(gt)

        # Allow extra time beyond trajectory duration for settling
        goal.goal_time_tolerance = Duration(sec=3, nanosec=0)

        # ── Send goal and wait for result ─────────────────────
        print(f'\n  Sending goal to panda_arm_controller ...')
        print(f'    Duration  : {duration:.1f} s  at {int(speed_factor*100)}% speed')
        print(f'    Interpolation : JTC cubic spline (internal)')

        send_future = self._fjt.send_goal_async(goal)
        while not send_future.done():
            time.sleep(0.05)

        goal_handle = send_future.result()
        if not goal_handle.accepted:
            return False, 'Goal rejected by controller'

        print('  ✓ Goal accepted — executing ', end='', flush=True)

        result_future = goal_handle.get_result_async()
        while not result_future.done():
            print('.', end='', flush=True)
            time.sleep(0.3)
        print()

        result = result_future.result().result
        code   = result.error_code
        label  = FJT_ERRORS.get(code,
                 f'code {code}')

        return (code == FollowJointTrajectory.Result.SUCCESSFUL), label


# ═══════════════════════════════════════════════════════════════
#  MAIN INTERACTIVE LOOP
# ═══════════════════════════════════════════════════════════════

def main():
    rclpy.init()

    print('\n')
    header('FRANKA PANDA  —  IK + JTC DIRECT EXECUTION')
    print("""
  Pipeline:
    /compute_ik  →  /panda_arm_controller/follow_joint_trajectory

  Step 1  /compute_ik              : KDL finds joint angles for target
  Step 2  FollowJointTrajectory    : JTC receives start + end positions
  Step 3  JTC cubic spline         : controller interpolates smoothly
  Step 4  Real robot executes      : position commands at 1 kHz

  Frame   : panda_link0  (robot base)
  Input   : x, y, z (metres)  +  roll, pitch, yaw (degrees)
  EE link : panda_link8  (flange)
""")

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
            print('  (Press Enter for defaults: roll=180°, pitch=0°, yaw=0°)')
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
                print('\n  ✗ Outside workspace:')
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
            print('  Solving ...')

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

            # ── 4. Execute ────────────────────────────────────
            print()
            divider()
            print('  [4/4] Execution  (panda_arm_controller JTC)')
            divider()

            speed = get_speed()
            dur   = estimate_duration(current, joints, speed)

            print(f'\n  Plan:')
            print(f'    Target     : x={x:.4f}  y={y:.4f}  z={z:.4f}')
            print(f'    Orient     : roll={roll:.1f}°  '
                  f'pitch={pitch:.1f}°  yaw={yaw:.1f}°')
            print(f'    Duration   : ~{dur:.1f} s  at {int(speed*100)}% speed')
            print(f'    Method     : JTC FollowJointTrajectory action')
            print(f'    Smoothing  : cubic spline (built into JTC)')
            print(f'\n  ⚠  Clear the robot workspace completely!')

            confirm = input(
                '\n  Execute on real robot? [y/N] : ').strip().lower()

            if confirm != 'y':
                print('  Execution skipped.')
            else:
                ok, msg = node.execute(joints, speed_factor=speed)

                if ok:
                    print(f'\n  ✓ Motion complete!')
                    time.sleep(0.3)
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
                    if 'PATH_TOLERANCE' in msg:
                        print('    Robot deviated too far from trajectory.')
                        print('    Try a lower speed or shorter movement.')
                    elif 'GOAL_TOLERANCE' in msg:
                        print('    Robot did not reach target accurately.')
                        print('    Check for mechanical resistance or errors.')
                    elif 'rejected' in msg.lower():
                        print('    Controller rejected the goal.')
                        print('    Check: ros2 control list_controllers')
                    else:
                        print(f'    Error: {msg}')

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
