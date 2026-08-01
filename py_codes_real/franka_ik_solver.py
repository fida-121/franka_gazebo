#!/usr/bin/env python3
"""
Franka Panda — Interactive IK Solver with Minimum-Jerk Trajectory
==================================================================
Requirements:
  - franka_bringup franka.launch.py running (hardware)
  - panda_arm_controller spawned with position command interface
  - MoveIt move_group running (for /compute_ik service)

Usage:
  python3 franka_ik_solver.py
"""

import math
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from moveit_msgs.srv import GetPositionIK
from moveit_msgs.msg import PositionIKRequest, RobotState

from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration


# ═══════════════════════════════════════════════════════════════
#  ROBOT CONSTANTS
# ═══════════════════════════════════════════════════════════════

JOINT_NAMES = [f'panda_joint{i}' for i in range(1, 8)]

# Official Franka Panda joint position limits (radians)
JOINT_LIMITS = {
    'panda_joint1': (-2.8973,  2.8973),   # ±166°
    'panda_joint2': (-1.7628,  1.7628),   # ±101°
    'panda_joint3': (-2.8973,  2.8973),   # ±166°
    'panda_joint4': (-3.0718, -0.0698),   # -176° to -4°
    'panda_joint5': (-2.8973,  2.8973),   # ±166°
    'panda_joint6': (-0.0175,  3.7525),   # -1° to 215°
    'panda_joint7': (-2.8973,  2.8973),   # ±166°
}

# Approximate reachable workspace from panda_link0 (meters)
WORKSPACE = {
    'x':          (-0.855,  0.855),
    'y':          (-0.855,  0.855),
    'z':          (-0.360,  1.190),
    'min_radius':  0.10,    # singularity near base
    'max_radius':  0.855,   # full arm reach
}

# MoveIt error code lookup
MOVEIT_ERRORS = {
     1: 'SUCCESS',
    -1: 'FAILURE',
    -6: 'TIMED_OUT',
   -10: 'START_STATE_IN_COLLISION',
   -12: 'GOAL_IN_COLLISION',
   -15: 'INVALID_GROUP_NAME',
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
        sr*cp*cy - cr*sp*sy,   # x
        cr*sp*cy + sr*cp*sy,   # y
        cr*cp*sy - sr*sp*cy,   # z
        cr*cp*cy + sr*sp*sy,   # w
    )


def quat_to_euler(x, y, z, w):
    """Quaternion → roll/pitch/yaw (radians)."""
    roll  = math.atan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    sinp  = 2*(w*y - z*x)
    pitch = math.copysign(math.pi / 2, sinp) if abs(sinp) >= 1 else math.asin(sinp)
    yaw   = math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    return roll, pitch, yaw


def minimum_jerk_pos(t, T):
    """
    Minimum-jerk position blend scalar [0 → 1] at time t over duration T.
    Zero velocity and acceleration at both endpoints.
    Profile: s = 10τ³ - 15τ⁴ + 6τ⁵
    """
    tau = max(0.0, min(1.0, t / T))
    return 10*tau**3 - 15*tau**4 + 6*tau**5


def minimum_jerk_vel(t, T):
    """
    Minimum-jerk velocity blend scalar at time t over duration T.
    Derivative of the position profile divided by T.
    """
    tau = max(0.0, min(1.0, t / T))
    return (30*tau**2 - 60*tau**3 + 30*tau**4) / T


def minimum_jerk_acc(t, T):
    """
    Minimum-jerk acceleration blend scalar at time t over duration T.
    Second derivative of the position profile divided by T².
    """
    tau = max(0.0, min(1.0, t / T))
    return (60*tau - 180*tau**2 + 120*tau**3) / (T * T)


def estimate_duration(current, target, speed_factor):
    """
    Estimate motion duration based on largest single-joint movement.
    Franka max joint speed: ~2.175 rad/s.
    """
    max_move = max(abs(t - c) for c, t in zip(current, target))
    speed    = 2.175 * speed_factor
    return max(5.0, max_move / speed)   # minimum 5 s for safety


# ═══════════════════════════════════════════════════════════════
#  DISPLAY HELPERS
# ═══════════════════════════════════════════════════════════════

def draw_bar(value, lo, hi, width=20):
    ratio  = max(0.0, min(1.0, (value - lo) / (hi - lo)))
    filled = int(ratio * width)
    return '#' * filled + '-' * (width - filled)


def divider(char='─', width=62):
    print(char * width)


def header(title):
    divider('═')
    print(f'  {title}')
    divider('═')


def get_float(prompt, default=None):
    while True:
        try:
            raw = input(prompt).strip()
            if raw == '' and default is not None:
                return default
            return float(raw)
        except ValueError:
            print('  ✗ Please enter a valid number.')


def get_speed():
    print('\n  Speed setting:')
    print('    [1]  Very slow  (5%)   ← safest, use for large movements')
    print('    [2]  Slow       (10%)  ← recommended for normal use')
    print('    [3]  Normal     (15%)  ← only for small nearby movements')
    opts = {'1': 0.05, '2': 0.10, '3': 0.15}
    while True:
        c = input('  Choose [1/2/3, default=1]: ').strip() or '1'
        if c in opts:
            return opts[c]
        print('  Enter 1, 2, or 3.')


# ═══════════════════════════════════════════════════════════════
#  IK SOLVER NODE
# ═══════════════════════════════════════════════════════════════

class IKSolverNode(Node):

    def __init__(self):
        super().__init__('franka_ik_solver')
        self.cb_group = ReentrantCallbackGroup()

        # ── /compute_ik service ───────────────────────────────
        self.ik_client = self.create_client(
            GetPositionIK, '/compute_ik',
            callback_group=self.cb_group)

        print('  Connecting to /compute_ik service ...')
        if not self.ik_client.wait_for_service(timeout_sec=8.0):
            raise RuntimeError(
                '✗ /compute_ik not found. Is MoveIt (move_group) running?\n'
                '  Run: ros2 launch franka_moveit_config moveit_on_hw.launch.py '
                'robot_ip:=<IP>')
        print('  ✓ /compute_ik connected.')

        # ── Joint state subscriber ────────────────────────────
        self._lock           = threading.Lock()
        self._current_joints = None

        self.create_subscription(
            JointState, '/joint_states',
            self._joint_state_cb, 10,
            callback_group=self.cb_group)

        # ── Trajectory publisher ──────────────────────────────
        self._traj_pub = self.create_publisher(
            JointTrajectory,
            '/panda_arm_controller/joint_trajectory',
            10)

        self.get_logger().info('IK Solver node ready.')

    # ── Callbacks ─────────────────────────────────────────────

    def _joint_state_cb(self, msg: JointState):
        """Cache latest joint positions from the real robot."""
        pos = {n: p for n, p in zip(msg.name, msg.position)}
        if all(j in pos for j in JOINT_NAMES):
            with self._lock:
                self._current_joints = [pos[j] for j in JOINT_NAMES]

    # ── Public API ────────────────────────────────────────────

    def wait_for_robot(self, timeout=8.0):
        """Block until joint states are received from the robot."""
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
        """
        Check that (x, y, z) is inside the Franka reachable workspace.
        Returns list of error strings (empty = OK).
        """
        errors = []
        r = math.sqrt(x**2 + y**2 + z**2)

        if r < WORKSPACE['min_radius']:
            errors.append(
                f'Too close to base: r={r:.3f} m '
                f'(minimum {WORKSPACE["min_radius"]} m)')
        if r > WORKSPACE['max_radius']:
            errors.append(
                f'Out of reach: r={r:.3f} m '
                f'(maximum {WORKSPACE["max_radius"]} m)')

        for axis, (lo, hi), v in [
                ('x', WORKSPACE['x'], x),
                ('y', WORKSPACE['y'], y),
                ('z', WORKSPACE['z'], z)]:
            if not (lo <= v <= hi):
                errors.append(
                    f'{axis}={v:.4f} m  out of range [{lo}, {hi}] m')
        return errors

    def solve_ik(self, x, y, z, roll_deg, pitch_deg, yaw_deg, timeout=1.5):
        """
        Call MoveIt /compute_ik to get joint angles for a Cartesian target.
        Seed state = current robot pose (finds closest solution).
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

        # Seed with current robot state so IK finds the nearest solution
        ik_req.robot_state = RobotState()
        current = self.get_current_joints()
        if current:
            ik_req.robot_state.joint_state.name     = JOINT_NAMES
            ik_req.robot_state.joint_state.position = current

        # Target end-effector pose
        target                  = PoseStamped()
        target.header.frame_id  = 'panda_link0'
        target.pose.position    = Point(x=x, y=y, z=z)
        target.pose.orientation = Quaternion(x=qx, y=qy, z=qz, w=qw)
        ik_req.pose_stamped     = target

        req            = GetPositionIK.Request()
        req.ik_request = ik_req

        future = self.ik_client.call_async(req)
        while not future.done():
            time.sleep(0.01)

        result = future.result()
        code   = result.error_code.val

        if code != 1:
            label = MOVEIT_ERRORS.get(code, f'code {code}')
            return None, label

        js      = result.solution.joint_state
        pos_map = {n: p for n, p in zip(js.name, js.position)}
        joints  = [pos_map.get(j, 0.0) for j in JOINT_NAMES]

        if len(joints) != 7:
            return None, 'Incomplete joint solution returned'

        return joints, None

    def check_joint_limits(self, joints):
        """
        Returns list of (name, value, lo, hi) for any violations.
        Empty list = all OK.
        """
        violations = []
        for name, val in zip(JOINT_NAMES, joints):
            lo, hi = JOINT_LIMITS[name]
            if not (lo <= val <= hi):
                violations.append((name, val, lo, hi))
        return violations

    def build_minimum_jerk_trajectory(self, current, target,
                                       duration, n_waypoints=60):
        """
        Build a JointTrajectory message using minimum-jerk interpolation.
        Each waypoint has position, velocity, and acceleration set for
        a perfectly smooth, zero-jerk-at-endpoints profile.

        Args:
            current:     list of 7 starting joint positions (rad)
            target:      list of 7 target joint positions (rad)
            duration:    total motion time (seconds)
            n_waypoints: number of intermediate points (more = smoother)

        Returns:
            JointTrajectory message ready to publish
        """
        now  = self.get_clock().now().to_msg()
        traj = JointTrajectory()
        traj.joint_names  = JOINT_NAMES

        for i in range(n_waypoints + 1):
            t   = duration * i / n_waypoints
            s   = minimum_jerk_pos(t, duration)   # position blend  [0,1]
            sd  = minimum_jerk_vel(t, duration)   # velocity blend  [0,...]
            sdd = minimum_jerk_acc(t, duration)   # accel blend     [0,...]

            pt = JointTrajectoryPoint()

            pt.positions = [
                c + (tgt - c) * s
                for c, tgt in zip(current, target)
            ]
            pt.velocities = [
                (tgt - c) * sd
                for c, tgt in zip(current, target)
            ]
            pt.accelerations = [
                (tgt - c) * sdd
                for c, tgt in zip(current, target)
            ]

            secs = int(t)
            nsec = int(round((t - secs) * 1e9))
            pt.time_from_start = Duration(sec=secs, nanosec=nsec)
            traj.points.append(pt)

        return traj

    def execute(self, target_joints, speed_factor=0.05, n_waypoints=60):
        """
        Execute a smooth minimum-jerk trajectory to target_joints.

        Args:
            target_joints: list of 7 target joint positions (rad)
            speed_factor:  fraction of max joint speed [0.05 … 0.20]
            n_waypoints:   smoothness (50-100 recommended)
        """
        current = self.get_current_joints()
        if current is None:
            print('  ✗ No joint state available.')
            return False

        duration = estimate_duration(current, target_joints, speed_factor)

        print(f'\n  Building trajectory:')
        print(f'    Waypoints : {n_waypoints + 1}')
        print(f'    Duration  : {duration:.1f} s')
        print(f'    Speed     : {int(speed_factor * 100)}% of max')

        traj = self.build_minimum_jerk_trajectory(
            current, target_joints, duration, n_waypoints)

        self._traj_pub.publish(traj)

        # Wait for motion to complete
        print('\n  Executing  ', end='', flush=True)
        steps = int(duration * 5)
        for i in range(steps):
            print('█' if i % 5 == 0 else '▒', end='', flush=True)
            time.sleep(0.2)
        print('  ✓')
        return True


# ═══════════════════════════════════════════════════════════════
#  INTERACTIVE MAIN LOOP
# ═══════════════════════════════════════════════════════════════

def print_joint_table(joints):
    """Print a formatted table of joint angles with limit bars."""
    print(f'\n  {"Joint":<14} {"Degrees":>9}  {"Radians":>9}  '
          f'{"Min°":>7}  {"Max°":>7}  {"Range":^22}  {"Margin":>10}')
    divider()

    all_ok = True
    for name, val in zip(JOINT_NAMES, joints):
        lo, hi  = JOINT_LIMITS[name]
        deg     = math.degrees(val)
        lo_d    = math.degrees(lo)
        hi_d    = math.degrees(hi)
        ok      = lo <= val <= hi
        bar     = draw_bar(val, lo, hi)
        margin  = min(abs(val - lo), abs(val - hi))
        status  = f'✓ {math.degrees(margin):.1f}° left' if ok else '✗ VIOLATION'
        if not ok:
            all_ok = False
        print(f'  {name:<14} {deg:>9.2f}° {val:>9.4f}r '
              f'{lo_d:>7.1f}° {hi_d:>7.1f}°  |{bar}|  {status}')
    return all_ok


def main():
    rclpy.init()

    print('\n')
    header('FRANKA PANDA  —  INTERACTIVE IK SOLVER')
    print("""
  Solves inverse kinematics for a target Cartesian pose and
  executes a smooth minimum-jerk trajectory on the real robot.

  Coordinate frame : panda_link0  (robot base centre)
  Position input   : x, y, z in metres
  Orientation input: roll, pitch, yaw in degrees
  End-effector link: panda_link8  (flange, no gripper)

  Prerequisites:
    Terminal 1 → ros2 launch franka_bringup franka.launch.py robot_ip:=<IP>
    Terminal 2 → spawner panda_arm_controller  (position interface)
    Terminal 3 → ros2 launch franka_moveit_config moveit_on_hw.launch.py robot_ip:=<IP>
""")

    # ── Initialise node ──────────────────────────────────────
    try:
        node = IKSolverNode()
    except RuntimeError as e:
        print(f'\n  {e}')
        rclpy.try_shutdown()
        return

    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    # ── Wait for robot ───────────────────────────────────────
    print('\n  Waiting for robot joint states ...')
    if not node.wait_for_robot(timeout=10.0):
        print('  ✗ No joint states received.')
        print('    Is franka_bringup/franka.launch.py running?')
        node.destroy_node()
        rclpy.try_shutdown()
        return

    # Show current robot state
    current = node.get_current_joints()
    print('  ✓ Robot connected.\n')
    print('  Current joint angles:')
    for name, val in zip(JOINT_NAMES, current):
        print(f'    {name}: {math.degrees(val):>8.2f}°  ({val:.4f} rad)')

    # ════════════════════════════════════════════════════════
    #  Main interactive loop
    # ════════════════════════════════════════════════════════
    while True:
        try:
            print()
            divider()
            print('  Enter target Cartesian pose:')
            print('  (Press Enter on orientation to use defaults: '
                  'roll=180°, pitch=0°, yaw=0°)')
            divider()

            x     = get_float('  x     (m)         : ')
            y     = get_float('  y     (m)         : ')
            z     = get_float('  z     (m)         : ')
            roll  = get_float('  roll  (deg) [180] : ', default=180.0)
            pitch = get_float('  pitch (deg) [0]   : ', default=0.0)
            yaw   = get_float('  yaw   (deg) [0]   : ', default=0.0)

            r = math.sqrt(x**2 + y**2 + z**2)

            # ── Step 1: Workspace validation ─────────────────
            print()
            divider()
            print('  [1/4] Workspace Validation')
            divider()
            print(f'  Target position    : '
                  f'x={x:.4f}  y={y:.4f}  z={z:.4f}  (r={r:.4f} m)')
            print(f'  Target orientation : '
                  f'roll={roll:.1f}°  pitch={pitch:.1f}°  yaw={yaw:.1f}°')

            ws_errors = node.validate_workspace(x, y, z)
            if ws_errors:
                print('\n  ✗ Target is outside the reachable workspace:')
                for e in ws_errors:
                    print(f'    · {e}')
                print('\n  Please enter a position within these bounds:')
                print(f'    x : {WORKSPACE["x"]}  m')
                print(f'    y : {WORKSPACE["y"]}  m')
                print(f'    z : {WORKSPACE["z"]}  m')
                print(f'    r : {WORKSPACE["min_radius"]} … '
                      f'{WORKSPACE["max_radius"]} m from base')
                continue

            print('  ✓ Target is within workspace bounds.')

            # ── Step 2: Solve IK ──────────────────────────────
            print()
            divider()
            print('  [2/4] Solving Inverse Kinematics  (MoveIt KDL solver)')
            divider()
            print('  Calling /compute_ik ...')

            joints, err = node.solve_ik(x, y, z, roll, pitch, yaw,
                                        timeout=1.5)
            if err:
                print(f'\n  ✗ IK failed: {err}')
                print('\n  Suggestions:')
                print('    · Try a different orientation (adjust roll/pitch/yaw)')
                print('    · The target may be near a kinematic singularity')
                print('    · Try slightly different x/y/z values')
                print('    · Make sure MoveIt move_group is running')
                continue

            print('  ✓ IK solution found.')

            # ── Step 3: Joint limit check ─────────────────────
            print()
            divider()
            print('  [3/4] Joint Limit Verification')
            divider()

            all_ok = print_joint_table(joints)

            if not all_ok:
                print('\n  ✗ Joint limit violation — cannot execute safely.')
                print('    Try a different target pose or orientation.')
                continue

            print('\n  ✓ All joint limits satisfied.')

            # FK verification (using current→target displacement as sanity check)
            print(f'\n  Largest joint movement : '
                  f'{math.degrees(max(abs(t - c) for c, t in zip(current, joints))):.1f}°')

            # ── Step 4: Execute ───────────────────────────────
            print()
            divider()
            print('  [4/4] Execution')
            divider()

            speed = get_speed()
            dur   = estimate_duration(
                node.get_current_joints(), joints, speed)

            print(f'\n  Motion plan:')
            print(f'    From    : current robot pose')
            print(f'    To      : x={x:.4f}  y={y:.4f}  z={z:.4f}')
            print(f'    Orient  : roll={roll:.1f}°  pitch={pitch:.1f}°  yaw={yaw:.1f}°')
            print(f'    Duration: ~{dur:.1f} s  at {int(speed*100)}% speed')
            print(f'    Profile : minimum-jerk (smooth accel/decel)')
            print(f'\n  ⚠  Ensure the robot workspace is completely clear!')

            confirm = input('\n  Execute on real robot? [y/N] : ').strip().lower()
            if confirm != 'y':
                print('  Execution skipped.')
            else:
                ok = node.execute(joints, speed_factor=speed, n_waypoints=60)
                if ok:
                    # Update current joints display
                    time.sleep(0.5)
                    current = node.get_current_joints()
                    print('\n  Robot reached target. Current joint angles:')
                    for name, val in zip(JOINT_NAMES, current or joints):
                        print(f'    {name}: {math.degrees(val):>8.2f}°')

        except KeyboardInterrupt:
            print('\n\n  Interrupted — exiting.')
            break

        # ── Continue prompt ───────────────────────────────────
        print()
        again = input('  Solve another target? [Y/n] : ').strip().lower()
        if again == 'n':
            print('\n  Goodbye.')
            break

    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
