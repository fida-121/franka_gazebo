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

import math
import threading
import time


# ─────────────────────────────────────────────
#  Franka Panda joint limits (radians)
# ─────────────────────────────────────────────
JOINT_LIMITS = {
    'panda_joint1': (-2.8973,  2.8973),
    'panda_joint2': (-1.7628,  1.7628),
    'panda_joint3': (-2.8973,  2.8973),
    'panda_joint4': (-3.0718, -0.0698),
    'panda_joint5': (-2.8973,  2.8973),
    'panda_joint6': (-0.0175,  3.7525),
    'panda_joint7': (-2.8973,  2.8973),
}

WORKSPACE = {
    'x':          (-0.855,  0.855),
    'y':          (-0.855,  0.855),
    'z':          (-0.360,  1.190),
    'min_radius':  0.10,
    'max_radius':  0.855,
}

JOINT_NAMES = [f'panda_joint{i}' for i in range(1, 8)]

MOVEIT_ERROR_CODES = {
     1: 'SUCCESS',        -1: 'FAILURE',
    -6: 'TIMED_OUT',     -10: 'START_STATE_IN_COLLISION',
   -12: 'GOAL_IN_COLLISION',
   -15: 'INVALID_GROUP_NAME',
   -17: 'INVALID_ROBOT_STATE',
   -18: 'INVALID_LINK_MODEL',
   -31: 'NO_IK_SOLUTION',
}


def euler_to_quat(roll, pitch, yaw):
    cr, sr = math.cos(roll/2),  math.sin(roll/2)
    cp, sp = math.cos(pitch/2), math.sin(pitch/2)
    cy, sy = math.cos(yaw/2),   math.sin(yaw/2)
    return (
        sr*cp*cy - cr*sp*sy,
        cr*sp*cy + sr*cp*sy,
        cr*cp*sy - sr*sp*cy,
        cr*cp*cy + sr*sp*sy,
    )


def quat_to_euler(x, y, z, w):
    roll  = math.atan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    sinp  = 2*(w*y - z*x)
    pitch = math.copysign(math.pi/2, sinp) if abs(sinp) >= 1 else math.asin(sinp)
    yaw   = math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    return roll, pitch, yaw


def estimate_duration(current, target, speed_factor=0.2):
    """Estimate travel time based on largest joint movement."""
    max_move = max(abs(t - c) for c, t in zip(current, target))
    # Franka max joint speed ~2.1750 rad/s, use 20% of that
    speed = 2.175 * speed_factor
    return max(3.0, max_move / speed)   # minimum 3 seconds


class IKSolver(Node):
    def __init__(self):
        super().__init__('ik_solver')
        self.cb_group = ReentrantCallbackGroup()

        # IK service
        self.ik_client = self.create_client(
            GetPositionIK, '/compute_ik',
            callback_group=self.cb_group)
        while not self.ik_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().info('Waiting for /compute_ik ...')
        self.get_logger().info('✓ /compute_ik service connected.')

        # Direct trajectory publisher — no move_group action needed
        self.traj_pub = self.create_publisher(
            JointTrajectory,
            '/panda_arm_controller/joint_trajectory',
            10)

        # Joint state subscriber
        self.current_joints = None
        self.lock = threading.Lock()
        self.create_subscription(
            JointState, '/joint_states',
            self._joint_cb, 10,
            callback_group=self.cb_group)

        self.get_logger().info('✓ IK Solver ready.')

    def _joint_cb(self, msg: JointState):
        pos = {n: p for n, p in zip(msg.name, msg.position)}
        if all(j in pos for j in JOINT_NAMES):
            with self.lock:
                self.current_joints = [pos[j] for j in JOINT_NAMES]

    def wait_for_joints(self, timeout=5.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            with self.lock:
                if self.current_joints is not None:
                    return True
            time.sleep(0.05)
        return False

    def validate_target(self, x, y, z):
        errors = []
        r = math.sqrt(x**2 + y**2 + z**2)
        if r < WORKSPACE['min_radius']:
            errors.append(f'Too close to base  r={r:.3f} m  (min={WORKSPACE["min_radius"]} m)')
        if r > WORKSPACE['max_radius']:
            errors.append(f'Out of reach       r={r:.3f} m  (max={WORKSPACE["max_radius"]} m)')
        for ax, (lo, hi) in [('x', WORKSPACE['x']),
                              ('y', WORKSPACE['y']),
                              ('z', WORKSPACE['z'])]:
            v = {'x': x, 'y': y, 'z': z}[ax]
            if not (lo <= v <= hi):
                errors.append(f'{ax}={v:.3f} out of range [{lo}, {hi}] m')
        return errors

    def solve_ik(self, x, y, z, roll_deg, pitch_deg, yaw_deg,
                 attempts=10, timeout=1.0):
        qx, qy, qz, qw = euler_to_quat(
            math.radians(roll_deg),
            math.radians(pitch_deg),
            math.radians(yaw_deg))

        req     = GetPositionIK.Request()
        ik_req  = PositionIKRequest()
        ik_req.group_name       = 'panda_arm'
        ik_req.avoid_collisions = True
        ik_req.timeout.sec      = int(timeout)
        ik_req.timeout.nanosec  = int((timeout % 1) * 1e9)

        # Seed with current robot state
        ik_req.robot_state = RobotState()
        with self.lock:
            if self.current_joints:
                ik_req.robot_state.joint_state.name     = JOINT_NAMES
                ik_req.robot_state.joint_state.position = list(self.current_joints)

        target                  = PoseStamped()
        target.header.frame_id  = 'panda_link0'
        target.pose.position    = Point(x=x, y=y, z=z)
        target.pose.orientation = Quaternion(x=qx, y=qy, z=qz, w=qw)
        ik_req.pose_stamped     = target
        req.ik_request          = ik_req

        future = self.ik_client.call_async(req)
        while not future.done():
            time.sleep(0.01)

        result = future.result()
        code   = result.error_code.val
        if code != 1:
            return None, MOVEIT_ERROR_CODES.get(code, f'code {code}')

        js      = result.solution.joint_state
        pos_map = {n: p for n, p in zip(js.name, js.position)}
        joints  = [pos_map[j] for j in JOINT_NAMES if j in pos_map]
        if len(joints) != 7:
            return None, 'Incomplete IK solution'
        return joints, None

    def check_joint_limits(self, joints):
        violations = []
        for name, val in zip(JOINT_NAMES, joints):
            lo, hi = JOINT_LIMITS[name]
            if not (lo <= val <= hi):
                violations.append((name, val, lo, hi))
        return violations

    def execute_trajectory(self, target_joints, speed_factor=0.2):
        """
        Publish JointTrajectory directly to panda_arm_controller.
        No move_group action server required.
        """
        with self.lock:
            current = list(self.current_joints) if self.current_joints else target_joints

        duration_sec = estimate_duration(current, target_joints, speed_factor)

        traj              = JointTrajectory()
        traj.joint_names  = JOINT_NAMES

        # Waypoint 1: current position at t=0 (smooth start)
        p0               = JointTrajectoryPoint()
        p0.positions     = current
        p0.velocities    = [0.0] * 7
        p0.accelerations = [0.0] * 7
        p0.time_from_start = Duration(sec=0, nanosec=0)
        traj.points.append(p0)

        # Waypoint 2: target position
        p1               = JointTrajectoryPoint()
        p1.positions     = target_joints
        p1.velocities    = [0.0] * 7
        p1.accelerations = [0.0] * 7
        secs             = int(duration_sec)
        nsecs            = int((duration_sec - secs) * 1e9)
        p1.time_from_start = Duration(sec=secs, nanosec=nsecs)
        traj.points.append(p1)

        print(f'\n  Publishing trajectory  (duration: {duration_sec:.1f} s  '
              f'speed: {int(speed_factor*100)}%)')
        self.traj_pub.publish(traj)

        # Wait for motion to complete
        print('  Executing ', end='', flush=True)
        for _ in range(int(duration_sec * 5)):
            print('.', end='', flush=True)
            time.sleep(0.2)
        print()
        return True


# ─────────────────────────────────────────────
#  Input helpers
# ─────────────────────────────────────────────
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
    options = {'1': 0.10, '2': 0.20, '3': 0.30}
    print('\n  Speed setting:')
    print('    [1] Slow   (10% max velocity)  ← recommended for first runs')
    print('    [2] Normal (20% max velocity)')
    print('    [3] Fast   (30% max velocity)')
    while True:
        c = input('  Choose [1/2/3, default=1]: ').strip() or '1'
        if c in options:
            return options[c]
        print('  Enter 1, 2, or 3.')


def draw_bar(value, lo, hi, width=18):
    ratio  = max(0.0, min(1.0, (value - lo) / (hi - lo)))
    filled = int(ratio * width)
    return '#' * filled + '-' * (width - filled)


# ─────────────────────────────────────────────
#  Main interactive loop
# ─────────────────────────────────────────────
def main():
    rclpy.init()
    node = IKSolver()

    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    print('\n' + '═' * 60)
    print('     FRANKA PANDA — INTERACTIVE IK SOLVER')
    print('═' * 60)
    print("""
  Frame  : panda_link0 (robot base)
  Input  : x, y, z (meters) + roll, pitch, yaw (degrees)
  Output : joint angles + optional execution on real robot
""")

    if not node.wait_for_joints():
        print('✗ No joint states received. Is the robot running?')
        rclpy.try_shutdown()
        return
    print('  ✓ Robot connected and joint states received.\n')

    while True:
        try:
            print('─' * 60)
            print('  Enter target Cartesian pose:')
            print('─' * 60)
            x     = get_float('  x     (m)   : ')
            y     = get_float('  y     (m)   : ')
            z     = get_float('  z     (m)   : ')
            roll  = get_float('  roll  (deg) [Enter=180]: ', default=180.0)
            pitch = get_float('  pitch (deg) [Enter=0]  : ', default=0.0)
            yaw   = get_float('  yaw   (deg) [Enter=0]  : ', default=0.0)

            # ── 1. Workspace check ────────────
            print('\n  [1/3] Workspace Validation')
            print('─' * 60)
            errors = node.validate_target(x, y, z)
            r = math.sqrt(x**2 + y**2 + z**2)
            print(f'  Position   : x={x:.4f}  y={y:.4f}  z={z:.4f}  '
                  f'|r={r:.4f} m|')
            print(f'  Orientation: roll={roll:.1f}°  '
                  f'pitch={pitch:.1f}°  yaw={yaw:.1f}°')

            if errors:
                print('\n  ✗ Workspace violations:')
                for e in errors:
                    print(f'    · {e}')
                print()
                continue
            print('  ✓ Within workspace limits.')

            # ── 2. Solve IK ───────────────────
            print('\n  [2/3] Solving IK ...')
            print('─' * 60)
            joints, err = node.solve_ik(x, y, z, roll, pitch, yaw,
                                        attempts=10, timeout=1.0)
            if err:
                print(f'\n  ✗ IK failed: {err}')
                print('  Tips:')
                print('    · Try adjusting roll/pitch/yaw orientation')
                print('    · Target may be near a singularity')
                print('    · Try a slightly different position\n')
                continue

            # Joint limit check
            violations = node.check_joint_limits(joints)

            print(f'\n  {"Joint":<14} {"Value °":>9}  {"Value r":>9}  '
                  f'{"Min °":>8}  {"Max °":>8}  {"Bar":^20}  {"":>6}')
            print('  ' + '─' * 80)

            for name, val in zip(JOINT_NAMES, joints):
                lo, hi  = JOINT_LIMITS[name]
                deg     = math.degrees(val)
                lo_d    = math.degrees(lo)
                hi_d    = math.degrees(hi)
                ok      = lo <= val <= hi
                bar     = draw_bar(val, lo, hi)
                margin  = min(abs(val - lo), abs(val - hi))
                status  = f'✓ ({math.degrees(margin):.1f}° margin)' if ok else '✗ LIMIT!'
                print(f'  {name:<14} {deg:>9.2f}° {val:>9.4f}r '
                      f'{lo_d:>8.1f}° {hi_d:>8.1f}°  |{bar}|  {status}')

            if violations:
                print('\n  ✗ Joint limit violations. Cannot execute safely.')
                continue

            print('\n  ✓ IK valid. All joint limits satisfied.')

            # ── 3. Execute ────────────────────
            print('\n  [3/3] Execution')
            print('─' * 60)
            speed = get_speed()

            with node.lock:
                curr = list(node.current_joints)
            dur = estimate_duration(curr, joints, speed)
            print(f'\n  Estimated motion time : {dur:.1f} s')
            print(f'  Speed factor          : {int(speed*100)}%')
            print(f'  Target position       : x={x:.4f}  y={y:.4f}  z={z:.4f}')
            print(f'\n  ⚠  Make sure the robot workspace is clear!')
            confirm = input('\n  Execute on real robot? [y/N] : ').strip().lower()

            if confirm == 'y':
                node.execute_trajectory(joints, speed_factor=speed)
                print('  ✓ Trajectory sent.')
            else:
                print('  Execution skipped.')

        except KeyboardInterrupt:
            print('\n\nExiting.')
            break

        print()
        if input('  Solve another target? [Y/n] : ').strip().lower() == 'n':
            break

    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
