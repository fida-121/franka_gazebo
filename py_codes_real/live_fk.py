import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from moveit_msgs.srv import GetPositionFK
from moveit_msgs.msg import RobotState
from sensor_msgs.msg import JointState
import math
import threading


class LiveFKAnalyzer(Node):
    def __init__(self):
        super().__init__('live_fk_analyzer')

        # Reentrant group allows service calls inside timer callbacks
        self.cb_group = ReentrantCallbackGroup()

        # FK service client
        self.fk_client = self.create_client(
            GetPositionFK, '/compute_fk',
            callback_group=self.cb_group)
        while not self.fk_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().info('Waiting for /compute_fk service...')

        self.joint_names = [
            'panda_joint1', 'panda_joint2', 'panda_joint3',
            'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7'
        ]

        self.current_joint_positions = None
        self.lock = threading.Lock()

        # Joint state subscriber
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10,
            callback_group=self.cb_group
        )

        # Timer at 5 Hz (FK service call has latency, don't overload)
        self.timer = self.create_timer(
            0.2, self.fk_timer_callback,
            callback_group=self.cb_group)

        self.get_logger().info('Live FK Analyzer started. Waiting for joint states...')

    def joint_state_callback(self, msg: JointState):
        """Store latest joint positions from the real robot."""
        positions = {}
        for i, name in enumerate(msg.name):
            if name in self.joint_names:
                positions[name] = msg.position[i]

        if all(j in positions for j in self.joint_names):
            with self.lock:
                self.current_joint_positions = [
                    positions[j] for j in self.joint_names
                ]

    def compute_fk(self, joint_positions_rad, ee_link='panda_link8'):
        """Call /compute_fk service synchronously (safe inside ReentrantCallbackGroup)."""
        request = GetPositionFK.Request()
        request.header.frame_id = 'panda_link0'
        request.fk_link_names = [ee_link]

        robot_state = RobotState()
        robot_state.joint_state.name = self.joint_names
        robot_state.joint_state.position = joint_positions_rad
        request.robot_state = robot_state

        future = self.fk_client.call_async(request)
        # Block until done — safe here because of MultiThreadedExecutor
        while not future.done():
            pass

        result = future.result()
        if result and result.error_code.val == 1:
            return result.pose_stamped[0]
        else:
            self.get_logger().warn(
                f'FK error code: {result.error_code.val if result else "None"}')
            return None

    def fk_timer_callback(self):
        """Compute FK from live joint angles and print to terminal."""
        with self.lock:
            if self.current_joint_positions is None:
                return
            positions = list(self.current_joint_positions)

        pose = self.compute_fk(positions)
        if pose is None:
            return

        p = pose.pose.position
        o = pose.pose.orientation
        roll, pitch, yaw = self.quat_to_euler(o.x, o.y, o.z, o.w)
        angles_deg = [round(math.degrees(a), 2) for a in positions]

        print('\033[2J\033[H', end='')
        print('=' * 58)
        print('        LIVE FRANKA FK ANALYZER  (Ctrl+C to stop)')
        print('=' * 58)

        print('\n  Joint Angles:')
        for i, (deg, rad) in enumerate(zip(angles_deg, positions)):
            bar = self.draw_bar(deg, -180, 180, width=20)
            print(f'    J{i+1}: {deg:>8.2f} deg  {rad:>7.4f} rad  |{bar}|')

        print('\n  End-Effector Position (m):')
        print(f'    x = {p.x:>8.4f}')
        print(f'    y = {p.y:>8.4f}')
        print(f'    z = {p.z:>8.4f}')
        print(f'    dist from base = {math.sqrt(p.x**2+p.y**2+p.z**2):.4f} m')

        print('\n  End-Effector Orientation:')
        print(f'    roll  = {math.degrees(roll):>8.2f} deg')
        print(f'    pitch = {math.degrees(pitch):>8.2f} deg')
        print(f'    yaw   = {math.degrees(yaw):>8.2f} deg')
        print(f'    quat  = [{o.x:.4f}, {o.y:.4f}, {o.z:.4f}, {o.w:.4f}]')
        print('=' * 58)

    def quat_to_euler(self, x, y, z, w):
        roll = math.atan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
        sinp = 2*(w*y - z*x)
        pitch = math.copysign(math.pi/2, sinp) if abs(sinp) >= 1 else math.asin(sinp)
        yaw = math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
        return roll, pitch, yaw

    def draw_bar(self, value, min_val, max_val, width=20):
        ratio = max(0.0, min(1.0, (value - min_val) / (max_val - min_val)))
        filled = int(ratio * width)
        return '#' * filled + '-' * (width - filled)


def main():
    rclpy.init()
    node = LiveFKAnalyzer()

    # MultiThreadedExecutor allows subscription + service call to run concurrently
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        print('\nStopping live FK analyzer.')
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
