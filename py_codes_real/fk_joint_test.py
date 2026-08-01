import rclpy
from rclpy.node import Node
from moveit_msgs.srv import GetPositionFK
from moveit_msgs.msg import RobotState
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
import math

class FKTester(Node):
    def __init__(self):
        super().__init__('fk_tester')

        # FK service client
        self.fk_client = self.create_client(GetPositionFK, '/compute_fk')
        while not self.fk_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().info('Waiting for /compute_fk service...')

        self.get_logger().info('FK service ready.')

    def compute_fk(self, joint_angles_deg):
        """
        Compute FK for given joint angles (in degrees).
        Returns end-effector PoseStamped.
        """
        # Convert degrees to radians
        joint_angles_rad = [math.radians(a) for a in joint_angles_deg]

        joint_names = [
            'panda_joint1', 'panda_joint2', 'panda_joint3',
            'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7'
        ]

        # Build the request
        request = GetPositionFK.Request()
        request.header.frame_id = 'panda_link0'
        request.fk_link_names = ['panda_link8']  # end-effector frame

        robot_state = RobotState()
        robot_state.joint_state.name = joint_names
        robot_state.joint_state.position = joint_angles_rad
        request.robot_state = robot_state

        # Call the service
        future = self.fk_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        response = future.result()

        if response.error_code.val == 1:
            pose = response.pose_stamped[0]
            return pose
        else:
            self.get_logger().error(f'FK failed with error code: {response.error_code.val}')
            return None


def main():
    rclpy.init()
    node = FKTester()

    # -----------------------------------------------
    # Define your joint angles here (in degrees)
    # Panda home/ready position
    joint_angles_deg = [0, -45, 0, -135, 0, 90, 45]
    # -----------------------------------------------

    print(f'\nComputing FK for joint angles (deg): {joint_angles_deg}')
    print(f'(rad): {[round(math.radians(a), 4) for a in joint_angles_deg]}')

    pose = node.compute_fk(joint_angles_deg)

    if pose:
        p = pose.pose.position
        o = pose.pose.orientation
        print(f'\n--- End-Effector Pose (frame: panda_link0) ---')
        print(f'Position:    x={p.x:.4f}  y={p.y:.4f}  z={p.z:.4f}  (meters)')
        print(f'Orientation: x={o.x:.4f}  y={o.y:.4f}  z={o.z:.4f}  w={o.w:.4f}  (quaternion)')

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
