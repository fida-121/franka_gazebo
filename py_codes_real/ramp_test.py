#!/usr/bin/env python3
"""
ramp_test.py

Minimal standalone test node for franka_example_controllers/JointImpedanceController.
Smoothly ramps ONE joint from its current position to a target offset over a
configurable duration, publishing intermediate setpoints at a fixed rate --
avoiding the step-input torque spike that trips libfranka's power_limit_violation
reflex.

This is NOT the full trajectory bridge -- it's a minimal single-move sanity
test to confirm smooth motion works before moving to full trajectory streaming.

Usage:
    python3 ramp_test.py --joint 4 --offset_deg 15 --duration 3.0

Safety notes:
    - Always run the hold-test (publish current position) immediately after
      switching controllers, BEFORE running this script.
    - Start with small offsets (5-10 deg) and long durations (3-5s) on your
      first few tests. Only increase once behavior looks smooth and stable.
    - Keep a hand near the e-stop / Desk emergency stop for every test.
"""

import argparse
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState


JOINT_ORDER = [
    'panda_joint1', 'panda_joint2', 'panda_joint3',
    'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7'
]


class RampTest(Node):
    def __init__(self, joint_idx, offset_rad, duration_s, rate_hz):
        super().__init__('ramp_test')
        self.joint_idx = joint_idx
        self.offset_rad = offset_rad
        self.duration_s = duration_s
        self.rate_hz = rate_hz

        self.pub = self.create_publisher(
            Float64MultiArray, '/joint_impedance/joints_desired', 10)

        self.current_positions = None
        self.sub = self.create_subscription(
            JointState, '/franka/joint_states', self._joint_state_cb, 10)

        self.get_logger().info('Waiting for /franka/joint_states...')

    def _joint_state_cb(self, msg: JointState):
        # Match by NAME, not array position -- /franka/joint_states does not
        # guarantee joint1..joint7 sequential order (confirmed on this repo).
        if self.current_positions is None:
            name_to_pos = dict(zip(msg.name, msg.position))
            try:
                self.current_positions = [name_to_pos[j] for j in JOINT_ORDER]
                self.get_logger().info(
                    f'Got current positions: {self.current_positions}'
                )
                self.run_ramp()
            except KeyError as e:
                self.get_logger().error(f'Missing joint name in joint_states: {e}')

    def run_ramp(self):
        start = list(self.current_positions)
        target = list(self.current_positions)
        target[self.joint_idx] += self.offset_rad

        n_steps = max(2, int(self.duration_s * self.rate_hz))
        period = 1.0 / self.rate_hz

        self.get_logger().info(
            f'Ramping joint{self.joint_idx + 1} by {self.offset_rad:.4f} rad '
            f'over {self.duration_s}s ({n_steps} steps)...'
        )

        for step in range(n_steps + 1):
            alpha = step / n_steps
            positions = [
                s + alpha * (t - s) for s, t in zip(start, target)
            ]
            msg = Float64MultiArray()
            msg.data = positions
            self.pub.publish(msg)
            time.sleep(period)

        self.get_logger().info('Ramp complete. Holding at target.')

        # Keep publishing the final position for a bit so the controller
        # has a steady setpoint to settle into, rather than the topic going
        # silent right after the last step.
        hold_end = time.time() + 2.0
        while time.time() < hold_end:
            msg = Float64MultiArray()
            msg.data = target
            self.pub.publish(msg)
            time.sleep(period)

        self.get_logger().info('Done.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--joint', type=int, required=True,
                         help='Joint number to move, 1-7 (e.g. 4 for panda_joint4)')
    parser.add_argument('--offset_deg', type=float, required=True,
                         help='Offset in degrees to ramp to, relative to current position')
    parser.add_argument('--duration', type=float, default=3.0,
                         help='Ramp duration in seconds (default 3.0)')
    parser.add_argument('--rate_hz', type=float, default=100.0,
                         help='Publish rate during ramp (default 100 Hz)')
    args = parser.parse_args()

    if not (1 <= args.joint <= 7):
        raise ValueError('joint must be between 1 and 7')

    joint_idx = args.joint - 1
    offset_rad = args.offset_deg * 3.14159265358979 / 180.0

    rclpy.init()
    node = RampTest(joint_idx, offset_rad, args.duration, args.rate_hz)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
