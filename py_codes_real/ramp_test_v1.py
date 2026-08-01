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
    def __init__(self, joint_offsets, duration_s, rate_hz):
        """joint_offsets: dict mapping joint_idx (0-based) -> offset_rad"""
        super().__init__('ramp_test')
        self.joint_offsets = joint_offsets
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
        for joint_idx, offset_rad in self.joint_offsets.items():
            target[joint_idx] += offset_rad

        n_steps = max(2, int(self.duration_s * self.rate_hz))
        period = 1.0 / self.rate_hz

        summary = ', '.join(
            f'joint{idx + 1} by {off:.4f} rad'
            for idx, off in self.joint_offsets.items()
        )
        self.get_logger().info(
            f'Ramping {summary} over {self.duration_s}s ({n_steps} steps)...'
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
    parser = argparse.ArgumentParser(
        description='Ramp one or more joints smoothly to an offset from current position.'
    )
    parser.add_argument('--joints', type=int, nargs='+', required=True,
                         help='Joint number(s) to move, 1-7, space separated (e.g. --joints 2 4 6)')
    parser.add_argument('--offsets_deg', type=float, nargs='+', required=True,
                         help='Offset(s) in degrees, one per joint listed in --joints, same order')
    parser.add_argument('--duration', type=float, default=3.0,
                         help='Ramp duration in seconds (default 3.0)')
    parser.add_argument('--rate_hz', type=float, default=100.0,
                         help='Publish rate during ramp (default 100 Hz)')
    args = parser.parse_args()

    if len(args.joints) != len(args.offsets_deg):
        raise ValueError('--joints and --offsets_deg must have the same number of values')

    joint_offsets = {}
    for joint_num, offset_deg in zip(args.joints, args.offsets_deg):
        if not (1 <= joint_num <= 7):
            raise ValueError(f'joint {joint_num} must be between 1 and 7')
        joint_idx = joint_num - 1
        joint_offsets[joint_idx] = offset_deg * 3.14159265358979 / 180.0

    rclpy.init()
    node = RampTest(joint_offsets, args.duration, args.rate_hz)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
