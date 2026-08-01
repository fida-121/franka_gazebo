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
    def __init__(self, joint_targets, duration_s, rate_hz, absolute, joint_state_topic='/franka/joint_states'):
        """
        joint_targets: dict mapping joint_idx (0-based) -> value_rad
            - if absolute=True, value_rad is the EXACT target position to move to
            - if absolute=False, value_rad is an OFFSET added to current position
        """
        super().__init__('ramp_test')
        self.joint_targets = joint_targets
        self.duration_s = duration_s
        self.rate_hz = rate_hz
        self.absolute = absolute

        self.pub = self.create_publisher(
            Float64MultiArray, '/joint_impedance/joints_desired', 10)

        self.current_positions = None
        self.sub = self.create_subscription(
            JointState, joint_state_topic, self._joint_state_cb, 10)

        self.get_logger().info(f'Waiting for {joint_state_topic}...')

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

    # Panda joint limits (rad), from Franka's published spec -- used only for
    # a sanity warning in absolute mode, not enforced/clamped.
    PANDA_JOINT_LIMITS = [
        (-2.8973, 2.8973),
        (-1.7628, 1.7628),
        (-2.8973, 2.8973),
        (-3.0718, -0.0698),
        (-2.8973, 2.8973),
        (-0.0175, 3.7525),
        (-2.8973, 2.8973),
    ]

    def run_ramp(self):
        start = list(self.current_positions)
        target = list(self.current_positions)

        for joint_idx, value_rad in self.joint_targets.items():
            if self.absolute:
                target[joint_idx] = value_rad
                lo, hi = self.PANDA_JOINT_LIMITS[joint_idx]
                if not (lo <= value_rad <= hi):
                    self.get_logger().warn(
                        f'joint{joint_idx + 1} target {value_rad:.4f} rad is '
                        f'OUTSIDE published Panda joint limits [{lo:.4f}, {hi:.4f}]! '
                        f'Double check this value before trusting it on hardware.'
                    )
            else:
                target[joint_idx] += value_rad

        n_steps = max(2, int(self.duration_s * self.rate_hz))
        period = 1.0 / self.rate_hz

        mode_str = 'absolute target' if self.absolute else 'offset'
        summary = ', '.join(
            f'joint{idx + 1} {mode_str} {val:.4f} rad'
            for idx, val in self.joint_targets.items()
        )
        self.get_logger().info(
            f'Ramping {summary} over {self.duration_s}s ({n_steps} steps)...'
        )
        self.get_logger().info(f'Final target pose: {target}')

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
        description=(
            'Ramp one or more joints smoothly to a target position. '
            'Default mode treats values as OFFSETS from current position; '
            'pass --absolute to treat values as EXACT target joint angles instead.'
        )
    )
    parser.add_argument('--joints', type=int, nargs='+', required=True,
                         help='Joint number(s), 1-7, space separated (e.g. --joints 1 2 3 4 5 6 7)')
    parser.add_argument('--offsets_deg', type=float, nargs='+', required=True,
                         help=('Value(s) in degrees, one per joint listed in --joints, same order. '
                               'Interpreted as OFFSETS unless --absolute is set, in which case these '
                               'are the exact target angles to move each listed joint to.'))
    parser.add_argument('--absolute', action='store_true',
                         help='Treat --offsets_deg values as exact target joint angles, not offsets.')
    parser.add_argument('--duration', type=float, default=3.0,
                         help='Ramp duration in seconds (default 3.0)')
    parser.add_argument('--rate_hz', type=float, default=100.0,
                         help='Publish rate during ramp (default 100 Hz)')
    parser.add_argument('--sim', action='store_true',
                         help=('Use MuJoCo simulation joint state topic (/joint_states) '
                               'instead of real hardware (/franka/joint_states). '
                               'Default: real hardware.'))
    args = parser.parse_args()

    if len(args.joints) != len(args.offsets_deg):
        raise ValueError('--joints and --offsets_deg must have the same number of values')

    joint_targets = {}
    for joint_num, value_deg in zip(args.joints, args.offsets_deg):
        if not (1 <= joint_num <= 7):
            raise ValueError(f'joint {joint_num} must be between 1 and 7')
        joint_idx = joint_num - 1
        joint_targets[joint_idx] = value_deg * 3.14159265358979 / 180.0

    joint_state_topic = '/joint_states' if args.sim else '/franka/joint_states'

    rclpy.init()
    node = RampTest(joint_targets, args.duration, args.rate_hz, args.absolute, joint_state_topic)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
