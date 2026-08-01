# franka_gazebo

This repository is a fork of **[multipanda_ros2](https://github.com/tenfoldpaper/multipanda_ros2)**, a
`ros2_control`-based framework for the Franka Emika Panda robot on ROS 2 Humble, developed by
Jon Škerlj, Seongjin Bien, Abdeldjallil Naceri, and Sami Haddadin. All of the underlying
real-hardware and MuJoCo simulation architecture — `franka_hardware`, the MoveIt configuration,
the controller framework, and the MuJoCo integration via `mujoco_ros_pkgs` — comes directly from
that project. See their paper for the full framework:

**[Bridging the Sim-to-Real Gap with multipanda_ros2: A Real-Time ROS2 Framework for Multimanual Systems](https://arxiv.org/abs/2602.02269)**

```bibtex
@misc{škerlj2026multipanda_ros2,
      title={Bridging the Sim-to-Real Gap with multipanda_ros2: A Real-Time ROS2 Framework for Multimanual Systems},
      author={Jon Škerlj and Seongjin Bien and Abdeldjallil Naceri and Sami Haddadin},
      year={2026},
      eprint={2602.02269},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2602.02269},
}
```

The original project itself is built on [mcbed's ROS 2 Humble port of franka_ros2](https://github.com/mcbed/franka_ros2/tree/humble).

---

## What this fork adds: Gazebo Sim support

The upstream project's simulation path is built around **MuJoCo**, via a
[forked `mujoco_ros_pkgs`](https://github.com/tenfoldpaper/mujoco_ros_pkgs) plugin. This fork adds
a **second, parallel simulation path using Gazebo Sim (Ignition Fortress)**, built on
`gz_ros2_control` instead of the MuJoCo plugin. The original MuJoCo path
(`franka_bringup franka_sim.launch.py`) is untouched — everything below is additive.

### Requirements (in addition to the base install)

```bash
sudo apt update
sudo apt install -y \
  ros-humble-ros-gz \
  ros-humble-gz-ros2-control \
  ros-humble-ros-gz-sim \
  ros-humble-ros-gz-bridge \
  ros-humble-moveit-planners-chomp
```

### Running it

```bash
source ~/multipanda_ws/install/setup.bash
ros2 launch franka_moveit_config gazebo.launch.py
```

This launches Gazebo Sim with the Panda arm and gripper spawned in, `move_group` with **both
OMPL and CHOMP** available as selectable planning pipelines (switchable live from RViz's
MotionPlanning panel), and RViz configured to plan and execute directly against the Gazebo
simulation.

### Joint-space motion (direct, outside MoveIt)

To command a joint-space trajectory directly to the arm controller, bypassing MoveIt planning:

```bash
ros2 topic pub --once \
  /panda_arm_controller/joint_trajectory \
  trajectory_msgs/msg/JointTrajectory \
  '{
    joint_names: [panda_joint1, panda_joint2, panda_joint3, panda_joint4, panda_joint5, panda_joint6, panda_joint7],
    points: [{
      positions: [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785],
      time_from_start: {sec: 3, nanosec: 0}
    }]
  }'
```

### Gripper control

The gripper is exposed as a standard `control_msgs/action/GripperCommand` action:

```bash
# Open
ros2 action send_goal /panda_hand_controller/gripper_cmd control_msgs/action/GripperCommand \
"{command: {position: 0.04, max_effort: 20.0}}"

# Close / grip
ros2 action send_goal /panda_hand_controller/gripper_cmd control_msgs/action/GripperCommand \
"{command: {position: 0.0, max_effort: 20.0}}"
```

No homing step is required in simulation.

### Implementation notes

- **Inertials**: the base `panda_arm.xacro`/`hand.xacro` macros ship with no `<inertial>` blocks
  (mass/inertia lives in the MuJoCo MJCF model instead). Gazebo's `urdf2sdf` conversion silently
  drops the entire model if any link is missing one. Both macros now accept an
  `add_inertials:=false` (default) parameter — only the Gazebo xacro path sets it `true`, so the
  MuJoCo/real-hardware output is unchanged.
- **Mimic joint**: `panda_finger_joint2` is a mimic joint. Only the leader
  (`panda_finger_joint1`) has a command interface in `ros2_control`; the follower has state
  interfaces only.
- **`position_proportional_gain`** is a global `gz_ros2_control` plugin parameter (a direct child
  of `<gazebo><plugin>`), not a per-joint setting.
- **Controller naming**: `sim_gazebo_panda_controllers.yaml` and
  `sim_gazebo_panda_ros_controllers.yaml` are Gazebo-specific copies of the shared MoveIt/
  ros2_control config files, so real-hardware and MuJoCo controller naming stay untouched.

---

## Original documentation (multipanda_ros2)

The MuJoCo simulation and real-hardware setup below is entirely from the upstream project.
See [their documentation](https://github.com/tenfoldpaper/multipanda_ros2) for the full,
up-to-date version.

### Working features

* Real robot:
    * FrankaState broadcaster
    * All control interfaces (torque, position, velocity, Cartesian)
    * Example controllers for all interfaces
    * Controllers are swappable using rqt_controller_manager
    * Runtime `franka::ControlException` error recovery via `~/service_server/error_recovery`
    * Runtime internal parameter setter services
* Sim robot (MuJoCo):
    * Same as the real robot, except no Cartesian command interface
    * Gripper server with identical interface to the real gripper
    * FrankaState implements torque, joint position/velocity, `O_T_EE`, `O_F_ext_hat`
    * Model provides `pose`, `zeroJacobian`, `bodyJacobian`, `mass`, `gravity`, `coriolis`

### Installation (MuJoCo path, one-click installer)

```bash
git clone --recursive https://github.com/tenfoldpaper/multipanda_ros2.git
cd multipanda_ros2
./tools/setup_env
./run
colcon build
source ~/multipanda_ws/install/setup.bash && \
  ros2 launch franka_bringup franka_sim.launch.py
```

## License

All packages are licensed under the [Apache 2.0 license](https://www.apache.org/licenses/LICENSE-2.0.html),
following `franka_ros2` and `multipanda_ros2`.
