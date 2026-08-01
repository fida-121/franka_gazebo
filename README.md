# franka_gazebo

This repository is a fork of `multipanda_ros2`, a `ros2_control`-based framework for the Franka Emika Panda robot on ROS 2 Humble, developed by Jon Škerlj, Seongjin Bien, Abdeldjallil Naceri, and Sami Haddadin. All of the underlying real-hardware and MuJoCo simulation architecture — `franka_hardware`, the MoveIt configuration, the controller framework, and the MuJoCo integration via `mujoco_ros_pkgs` — comes directly from that project. See their paper for the full framework:

**Bridging the Sim-to-Real Gap with multipanda_ros2: A Real-Time ROS2 Framework for Multimanual Systems**
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

The original project itself is built on `mcbed`'s ROS 2 Humble port of `franka_ros2`.

### What this fork adds: Gazebo Sim support
The upstream project's simulation path is built around MuJoCo, via a forked `mujoco_ros_pkgs` plugin. This fork adds a **second, parallel simulation path** using Gazebo Sim (Ignition Fortress), built on `gz_ros2_control` instead of the MuJoCo plugin. The original MuJoCo path (`franka_bringup franka_sim.launch.py`) is untouched — everything below is additive.

---

## Working Features

More thorough information is available in the documentation.

**Real Robot**
*   `FrankaState` broadcaster
*   All control interfaces (torque, position, velocity, Cartesian)
*   Example controllers for all interfaces
*   Controllers are swappable using `rqt_controller_manager`
*   Runtime `franka::ControlException` error recovery via `~/service_server/error_recovery`. Upon recovery, the previously executed control loop will be executed again, so no reloading necessary.
*   Runtime internal parameter setter services much like what is offered in the updated `franka_ros2`

**Sim Robot**
*   Same as the real robot, except Cartesian command interface is not available, and there is no plan to implement this for now.
*   Gripper server with identical interface to the real gripper (i.e. action servers).
*   Example controllers for the real single-arm listed above, that correspond to those interfaces, work out of the box.
*   `FrankaState` implements the basics: torque, joint position/velocity, `O_T_EE` and `O_F_ext_hat`.
*   Model provides all the existing functions: `pose`, `zeroJacobian`, `bodyJacobian`, `mass`, `gravity`, `coriolis`.

---

## Installation — Windows 11

1. Open `cmd` as administrator and paste:
   ```cmd
   wsl --install -d Ubuntu
   ```

2. **Two cases:**
   *   **If it asks for a username and password:** Enter any username and password of your choice. This creates a new Linux user account, separate from your Windows account. It may also ask whether to enable metrics collection; either option is fine. Once the setup is complete, **reboot your device**.
   *   **If it does not ask for a username or password:** Wait for the command to finish, then **reboot your device**. After restarting, the Ubuntu setup may resume automatically. Let it finish, then create a new Linux username and password when prompted. If nothing opens automatically after reboot, launch the **Ubuntu** app from the Start menu to continue the setup.

3. Open a **separate** `cmd` window as administrator and run:
   ```cmd
   wsl -l -v
   ```
   Confirm `VERSION` shows `2`, e.g.:
   ```text
   NAME      STATE           VERSION
   * Ubuntu    Running         2
   ```

4. Still in that same `cmd` window (not the Ubuntu one), run:
   ```cmd
   notepad "%USERPROFILE%\.wslconfig"
   ```
   Press **Yes** to create the file. Paste:
   ```ini
   [wsl2]
   networkingMode=mirrored
   ```
   Save, close, then:
   ```cmd
   wsl --shutdown
   ```

5. Reopen the Ubuntu app from the Start menu.

6. Install **Docker Desktop for Windows** (Download from their website), Run the `.exe` file and install. Make sure "Use WSL 2 instead of Hyper-V" is checked during install. After install, go to **Docker Desktop App → Settings → Resources → WSL Integration**:
   *   "Enable integration with my default WSL distro" — **ON**
   *   Your Ubuntu distro toggled **ON** individually
   *   Apply & Restart

7. Back in the Ubuntu WSL terminal:
   ```bash
   docker info
   ```
   *If you see "permission denied" — close the Ubuntu window entirely, reopen, and retry.*

8. *(Optional, if you have an NVIDIA GPU):*
   ```bash
   nvidia-smi
   ```
   *If this shows your GPU, Docker will auto-detect it later.*

9. **Clone the repository:**
   ```bash
   cd ~
   mkdir -p franka_gazebo
   cd franka_gazebo
   git clone --recursive https://github.com/fida-121/franka_gazebo.git
   cd franka_gazebo
   ```

10. **Set up and enter the container:**
    ```bash
    ./tools/setup_env
    sudo apt update
    sudo apt install -y x11-xserver-utils
    ./run
    ```
    *If asked for a password, enter the password you created for your Linux user account.*

11. You're now **inside** the container at a `developer@docker-desktop` prompt. Still inside the container, build the workspace:
    ```bash
    colcon build
    ```
    *If you get stuck in colcon build while setting up `multi_mode_controller`. Close the terminal. Try the following:*
    ```bash
    colcon build --packages-select multi_mode_controller
    colcon build --packages-skip multi_mode_controller
    ```
    *If Ubuntu terminal doesn't open. Run the following in the cmd and then try again:*
    ```cmd
    wsl --shutdown
    ```

12. Make ROS see your built packages in every future shell:
    ```bash
    source ~/multipanda_ws/install/setup.bash 
    echo "source ~/multipanda_ws/install/setup.bash" >> ~/.bashrc
    ```

13. To open a **second** terminal into the same running container later (open a new WSL/Ubuntu terminal window, then run):
    ```bash
    docker exec -it --user developer gazebo-container /bin/bash -c "source /home/developer/.bashrc && bash"
    ```

---

## Installation — Windows 10 (Not Verified)

Everything above is identical, **except**:
1. Mirrored networking (the `.wslconfig` step) may not be supported on older Windows 10 builds. Check with `wsl --version`, if `networkingMode` isn't recognized after `wsl --shutdown`, skip that step and flag it so the `run` script's `--network` flag can be adjusted instead.
2. Install **VcXsrv** (search "VcXsrv" on SourceForge), launch it via XLaunch with "Disable access control" checked, and leave it running in the background.
3. Add this line to `~/.bashrc` inside WSL *(not the container)* before running `./run`:
   ```bash
   export DISPLAY=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}'):0
   ```
   Then:
   ```bash
   source ~/.bashrc
   ```
4. Make sure VcXsrv is already running before `./run`.

---

## Installation — Ubuntu 22.04

1. Clone the repository recursively to include `mujoco_ros_pkgs`:
   ```bash
   git clone --recursive https://github.com/fida-121/franka_gazebo.git
   ```

2. Change into the cloned repository:
   ```bash
   cd franka_gazebo
   ```

3. Build the docker image by running one command (takes some time):
   ```bash
   ./tools/setup_env
   ```

4. Once the image is built, start the development container:
   ```bash
   ./run
   ```

The default config allows for communication in the network, GPU access, display forwarding for GUI applications, hardware devices, etc. By default the script opens a bash shell inside the container as a `developer` user in the ROS2 workspace under `~/multipanda_ws`.

5. Build ROS2 packages with:
   ```bash
   colcon build
   ```

6. Make ROS see your built packages in every future shell:
   ```bash
   source ~/multipanda_ws/install/setup.bash 
   echo "source ~/multipanda_ws/install/setup.bash" >> ~/.bashrc
   ```

*(In case there are problems with missing packages, try running the following commands inside the container before `colcon build`:)*
```bash
sudo apt update && \
rosdep update && \
rosdep install --from-paths src --ignore-src -y -r
```

To open the docker container in an additional terminal, use the `docker exec` command:
```bash
docker exec -it --user developer gazebo-container bash
```

---

## Gazebo Prerequisites

In addition to the base install, ensure the following packages are installed inside your environment/container to run Gazebo Sim:

```bash
sudo apt update
sudo apt install -y \
  ros-humble-ros-gz \
  ros-humble-gz-ros2-control \
  ros-humble-ros-gz-sim \
  ros-humble-ros-gz-bridge \
  ros-humble-moveit-planners-chomp
```

---

## Usage Commands

Run the commands below inside the container unless noted otherwise.

### 1. Launching RViz and Gazebo Simulation
This launches Gazebo Sim with the Panda arm and gripper spawned in. `move_group` is available with **both OMPL and CHOMP** as selectable planning pipelines (switchable live from RViz's MotionPlanning panel), and RViz is configured to plan and execute directly against the Gazebo simulation.

```bash
source ~/multipanda_ws/install/setup.bash
ros2 launch franka_moveit_config gazebo.launch.py
```

### 2. Joint-Space Motion (Directly via Controller)
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

### 3. Gripper Control
The gripper is exposed as a standard `control_msgs/action/GripperCommand` action. No homing step is required in simulation.

**Open Gripper:**
```bash
ros2 action send_goal /panda_hand_controller/gripper_cmd control_msgs/action/GripperCommand "{command: {position: 0.04, max_effort: 20.0}}"
```

**Close / Grasp:**
```bash
ros2 action send_goal /panda_hand_controller/gripper_cmd control_msgs/action/GripperCommand "{command: {position: 0.0, max_effort: 20.0}}"
```

---

## Implementation Notes

*   **Inertials:** The base `panda_arm.xacro`/`hand.xacro` macros ship with no `<inertial>` blocks (mass/inertia lives in the MuJoCo MJCF model instead). Gazebo's `urdf2sdf` conversion silently drops the entire model if any link is missing one. Both macros now accept an `add_inertials:=false` (default) parameter — only the Gazebo xacro path sets it `true`, so the MuJoCo/real-hardware output is unchanged.
*   **Mimic joint:** `panda_finger_joint2` is a mimic joint. Only the leader (`panda_finger_joint1`) has a command interface in `ros2_control`; the follower has state interfaces only.
*   **Gain Control:** `position_proportional_gain` is a global `gz_ros2_control` plugin parameter (a direct child of `<gazebo><plugin>`), not a per-joint setting.
*   **Controller naming:** `sim_gazebo_panda_controllers.yaml` and `sim_gazebo_panda_ros_controllers.yaml` are Gazebo-specific copies of the shared MoveIt/ `ros2_control` config files, so real-hardware and MuJoCo controller naming stay untouched.

---

## License

All packages of `multipanda_ros2` (and this fork) are licensed under the **Apache 2.0 license**, following `franka_ros2`.
