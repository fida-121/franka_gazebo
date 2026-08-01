#  Copyright (c) 2021 Franka Emika GmbH
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

# Adapted from panda_arm_sim demo.launch.py (MuJoCo) to run on Gazebo Sim (Fortress)
# with OMPL + CHOMP as selectable planning pipelines.
#
# BEFORE RUNNING THIS FILE, MAKE SURE:
#   1. franka_moveit_config/config/panda_controllers.yaml has:
#        panda_arm_controller:      default: true
#        joint_impedance_controller: default: false
#      (currently the reverse -- MoveIt will otherwise keep targeting the
#      impedance controller no matter what you spawn in Gazebo)
#   2. franka_moveit_config/config/sim_gazebo_panda_ros_controllers.yaml exists
#      (copy of sim_panda_ros_controllers.yaml)
#   3. franka_moveit_config/config/chomp_planning.yaml exists
#   4. franka_description/robots/sim/panda_arm_gazebo.urdf.xacro and
#      panda_arm_gazebo.ros2_control.xacro exist (from the previous message)
#   5. ros-humble-gz-ros2-control, ros-humble-ros-gz-sim, ros-humble-moveit-planners-chomp
#      are installed

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, Shutdown
from launch.conditions import IfCondition
from launch.substitutions import Command, FindExecutable, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch.actions import SetEnvironmentVariable
import yaml


def concatenate_ns(ns1, ns2, absolute=False):
    if len(ns1) == 0:
        return ns2
    if len(ns2) == 0:
        return ns1
    if ns1[0] == '/':
        ns1 = ns1[1:]
    if ns1[-1] == '/':
        ns1 = ns1[:-1]
    if ns2[0] == '/':
        ns2 = ns2[1:]
    if ns2[-1] == '/':
        ns2 = ns2[:-1]
    if absolute:
        ns1 = '/' + ns1
    return ns1 + '/' + ns2


def load_yaml(package_name, file_path):
    package_path = get_package_share_directory(package_name)
    absolute_file_path = os.path.join(package_path, file_path)
    try:
        with open(absolute_file_path, 'r') as file:
            return yaml.safe_load(file)
    except EnvironmentError:
        return None


def generate_launch_description():

    # Parameters as launch arguments
    arm_id_param = 'arm_id'
    initial_positions_param = 'initial_positions'

    arm_id = LaunchConfiguration(arm_id_param)
    initial_positions = LaunchConfiguration(initial_positions_param)

    # Command-line arguments
    db_arg = DeclareLaunchArgument(
        'db', default_value='False', description='Database flag'
    )

    load_gripper = True  # arm-only ros2_control for now -- see note at bottom of chat
    use_sim_time = {'use_sim_time': True}

    # ---- Robot description (Gazebo variant) ----
    franka_xacro_file = os.path.join(
        get_package_share_directory('franka_description'), 'robots', 'sim',
        'panda_arm_gazebo.urdf.xacro')
    franka_bringup_path = get_package_share_directory('franka_bringup')

    set_ign_resource_path = SetEnvironmentVariable(
        'IGN_GAZEBO_RESOURCE_PATH',
        os.path.join(get_package_share_directory('franka_description'), '..')
    )

    set_ign_plugin_path = SetEnvironmentVariable(
        'IGN_GAZEBO_SYSTEM_PLUGIN_PATH',
        '/opt/ros/humble/lib'
    )
    robot_description_config = Command(
        [FindExecutable(name='xacro'), ' ', franka_xacro_file,
         ' arm_id:=', arm_id,
         ' hand:=', str(load_gripper).lower(),
         ' initial_positions:=', initial_positions])

    robot_description = {
        'robot_description': ParameterValue(robot_description_config, value_type=str)
    }

    # ---- Semantic description (unchanged) ----
    franka_semantic_xacro_file = os.path.join(get_package_share_directory('franka_moveit_config'),
                                              'srdf',
                                              'panda_arm.srdf.xacro')
    robot_description_semantic_config = Command(
        [FindExecutable(name='xacro'), ' ', franka_semantic_xacro_file, ' hand:=', str(load_gripper).lower()]
    )
    robot_description_semantic = {
        'robot_description_semantic': ParameterValue(robot_description_semantic_config, value_type=str)
    }

    kinematics_yaml = load_yaml(
        'franka_moveit_config', 'config/kinematics.yaml'
    )

    # ---- Planning pipelines: OMPL + CHOMP, both selectable at runtime in RViz ----
    # NOTE: these dicts must NOT be wrapped in an extra 'move_group': {...} key.
    # move_group_node is already the node these parameters get attached to --
    # adding 'move_group' as a key here creates the parameter
    # 'move_group.planning_pipelines...' instead of 'planning_pipelines...',
    # which MoveIt silently never finds. That was the root cause of the
    # pipeline selector never working and CHOMP always being force-picked.
    planning_pipelines_config = {
        'planning_pipelines': {
            'pipeline_names': ['ompl', 'chomp'],
        },
        'default_planning_pipeline': 'ompl',
    }

    ompl_planning_pipeline_config = {
        'ompl': {
            'planning_plugin': 'ompl_interface/OMPLPlanner',
            'request_adapters': 'default_planner_request_adapters/AddTimeOptimalParameterization '
                                'default_planner_request_adapters/ResolveConstraintFrames '
                                'default_planner_request_adapters/FixWorkspaceBounds '
                                'default_planner_request_adapters/FixStartStateBounds '
                                'default_planner_request_adapters/FixStartStateCollision '
                                'default_planner_request_adapters/FixStartStatePathConstraints',
            'start_state_max_bounds_error': 0.1,
        }
    }
    ompl_planning_yaml = load_yaml(
        'franka_moveit_config', 'config/ompl_planning.yaml'
    )
    ompl_planning_pipeline_config['ompl'].update(ompl_planning_yaml)

    chomp_planning_pipeline_config = {
        'chomp': {
            'planning_plugin': 'chomp_interface/CHOMPPlanner',
            'request_adapters': 'default_planner_request_adapters/AddTimeOptimalParameterization '
                                'default_planner_request_adapters/ResolveConstraintFrames '
                                'default_planner_request_adapters/FixWorkspaceBounds '
                                'default_planner_request_adapters/FixStartStateBounds '
                                'default_planner_request_adapters/FixStartStateCollision '
                                'default_planner_request_adapters/FixStartStatePathConstraints',
            'start_state_max_bounds_error': 0.1,
        }
    }
    chomp_planning_yaml = load_yaml(
        'franka_moveit_config', 'config/chomp_planning.yaml'
    )
    chomp_planning_pipeline_config['chomp'].update(chomp_planning_yaml)

    robot_description_planning = {
        'robot_description_planning': load_yaml(
            'franka_moveit_config', 'config/joint_limits.yaml'
        )
    }

    # ---- Trajectory execution: panda_arm_controller (make sure default:true, see note above) ----
    moveit_simple_controllers_yaml = load_yaml(
        'franka_moveit_config', 'config/panda_controllers.yaml'
    )
    moveit_controllers = {
        'moveit_simple_controller_manager': moveit_simple_controllers_yaml,
        'moveit_controller_manager': 'moveit_simple_controller_manager'
                                     '/MoveItSimpleControllerManager',
    }

    trajectory_execution = {
        'moveit_manage_controllers': True,
        'trajectory_execution.allowed_execution_duration_scaling': 3.0,
        'trajectory_execution.allowed_goal_duration_margin': 5.0,
        'trajectory_execution.allowed_start_tolerance': 0.05,
    }

    planning_scene_monitor_parameters = {
        'publish_planning_scene': True,
        'publish_geometry_updates': True,
        'publish_state_updates': True,
        'publish_transforms_updates': True,
    }

    # ---- move_group node ----
    run_move_group_node = Node(
        package='moveit_ros_move_group',
        executable='move_group',
        output='screen',
        parameters=[
            robot_description,
            robot_description_semantic,
            kinematics_yaml,
            robot_description_planning,
            planning_pipelines_config,
            ompl_planning_pipeline_config,
            chomp_planning_pipeline_config,
            trajectory_execution,
            moveit_controllers,
            planning_scene_monitor_parameters,
            use_sim_time,          # <-- add
        ],
    )

    # ---- RViz ----
    rviz_base = os.path.join(get_package_share_directory('franka_moveit_config'), 'rviz')
    rviz_full_config = os.path.join(rviz_base, 'moveit.rviz')

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='log',
        arguments=['-d', rviz_full_config],
        parameters=[
            robot_description,
            robot_description_semantic,
            planning_pipelines_config,
            ompl_planning_pipeline_config,
            chomp_planning_pipeline_config,
            kinematics_yaml,
            robot_description_planning,
            use_sim_time,          # <-- add
        ],
    )

    # ---- TF ----
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='both',
        parameters=[robot_description, use_sim_time],
    )

    # ---- Gazebo Sim (Fortress) ----
    gz_sim = ExecuteProcess(
        cmd=['ign', 'gazebo', '-r', 'empty.sdf'],
        output='screen',
        on_exit=Shutdown(),
    )

    # Spawn the robot entity from the /robot_description topic
    spawn_entity = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=['-topic', 'robot_description', '-name', arm_id],
        output='screen'
    )

    # ---- Load controllers (same spawner pattern as MuJoCo) ----
    load_controllers = []
    for controller in ['panda_arm_controller', 'joint_state_broadcaster', 'panda_hand_controller']:
        load_controllers += [
            ExecuteProcess(
                cmd=['ros2 run controller_manager spawner {}'.format(controller)],
                shell=True,
                output='screen',
            )
        ]

    # ---- Warehouse mongodb server (unchanged) ----
    db_config = LaunchConfiguration('db')
    mongodb_server_node = Node(
        package='warehouse_ros_mongo',
        executable='mongo_wrapper_ros.py',
        parameters=[
            {'warehouse_port': 33829},
            {'warehouse_host': 'localhost'},
            {'warehouse_plugin': 'warehouse_ros_mongo::MongoDatabaseConnection'},
        ],
        output='screen',
        condition=IfCondition(db_config)
    )

    clock_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=['/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock'],
        output='screen',
    )

    # ---- Launch arguments ----
    arm_id_arg = DeclareLaunchArgument(
        arm_id_param,
        default_value='panda',
        description='The name of the robot. Defaults to panda.')

    initial_position_arg = DeclareLaunchArgument(
        initial_positions_param,
        default_value='0.0,-0.785,0.0,-2.356,0.0,1.571,0.785',
        description='Initial joint positions, comma-separated, no spaces or quotes.'
    )

    return LaunchDescription(
        [arm_id_arg,
         initial_position_arg,
         db_arg,
         set_ign_resource_path,
         set_ign_plugin_path,   # <-- add this
         rviz_node,
         robot_state_publisher,
         run_move_group_node,
         gz_sim,
         spawn_entity,
         mongodb_server_node,
         clock_bridge,          # <-- add this
         ]
        + load_controllers
    )
