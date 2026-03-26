import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.conditions import IfCondition
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory
from moveit_configs_utils import MoveItConfigsBuilder


def _launch_setup(context):
    use_retimed = (
        LaunchConfiguration("use_retimed_pointclouds").perform(context).lower()
        in ("true", "1", "yes", "on")
    )
    sensors_file = "sensors_3d.yaml" if use_retimed else "sensors_3d_raw.yaml"
    sim_time_param = {"use_sim_time": LaunchConfiguration("use_sim_time")}

    moveit_config = (
        MoveItConfigsBuilder("moveit_resources_panda")
        .robot_description(
            file_path="config/panda.urdf.xacro",
            mappings={
                "ros2_control_hardware_type": LaunchConfiguration(
                    "ros2_control_hardware_type"
                )
            },
        )
        .robot_description_semantic(file_path="config/panda.srdf")
        .robot_description_kinematics(file_path="config/kinematics.yaml")
        .planning_scene_monitor(
            publish_robot_description=True, publish_robot_description_semantic=True
        )
        .trajectory_execution(file_path="config/gripper_moveit_controllers.yaml")
        .planning_pipelines(pipelines=["ompl"])
        .sensors_3d(
            file_path=os.path.join(
                get_package_share_directory("perception_pipeline_humble_demo"),
                "config",
                sensors_file,
            )
        )
        .to_moveit_configs()
    )

    # Start the actual move_group node/action server
    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[moveit_config.to_dict(), sim_time_param],
        arguments=["--ros-args", "--log-level", "info"],
    )

    pointcloud_retimer_node = Node(
        package="perception_pipeline_humble_demo",
        executable="retime_pointclouds.py",
        output="screen",
        condition=IfCondition(LaunchConfiguration("use_retimed_pointclouds")),
        parameters=[sim_time_param],
    )

    # RViz
    rviz_base = LaunchConfiguration("rviz_config")
    rviz_config = PathJoinSubstitution(
        [FindPackageShare("perception_pipeline_humble_demo"), "rviz2", rviz_base]
    )
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        output="log",
        condition=IfCondition(LaunchConfiguration("use_rviz")),
        arguments=["-d", rviz_config],
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.planning_pipelines,
            moveit_config.robot_description_kinematics,
            moveit_config.joint_limits,
            sim_time_param,
        ],
    )

    # Static TF
    static_tf_node = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="static_transform_publisher",
        output="log",
        parameters=[sim_time_param],
        arguments=["0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "world", "panda_link0"],
    )

    # Publish TF
    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="both",
        parameters=[
            moveit_config.robot_description,
            sim_time_param,
            {"publish_frequency": 50.0, "ignore_timestamp": True},
        ],
    )

    # ros2_control using FakeSystem as hardware
    ros2_controllers_path = os.path.join(
        get_package_share_directory("moveit_resources_panda_moveit_config"),
        "config",
        "ros2_controllers.yaml",
    )
    ros2_control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[ros2_controllers_path, sim_time_param],
        remappings=[
            ("/controller_manager/robot_description", "/robot_description"),
        ],
        output="screen",
    )

    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "joint_state_broadcaster",
            "--controller-manager",
            "/controller_manager",
        ],
    )

    panda_arm_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["panda_arm_controller", "-c", "/controller_manager"],
    )

    panda_hand_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["panda_hand_controller", "-c", "/controller_manager"],
        condition=IfCondition(LaunchConfiguration("spawn_hand_controller")),
    )

    return [
        rviz_node,
        static_tf_node,
        robot_state_publisher,
        move_group_node,
        pointcloud_retimer_node,
        ros2_control_node,
        joint_state_broadcaster_spawner,
        panda_arm_controller_spawner,
        panda_hand_controller_spawner,
    ]


def generate_launch_description():
    # Command-line arguments
    rviz_config_arg = DeclareLaunchArgument(
        "rviz_config",
        default_value="perception_pipeline.rviz",
        description="RViz configuration file",
    )

    ros2_control_hardware_type = DeclareLaunchArgument(
        "ros2_control_hardware_type",
        default_value="mock_components",
        description="ROS 2 control hardware interface type to use for the launch file -- possible values: [mock_components, isaac]",
    )
    use_rviz_arg = DeclareLaunchArgument(
        "use_rviz",
        default_value="true",
        description="Set to false to skip RViz for headless runs",
    )
    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time",
        default_value="false",
        description="Use simulation clock for all demo nodes",
    )
    use_retimed_pointclouds_arg = DeclareLaunchArgument(
        "use_retimed_pointclouds",
        default_value="true",
        description="Use retimed point cloud topics for stable looped playback",
    )
    spawn_hand_controller_arg = DeclareLaunchArgument(
        "spawn_hand_controller",
        default_value="true",
        description="Spawn panda_hand_controller (set false for arm-only demos)",
    )

    return LaunchDescription(
        [
            rviz_config_arg,
            ros2_control_hardware_type,
            use_rviz_arg,
            use_sim_time_arg,
            use_retimed_pointclouds_arg,
            spawn_hand_controller_arg,
            OpaqueFunction(function=_launch_setup),
        ]
    )
