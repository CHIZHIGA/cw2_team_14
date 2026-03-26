from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('pcl_tutorial'),
        'config',
        'pcl_tutorial.yaml'
    )

    return LaunchDescription([
        Node(
            package='pcl_tutorial',
            executable='pcl_tutorial_node',
            name='pcl_tutorial_node',
            output='screen',
            parameters=[config],
        )
    ])
