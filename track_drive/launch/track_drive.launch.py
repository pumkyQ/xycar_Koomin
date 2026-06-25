import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    # 1. Include the ros_tcp_endpoint package's endpoint.py launch description
    endpoint_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(
                get_package_share_directory('ros_tcp_endpoint'),
                'launch',
                'endpoint.py'
            )
        ])
    )

    # 2. Launch the track_drive package's track_drive node (entry point: track_drive.track_drive:main)
    track_drive_node = Node(
        package='track_drive',
        executable='track_drive',
        name='driver',
        output='screen'
    )

    return LaunchDescription([
        endpoint_launch,
        track_drive_node
    ])
