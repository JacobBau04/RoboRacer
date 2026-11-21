from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():

    bringup_share = get_package_share_directory('mppi_bringup')

    map_yaml = os.path.join(
        bringup_share,
        'maps',
        'levine_blocked.yaml'
    )

    csv_path = os.path.join(
        bringup_share,
        'waypoints',
        'old',
        'lev_testing',
        'lev_blocked.csv'
    )

    mppi_node = Node(
        package='mppi',
        executable='mppi_node.py',
        name='mppi_node',
        output='screen',
        parameters=[{
            'track': 'levine_blocked',
            'wall_map_yaml': map_yaml,
            'waypoint_path': csv_path,
        }],
    )

    return LaunchDescription([mppi_node])