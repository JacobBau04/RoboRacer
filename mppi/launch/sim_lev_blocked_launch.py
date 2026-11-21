from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os
from launch.actions import TimerAction

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
        'lev_blocked.csv')

    # Keep your existing map_yaml and csv_path definitions here.

    opponent_detector = Node(
        package='mppi',
        executable='opponent_lidar_detector_node',
        name='opponent_lidar_detector',
        output='screen'
    )

    opponent_predictor = Node(
        package='mppi',
        executable='opponent_predictor_node.py',
        name='opponent_predictor',
        output='screen'
    )

    mppi_controller = Node(
        package='mppi',
        executable='mppi_node.py',
        name='mppi_node',
        output='screen',
        parameters=[{
            'track': 'levine_blocked',
            'wall_map_yaml': map_yaml,
            'waypoint_path': csv_path
        }]
    )

    return LaunchDescription([
        opponent_detector,
        opponent_predictor,

        TimerAction(
            period=3.0,
            actions=[mppi_controller]
        )
    ])