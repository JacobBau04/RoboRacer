"""race1 in the simulator (f1tenth_gym_ros) with obstacle avoidance.

Start the simulator first (its own workspace ~/sim_ws, ROS_DOMAIN_ID 42, never the car's 12):
    ros2 launch f1tenth_gym_ros gym_bridge_launch.py config:=<sim config with a race1 map>
Then (in a shell WITHOUT the sim's venv):
    ros2 launch mpc race1_sim_launch.py
    ros2 launch mpc race1_sim_launch.py avoidance:=false record:=true
tools/sim_batch.py does both for many obstacle layouts.

Differences from the car: pose = /ego_racecar/odom (exact, base_link = rear axle, so the body
reaches 0.46 m ahead and 0.12 m behind it), no delay compensation, scans thinned to 15 Hz.
"""
import os
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

ENV = {'OPENBLAS_NUM_THREADS': '1'}
CAR_DOMAIN = '12'           # the car's ROS_DOMAIN_ID (~/.bashrc); the sim must never use it
FRONT, REAR = 0.46, 0.12    # sim car body around base_link (rear axle)


def generate_launch_description():
    if os.environ.get('ROS_DOMAIN_ID', '0') == CAR_DOMAIN:
        # the sim's MPC publishes /drive: on the car's domain the car would drive
        raise RuntimeError(f'ROS_DOMAIN_ID is {CAR_DOMAIN} (the car). Run the sim on its own '
                           'domain: export ROS_DOMAIN_ID=42')
    lc = LaunchConfiguration
    recorder = Path(__file__).resolve().parents[1] / 'tools' / 'run_recorder.py'
    return LaunchDescription([
        DeclareLaunchArgument('track', default_value='race1'),
        DeclareLaunchArgument('map_yaml', default_value='race1_cleaned_blocked.yaml',
                              description='walls map in mpc/maps (obstacle detection + planning)'),
        DeclareLaunchArgument('model', default_value='kinematic'),
        DeclareLaunchArgument('delay_comp', default_value='0.0'),
        DeclareLaunchArgument('avoidance', default_value='true'),
        DeclareLaunchArgument('record', default_value='false'),
        DeclareLaunchArgument('note', default_value=''),
        Node(package='mpc', executable='mpc_node.py', output='screen', additional_env=ENV,
             parameters=[{
                 'track': lc('track'),
                 'model': lc('model'),
                 'delay_comp': ParameterValue(lc('delay_comp'), value_type=float),
                 'avoidance': ParameterValue(lc('avoidance'), value_type=bool),
                 'publish_debug': True,
                 'pose_topic': '/ego_racecar/odom',
                 'safety_front': FRONT,
                 'safety_rear': REAR,
             }]),
        Node(package='mpc', executable='obstacle_avoidance_node.py', output='screen',
             additional_env=ENV, condition=IfCondition(lc('avoidance')),
             parameters=[{
                 'track': lc('track'),
                 'map_yaml': lc('map_yaml'),
                 'scan_topic': '/scan',
                 'pose_topic': '/ego_racecar/odom',
                 'pose_frame': 'ego_racecar/base_link',
                 'pose_source': 'odom',         # exact sim pose (its TF rate is too high for Python)
                 'laser_offset': [0.275, 0.0, 0.0],   # sim.yaml lidar_base_link_to_lidar_tf
                 'max_scan_rate': 15.0,        # like the car's SICK (the sim publishes 250 Hz)
                 'scan_stamp_at_end': False,
                 'planner.front': FRONT,
                 'planner.rear': REAR,
             }]),
        ExecuteProcess(
            condition=IfCondition(lc('record')),
            cmd=['python3', str(recorder), '--track', lc('track'), '--platform', 'sim',
                 ['--note=', lc('note')]],
            output='screen', additional_env={'PYTHONUNBUFFERED': '1', **ENV},
            sigterm_timeout='60'),
    ])
