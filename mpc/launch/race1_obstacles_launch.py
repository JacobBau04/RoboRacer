"""Real car on race1 with obstacle avoidance and the run recorder.

Start the car stack and the particle filter first, then:
    ros2 launch mpc race2_obstacles_launch.py                    # avoidance + recorder
    ros2 launch mpc race2_obstacles_launch.py avoidance:=false   # plain MPC (clean-lap baseline)
    ros2 launch mpc race2_obstacles_launch.py record:=false note:="..." delay_comp:=0.20 pose_offset:=0.27

The recorder keeps the rosbag by default here (keep_bag:=true): it holds /scan and /tf, so
tools/replay_detection.py can rerun detection on it offline.
"""
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

ENV = {'OPENBLAS_NUM_THREADS': '1'}   # numpy's OpenBLAS otherwise spins idle threads at 100 % CPU


def find_recorder():
    """tools/ is not installed; with --symlink-install this file resolves into the source tree."""
    here = Path(__file__).resolve().parents[1] / 'tools' / 'run_recorder.py'
    if here.exists():
        return here
    raise FileNotFoundError(f'{here} not found (build with --symlink-install)')



def generate_launch_description():
    lc = LaunchConfiguration
    return LaunchDescription([
        DeclareLaunchArgument('track', default_value='race1'),
        DeclareLaunchArgument('map_yaml', default_value='race1_cleaned_blocked.yaml',
                              description='walls map in mpc/maps (obstacle detection + planning)'),
        DeclareLaunchArgument('model', default_value='kinematic'),
        DeclareLaunchArgument('delay_comp', default_value='0.20',
                              description='MPC delay compensation [s] (measured 0.20 after the '
                                          'VESC swap, race2 runs 2026-11-04)'),
        DeclareLaunchArgument('pose_offset', default_value='0.27',
                              description='particle-filter pose (LiDAR) ahead of the rear axle [m]; '
                                          'the MPC model is for the rear axle'),
        DeclareLaunchArgument('avoidance', default_value='true'),
        DeclareLaunchArgument('record', default_value='true'),
        DeclareLaunchArgument('keep_bag', default_value='true'),
        DeclareLaunchArgument('note', default_value=''),
        Node(package='mpc', executable='mpc_node.py', output='screen', additional_env=ENV,
             parameters=[{
                 'track': lc('track'),
                 'model': lc('model'),
                 'delay_comp': ParameterValue(lc('delay_comp'), value_type=float),
                 'pose_offset': ParameterValue(lc('pose_offset'), value_type=float),
                 'avoidance': ParameterValue(lc('avoidance'), value_type=bool),
                 'publish_debug': True,
                 'pose_topic': 'pf/pose/odom',
             }]),
        Node(package='mpc', executable='obstacle_avoidance_node.py', output='screen',
             additional_env=ENV, condition=IfCondition(lc('avoidance')),
             parameters=[{
                 'track': lc('track'),
                 'map_yaml': lc('map_yaml'),
                 'scan_topic': '/scan',
                 'pose_topic': 'pf/pose/odom',
                 'pose_frame': '',             # the particle filter pose = the LiDAR frame
                 'scan_stamp_at_end': True,    # SICK TiM571: stamp ~ arrival of the last beam
                 'detection.wall_margin': 0.25,   # race2 map is up to ~0.2 m off in places
                 'beam_time_offset': 0.04,        # best wall fit in tools/replay_detection.py
                 'detection.occlusion_extend': 0.1,  # was 0.3: hidden-edge growth boxed the car in
                 'planner.tight_speed': 1.6,      # was 1.2 m/s: speed through tight gaps
                 'planner.crawl_speed': 0.9,      # was 0.6 m/s: speed at the crawl level
             }]),
        ExecuteProcess(
            condition=IfCondition(lc('record')),
            cmd=['python3', str(find_recorder()), '--track', lc('track'), '--platform', 'real',
                 ['--note=', lc('note')], ['--keep-bag=', lc('keep_bag')]],
            output='screen', additional_env={'PYTHONUNBUFFERED': '1', **ENV},
            sigterm_timeout='60'),
    ])
