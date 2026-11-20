"""Real car on race1 with obstacle avoidance and the run recorder.

Start the car stack and the state estimation first (`ros2 launch state_estimation estimation_launch.py`,
which replaced `particle_filter localize_launch.py`), then:
    ros2 launch mpc race2_obstacles_launch.py                    # avoidance + recorder
    ros2 launch mpc race2_obstacles_launch.py avoidance:=false   # plain MPC (clean-lap baseline)
    ros2 launch mpc race2_obstacles_launch.py record:=false note:="..." delay_comp:=0.20 pose_offset:=0.27

State estimator: `estimator` (default: ESTIMATOR in state_estimation/launch/estimation_launch.py, the one place
to change it) picks the MPC's pose_topic, pose_offset and delay_comp from state_estimation's
config/estimators/<name>.yaml. race = the race-2 values (pf/pose/odom, 0.27, 0.20). delay_comp:= / pose_offset:=
on the command line still override. Use the same estimator here as in estimation_launch.py.

The recorder keeps the rosbag by default here (keep_bag:=true): it holds /scan and /tf, so
tools/replay_detection.py can rerun detection on it offline.
"""
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
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



def _nodes(context):
    """MPC, obstacle avoidance and recorder, with the pose settings of the chosen estimator."""
    from state_estimation.estimators import load
    lc = LaunchConfiguration
    cfg = context.launch_configurations
    est = load(cfg['estimator'])
    pose_topic = est['mpc']['pose_topic']
    delay_comp = cfg['delay_comp'] or str(est['mpc']['delay_comp'])     # '' = the estimator's value
    pose_offset = cfg['pose_offset'] or str(est['mpc']['pose_offset'])
    print(f'[race1_obstacles_launch] estimator {est["name"]}: pose_topic {pose_topic}, '
          f'pose_offset {pose_offset}, delay_comp {delay_comp}', flush=True)
    return [
        Node(package='mpc', executable='mpc_node.py', output='screen', additional_env=ENV,
             parameters=[{
                 'track': lc('track'),
                 'model': lc('model'),
                 'delay_comp': float(delay_comp),
                 'pose_offset': float(pose_offset),
                 'avoidance': ParameterValue(lc('avoidance'), value_type=bool),
                 'publish_debug': True,
                 'pose_topic': pose_topic,
             }]),
        Node(package='mpc', executable='obstacle_avoidance_node.py', output='screen',
             additional_env=ENV, condition=IfCondition(lc('avoidance')),
             parameters=[{
                 'track': lc('track'),
                 'map_yaml': lc('map_yaml'),
                 'scan_topic': '/scan',
                 'pose_topic': pose_topic,     # speed only; the pose comes from TF map -> laser
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
                 ['--note=', lc('note')], ['--keep-bag=', lc('keep_bag')], '--estimator', est['name']],
            output='screen', additional_env={'PYTHONUNBUFFERED': '1', **ENV},
            sigterm_timeout='60'),
    ]


def generate_launch_description():
    from state_estimation.estimators import default_estimator
    return LaunchDescription([
        DeclareLaunchArgument('track', default_value='race1'),
        DeclareLaunchArgument('map_yaml', default_value='race1_cleaned_blocked.yaml',
                              description='walls map in mpc/maps (obstacle detection + planning)'),
        DeclareLaunchArgument('model', default_value='kinematic'),
        DeclareLaunchArgument('estimator', default_value=default_estimator(),
                              description='race | 1A | 1B | 1C | 2A (default: ESTIMATOR in '
                                          'state_estimation/launch/estimation_launch.py)'),
        DeclareLaunchArgument('delay_comp', default_value='',
                              description="MPC delay compensation [s]; '' = the estimator's "
                                          '(race 0.20, measured after the VESC swap; 1A/1B 0.14)'),
        DeclareLaunchArgument('pose_offset', default_value='',
                              description="pose point ahead of the rear axle [m]; '' = the estimator's "
                                          '(race 0.27 = particle-filter LiDAR pose; 1A/1B 0)'),
        DeclareLaunchArgument('avoidance', default_value='true'),
        DeclareLaunchArgument('record', default_value='true'),
        DeclareLaunchArgument('keep_bag', default_value='true'),
        DeclareLaunchArgument('note', default_value=''),
        OpaqueFunction(function=_nodes),
    ])
