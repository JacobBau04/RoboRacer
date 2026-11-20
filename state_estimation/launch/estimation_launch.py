"""State estimation for the car: one place to choose what the MPC drives on.

    ros2 launch state_estimation estimation_launch.py                    # uses ESTIMATOR below
    ros2 launch state_estimation estimation_launch.py estimator:=1A      # override for one run
    ros2 launch state_estimation estimation_launch.py estimator:=1A initial_pose:='[x, y, theta]'

Replaces `ros2 launch particle_filter localize_launch.py` (T2). Start the car bringup first (and hold LB once, so
/odom starts). The MPC launch (mpc race1/race2_obstacles_launch.py) reads the same ESTIMATOR and sets its
pose_topic, pose_offset and delay_comp to match, so only this line changes.

  race  particle filter on raw /odom, its own map -> laser TF                    (race 2 setup, A_car)
  1A    sensor_prep -> robot_localization EKF (ekf1) -> particle filter (scan mode) -> state_assembler
        -> /state_estimate; shadow EKF (1B) runs alongside for comparison; viz_tf publishes map -> odom
  1B    as 1A, the assembler predicts with the shadow EKF instead of ekf1
  1C, 2A  not on the car yet

Arguments: estimator, initial_pose ('[0.0]' = global init; else [x, y, theta] of the LASER in map; RViz/Foxglove
/initialpose also works), shadow (1A only: run 1B alongside, default true), use_sim_time (bag replay).
The map is particle_filter's localize.yaml `map` (one setting for every estimator).
"""
# TODO (Jacob): think about implementing the commanded steering angle into the dynamics model or measurement for
# EKF_1. Would this be helpful? (race uses it through /odom; 1A doesn't use it; 1B uses it in its yaw model)
ESTIMATOR = '1A'   # race | 1A | 1B | 1C | 2A   <- the one setting to change

import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch_ros.actions import Node

ENV = {'OPENBLAS_NUM_THREADS': '1'}   # TUNING_LOG H3: numpy's idle OpenBLAS threads ate 5 of 6 cores


def _initial_pose(text):
    v = yaml.safe_load(text)
    v = [float(x) for x in (v if isinstance(v, list) else [v])]
    if len(v) not in (1, 3):
        raise RuntimeError(f"initial_pose must be '[x, y, theta]' or '[0.0]', got {text!r}")
    return v


def _setup(context):
    from state_estimation.estimators import load
    a = context.launch_configurations
    est = load(a['estimator'])
    sim = {'use_sim_time': a['use_sim_time'].lower() == 'true'}
    se = get_package_share_directory('state_estimation')
    cfg = lambda f: os.path.join(se, 'config', f)  # noqa: E731
    pf_share = get_package_share_directory('particle_filter')
    localize = os.path.join(pf_share, 'config', 'localize.yaml')
    map_name = yaml.safe_load(open(localize))['map_server']['ros__parameters']['map']
    init = _initial_pose(a['initial_pose'])

    actions = [
        Node(package='nav2_map_server', executable='map_server', name='map_server', output='screen',
             parameters=[{'yaml_filename': os.path.join(pf_share, 'maps', map_name + '.yaml'),
                          'topic': 'map', 'frame_id': 'map'}, sim]),
        Node(package='nav2_lifecycle_manager', executable='lifecycle_manager', name='lifecycle_manager_localization',
             output='screen', parameters=[{'autostart': True, 'node_names': ['map_server']}, sim]),
    ]
    if est['name'] == 'race':
        actions.append(Node(package='particle_filter', executable='particle_filter', name='particle_filter',
                            output='screen', additional_env=ENV,
                            parameters=[localize, {'initial_pose': init}, sim]))
        return actions

    # 1A / 1B
    actions += [
        Node(package='state_estimation', executable='sensor_prep', name='sensor_prep', output='screen',
             additional_env=ENV, parameters=[cfg('prep.yaml'), sim]),
        Node(package='robot_localization', executable='ekf_node', name='ekf1', output='screen',
             parameters=[cfg('ekf1.yaml'), sim], remappings=[('odometry/filtered', '/ekf1/odometry')]),
        Node(package='particle_filter', executable='particle_filter', name='particle_filter', output='screen',
             additional_env=ENV, parameters=[cfg('pf.yaml'), {'initial_pose': init}, sim]),
        Node(package='state_estimation', executable='state_assembler', name='state_assembler', output='screen',
             additional_env=ENV, parameters=[cfg('assembler.yaml'), {'ekf_topic': est['assembler_ekf_topic']}, sim]),
        # map -> odom for viewers AND for obstacle avoidance (it looks up map -> laser through TF)
        Node(package='state_estimation', executable='viz_tf', name='viz_tf', output='screen', parameters=[sim]),
    ]
    if est['name'] == '1B' or a['shadow'].lower() == 'true':
        actions.append(Node(package='state_estimation', executable='shadow_ekf1', name='shadow_ekf1', output='screen',
                            additional_env=ENV, parameters=[cfg('shadow.yaml'), sim]))
    return actions


def generate_launch_description():
    args = [
        ('estimator', ESTIMATOR, 'race | 1A | 1B | 1C | 2A (default: ESTIMATOR in this file)'),
        ('initial_pose', '[0.0]', "[x, y, theta] of the LASER in map; '[0.0]' = global init"),
        ('shadow', 'true', '1A: also run the 1B shadow EKF for comparison (logged, not used)'),
        ('use_sim_time', 'false', 'true for bag replay'),
    ]
    return LaunchDescription([DeclareLaunchArgument(n, default_value=d, description=h) for n, d, h in args]
                             + [OpaqueFunction(function=_setup)])
