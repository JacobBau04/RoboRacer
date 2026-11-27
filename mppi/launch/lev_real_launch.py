from launch import LaunchDescription
from launch.actions import TimerAction
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    mpc_wpts = os.path.join(get_package_share_directory('mpc'), 'waypoints')

    # Same map the particle filter localizes on (identical to particle_filter's levine_2nd)
    map_yaml = os.path.join(mpc_wpts, 'levine_2nd.yaml')
    # Raceline made on levine_2nd: x, y, yaw, v (2.1-2.3 m/s)
    csv_path = os.path.join(mpc_wpts, 'levine_race.csv')
    # The same raceline in the semicolon format (s; x; y; psi; kappa; vx; ...): the opponent
    # detector and predictor read only this format
    opp_csv_path = os.path.join(get_package_share_directory('mppi'), 'waypoints', 'levine_race_semicolon.csv')

    # Finds the opponent in /scan: drops points on /map walls (map from particle_filter's map_server),
    # clusters the rest near the raceline, publishes /opponents/detection_odom_{0,1,2}
    opponent_detector = Node(
        package='mppi',
        executable='opponent_lidar_detector_node_4_car',
        name='opponent_lidar_detector',
        output='screen',
        parameters=[{
            'scan_topic': '/scan',
            'ego_odom_topic': '/pf/pose/odom',   # car: particle filter pose (default is the sim's /ego_racecar/odom)
            'map_topic': '/map',
            'laser_x_offset': 0.0,               # the particle-filter pose is already the LiDAR
            'waypoint_path': opp_csv_path,
            'waypoint_path_absolute': True,
        }],
    )

    # Filters the detections along the raceline and publishes /opponents/predicted_path_{0,1,2} (2 s ahead),
    # which mppi_node uses for its opponent-avoidance cost
    opponent_predictor = Node(
        package='mppi',
        executable='opponent_predictor_node.py',
        name='opponent_predictor',
        output='screen',
        parameters=[{
            'waypoint_path': opp_csv_path,
            'waypoint_path_absolute': True,
        }],
    )

    mppi_controller = Node(
        package='mppi',
        executable='mppi_node.py',
        name='mppi_node',
        output='screen',
        parameters=[{
            'wall_map_yaml': map_yaml,
            'waypoint_path': csv_path,
            # same values as the MPC's 'race' estimator (particle filter on /pf/pose/odom)
            'pose_offset': 0.27,   # LiDAR is 0.27 m ahead of the rear axle (base_link -> laser TF)
            'prediction_start_delay': 0.14,  # actuation delay only; prediction is already current
            'delay_comp': 0.20,    # measured steering command -> turn in pose delay
        }],
        # node hard-codes the sim topic; feed it the particle filter pose
        remappings=[('/ego_racecar/odom', '/pf/pose/odom')],
    )

    return LaunchDescription([
        opponent_detector,
        opponent_predictor,
        TimerAction(period=3.0, actions=[mppi_controller]),
    ])