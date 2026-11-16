"""race2 in the simulator with obstacle avoidance.

Same as race1_sim_launch.py (see there for all arguments) with the race2 raceline and
walls. Start the simulator with a race2 config first (ROS_DOMAIN_ID 42), then:
    ros2 launch mpc race2_sim_launch.py
    ros2 launch mpc race2_sim_launch.py avoidance:=false record:=true
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    base = os.path.join(get_package_share_directory('mpc'), 'launch', 'race1_sim_launch.py')
    return LaunchDescription([IncludeLaunchDescription(
        PythonLaunchDescriptionSource(base),
        launch_arguments={'track': 'race2', 'map_yaml': 'race2_blocked.yaml'}.items())])
