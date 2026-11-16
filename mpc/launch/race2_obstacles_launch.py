"""Real car on race2 with obstacle avoidance and the run recorder.

Same as race1_obstacles_launch.py (see there for all arguments) with the race2 raceline
(waypoints/race2.csv) and walls (maps/race2_blocked.yaml). Start the car stack and the
particle filter (localize.yaml map: race2_remapped_edited) first, then:
    ros2 launch mpc race2_obstacles_launch.py
    ros2 launch mpc race2_obstacles_launch.py avoidance:=false note:="clean laps"
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    base = os.path.join(get_package_share_directory('mpc'), 'launch', 'race1_obstacles_launch.py')
    return LaunchDescription([IncludeLaunchDescription(
        PythonLaunchDescriptionSource(base),
        launch_arguments={'track': 'race2', 'map_yaml': 'race2_blocked.yaml'}.items())])
