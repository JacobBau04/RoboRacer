"""Testing only (the autograder never runs this file): the graded Levine launch
plus a recorder that saves lap times, wall hits, tracking error and MPC solve
times to mpc/data/<date_time>/ when you press Ctrl+C.

    ros2 launch mpc levine_debug_launch.py
    ros2 launch mpc levine_debug_launch.py track:=levine_curv   # override WAYPOINTS for one run
    ros2 launch mpc levine_debug_launch.py note:="Qk yaw 13 -> 20"
    ros2 launch mpc levine_debug_launch.py keep_bag:=true   # also keep the rosbag
    ros2 launch mpc levine_debug_launch.py model:=kinematic # override MODEL for one run
    ros2 launch mpc levine_debug_launch.py delay_comp:=0.25 # delay compensation [s] for one run
    ros2 launch mpc levine_debug_launch.py platform:=sim    # record the sim's topics, not the car's

It starts the same node with the same parameters as levine_launch.py (read from
that file, so they stay in sync), plus publish_debug:=true (solve-time stats for
the recorder). You can change `track` (WAYPOINTS below), which picks the
waypoints CSV, and `model` (MODEL below), the MPC's prediction model.
The map comes from f1tenth_gym_ros/config/sim.yaml, not from here.
The recorder is tools/run_recorder.py. It records the collision flag and the lap
timer, which is why it must never go into levine_launch.py.
"""
import importlib.util
import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, SetParameter

GRADED = 'levine'   # use the node and parameters from launch/levine_launch.py

# Waypoints the MPC follows: waypoints/<WAYPOINTS>.csv (name without .csv).
# Change it here, or for one run with  track:=<name>  on the command line.
# Run colcon build after changing it: ros2 launch uses the installed copy.
WAYPOINTS = 'levine_real_mincurv'

# MPC prediction model: 'kinematic' (original, no slip) or 'slip' (the sim's dynamic
# single-track model, scripts/slip_model.py). Change it here, or for one run with
# model:=<name>. This overrides levine_launch.py, so graded runs are not affected.
MODEL = 'kinematic' # slip

# Delay compensation [s] (mpc_node.py predict_delay), 0 = off. The car measured ~0.26 s from
# /drive to the turn showing in the pose (summary.txt STEERING). Change here, or for one run
# with delay_comp:=<s>.
DELAY_COMP = '0.26'


def load_graded():
    """Import levine_launch.py so we can reuse its EXECUTABLE and PARAMETERS.

    We can't just include that file and override `track` with SetParameter: a
    parameter set on the node itself ({'track': 'levine'} there) beats a global one.
    """
    path = os.path.join(get_package_share_directory('mpc'), 'launch', f'{GRADED}_launch.py')
    spec = importlib.util.spec_from_file_location(f'{GRADED}_launch', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def find_recorder():
    """tools/ is not installed, so find it in the source tree.

    The installed copy of this file lives in install/mpc/share/mpc/launch, so
    look for the package source under <workspace>/src instead.
    """
    here = Path(__file__).resolve().parents[1] / 'tools' / 'run_recorder.py'
    if here.exists():                      # running from source / symlink install
        return here
    workspace = Path(get_package_share_directory('mpc')).parents[3]
    for depth in ('*', '*/*', '*/*/*'):
        for hit in (workspace / 'src').glob(f'{depth}/tools/run_recorder.py'):
            if (hit.parents[1] / 'package.xml').read_text().find('<name>mpc</name>') >= 0:
                return hit
    raise FileNotFoundError('could not find mpc/tools/run_recorder.py under '
                            f'{workspace / "src"}')


def generate_launch_description():
    graded = load_graded()
    track = LaunchConfiguration('track')
    # same parameters as the graded launch, except which waypoints to follow and the model
    params = dict(graded.PARAMETERS, track=track, model=LaunchConfiguration('model'),
                  delay_comp=LaunchConfiguration('delay_comp'))
    return LaunchDescription([
        DeclareLaunchArgument('track', default_value=WAYPOINTS,
                              description='waypoints/<track>.csv for the MPC to follow'),
        DeclareLaunchArgument('delay_comp', default_value=DELAY_COMP,
                              description='delay compensation in seconds, 0 = off'),
        DeclareLaunchArgument('model', default_value=MODEL,
                              description="MPC prediction model: 'kinematic' or 'slip'"),
        DeclareLaunchArgument('note', default_value='',
                              description='free-text note saved with the run'),
        DeclareLaunchArgument('keep_bag', default_value='false',
                              description='keep the raw rosbag in the run folder'),
        DeclareLaunchArgument('platform', default_value='real',
                              description="recorder topics: 'real' (car + particle filter) or 'sim'"),
        # applies to every node started after it, i.e. the MPC below
        SetParameter(name='publish_debug', value=True),
        Node(package='mpc', executable=graded.EXECUTABLE, output='screen',
             parameters=[params],
             # stops numpy's OpenBLAS from keeping idle threads spinning at 100% CPU
             additional_env={'OPENBLAS_NUM_THREADS': '1'}),
        ExecuteProcess(
            # 'python3' from PATH = your .venv when it is active (ros2 launch itself
            # runs on /usr/bin/python3, whose matplotlib is broken by NumPy 2)
            cmd=['python3', str(find_recorder()),
                 '--track', track,
                 ['--platform=', LaunchConfiguration('platform')],
                 # one argument "--note=<text>", so an empty note stays valid
                 ['--note=', LaunchConfiguration('note')],
                 ['--keep-bag=', LaunchConfiguration('keep_bag')]],
            output='screen',
            additional_env={'PYTHONUNBUFFERED': '1'},
            # after Ctrl+C give it time to write files and the plot before killing it
            sigterm_timeout='60',
        ),
    ])
