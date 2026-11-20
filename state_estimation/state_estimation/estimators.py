"""Estimator table shared by state_estimation's estimation_launch.py and mpc's race launch files.

The default estimator is ESTIMATOR at the top of launch/estimation_launch.py (the one line to change). Each estimator's
settings are in config/estimators/<name>.yaml: whether it can run on the car, and the MPC settings that match it
(pose_topic, pose_offset, delay_comp).
"""
import importlib.util
import os

import yaml
from ament_index_python.packages import get_package_share_directory

NAMES = ('race', '1A', '1B', '1C', '2A')


def _share():
    return get_package_share_directory('state_estimation')


def default_estimator():
    """ESTIMATOR from the installed estimation_launch.py."""
    path = os.path.join(_share(), 'launch', 'estimation_launch.py')
    spec = importlib.util.spec_from_file_location('estimation_launch', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ESTIMATOR


def load(name):
    """Settings of one estimator; raises with a clear message if it can't run on the car."""
    if name not in NAMES:
        raise RuntimeError(f'unknown estimator {name!r}; choose one of {", ".join(NAMES)}')
    with open(os.path.join(_share(), 'config', 'estimators', f'{name}.yaml')) as f:
        cfg = yaml.safe_load(f)
    if not cfg.get('available', False):
        raise RuntimeError(f'estimator {name!r} is not available on the car: {cfg.get("reason", "")}')
    cfg['name'] = name
    return cfg
