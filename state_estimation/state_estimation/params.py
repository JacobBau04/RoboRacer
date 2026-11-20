"""Parameter dataclasses shared by the nodes, the offline sim and the tests."""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class ModelParams:
    """Shadow-EKF vehicle model parameters (the `p` argument of student_ekf functions)."""
    L: float = 0.3302    # wheelbase [m] (sim: lf + lr)
    tau: float = 0.1     # yaw-rate response time constant [s]


@dataclass
class ShadowNoise:
    """Process noise per second (diagonal, for x, y, theta, v, omega) and the initial covariance."""
    q_diag: tuple = (1e-6, 1e-6, 1e-6, 0.5, 0.3)   # v, omega tuned on a sim bag with tools/shadow_replay.py (NIS ~1)
    sigma0_diag: tuple = (1e-4, 1e-4, 1e-4, 0.01, 0.01)

    def Q(self, scale=1.0):
        return np.diag(np.asarray(self.q_diag, dtype=float) * scale)

    def Sigma0(self):
        return np.diag(np.asarray(self.sigma0_diag, dtype=float))


@dataclass
class PrepParams:
    """sensor_prep conversion and noise settings (what the filters ASSUME)."""
    speed_scale: float = 1.0          # measured speed / true speed (calibration); prep divides by it
    sigma_v: float = 0.05             # speed noise std [m/s]
    low_speed: float = 0.5            # below this speed, variance is inflated
    low_speed_inflation: float = 4.0
    accel_threshold: float = 4.0      # |dv/dt| above this [m/s^2] -> variance inflated
    accel_inflation: float = 4.0
    gyro_scale: float = 1.0           # raw gyro units -> rad/s (car VESC driver publishes deg/s: pi/180)
    gyro_offset: float = 0.0          # stored gyro offset [rad/s], subtracted
    sigma_w: float = 0.02             # gyro noise std [rad/s]
    gyro_calib_seconds: float = 0.0   # >0: average the gyro while stationary for this long at startup
    servo_gain: float = -1.2135       # servo = gain * delta + offset (F1TENTH VESC defaults)
    servo_offset: float = 0.5304
    variance_floor: float = 1e-6      # R is never zero


@dataclass
class AssemblerParams:
    lidar_offset: float = 0.275       # LiDAR ahead of the rear axle [m]
    compensation: bool = True
    buffer_seconds: float = 1.0
    reference_point: str = 'rear_axle'   # or 'cog'
    lr: float = 0.17145               # rear axle -> CoG distance, used if reference_point == 'cog'
    wheelbase: float = 0.3302


@dataclass
class Counters:
    """Diagnostic counters for the shadow EKF wrapper."""
    dropped_old: int = 0
    substeps: int = 0
    resets: int = 0
    updates: int = 0
    predicts: int = 0
    stale_steering: int = 0
    extra: dict = field(default_factory=dict)
