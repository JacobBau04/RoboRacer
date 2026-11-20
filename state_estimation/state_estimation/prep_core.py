"""sensor_prep logic (no ROS): raw sensor values -> filter measurements with variances (R).

This is the ONE place where assumed sensor noise lives. robot_localization and the shadow EKF read
R from the message covariances that sensor_prep writes.
"""
import math

from .params import PrepParams


class SpeedPrep:
    """Wheel speed -> (v, var_v). Variance is sigma_v^2 * r_scale, inflated at low speed and during
    hard acceleration/braking (a = smoothed dv/dt over `accel_tau` seconds)."""

    def __init__(self, p: PrepParams, r_scale=1.0, accel_tau=0.1):
        self.p = p
        self.r_scale = float(r_scale)
        self.accel_tau = float(accel_tau)
        self._t = None
        self._v = None
        self.accel = 0.0

    def __call__(self, t, v_raw):
        v = v_raw / self.p.speed_scale
        if self._t is not None and t > self._t:
            dt = t - self._t
            a_inst = (v - self._v) / dt
            k = 1.0 - math.exp(-dt / self.accel_tau)
            self.accel += k * (a_inst - self.accel)
        self._t, self._v = t, v
        var = self.p.sigma_v ** 2 * self.r_scale
        if abs(v) < self.p.low_speed:
            var *= self.p.low_speed_inflation
        if abs(self.accel) > self.p.accel_threshold:
            var *= self.p.accel_inflation
        return v, max(var, self.p.variance_floor)


class GyroPrep:
    """Gyro z -> (w, var_w). Optional start-up calibration: while the car is stationary for the first
    `gyro_calib_seconds`, readings are averaged into the offset and None is returned (nothing is
    published until calibration is done).

    "Stationary" uses a SMOOTHED speed (time constant speed_tau) against a threshold well above the
    speed noise: single noisy readings at rest (sigma_v = 0.05 m/s) would otherwise end the
    calibration after a few samples. Each sample is also gated on its own speed reading, so the
    samples taken while the smoothed speed is still rising after the car drives off are not averaged."""

    def __init__(self, p: PrepParams, r_scale=1.0, stationary_speed=0.3, speed_tau=0.2, sample_gate=0.25):
        self.p = p
        self.r_scale = float(r_scale)
        self.offset = float(p.gyro_offset)
        self.calibrating = p.gyro_calib_seconds > 0.0
        self.stationary_speed = stationary_speed
        self.speed_tau = speed_tau
        self.sample_gate = sample_gate     # a sample whose own speed reading exceeds this is never averaged
        self._speed_f = 0.0
        self._t_last = None
        self._t0 = None
        self._sum = 0.0
        self._n = 0

    def __call__(self, t, w_raw, speed=0.0):
        w_raw = w_raw * self.p.gyro_scale   # to rad/s (offset and calibration are in rad/s)
        if self.calibrating:
            if self._t0 is None:
                self._t0 = t
            if self._t_last is not None and t > self._t_last:
                k = 1.0 - math.exp(-(t - self._t_last) / self.speed_tau)
                self._speed_f += k * (abs(speed) - self._speed_f)
            self._t_last = t
            moving = self._speed_f >= self.stationary_speed
            if not moving and abs(speed) < self.sample_gate:
                self._sum += w_raw
                self._n += 1
            if t - self._t0 >= self.p.gyro_calib_seconds or moving:
                if self._n > 0:
                    self.offset = self._sum / self._n
                self.calibrating = False
            else:
                return None
        var = self.p.sigma_w ** 2 * self.r_scale
        return w_raw - self.offset, max(var, self.p.variance_floor)


def steering_from_servo(servo, p: PrepParams):
    """Inverse of the VESC servo mapping: delta = (servo - offset) / gain [rad]."""
    return (servo - p.servo_offset) / p.servo_gain
