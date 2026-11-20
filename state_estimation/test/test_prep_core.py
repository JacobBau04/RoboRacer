import math

import numpy as np

import pytest

from state_estimation.params import PrepParams
from state_estimation.prep_core import GyroPrep, SpeedPrep, steering_from_servo


def test_speed_scale_and_variance():
    p = PrepParams(speed_scale=1.03, sigma_v=0.05)
    sp = SpeedPrep(p)
    v, var = sp(0.0, 5.15)
    assert v == pytest.approx(5.0) and var == pytest.approx(0.0025)


def test_speed_low_speed_inflation():
    p = PrepParams(sigma_v=0.05)
    v, var = SpeedPrep(p)(0.0, 0.2)
    assert var == pytest.approx(0.0025 * p.low_speed_inflation)


def test_speed_accel_inflation_and_r_scale():
    p = PrepParams(sigma_v=0.05)
    sp = SpeedPrep(p, r_scale=2.0)
    for k in range(50):            # 10 m/s^2 ramp for 1 s
        v, var = sp(k * 0.02, 1.0 + 10.0 * k * 0.02)
    assert sp.accel == pytest.approx(10.0, rel=0.05)
    assert var == pytest.approx(0.0025 * 2.0 * p.accel_inflation)


def test_variance_floor():
    p = PrepParams(sigma_v=0.0, sigma_w=0.0)
    assert SpeedPrep(p)(0.0, 3.0)[1] == p.variance_floor
    assert GyroPrep(p)(0.0, 0.1)[1] == p.variance_floor


def test_gyro_calibration_removes_offset_then_publishes():
    p = PrepParams(gyro_calib_seconds=2.0, sigma_w=0.02)
    g = GyroPrep(p)
    outs = [g(k * 0.01, 0.013, speed=0.0) for k in range(250)]
    assert all(o is None for o in outs[:200])
    assert outs[-1][0] == pytest.approx(0.0, abs=1e-12)
    assert g.offset == pytest.approx(0.013)


def test_gyro_calibration_stops_when_car_moves():
    g = GyroPrep(PrepParams(gyro_calib_seconds=5.0))
    assert g(0.0, 0.01, speed=0.0) is None
    out = None
    k = 1
    while out is None:                         # car drives off at 2 m/s: calibration ends quickly
        out = g(0.01 * k, 0.51, speed=2.0)
        k += 1
    assert k < 30 and g.offset == pytest.approx(0.01, abs=0.05)


def test_gyro_calibration_ignores_speed_noise_at_rest():
    """Regression: noisy speed readings at rest (sigma 0.05 m/s) must not end the calibration early."""
    rng = np.random.default_rng(3)
    g = GyroPrep(PrepParams(gyro_calib_seconds=2.0))
    for k in range(250):
        out = g(k * 0.01, 0.01 + 0.02 * rng.standard_normal(), speed=0.05 * rng.standard_normal())
        if k < 199:
            assert out is None
    assert g.offset == pytest.approx(0.01, abs=0.005)


def test_servo_round_trip():
    p = PrepParams()
    for d in (-0.4, 0.0, 0.25):
        assert steering_from_servo(p.servo_gain * d + p.servo_offset, p) == pytest.approx(d)


def test_gyro_scale_converts_deg_to_rad():
    import math
    from state_estimation.params import PrepParams
    from state_estimation.prep_core import GyroPrep
    g = GyroPrep(PrepParams(gyro_scale=math.pi / 180.0, gyro_offset=0.01))
    w, _ = g(0.0, 180.0)
    assert abs(w - (math.pi - 0.01)) < 1e-12
    assert GyroPrep(PrepParams())(0.0, 1.5)[0] == 1.5   # default scale 1: sim behaviour unchanged
