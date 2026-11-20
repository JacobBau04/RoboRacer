import math

import numpy as np
import pytest

from state_estimation import se2

rng = np.random.default_rng(0)


def rand_pose():
    return (rng.uniform(-10, 10), rng.uniform(-10, 10), rng.uniform(-math.pi, math.pi))


def close(a, b, tol=1e-9):
    return abs(a[0] - b[0]) < tol and abs(a[1] - b[1]) < tol and abs(se2.wrap(a[2] - b[2])) < tol


def test_wrap_range_and_seam():
    assert se2.wrap(math.pi + 0.1) == pytest.approx(-math.pi + 0.1)
    assert se2.wrap(-math.pi) == pytest.approx(-math.pi)
    assert se2.wrap(math.pi) == pytest.approx(-math.pi)      # [-pi, pi)
    a = rng.uniform(-100, 100, 1000)
    w = se2.wrap(a)
    assert np.all(w >= -math.pi) and np.all(w < math.pi)
    assert np.allclose(np.sin(w), np.sin(a)) and np.allclose(np.cos(w), np.cos(a))


def test_compose_inverse_round_trip():
    for _ in range(200):
        a, b = rand_pose(), rand_pose()
        assert close(se2.compose(a, se2.inverse(a)), (0, 0, 0))
        assert close(se2.compose(se2.inverse(a), a), (0, 0, 0))
        assert close(se2.compose(a, se2.between(a, b)), b)


def test_compose_associative():
    for _ in range(100):
        a, b, c = rand_pose(), rand_pose(), rand_pose()
        assert close(se2.compose(se2.compose(a, b), c), se2.compose(a, se2.compose(b, c)))


def test_compose_known_value():
    # A frame at (1, 2) rotated 90 deg; a point 1 m ahead in it is at (1, 3).
    p = se2.compose((1.0, 2.0, math.pi / 2), (1.0, 0.0, 0.0))
    assert close(p, (1.0, 3.0, math.pi / 2))


def test_interpolate_endpoints_and_seam():
    p0, p1 = (0.0, 0.0, math.pi - 0.1), (1.0, 2.0, -math.pi + 0.1)
    assert close(se2.interpolate(p0, p1, 0.0), p0)
    assert close(se2.interpolate(p0, p1, 1.0), p1)
    mid = se2.interpolate(p0, p1, 0.5)
    # shortest way across the seam goes through pi, not through 0
    assert abs(abs(mid[2]) - math.pi) < 1e-9
    assert mid[0] == pytest.approx(0.5) and mid[1] == pytest.approx(1.0)


def test_laser_rear_axle_round_trip_and_value():
    d = 0.275
    for _ in range(100):
        p = rand_pose()
        assert close(se2.rear_axle_to_laser(se2.laser_to_rear_axle(p, d), d), p)
    # heading +y: the rear axle is d behind the laser in -y
    assert close(se2.laser_to_rear_axle((0.0, 0.0, math.pi / 2), d), (0.0, -d, math.pi / 2))


def test_cog_velocity_to_rear_axle():
    # Pure rotation about the rear axle: the CoG (lr ahead) moves sideways at lr*w,
    # the rear axle does not move sideways.
    lr, w = 0.17145, 2.0
    vx, vy = se2.cog_velocity_to_rear_axle(3.0, lr * w, w, lr)
    assert vx == 3.0 and vy == pytest.approx(0.0)


def test_quaternion_round_trip():
    class Q:
        pass

    for th in np.linspace(-math.pi, math.pi, 37)[:-1]:
        q = Q()
        q.x, q.y, q.z, q.w = se2.quat_from_yaw(th)
        assert se2.wrap(se2.yaw_from_quat(q) - th) == pytest.approx(0.0, abs=1e-12)
