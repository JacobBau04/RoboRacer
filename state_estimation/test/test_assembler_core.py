"""Latency compensation on a synthetic trajectory (plan section 12, layer 1)."""
import math

import numpy as np
import pytest

from state_estimation import se2
from state_estimation.assembler_core import Assembler

D = 0.275


def circle_truth(t, v=5.0, R=5.0):
    w = v / R
    th = w * t
    return (R * math.sin(th), R * (1 - math.cos(th)), se2.wrap(th))


def run(compensation, delay=0.030, odom_rate=50.0, pf_rate=40.0, T=4.0):
    T_map_odom_true = (3.0, -2.0, 0.7)          # odom frame is offset/rotated from map
    a = Assembler(lidar_offset=D, compensation=compensation)
    errs = []
    t_odom = np.arange(0.0, T, 1.0 / odom_rate)
    t_scan = list(np.arange(0.0, T, 1.0 / pf_rate))
    pending = []
    for t in t_odom:
        while t_scan and t_scan[0] <= t:          # PF result for scan t_s arrives at t_s + delay
            ts = t_scan.pop(0)
            pending.append((ts + delay, ts))
        odom_pose = se2.compose(se2.inverse(T_map_odom_true), circle_truth(t))
        a.add_odom(t, odom_pose)
        for arr, ts in [p for p in pending if p[0] <= t]:
            a.add_pf(ts, se2.rear_axle_to_laser(circle_truth(ts), D))
            pending.remove((arr, ts))
        est = a.estimate(odom_pose)
        if est is not None and t > 0.5:
            tr = circle_truth(t)
            errs.append(math.hypot(est[0] - tr[0], est[1] - tr[1]))
    return np.array(errs)


def test_compensation_under_1mm_with_30ms_delay():
    e = run(True)
    assert e.max() < 1e-3


def test_without_compensation_error_is_speed_times_latency():
    e = run(False)
    # 5 m/s x (30 ms + up to one scan period) -> clearly > 10 cm on average
    assert e.mean() > 0.1


def test_no_output_before_first_pf():
    a = Assembler()
    a.add_odom(0.0, (0, 0, 0))
    assert a.estimate((0, 0, 0)) is None


def test_lookup_status_counts():
    a = Assembler()
    assert a.add_pf(1.0, (0, 0, 0)) == 'empty'
    a.add_odom(1.0, (0, 0, 0))
    a.add_odom(1.1, (0.1, 0, 0))
    assert a.add_pf(0.5, (0, 0, 0)) == 'older'
    assert a.add_pf(2.0, (0, 0, 0)) == 'newer'
    assert a.counts['pf_older_than_buffer'] == 1 and a.counts['pf_newer_than_odom'] == 1


def test_relocalisation_jump_moves_estimate():
    a = Assembler(lidar_offset=0.0)
    a.add_odom(0.0, (0, 0, 0))
    a.add_odom(1.0, (1, 0, 0))
    a.add_pf(1.0, (1, 0, 0))
    assert a.estimate((1, 0, 0)) == pytest.approx((1, 0, 0))
    a.add_pf(1.0, (1, 0.5, 0))                    # PF corrects by 0.5 m sideways
    assert a.estimate((1, 0, 0)) == pytest.approx((1, 0.5, 0))


def test_integrate_twist_arc_and_straight():
    p = se2.integrate_twist((0.0, 0.0, 0.0), 2.0, 0.0, 0.5)
    assert p == pytest.approx((1.0, 0.0, 0.0))
    # quarter circle of radius 1: (1, 1, pi/2)
    q = se2.integrate_twist((0.0, 0.0, 0.0), 1.0, 1.0, math.pi / 2)
    assert q == pytest.approx((1.0, 1.0, math.pi / 2))


def test_predict_forward_matches_truth_on_circle():
    a = Assembler(lidar_offset=D)
    a.add_odom(0.0, circle_truth(0.0))
    a.add_odom(1.0, circle_truth(1.0))
    a.add_pf(1.0, se2.rear_axle_to_laser(circle_truth(1.0), D))
    est = a.estimate(circle_truth(1.0), v=5.0, w=1.0, dt_forward=0.02)
    tr = circle_truth(1.02)
    assert math.hypot(est[0] - tr[0], est[1] - tr[1]) < 1e-9


def test_correction_tau_zero_is_jump_and_positive_is_gradual():
    for tau, expect_full in ((0.0, True), (0.1, False)):
        a = Assembler(lidar_offset=0.0, correction_tau=tau)
        a.add_odom(0.0, (0.0, 0.0, 0.0))
        a.add_odom(1.0, (1.0, 0.0, 0.0))
        a.add_pf(0.9, (0.9, 0.0, 0.0))
        a.add_pf(0.925, (0.925, 0.1, 0.0))          # PF says 10 cm to the left
        y = a.estimate((0.925, 0.0, 0.0))[1]
        if expect_full:
            assert y == pytest.approx(0.1)
        else:
            assert y == pytest.approx(0.1 * (1 - math.exp(-0.025 / 0.1)), rel=1e-6)


def test_correction_blend_rotates_about_the_car_not_the_odom_origin():
    a = Assembler(lidar_offset=0.0, correction_tau=0.1)
    a.add_odom(0.0, (20.0, 0.0, 0.0))
    a.add_odom(1.0, (21.0, 0.0, 0.0))
    a.add_pf(0.5, (20.5, 0.0, 0.0))
    a.add_pf(0.525, (20.525, 0.0, 0.02))         # heading correction only
    x, y, th = a.estimate((20.525, 0.0, 0.0))
    assert abs(y) < 1e-9 and x == pytest.approx(20.525)   # no position jump 20 m from the origin
    assert 0.0 < th < 0.02


def test_compensation_still_under_1mm_with_smoothing_when_pf_is_exact():
    # with exact PF poses the blended correction equals the true one, so smoothing costs nothing
    T_true = (3.0, -2.0, 0.7)
    a = Assembler(lidar_offset=D, correction_tau=0.2)
    errs = []
    for k in range(200):
        t = k * 0.02
        odom = se2.compose(se2.inverse(T_true), circle_truth(t))
        a.add_odom(t, odom)
        if k % 2 == 0 and t >= 0.04:
            ts = t - 0.03
            a.add_pf(ts, se2.rear_axle_to_laser(circle_truth(ts), D))
        est = a.estimate(odom)
        if est is not None and t > 0.5:
            tr = circle_truth(t)
            errs.append(math.hypot(est[0] - tr[0], est[1] - tr[1]))
    assert max(errs) < 1e-3
