import math

import pytest

from state_estimation.pose_buffer import PoseBuffer


def test_empty():
    b = PoseBuffer()
    assert b.lookup(0.0) == (None, 'empty')
    assert b.lookup_extra(0.0) is None


def test_rejects_out_of_order_and_duplicates():
    b = PoseBuffer()
    assert b.add(1.0, (0, 0, 0))
    assert not b.add(1.0, (1, 0, 0))
    assert not b.add(0.5, (1, 0, 0))
    assert len(b) == 1


def test_horizon_trims_oldest():
    b = PoseBuffer(horizon=1.0)
    for i in range(30):
        b.add(i * 0.1, (i, 0, 0))
    newest = b.newest()[0]
    # the oldest entry is at or just before newest - horizon, and nothing older is kept
    assert b.oldest()[0] <= newest - 1.0 < b.oldest()[0] + 0.1 + 1e-9
    assert b.lookup(newest - 1.0)[1] == 'ok'


def test_lookup_interpolates_and_reports_status():
    b = PoseBuffer(horizon=5.0)
    b.add(0.0, (0.0, 0.0, 0.0))
    b.add(1.0, (2.0, 4.0, 1.0))
    b.add(2.0, (4.0, 4.0, 1.0))
    p, s = b.lookup(0.25)
    assert s == 'ok' and p == pytest.approx((0.5, 1.0, 0.25))
    p, s = b.lookup(1.5)
    assert s == 'ok' and p == pytest.approx((3.0, 4.0, 1.0))
    assert b.lookup(1.0) == ((2.0, 4.0, 1.0), 'ok')
    assert b.lookup(-1.0) == ((0.0, 0.0, 0.0), 'older')
    assert b.lookup(3.0) == ((4.0, 4.0, 1.0), 'newer')


def test_lookup_across_angle_seam():
    b = PoseBuffer()
    b.add(0.0, (0, 0, math.pi - 0.05))
    b.add(1.0, (0, 0, -math.pi + 0.05))
    p, _ = b.lookup(0.5)
    assert abs(abs(p[2]) - math.pi) < 1e-9


def test_lookup_extra_zero_order_hold():
    b = PoseBuffer()
    b.add(0.0, (0, 0, 0), 'a')
    b.add(1.0, (0, 0, 0), 'b')
    assert b.lookup_extra(-0.1) is None
    assert b.lookup_extra(0.0) == 'a'
    assert b.lookup_extra(0.99) == 'a'
    assert b.lookup_extra(5.0) == 'b'
