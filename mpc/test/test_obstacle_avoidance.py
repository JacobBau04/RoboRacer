"""Unit tests for obstacle avoidance on race1 (no ROS needed).

    cd mpc && python3 -m pytest test/test_obstacle_avoidance.py -q
"""
import math
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(HERE, '..')
sys.path[:0] = [os.path.join(PKG, 'scripts'), os.path.join(PKG, 'tools')]

import avoidance_msgs as am                                         # noqa: E402
from detour_planner import DetourPlanner, PlannerParams             # noqa: E402
from obstacle_tracker import (Cluster, DetectionParams, Obstacle,  # noqa: E402
                              ObstacleTracker, ScanView, detect)
from scan_sim import ScanSim, box_corners                           # noqa: E402
from track_model import load_track                                  # noqa: E402


@pytest.fixture(scope='module')
def tr():
    return load_track(os.path.join(PKG, 'waypoints', 'race1.csv'),
                      os.path.join(PKG, 'maps', 'race1_cleaned_blocked.yaml'))


def box_obstacle(tr, s_mid, dc, w, depth=0.3, oid=1):
    return Obstacle(oid, s_min=float(tr.wrap_s(s_mid - depth / 2)), s_len=depth,
                    d_min=dc - w / 2, d_max=dc + w / 2, confirmed=True, hits=2)


# ---- track ---------------------------------------------------------------------------
def test_track_geometry(tr):
    assert 24.0 < tr.length < 26.5
    wall = tr.occ.wall_distance(tr.xy[:, 0], tr.xy[:, 1])
    assert wall.min() > 0.3                       # raceline keeps away from the walls
    assert np.all(tr.d_left > 0.3) and np.all(tr.d_right > 0.3)


def test_track_roundtrip(tr):
    rng = np.random.default_rng(1)
    s = rng.uniform(0, tr.length, 300)
    d = rng.uniform(-0.4, 0.4, 300)
    s2, d2, ok = tr.project(tr.frenet_to_xy(s, d))
    assert ok.all()
    assert np.abs(tr.rel_s(s2, s)).max() < 0.03
    assert np.abs(d2 - d).max() < 0.01


# ---- detection -----------------------------------------------------------------------
def scan_at(tr, s_car, boxes):
    sim = ScanSim(tr.occ, boxes)
    x, y = tr.frenet_to_xy(s_car, 0.0)
    yaw = float(tr.yaw_at(s_car))
    return sim, sim.scan(x, y, yaw), (x, y, yaw)


def test_detects_box_ahead(tr):
    s_car, s_obs = 3.0, 7.0
    c = tr.frenet_to_xy(s_obs, 0.1)
    box = box_corners(c[0], c[1], float(tr.yaw_at(s_obs)), 0.3, 0.3)
    sim, r, (x, y, yaw) = scan_at(tr, s_car, [box])
    cl, _ = detect(tr, r, sim.angle_min, sim.angle_inc, x, y, yaw, s_car, DetectionParams())
    assert len(cl) == 1
    assert abs(cl[0].u_min - 3.85) < 0.1
    assert cl[0].d_min < 0.0 < cl[0].d_max


def test_hidden_edge_flagged(tr):
    # far box partly hidden on its left (+d) side by a nearer box
    s_car = 3.0
    near = tr.frenet_to_xy(5.0, 0.30)
    far = tr.frenet_to_xy(8.0, 0.0)
    boxes = [box_corners(near[0], near[1], float(tr.yaw_at(5.0)), 0.2, 0.3),
             box_corners(far[0], far[1], float(tr.yaw_at(8.0)), 0.3, 0.6)]
    sim, r, (x, y, yaw) = scan_at(tr, s_car, boxes)
    cl, _ = detect(tr, r, sim.angle_min, sim.angle_inc, x, y, yaw, s_car, DetectionParams())
    far_cl = [c for c in cl if c.u_min > 4.0]
    # which side is hidden depends on the track's bend; exactly one edge must be
    assert far_cl and (far_cl[0].hidden_hi != far_cl[0].hidden_lo)
    near_cl = [c for c in cl if c.u_min < 3.0]
    assert near_cl and not near_cl[0].hidden_lo and not near_cl[0].hidden_hi


def test_no_false_points_on_clean_lap(tr):
    p = DetectionParams()
    rng = np.random.default_rng(2)
    for s_car in np.arange(0.0, tr.length, 0.7):
        sim, r, (x, y, yaw) = scan_at(tr, s_car, [])
        # particle-filter-like pose error
        cl, _ = detect(tr, r + rng.normal(0, 0.01, len(r)), sim.angle_min, sim.angle_inc,
                       x + rng.normal(0, 0.03), y + rng.normal(0, 0.03), yaw + rng.normal(0, 0.01),
                       s_car, p)
        assert cl == [], f'false obstacle at s={s_car:.1f}: {cl}'


def test_zero_ranges_ignored(tr):
    sim, r, (x, y, yaw) = scan_at(tr, 5.0, [])
    r = r.copy()
    r[::3] = 0.0                                  # SICK: 0 = no echo
    cl, _ = detect(tr, r, sim.angle_min, sim.angle_inc, x, y, yaw, 5.0, DetectionParams())
    assert cl == []


# ---- memory --------------------------------------------------------------------------
def cluster(u0, u1, d0, d1):
    return Cluster(u0, u1, d0, d1, 5, 3.0, np.zeros((5, 2)))


def test_confirm_after_two_scans_and_remember(tr):
    tk = ObstacleTracker(tr, DetectionParams())
    assert tk.update([cluster(4.0, 4.3, -0.1, 0.1)], 0.0) == []
    conf = tk.update([cluster(3.8, 4.1, -0.1, 0.12)], 0.2)
    assert len(conf) == 1
    for k in range(5):                            # not seen: still remembered
        assert len(tk.update([], 0.4 + 0.2 * k)) == 1
    assert tk.update([], 5.5) == []               # passed


def test_tentative_dropped_if_not_seen_again(tr):
    tk = ObstacleTracker(tr, DetectionParams())
    tk.update([cluster(4.0, 4.3, -0.1, 0.1)], 0.0)
    tk.update([], 0.2)
    assert tk.obstacles == []


def test_see_through_deletes(tr):
    tk = ObstacleTracker(tr, DetectionParams())
    tk.update([cluster(4.0, 4.3, -0.1, 0.1)], 0.0)
    tk.update([cluster(4.0, 4.3, -0.1, 0.1)], 0.0)
    sim, r, (x, y, yaw) = scan_at(tr, 0.0, [])    # empty track: beams pass the box's spot
    view = ScanView(np.array([x, y]), yaw, r, sim.angle_min, sim.angle_inc)
    for _ in range(3):
        assert len(tk.update([], 0.0, view)) == 1
    assert tk.update([], 0.0, view) == []


# ---- planner -------------------------------------------------------------------------
def check_plan(tr, plan, obstacles, pp):
    assert plan is not None and not plan.stop
    ahead = plan.u >= 0
    for ob in obstacles:
        assert plan.margins[ob.id] >= (pp.margin_min if plan.level == 0 else pp.crawl_margin) - 2e-3
    assert abs(plan.d[-1]) < 1e-3                 # back on the raceline
    _, _, kappa = DetourPlanner._geometry(plan.xy)
    assert np.abs(kappa[ahead][1:-1]).max() <= pp.crawl_kappa + 1e-6
    assert np.all(plan.v >= 0)


def test_detour_around_box_on_raceline(tr):
    pp = PlannerParams()
    pl = DetourPlanner(tr, pp)
    ob = box_obstacle(tr, 6.0, 0.0, 0.35)
    plan = pl.plan(1.0, 3.0, [ob])
    check_plan(tr, plan, [ob], pp)
    assert plan.level == 0
    assert np.abs(plan.d).max() > 0.3


def test_no_detour_for_box_off_the_line(tr):
    pl = DetourPlanner(tr)
    lo, hi = tr.bounds_at(6.0)
    ob = box_obstacle(tr, 6.0, hi - 0.15, 0.2)    # against the left wall
    assert pl.plan(1.0, 3.0, [ob]) is None


def test_stop_when_blocked(tr):
    pl = DetourPlanner(tr)
    lo, hi = tr.bounds_at(6.0)
    ob = box_obstacle(tr, 6.0, 0.5 * (lo + hi), (hi - lo) - 0.05)   # wall to wall
    plan = pl.plan(1.0, 3.0, [ob])
    assert plan is not None and plan.stop
    u_ob = float(tr.rel_s(ob.s_min, 1.0))
    assert np.all(plan.v[plan.u >= u_ob - pl.p.front - 0.05] == 0.0)


def test_keeps_side_when_box_grows(tr):
    pl = DetourPlanner(tr)
    ob = box_obstacle(tr, 6.0, 0.0, 0.3)
    p1 = pl.plan(1.0, 3.0, [ob])
    side = dict(p1.key[:-1])[1]
    ob2 = box_obstacle(tr, 6.0, 0.0 - 0.02 * side, 0.34)        # grows a bit toward the car's side
    p2 = pl.plan(1.4, 3.0, [ob2])
    assert dict(p2.key[:-1])[1] == side


def test_two_obstacles(tr):
    pp = PlannerParams()
    pl = DetourPlanner(tr, pp)
    obs = [box_obstacle(tr, 4.5, 0.25, 0.3, oid=1), box_obstacle(tr, 6.5, -0.25, 0.3, oid=2)]
    plan = pl.plan(0.5, 3.0, obs)
    check_plan(tr, plan, obs, pp)


# ---- messages and splice -------------------------------------------------------------
def test_plan_message_and_splice(tr):
    pl = DetourPlanner(tr)
    for s_car in (1.0, tr.length - 3.0):           # also across the lap start
        ob = box_obstacle(tr, s_car + 5.0, 0.0, 0.35)
        plan = pl.plan(s_car, 3.0, [ob])
        i0, i1, pts = am.resample_plan(tr, plan)
        msg = am.decode_plan(am.encode_plan(7, i0, i1, pts, plan.stop))
        assert msg['plan_id'] == 7 and msg['i_start'] == i0 and np.allclose(msg['pts'], pts)
        race = np.column_stack((tr.xy, tr.yaw, tr.v))
        wp = am.splice(race, i0, i1, pts)
        step = np.hypot(*np.diff(wp[:, :2], axis=0).T)
        assert np.all(step < 0.2) and np.all(step > 0.02)   # continuous, no duplicates
        assert np.allclose(wp[0, :2], race[(i1 + 1) % tr.n, :2])   # first segment = raceline
        pl.reset()
    assert am.decode_plan(am.encode_plan(3))['pts'] is None


def test_obstacles_message(tr):
    obs = [box_obstacle(tr, 6.0, 0.0, 0.3, oid=4)]
    boxes, pts = am.decode_obstacles(am.encode_obstacles(tr, obs))
    assert boxes.shape == (1, 5) and boxes[0, 0] == 4
    s, d, _ = tr.project(pts)
    assert np.all(np.abs(d) <= 0.151) and np.all(np.abs(tr.rel_s(s, 6.0)) <= 0.16)
