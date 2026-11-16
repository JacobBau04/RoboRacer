#!/usr/bin/env python3
"""Planner-only statistics (race1 default, --waypoints race2.csv --map race2_blocked.yaml for race2): random single boxes (5-30 % of the local width,
within 0.6 m of the raceline), car 5 m before the box. Counts plans per level, stops and
planning time. No detection, no driving (see offline_driveby.py for that).

    python3 tools/planner_stats.py --n 300
"""
import argparse
import collections
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, '..', 'scripts')]
from track_model import load_track                 # noqa: E402
from detour_planner import DetourPlanner          # noqa: E402
from obstacle_tracker import Obstacle             # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--n', type=int, default=300)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--ahead', type=float, default=5.0)
ap.add_argument('--waypoints', default='race1.csv', help='file in mpc/waypoints')
ap.add_argument('--map', default='race1_cleaned_blocked.yaml', help='file in mpc/maps')
a = ap.parse_args()
PKG = os.path.join(HERE, '..')
tr = load_track(os.path.join(PKG, 'waypoints', a.waypoints), os.path.join(PKG, 'maps', a.map))
rng = np.random.default_rng(a.seed)
stats, levels, ms = collections.Counter(), collections.Counter(), []
for k in range(a.n):
    s = rng.uniform(0, tr.length)
    lo, hi = tr.bounds_at(s)
    w = rng.uniform(0.05, 0.3) * (hi - lo)
    dc = rng.uniform(lo + w / 2, hi - w / 2)
    if not (dc - w / 2 < 0.6 and dc + w / 2 > -0.6):
        continue
    pl = DetourPlanner(tr)
    ob = Obstacle(1, s_min=float(tr.wrap_s(s - 0.15)), s_len=0.3, d_min=dc - w / 2,
                  d_max=dc + w / 2, confirmed=True)
    s_car = float(tr.wrap_s(s - a.ahead))
    t0 = time.perf_counter()
    p = pl.plan(s_car, float(tr.v_at(s_car)), [ob])
    ms.append((time.perf_counter() - t0) * 1e3)
    res = 'raceline' if p is None else ('stop' if p.stop else f'level {p.level}')
    stats[res] += 1
    if res == 'stop':
        print(f'  stop: s {s:.1f} w {w:.2f} dc {dc:+.2f} walls {lo:+.2f} {hi:+.2f}')
print(dict(stats), f'| planning ms mean {np.mean(ms):.1f} p95 {np.percentile(ms, 95):.1f} max {np.max(ms):.1f}')
