#!/usr/bin/env python3
"""Offline test of detection + memory + planner on race1 (no ROS, no MPC).

The car's pose point follows the active plan (or the raceline) exactly at the planned
speed; scans are synthetic (tools/scan_sim.py, SICK pattern, 15 Hz). Detection sees the
pose with a slowly drifting error, like the particle filter. For random boxes it reports
when they were first seen and confirmed, the true clearance between the car's outline and
the box, and the time lost against the raceline. Clean laps count false confirmations.

    python3 tools/offline_driveby.py --n 200 --v-scale 1.0 --noise-xy 0.03 --noise-yaw 0.01
"""
import argparse
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, '..', 'scripts'), HERE]
from track_model import load_track                          # noqa: E402
from obstacle_tracker import DetectionParams, ObstacleTracker, detect   # noqa: E402
from detour_planner import DetourPlanner, PlannerParams    # noqa: E402
from scan_sim import ScanSim, box_corners                  # noqa: E402

PKG = os.path.join(HERE, '..')


def polygon_distance(P, Q):
    """Smallest distance between two convex polygons (0 if they overlap)."""
    def inside(pt, poly):
        sgn = None
        for k in range(len(poly)):
            a, b = poly[k], poly[(k + 1) % len(poly)]
            cr = (b[0] - a[0]) * (pt[1] - a[1]) - (b[1] - a[1]) * (pt[0] - a[0])
            if sgn is None:
                sgn = cr >= 0
            elif (cr >= 0) != sgn:
                return False
        return True

    def seg_dist(p, a, b):
        ab = b - a
        t = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-12), 0, 1)
        return np.hypot(*(a + t * ab - p))

    if any(inside(p, Q) for p in P) or any(inside(q, P) for q in Q):
        return 0.0
    best = math.inf
    for X, Y in ((P, Q), (Q, P)):
        for p in X:
            for k in range(len(Y)):
                best = min(best, seg_dist(p, Y[k], Y[(k + 1) % len(Y)]))
    return best


def car_outline(xy, yaw, pp):
    c, s = math.cos(yaw), math.sin(yaw)
    pts = [(pp.front, pp.half_width), (-pp.rear, pp.half_width),
           (-pp.rear, -pp.half_width), (pp.front, -pp.half_width)]
    return np.array([(xy[0] + c * a - s * b, xy[1] + s * a + c * b) for a, b in pts])


class Drift:
    """AR(1) pose error: std sigma, correlation time tau [s]."""
    def __init__(self, sigma, tau, dt, rng):
        self.a = math.exp(-dt / tau)
        self.sigma, self.rng, self.x = sigma, rng, 0.0

    def step(self):
        self.x = self.a * self.x + math.sqrt(1 - self.a ** 2) * self.sigma * self.rng.normal()
        return self.x


def run_pass(tr, box, s_start, s_stop, args, rng, dp, pp):
    """Drive from s_start to s_stop with one box (or none). Returns a result dict."""
    sim = ScanSim(tr.occ, [box] if box is not None else [])
    tracker = ObstacleTracker(tr, dp)
    planner = DetourPlanner(tr, pp)
    dt = 1.0 / 15.0
    ex, ey, eyaw = (Drift(args.noise_xy, 1.0, dt, rng), Drift(args.noise_xy, 1.0, dt, rng),
                    Drift(args.noise_yaw, 1.0, dt, rng))
    s, v = s_start, float(tr.v_at(s_start)) * args.v_scale
    t_total, min_clear, first_u, conf_u = 0.0, math.inf, None, None
    stopped, planned, n_conf_max, plan_ms = False, False, 0, []
    plan = None
    travelled, distance = 0.0, s_stop - s_start
    while travelled < distance and t_total < 30.0:
        if plan is not None:
            d = float(np.interp(0.0, plan.u, plan.d))
            v_cmd = float(np.interp(0.0, plan.u, plan.v))
        else:
            d, v_cmd = 0.0, float(tr.v_at(s)) * args.v_scale
        xy = tr.frenet_to_xy(s, d)
        s2 = s + 0.05
        d2 = float(np.interp(0.05, plan.u, plan.d)) if plan is not None else 0.0
        xy2 = tr.frenet_to_xy(s2, d2)
        yaw = math.atan2(xy2[1] - xy[1], xy2[0] - xy[0])
        if box is not None:
            min_clear = min(min_clear, polygon_distance(car_outline(xy, yaw, pp), box))
        r = sim.scan(xy[0], xy[1], yaw, noise=0.01, rng=rng)
        nx, ny, nyaw = xy[0] + ex.step(), xy[1] + ey.step(), yaw + eyaw.step()
        s_est = float(tr.project(np.array([[nx, ny]]), s_hint=s, window=(-2, 2))[0][0])
        clusters, _ = detect(tr, r, sim.angle_min, sim.angle_inc, nx, ny, nyaw, s_est, dp)
        conf = tracker.update(clusters, s_est)
        n_conf_max = max(n_conf_max, len(conf))
        if clusters and first_u is None and box is not None:
            first_u = min(c.u_min for c in clusters)
        if conf and conf_u is None:
            conf_u = float(tr.rel_s(conf[0].s_min, s_est))
        t0 = time.perf_counter()
        plan = planner.plan(s_est, v, conf)
        plan_ms.append((time.perf_counter() - t0) * 1e3)
        if getattr(args, 'verbose', False):
            obs_txt = ' '.join(f'[u {float(tr.rel_s(o.s_min, s_est)):5.2f}+{o.s_len:.2f} d {o.d_min:+.2f}..{o.d_max:+.2f}]'
                               for o in conf)
            ptxt = 'raceline' if plan is None else ('STOP' if plan.stop else f'L{plan.level} {plan.key}')
            print(f'  t {t_total:5.2f} s {s:6.2f} v {v:4.2f} | {obs_txt} | {ptxt}')
        if plan is not None:
            planned = True
            stopped |= plan.stop
            # plan is relative to the estimated s: re-express around the true s
            plan.u = plan.u + float(tr.rel_s(s_est, s))
        # advance with the speed the plan asks for (perfect speed tracking, accel limits apply
        # through the plan's speed profile)
        v = max(v_cmd, 0.0) if plan is not None else v_cmd
        if v < 0.05 and plan is not None and plan.stop:
            break
        ds = max(v, 0.0) * dt
        s = float(tr.wrap_s(s + ds))
        travelled += ds
        t_total += dt
    return dict(time=t_total, min_clear=min_clear, first_u=first_u, conf_u=conf_u,
                stopped=stopped, planned=planned, n_conf_max=n_conf_max,
                plan_ms=float(np.mean(plan_ms)) if plan_ms else 0.0,
                plan_ms_max=float(np.max(plan_ms)) if plan_ms else 0.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--n', type=int, default=100, help='random boxes')
    ap.add_argument('--v-scale', type=float, default=1.0, help='raceline speed multiplier')
    ap.add_argument('--noise-xy', type=float, default=0.03)
    ap.add_argument('--noise-yaw', type=float, default=0.01)
    ap.add_argument('--clean-laps', type=int, default=2)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--case', default='', help='one box: "s,width,dc[,depth]" (prints every scan)')
    ap.add_argument('--waypoints', default=os.path.join(PKG, 'waypoints', 'race1.csv'))
    ap.add_argument('--map', default=os.path.join(PKG, 'maps', 'race1_cleaned_blocked.yaml'))
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    tr = load_track(args.waypoints, args.map)
    dp, pp = DetectionParams(), PlannerParams()
    pp.v_max = max(pp.v_max, 4.0 * args.v_scale)

    if args.case:
        vals = [float(x) for x in args.case.split(',')]
        s_obs, w, dc = vals[:3]
        depth = vals[3] if len(vals) > 3 else 0.3
        c = tr.frenet_to_xy(s_obs, dc)
        box = box_corners(c[0], c[1], float(tr.yaw_at(s_obs)), depth, w)
        args.verbose = True
        res = run_pass(tr, box, s_obs - 12.0, s_obs + 4.0, args, rng, dp, pp)
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items()})
        return
    fp = 0
    for lap in range(args.clean_laps):
        res = run_pass(tr, None, 0.0, tr.length - 0.2, args, rng, dp, pp)
        fp += res['n_conf_max'] > 0
        print(f'clean lap {lap}: {res["time"]:.2f} s, confirmed obstacles {res["n_conf_max"]}, '
              f'planned {res["planned"]}, planner {res["plan_ms"]:.1f} ms')

    rows = []
    for k in range(args.n):
        s_obs = rng.uniform(0, tr.length)
        lo, hi = tr.bounds_at(s_obs)
        width = hi - lo
        w = rng.uniform(0.05, 0.30) * width
        depth = rng.uniform(0.15, 0.40)
        dc = rng.uniform(lo + 0.5 * w, hi - 0.5 * w)
        c = tr.frenet_to_xy(s_obs, dc)
        box = box_corners(c[0], c[1], float(tr.yaw_at(s_obs)), depth, w)
        if tr.occ.wall_distance(*box.T).min() < 0.02:
            continue                       # box drawn into a wall
        relevant = (dc - 0.5 * w) < 0.6 and (dc + 0.5 * w) > -0.6
        if not relevant:
            continue
        base = run_pass(tr, None, s_obs - 12.0, s_obs + 4.0, args, rng, dp, pp)
        res = run_pass(tr, box, s_obs - 12.0, s_obs + 4.0, args, rng, dp, pp)
        res.update(s=s_obs, w=w, dc=dc, lost=res['time'] - base['time'],
                   gap_raceline=max(dc - 0.5 * w, -(dc + 0.5 * w), 0.0))
        rows.append(res)
        flag = 'HIT' if res['min_clear'] <= 0 else ('STOP' if res['stopped'] else '')
        print(f'{len(rows):3d} s {s_obs:5.1f} w {w:.2f} dc {dc:+.2f}: first {res["first_u"]}, '
              f'conf {res["conf_u"] and round(res["conf_u"], 2)}, clear {res["min_clear"]:.2f}, '
              f'lost {res["lost"]:+.2f} s {flag}')
    if not rows:
        return
    clear = np.array([r['min_clear'] for r in rows])
    lost = np.array([r['lost'] for r in rows])
    conf = np.array([r['conf_u'] if r['conf_u'] is not None else np.nan for r in rows])
    print(f'\n{len(rows)} relevant boxes: hits {np.sum(clear <= 0)}, '
          f'stops {sum(r["stopped"] for r in rows)}, clean laps with false obstacles {fp}/{args.clean_laps}')
    print(f'clearance min {clear.min():.3f} p5 {np.percentile(clear, 5):.3f} median {np.median(clear):.3f}')
    print(f'confirmed at u: median {np.nanmedian(conf):.2f} m, p5 {np.nanpercentile(conf, 5):.2f} m, '
          f'never {np.isnan(conf).sum()}')
    print(f'time lost per box: mean {lost.mean():+.3f} s, p95 {np.percentile(lost, 95):+.3f} s')
    print(f'planner time mean {np.mean([r["plan_ms"] for r in rows]):.1f} ms, '
          f'max {np.max([r["plan_ms_max"] for r in rows]):.1f} ms')


if __name__ == '__main__':
    main()
