#!/usr/bin/env python3
"""
raceline_to_waypoints.py: turn the simulator's Spielberg raceline into Lab 7 waypoints.

Reads  f1tenth_gym_ros/maps/Spielberg_raceline.csv  (columns s; x; y; psi; kappa; vx; ax,
one row every 0.2 m) and writes  mpc/waypoints/spielberg.csv  with columns  x, y, yaw, v,
the same format make_waypoints.py writes.

What it does
  - drops the raceline's last row (a copy of the first, which would leave a 0 m gap)
  - fits a closed spline and resamples it every --ds metres (even spacing, as
    calc_interpolated_ref_trajectory in scripts/utils.py assumes)
  - yaw = the spline's tangent direction
  - v   = the raceline's own speed profile x --v-scale, clipped to [--v-min, --v-max]
    (the raceline is planned for up to 8 m/s, more than the MPC's MAX_SPEED)
  - checks the driving direction against the autograder's start pose and flags rows
    closer than --wall-margin to a wall

Direction: the raceline already runs the same way as the centerline (clockwise), which
is the direction the autograder drives Spielberg from (14.59, 3.92, -2.877).

How to run (from this folder, mpc/tools)
  python3 raceline_to_waypoints.py
  python3 raceline_to_waypoints.py --v-scale 0.7 --v-max 6.0
  Afterwards, run colcon build so the CSV is installed with the package.
"""
import argparse
import math
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

from make_waypoints import OUT_DIR, TRACKS, MapImage, default_maps_dir


def main():
    ap = argparse.ArgumentParser(
        description='Convert Spielberg_raceline.csv into waypoints/spielberg.csv (x, y, yaw, v).',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--ds', type=float, default=0.1, help='spacing between output rows [m]')
    ap.add_argument('--v-scale', type=float, default=0.5,
                    help="multiply the raceline's speeds (4.5-8 m/s) by this")
    ap.add_argument('--v-max', type=float, default=6.0, help='speed cap [m/s]')
    ap.add_argument('--v-min', type=float, default=1.5, help='speed floor [m/s]')
    ap.add_argument('--wall-margin', type=float, default=0.25,
                    help='flag rows closer than this to a wall [m]')
    ap.add_argument('--maps-dir', default=None, help='folder with Spielberg.yaml and the raceline '
                    '(if not given: f1tenth_gym_ros/maps next to this repo)')
    p = ap.parse_args()
    maps_dir = Path(p.maps_dir) if p.maps_dir else default_maps_dir()

    # s_m; x_m; y_m; psi_rad; kappa_radpm; vx_mps; ax_mps2
    raw = np.loadtxt(maps_dir / 'Spielberg_raceline.csv', delimiter=';', comments='#', ndmin=2)
    x, y, vx = raw[:, 1], raw[:, 2], raw[:, 5]
    if math.hypot(x[-1] - x[0], y[-1] - y[0]) < 1e-6:   # last row repeats the first
        x, y, vx = x[:-1], y[:-1], vx[:-1]

    # Closed spline through the raceline, parameterised by distance along it.
    closed = np.column_stack([np.append(x, x[0]), np.append(y, y[0])])
    knots = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(closed, axis=0).T))])
    spline = CubicSpline(knots, closed, bc_type='periodic')
    speed = CubicSpline(knots, np.append(vx, vx[0]), bc_type='periodic')

    length = knots[-1]
    n = int(round(length / p.ds))
    s = np.arange(n) * (length / n)             # last gap (back to row 0) matches too
    xs, ys = spline(s).T
    dx, dy = spline(s, 1).T
    yaw = np.arctan2(dy, dx)
    v = np.clip(speed(s) * p.v_scale, p.v_min, p.v_max)

    # Direction check against the autograder's start pose.
    sx, sy, sth = TRACKS['spielberg']['start']

    def heading_error():
        i = int(np.argmin(np.hypot(xs - sx, ys - sy)))
        return i, abs(math.atan2(math.sin(yaw[i] - sth), math.cos(yaw[i] - sth)))

    i, err = heading_error()
    if err >= math.pi / 2:                      # not expected, but be safe
        xs, ys, v = xs[::-1], ys[::-1], v[::-1]
        yaw = np.arctan2(-dy[::-1], -dx[::-1])
        print('Raceline ran against the start pose: reversed it.')
        i, err = heading_error()
    print(f'direction OK: path heading {math.degrees(yaw[i]):.1f} deg near the start pose '
          f'(start pose {math.degrees(sth):.1f} deg, {math.hypot(xs[i] - sx, ys[i] - sy):.2f} m away)')

    data = np.column_stack([xs, ys, yaw, v])
    header = ('x, y, yaw, v\n'
              f'tools/raceline_to_waypoints.py: Spielberg_raceline.csv, {n} rows every '
              f'{length / n:.3f} m, v = raceline vx x {p.v_scale}, clipped to [{p.v_min}, {p.v_max}]')
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / 'spielberg.csv'
    np.savetxt(out, data, fmt='%.4f', delimiter=', ', header=header)

    ds = length / n
    lap = float(np.sum(ds / v))
    print(f'wrote {n} rows to {out} ({length:.1f} m, v {v.min():.2f}-{v.max():.2f} m/s, '
          f'lap ~{lap:.1f} s at those speeds)')

    walls = MapImage(maps_dir / 'Spielberg.yaml', p.wall_margin)
    n_hit = int(walls.is_blocked(data[:, :2]).sum())
    if n_hit:
        print(f'  WARNING: {n_hit} rows are within {p.wall_margin} m of a wall')
    else:
        print(f'  no rows within {p.wall_margin} m of a wall')


if __name__ == '__main__':
    main()
