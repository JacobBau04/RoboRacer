#!/usr/bin/env python3
"""Plot the steering the MPC commanded during a recorded run, next to the steering
the reference line needs, and print what was happening at each wall hit.

Uses the files run_recorder.py writes (raw.npz, meta.json, waypoints.csv).

  python3 plot_steering.py                        # newest run in mpc/data
  python3 plot_steering.py ../data/2026-10-30_09-27-21
  python3 plot_steering.py --show                 # also open an interactive window

Writes <run>/steering.png. Lines on the plot:
  commanded   delta sent on /drive
  kinematic   delta = WB * kappa_ref: what a no-slip car needs to follow the line
              (the MPC's own model)
  + slip      delta = (WB + K v^2) * kappa_ref: what the sim's tyres need (understeer)
Sign convention: delta > 0 and kappa > 0 turn left; lateral error > 0 = left of the line.

QP cost panels (runs recorded after mpc_node.py started publishing cost_terms):
  every solve's objective split into Q (stage), Qf (terminal), R (input size) and
  Rd (input change) parts, so you can see which term the optimiser is trading
  against which.
    cost         each term's value per solve, all on one log scale, plus the total
    % of cost    each term as a % of that solve's total. Below the dark line =
                 tracking (Q, Qf), above = inputs (R, Rd). On straights the total
                 is tiny, so the shares there say little.
  Older runs skip these panels.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
if '--show' not in sys.argv and 'MPLBACKEND' not in os.environ:
    matplotlib.use('Agg')                       # just write the PNG
elif 'MPLBACKEND' not in os.environ:
    matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

from run_analysis import build_track, project

DATA_DIR = Path(__file__).resolve().parents[1] / 'data'
# Understeer gradient of the sim's f1tenth tyres: (1/(mu g)) (1/C_Sf - 1/C_Sr)
K_US = (1 / (1.0489 * 9.81)) * (1 / 4.718 - 1 / 5.4562)    # ~0.0028 rad per m/s^2

# /mpc/solve_stats columns in raw.npz['solve'] (column 0 = time); see mpc_node.cost_terms
COST_COLS = {'total': 5,
             'Q_x': 6, 'Q_y': 7, 'Q_v': 8, 'Q_yaw': 9,
             'Qf_x': 10, 'Qf_y': 11, 'Qf_v': 12, 'Qf_yaw': 13,
             'R_accel': 14, 'R_steer': 15, 'Rd_accel': 16, 'Rd_steer': 17}
# Plotted cost series: (label, columns summed, colour). Colours are the dataviz
# skill's validated categorical slots 1-7, one per series, in fixed order.
TRACKING = [('Q position (x+y)', ['Q_x', 'Q_y'], '#2a78d6'),
            ('Q heading', ['Q_yaw'], '#eb6834'),
            ('Q speed', ['Q_v'], '#1baf7a'),
            ('Qf terminal (all states)', ['Qf_x', 'Qf_y', 'Qf_v', 'Qf_yaw'], '#eda100')]
EFFORT = [('R steering', ['R_steer'], '#e87ba4'),
          ('Rd steering change', ['Rd_steer'], '#008300'),
          ('R + Rd acceleration', ['R_accel', 'Rd_accel'], '#6250d6')]


def load_costs(solve):
    """(t, {name: values}) for solves that report costs, or None for older runs."""
    if solve.ndim != 2 or solve.shape[1] <= max(COST_COLS.values()):
        return None
    ok = np.isfinite(solve[:, COST_COLS['total']])
    if not ok.any():
        return None
    return solve[ok, 0], {k: solve[ok, c] for k, c in COST_COLS.items()}


def series(costs, cols):
    return sum(costs[c] for c in cols)


def shares(costs):
    """% of each solve's total for every plotted series (rows follow TRACKING + EFFORT)."""
    parts = np.array([series(costs, cols) for _, cols, _ in TRACKING + EFFORT])
    total = parts.sum(axis=0)                  # = solver objective (checked in the summary)
    return 100 * parts / np.where(total > 0, total, np.nan), total


def hold(t_query, t_cmd, values):
    """Value of the latest command at each query time (commands are held until replaced)."""
    i = np.clip(np.searchsorted(t_cmd, t_query, side='right') - 1, 0, len(t_cmd) - 1)
    return values[i]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('run_dir', nargs='?', help='run folder (default: newest in mpc/data)')
    ap.add_argument('--show', action='store_true', help='open an interactive window too')
    ap.add_argument('--window', type=float, default=1.0,
                    help='seconds before each hit to summarise')
    a = ap.parse_args()

    if a.run_dir:
        run = Path(a.run_dir)
    else:
        runs = sorted(p for p in DATA_DIR.iterdir() if (p / 'raw.npz').is_file())
        run = runs[-1]
    raw = np.load(run / 'raw.npz')
    meta = json.loads((run / 'meta.json').read_text())
    cfg = (meta.get('mpc') or {}).get('mpc_config', {})
    wb, max_steer = cfg.get('WB', 0.33), cfg.get('MAX_STEER', 0.4189)

    odom, drive, coll = raw['odom'], raw['drive'], raw['collisions']
    cost = load_costs(raw['solve'])
    t, x, y = odom[:, 0], odom[:, 1], odom[:, 2]
    v = np.hypot(odom[:, 4], odom[:, 5])
    track = build_track(np.loadtxt(run / 'waypoints.csv', delimiter=',', ndmin=2))
    j, _, e = project(track, odom[:, 1:3])
    kappa = track['kappa'][j]                    # reference curvature where the car is
    corner = track['corner_wp'][j]

    delta = hold(t, drive[:, 0], drive[:, 2])    # commanded steering at each odom sample
    d_kin = wb * kappa
    d_slip = (wb + K_US * v ** 2) * kappa
    hits = [te for te, state in coll if state > 0.5]

    # ---- text summary
    print(f'run: {run.name}   WB={wb}  MAX_STEER={max_steer:.4f} rad')
    sat = np.mean(np.abs(delta) >= max_steer - 1e-3)
    print(f'steering at the limit {100 * sat:.1f}% of the time; '
          f'max |delta| {np.abs(delta).max():.3f} rad')
    c = corner & (np.abs(d_slip) > 0.02)
    if c.any():
        ratio = np.median(delta[c] / d_slip[c])
        print(f'in corners: commanded / needed-with-slip = {ratio:.2f} (median; <1 = steering too little)')
    for k, te in enumerate(hits, 1):
        i = int(np.clip(np.searchsorted(t, te), 0, len(t) - 1))
        w = (t >= te - a.window) & (t <= te)
        print(f'\nhit {k} at t={te:.2f} s  ({x[i]:.2f}, {y[i]:.2f})  v={v[i]:.2f} m/s  '
              f'lateral error {e[i]:+.3f} m')
        print(f'  delta now {delta[i]:+.3f}   kinematic needs {d_kin[i]:+.3f}   '
              f'with slip {d_slip[i]:+.3f}   limit +/-{max_steer:.3f}')
        print(f'  last {a.window:.1f} s: delta {delta[w].min():+.3f} .. {delta[w].max():+.3f}, '
              f'needed (slip) {d_slip[w].min():+.3f} .. {d_slip[w].max():+.3f}, '
              f'v {v[w].min():.2f} .. {v[w].max():.2f}')

    # ---- QP cost summary
    if cost is None:
        print('\nno QP cost data in this run (recorded before mpc_node published '
              'cost_terms): record a new run to get the cost panels')
    else:
        tc, c = cost
        in_corner = corner[np.clip(np.searchsorted(t, tc), 0, len(t) - 1)]
        parts = sum(c[k] for k in COST_COLS if k != 'total')
        mismatch = np.nanmax(np.abs(parts - c['total']) / np.maximum(np.abs(c['total']), 1e-9))
        pct, total = shares(c)
        print(f'\nQP cost: each term as % of its solve\'s total, averaged over solves '
              f'(parts add up to the solver total within {100 * mismatch:.2g}%)')
        print(f'  {"term":28s} {"corners":>8s} {"straights":>10s}')
        for (label, _, _), row in zip(TRACKING + EFFORT, pct):
            print(f'  {label:28s} {np.nanmean(row[in_corner]):7.1f}% '
                  f'{np.nanmean(row[~in_corner]):9.1f}%')
        print(f'  {"(median total cost)":28s} {np.median(total[in_corner]):8.3f} '
              f'{np.median(total[~in_corner]):10.3f}')

    # ---- figure
    rows = 5 if cost else 3
    fig = plt.figure(figsize=(14, 3 * rows))
    gs = fig.add_gridspec(rows, 2, width_ratios=[2.2, 1])
    ax_d = fig.add_subplot(gs[0, 0])
    ax_e = fig.add_subplot(gs[1, 0], sharex=ax_d)
    ax_v = fig.add_subplot(gs[2, 0], sharex=ax_d)
    ax_m = fig.add_subplot(gs[:, 1])
    time_axes = [ax_d, ax_e, ax_v]
    if cost:
        ax_k = fig.add_subplot(gs[3, 0], sharex=ax_d)     # cost values
        ax_c = fig.add_subplot(gs[4, 0], sharex=ax_d)     # % of total
        time_axes += [ax_k, ax_c]

    for ax in time_axes:                         # shade time spent in corners
        ax.fill_between(t, 0, 1, where=corner, color='0.92', transform=ax.get_xaxis_transform(),
                        step='mid', lw=0)
        for te in hits:
            ax.axvline(te, color='red', lw=1.2)
        ax.grid(alpha=0.3)

    ax_d.plot(t, delta, color='C0', lw=1.2, label='commanded')
    ax_d.plot(t, d_kin, color='0.4', lw=1, ls='--', label='kinematic need (WB·κ)')
    ax_d.plot(t, d_slip, color='C1', lw=1, ls=':', label='need with slip ((WB+Kv²)·κ)')
    for s in (1, -1):
        ax_d.axhline(s * max_steer, color='red', lw=0.8, ls='--')
    ax_d.set_ylabel('steering δ [rad]')
    ax_d.legend(loc='upper right', fontsize=8)
    ax_d.set_title(f'{run.name}: grey = corners, red lines = wall hits')

    ax_e.plot(t, e, color='C2', lw=1)
    ax_e.axhline(0, color='k', lw=0.6)
    ax_e.set_ylabel('lateral error [m]\n(+ = left of line)')

    ax_v.plot(t, v, color='C3', lw=1, label='v (odom)')
    ax_v.plot(drive[:, 0], drive[:, 1], color='C3', lw=0.6, alpha=0.4, label='v command')
    ax_v.set_ylabel('speed [m/s]')
    ax_v.set_xlabel('time [s]')
    ax_v.legend(loc='upper right', fontsize=8)

    if cost:
        tc, c = cost
        pct, total = shares(c)
        group = TRACKING + EFFORT
        floor = max(np.nanpercentile([series(c, g[1]) for g in group], 5), 1e-6)
        for label, cols, colour in group:
            ax_k.plot(tc, np.maximum(series(c, cols), floor), color=colour, lw=1.1, label=label)
        ax_k.plot(tc, total, color='0.1', lw=0.9, ls='--', label='total')
        ax_k.set_yscale('log')
        ax_k.set_ylim(floor, np.nanmax(total) * 1.5)
        ax_k.set_ylabel('QP cost per solve\n(log scale)')
        ax_k.legend(loc='lower left', bbox_to_anchor=(0, 1.0), ncol=4, fontsize=8, frameon=False)

        pct = np.nan_to_num(pct)
        ax_c.stackplot(tc, pct, colors=[g[2] for g in group], edgecolor='white', linewidth=0.4)
        # boundary between tracking (Q, Qf) and input (R, Rd) cost
        ax_c.plot(tc, pct[:len(TRACKING)].sum(axis=0), color='0.1', lw=1.0)
        ax_c.set_ylim(0, 100)
        ax_c.set_ylabel('% of QP cost\n(below line: tracking,\nabove: inputs)')
        ax_v.set_xlabel('')
        ax_c.set_xlabel('time [s]')

    wp = track['wp']
    ax_m.plot(np.r_[wp[:, 0], wp[0, 0]], np.r_[wp[:, 1], wp[0, 1]], color='0.6', lw=1)
    sc = ax_m.scatter(x, y, c=delta, s=3, cmap='coolwarm', vmin=-max_steer, vmax=max_steer)
    for te in hits:
        i = int(np.clip(np.searchsorted(t, te), 0, len(t) - 1))
        ax_m.plot(x[i], y[i], 'x', color='red', ms=12, mew=3)
    ax_m.set_aspect('equal')
    ax_m.set_title('path coloured by δ, x = wall hit')
    fig.colorbar(sc, ax=ax_m, fraction=0.04, label='δ [rad]')

    fig.tight_layout()
    out = run / 'steering.png'
    fig.savefig(out, dpi=110)
    print(f'\nplot: {out}')
    if a.show:
        plt.show()


if __name__ == '__main__':
    main()
