#!/usr/bin/env python3
"""Figures for TUNING_LOG.md (real car, race2, 2026-11-04), built from recorded runs.

    python3 ~/f1tenth_ws/src/jacob_opt_obstacles/mpc/tools/make_log_figures.py

Writes PNGs to jacob_opt_obstacles/figures/ and prints the numbers the log quotes.
Reads only: mpc/data/<run>/ (raw.npz, waypoints.csv, meta.json), the particle-filter
map, and ~/steering_tests/<folder>/results.csv.

Tracking numbers use one rule for every run: the rear axle (pose moved 0.27 m back from
the LiDAR) against the raceline, only while the MPC drives above 1.0 m/s, at least 2 s
after it took over, and after the car first came within 15 cm of the line (so starting
parked off the line, or rejoining after a joystick takeover, doesn't count).
Wall clearance is the closest any corner of the car body (rear axle -0.12 .. +0.46 m,
+-0.155 m) came to a wall on the particle-filter map.
"""
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from PIL import Image  # noqa: E402
from scipy.ndimage import distance_transform_edt  # noqa: E402

PKG = Path(__file__).resolve().parents[1]           # .../jacob_opt_obstacles/mpc
DATA = PKG / 'data'
FIG = PKG.parent / 'figures'
STEER_TESTS = Path.home() / 'steering_tests'
LIDAR_AHEAD = 0.27
BODY = [(f, w) for f in (-0.12, 0.17, 0.46) for w in (-0.155, 0.155)]

# reference palette (dataviz skill, light mode); first three slots pass all-pairs CVD checks
SURFACE, INK, INK2, MUTED, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#8a8984', '#e6e5e1'
BLUE, ORANGE, AQUA, CRITICAL = '#2a78d6', '#eb6834', '#1baf7a', '#d03b3b'

plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'axes.edgecolor': MUTED, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
    'text.color': INK, 'axes.titlecolor': INK, 'axes.titlesize': 12, 'axes.titleweight': 'bold',
    'axes.labelsize': 10, 'font.size': 10, 'axes.grid': True, 'grid.color': GRID,
    'grid.linewidth': 0.8, 'axes.spines.top': False, 'axes.spines.right': False,
    'legend.frameon': False, 'lines.linewidth': 2.0,
})

# the tracking runs, in the order the fixes went in
RUNS = [
    ('2026-11-04_06-30-03', 'Start: servo gain -1.0, delay 0.26 s'),
    ('2026-11-04_06-46-16', 'Servo gain -1.0 -> -0.8'),
    ('2026-11-04_07-01-18', '+ track rear axle, delay 0.20 s'),
    ('2026-11-04_07-15-55', '+ steering-change cost Rdk 100 -> 30'),
    ('2026-11-04_07-26-25', '+ steering cost Rk 20 -> 5'),
    ('2026-11-04_07-34-49', '+ raceline grip mu 0.4 -> 0.3'),
    ('2026-11-04_07-46-18', '+ MPC time step 0.1 -> 0.05 s'),
    ('2026-11-04_08-29-15', '+ trim 0.469, wheelbase 0.324 m'),
]


# ----------------------------------------------------------------------------- helpers
class Run:
    def __init__(self, name):
        self.name = name
        self.dir = DATA / name
        d = np.load(self.dir / 'raw.npz', allow_pickle=True)
        self.raw = d
        self.meta = json.load(open(self.dir / 'meta.json'))
        self.wp = np.loadtxt(self.dir / 'waypoints.csv', delimiter=',', comments='#')
        o, c = d['odom'], d['cmd']
        self.t, self.x, self.y, self.yaw = o[:, 0], o[:, 1], o[:, 2], o[:, 3]
        self.v = np.hypot(o[:, 4], o[:, 5])
        self.xr = self.x - LIDAR_AHEAD * np.cos(self.yaw)
        self.yr = self.y - LIDAR_AHEAD * np.sin(self.yaw)
        k = np.searchsorted(c[:, 0], self.t).clip(0, len(c) - 1)
        self.mpc = c[k, 3] > 0.5
        # time since the MPC (re)took control
        since = np.zeros(len(self.t))
        t_on = None
        for i in range(len(self.t)):
            if self.mpc[i] and (i == 0 or not self.mpc[i - 1]):
                t_on = self.t[i]
            since[i] = self.t[i] - t_on if (self.mpc[i] and t_on is not None) else -1
        self.e, self.j = project(self.wp, self.xr, self.yr)
        # after each takeover, ignore the car until it has first come within 15 cm of the
        # line (a run often starts parked off the line; rejoining isn't tracking)
        joined = np.zeros(len(self.t), bool)
        on = False
        for i in range(len(self.t)):
            if not self.mpc[i]:
                on = False
            elif not on and since[i] >= 0 and abs(self.e[i]) < 0.15 and self.v[i] > 1.0:
                on = True
            joined[i] = on
        self.use = self.mpc & (self.v > 1.0) & (since >= 2.0) & joined


def project(wp, px, py):
    xy = wp[:, :2]
    n = len(xy)
    seg = np.roll(xy, -1, 0) - xy
    L = np.linalg.norm(seg, axis=1)
    E, J = np.zeros(len(px)), np.zeros(len(px), int)
    for i, p in enumerate(np.c_[px, py]):
        j = int(np.linalg.norm(xy - p, axis=1).argmin())
        a, b = xy[j], xy[(j + 1) % n]
        E[i] = np.cross(b - a, p - a) / L[j]           # + = left of the line
        J[i] = j
    return E, J


_MAPS = {}


def load_map(yaml_path):
    if yaml_path not in _MAPS:
        m = yaml.safe_load(open(yaml_path))
        img = np.array(Image.open(Path(yaml_path).with_suffix('.pgm')))
        dist = distance_transform_edt(img > 250) * m['resolution']
        _MAPS[yaml_path] = (img, m['resolution'], m['origin'][0], m['origin'][1], dist)
    return _MAPS[yaml_path]


def clearance(run):
    img, res, ox, oy, dist = load_map(run.meta['map']['yaml'])
    H, W = img.shape
    out = np.full(len(run.t), np.inf)
    for f, w in BODY:
        px = run.xr + f * np.cos(run.yaw) - w * np.sin(run.yaw)
        py = run.yr + f * np.sin(run.yaw) + w * np.cos(run.yaw)
        c = ((px - ox) / res).astype(int).clip(0, W - 1)
        r = (H - 1 - ((py - oy) / res).astype(int)).clip(0, H - 1)
        out = np.minimum(out, dist[r, c])
    return out


def curvature(wp):
    L = np.linalg.norm(np.roll(wp[:, :2], -1, 0) - wp[:, :2], axis=1)
    k = np.gradient(np.unwrap(wp[:, 2])) / L
    return np.convolve(np.r_[k[-7:], k, k[:7]], np.ones(7) / 7, 'same')[7:-7]


def draw_map(ax, run, pad=1.2):
    img, res, ox, oy, _ = load_map(run.meta['map']['yaml'])
    H, W = img.shape
    shown = np.where(img > 250, 1.0, np.where(img < 50, 0.0, 0.85))
    ax.imshow(shown, cmap='gray', vmin=0, vmax=1, origin='upper',
              extent=[ox, ox + W * res, oy, oy + H * res], interpolation='nearest')
    wp = run.wp
    ax.set_xlim(wp[:, 0].min() - pad, wp[:, 0].max() + pad)
    ax.set_ylim(wp[:, 1].min() - pad, wp[:, 1].max() + pad)
    ax.set_aspect('equal')
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def save(fig, name):
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / name, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote figures/{name}')


# ----------------------------------------------------------------------------- figures
def fig_tracking_progress(stats):
    labels = [lab for _, lab in RUNS]
    mean = [stats[r]['mean'] * 100 for r, _ in RUNS]
    clear = [stats[r]['clear'] * 100 for r, _ in RUNS]
    y = np.arange(len(RUNS))[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True,
                             gridspec_kw={'width_ratios': [1, 1], 'wspace': 0.08})
    for ax, vals, title, unit in [(axes[0], mean, 'Average distance from the raceline', 'cm'),
                                  (axes[1], clear, 'Closest the car body came to a wall', 'cm')]:
        ax.barh(y, vals, height=0.62, color=BLUE, edgecolor=SURFACE, linewidth=2)
        for yi, v in zip(y, vals):
            txt = 'contact' if (title.startswith('Closest') and v < 2) else f'{v:.1f} {unit}'
            ax.text(v + max(vals) * 0.02, yi, txt, va='center', color=INK2, fontsize=9)
        ax.set_title(title, loc='left')
        ax.set_xlim(0, max(vals) * 1.25)
        ax.grid(axis='y', visible=False)
        ax.set_xlabel(unit)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels)
    fig.suptitle('race2, real car: each fix added on top of the previous ones (2026-11-04)',
                 x=0.0, ha='left', fontsize=10, color=INK2, y=1.0)
    save(fig, 'r_tracking_progress.png')


def fig_paths(first, last):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    for ax, run, title in [(axes[0], first, f'Before ({first.name[11:]})'),
                           (axes[1], last, f'After all fixes ({last.name[11:]})')]:
        draw_map(ax, run)
        ax.plot(run.wp[:, 0], run.wp[:, 1], '--', color=INK2, lw=1.4, label='raceline')
        m = run.mpc & (run.v > 0.3)
        xr = np.where(m, run.xr, np.nan)
        ax.plot(xr, np.where(m, run.yr, np.nan), color=ORANGE if run is first else BLUE, lw=1.6,
                label='car (rear axle, MPC driving)')
        if run is first:
            cl = clearance(run)
            hit = m & (cl < 0.02)
            # one marker per contact
            idx = np.flatnonzero(hit)
            keep = [i for n, i in enumerate(idx) if n == 0 or run.t[i] - run.t[idx[n - 1]] > 2.0]
            ax.scatter(run.xr[keep], run.yr[keep], s=70, marker='X', color=CRITICAL,
                       edgecolor=SURFACE, linewidth=1.5, zorder=5, label='wall contact')
        ax.set_title(title, loc='left')
    handles, labels = [], []
    for ax in axes:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            if lab not in labels:
                handles.append(h)
                labels.append(lab)
    labels = [lab.replace('car (rear axle, MPC driving)', 'car, rear axle (orange: before, blue: after)')
              for lab in labels]
    fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=9, bbox_to_anchor=(0.5, 0.04))
    fig.subplots_adjust(bottom=0.08)
    save(fig, 'r_paths_before_after.png')


def fig_entry_exit(before, after):
    turns = [(76, 147, 'C1 hairpin\n(left)'), (154, 184, 'S-bend\n(right)'),
             (222, 263, 'C2 hairpin\n(left)')]
    groups, vb, va = [], [], []
    for run, out in [(before, vb), (after, va)]:
        kap = curvature(run.wp)
        for a, b, name in turns:
            s = np.sign(kap[(a + b) // 2])
            t3 = (b - a) / 3
            ent = run.use & (run.j >= a) & (run.j < a + t3)
            ex = run.use & (run.j >= a + 2 * t3) & (run.j < b + 12)
            out += [100 * s * run.e[ent].mean(), 100 * s * run.e[ex].mean()]
    for _, _, name in turns:
        groups += [f'{name}\nentry', f'{name}\nexit']
    x = np.arange(len(groups))
    w = 0.38
    fig, ax = plt.subplots(figsize=(10, 4.4))
    ax.bar(x - w / 2, vb, w, color=ORANGE, edgecolor=SURFACE, linewidth=2,
           label=f'{before.name[11:]}: Rdk 30, Rk 20, MPC step 0.1 s, mu 0.4')
    ax.bar(x + w / 2, va, w, color=BLUE, edgecolor=SURFACE, linewidth=2,
           label=f'{after.name[11:]}: Rk 5, MPC step 0.05 s, mu 0.3, trim fixed')
    ax.axhline(0, color=INK2, lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([g.replace('\n', ' ', 1) for g in groups], fontsize=8.5)
    ax.set_ylabel('sideways offset from raceline [cm]\n+ inside the turn,  - wide')
    ax.set_title('Turn-in: early (inside) at entry, wide at exit', loc='left')
    ax.grid(axis='x', visible=False)
    ax.legend(loc='lower left', fontsize=9)
    save(fig, 'r_turn_entry_exit.png')
    return groups, vb, va


def fig_steering():
    rows = []
    for folder in sorted(STEER_TESTS.glob('*_circle')):
        f = folder / 'results.csv'
        if not f.exists():
            continue
        import csv
        for r in csv.DictReader(open(f)):
            q, cmd = r['quality'], float(r['steer_cmd'])
            # 08-12-02 +0.3875: flagged only for an early speed bump; three methods agree
            ok = q == 'OK' or (folder.name.startswith('2026-11-04_08-12-02') and abs(cmd) > 0.38)
            if ok:
                rows.append((cmd, float(r['effective_rad']), folder.name[11:19]))
    if not rows:
        print('  no circle-test results found, skipping steering figure')
        return []
    rows.sort()
    c = np.array([r[0] for r in rows])
    e = np.array([r[1] for r in rows])
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    lim = 0.45
    ax.plot([-lim, lim], [-lim, lim], '--', color=INK2, lw=1.2)
    ax.text(0.04, 0.10, 'wheels = command', color=INK2, fontsize=9, rotation=45)
    ax.scatter(c, e, s=60, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=3)
    # selective labels: the gentlest command and the largest each way
    for ci, ei in zip(c, e):
        if ci in (c.min(), c.max(), 0.2):
            ax.annotate(f'{ei / ci:.2f}x', (ci, ei), textcoords='offset points',
                        xytext=(8, -12) if ci > 0 else (-6, -18), color=INK2, fontsize=9)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect('equal')
    ax.set_xlabel('commanded steering [rad]   (left +)')
    ax.set_ylabel('real wheel angle from the driven circle [rad]')
    ax.set_title('Slow-circle steering test, 0.5 m/s', loc='left')
    save(fig, 'r_steering_circle_test.png')
    return rows


def fig_accel(run_name='2026-11-04_08-45-12'):
    run = Run(run_name)
    gain = abs(run.meta['car']['vesc']['speed_to_erpm_gain'])
    vs, cmd = run.raw['vesc'], run.raw['cmd']
    t, v = vs[:, 0], np.abs(vs[:, 1]) / gain
    tg = np.arange(t[0] + 0.2, t[-1] - 0.2, 0.02)
    vg = np.interp(tg, t, v)
    k = 5
    a = (vg[2 * k:] - vg[:-2 * k]) / (2 * k * 0.02)
    tm, vm = tg[k:-k], vg[k:-k]
    vc = np.interp(tm, cmd[:, 0], cmd[:, 1])
    mpc = np.interp(tm, cmd[:, 0], cmd[:, 3]) > 0.5
    err = vc - vm
    ok = mpc & (vm > 0.5)
    edges = np.array([-0.8, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.8])
    mid, med, lo, hi, n = [], [], [], [], []
    for e0, e1 in zip(edges[:-1], edges[1:]):
        m = ok & (err >= e0) & (err < e1)
        if m.sum() < 10:
            continue
        mid.append((e0 + e1) / 2)
        med.append(np.median(a[m]))
        lo.append(np.percentile(a[m], 25))
        hi.append(np.percentile(a[m], 75))
        n.append(int(m.sum()))
    mid, med, lo, hi = map(np.array, (mid, med, lo, hi))
    ramp = 14000 / gain
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.axhline(ramp, color=CRITICAL, lw=1.4, ls='--')
    ax.axhline(-ramp, color=CRITICAL, lw=1.4, ls='--')
    ax.text(-0.78, ramp + 0.15, f'VESC speed ramp 14000 ERPM/s = {ramp:.1f} m/s²', color=INK2, fontsize=9)
    ax.text(-0.78, -ramp - 0.35, 'the same ramp also limits braking', color=INK2, fontsize=9)
    ax.axhline(0, color=INK2, lw=1)
    ax.vlines(mid, lo, hi, color=BLUE, lw=2)
    ax.plot(mid, med, 'o-', color=BLUE, ms=8, mec=SURFACE, mew=2)
    ax.set_xlabel('commanded speed minus actual speed [m/s]')
    ax.set_ylabel('measured acceleration [m/s²]\n(wheel speed, median and middle 50 %)')
    ax.set_title('Asking for more does not give more: the ceiling is in the VESC', loc='left')
    ax.set_ylim(-4.2, 4.6)
    save(fig, 'r_accel_ceiling.png')
    return list(zip(mid, med, n)), ramp


def fig_avoidance(run_name='2026-11-04_09-47-03', lap_index=5):
    run = Run(run_name)
    d = run.raw
    s_line = np.r_[0, np.cumsum(np.linalg.norm(np.diff(run.wp[:, :2], axis=0), axis=1))]
    L = s_line[-1] + np.linalg.norm(run.wp[0, :2] - run.wp[-1, :2])
    s = s_line[run.j]
    ds = np.diff(s)
    ds[ds < -L / 2] += L
    ds[ds > L / 2] -= L
    lapnum = np.floor((s[0] + np.r_[0, np.cumsum(ds)]) / L).astype(int)
    laps = sorted(set(lapnum[run.mpc]))
    use_lap = laps[min(lap_index, len(laps) - 1)]
    m = (lapnum == use_lap)
    t0, t1 = run.t[m].min(), run.t[m].max()
    boxes, box_pts = d['boxes'], d['box_pts']
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    draw_map(ax, run, pad=0.9)
    ax.plot(run.wp[:, 0], run.wp[:, 1], '--', color=INK2, lw=1.3, label='raceline')
    # last obstacle message of each box id seen during this lap
    inlap = (boxes[:, 0] >= t0) & (boxes[:, 0] <= t1)
    last_k = {}
    for row in boxes[inlap]:
        last_k[int(row[2])] = int(row[1])
    for k in sorted(set(last_k.values())):
        p = box_pts[box_pts[:, 0] == k]
        ax.scatter(p[:, 1], p[:, 2], s=4, color=CRITICAL, zorder=4)
    ax.scatter([], [], s=25, color=CRITICAL, label='obstacle boxes (as detected)')
    ax.plot(np.where(m, run.xr, np.nan), np.where(m, run.yr, np.nan), color=BLUE, lw=2,
            label='car (rear axle)')
    ax.set_title(f'One lap with obstacles, run {run_name[11:]} (lap {use_lap})', loc='left')
    ax.legend(loc='upper left', fontsize=9)
    save(fig, 'r_avoidance_lap.png')
    return len(last_k), t1 - t0


# ----------------------------------------------------------------------------- main
def main():
    print('Tracking runs (rear axle, MPC > 1 m/s, joined the line):')
    stats, runs = {}, {}
    for name, lab in RUNS:
        r = Run(name)
        runs[name] = r
        cl = clearance(r)
        e = np.abs(r.e[r.use])
        stats[name] = {'mean': e.mean(), 'p95': np.percentile(e, 95), 'max': e.max(),
                       'clear': cl[r.use].min(), 'n': int(r.use.sum())}
        st = stats[name]
        print(f'  {name}  mean {st["mean"]*100:5.1f} cm  p95 {st["p95"]*100:5.1f}  max {st["max"]*100:5.1f}'
              f'  closest wall {st["clear"]*100:5.1f} cm  ({st["n"]} samples)  {lab}')
    fig_tracking_progress(stats)
    fig_paths(runs[RUNS[0][0]], runs[RUNS[-1][0]])
    groups, vb, va = fig_entry_exit(runs['2026-11-04_07-15-55'], runs['2026-11-04_08-29-15'])
    print('Entry/exit (cm, + inside):')
    for g, b, a in zip(groups, vb, va):
        print(f'  {g.replace(chr(10), " "):28s} {b:+5.1f} -> {a:+5.1f}')
    rows = fig_steering()
    print('Steering circle test (cmd -> effective):', [(c, round(e, 3)) for c, e, _ in rows])
    acc, ramp = fig_accel()
    print(f'Acceleration vs speed error (ramp {ramp:.2f} m/s2):', [(round(a, 2), round(b, 2), n) for a, b, n in acc])
    nbox, dur = fig_avoidance()
    print(f'Avoidance lap: {nbox} boxes, {dur:.1f} s')


if __name__ == '__main__':
    main()
