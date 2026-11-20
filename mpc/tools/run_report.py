#!/usr/bin/env python3
"""Run report: one figure per real-car run, and a summary of all runs grouped by state estimator.

    python3 run_report.py <run_dir> [<run_dir> ...]   # report.png + report.json in each run folder
    python3 run_report.py --summary                   # data/estimator_summary.png + .csv from all report.json
    python3 run_report.py --all                       # reports for every real run without one, then the summary

run_recorder.py starts this automatically (in the background) after every run: report + summary.

The figure:
  - map with the raceline and the driven rear-axle path, coloured by sideways distance from the raceline;
    particle-filter pose jumps (x) and joystick takeovers (triangle)
  - sideways distance along the raceline, one thin line per clean lap (coloured by distance) and the mean (bold);
    corners shaded
  - control and state-estimation metrics, and the run's settings (estimator, delay_comp, pose_offset, VESC gains)

One rule for every run, so runs and estimators compare fairly:
  - tracking = REAR AXLE (the particle-filter LiDAR pose moved 0.27 m back) against the raceline
  - laps are timed at a fixed line per track (LAP_LINE_SEGMENT, default the raceline's first waypoint);
    a lap is clean when the MPC drove all of it
  - state estimation is judged against the map: the share of LiDAR beams that end within 10 cm of a map wall when
    projected from the particle-filter pose; scans with a person/obstacle in view (>= 15 beams ending in open
    hallway within 4 m) are left out; pose jumps; jitter of the pose the MPC received (second difference at 25 Hz)
"""
import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path

import warnings

import numpy as np

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / 'data'
sys.path.insert(0, str(HERE))

LIDAR_AHEAD = 0.27                       # base_link (rear axle) -> laser, static TF in the car bringup
ESTIMATORS = ['race', '1A', '1B', '1C', '2A']    # fixed order = fixed colour
LAP_LINE_SEGMENT = {'levine_race': 'S3'}          # lap line in the middle of this segment (the start straight)
MAP_FOR_TRACK = [('levine', 'levine_2nd'), ('race2', 'race2_remapped_edited'), ('race1', 'race1_cleaned_fixed')]
WALL_TOL = 0.10                          # m: a beam "hits a wall" when it ends this close to a map wall
PERSON_BEAMS, PERSON_RANGE, PERSON_GAP = 15, 4.0, 0.4

# colours: reference palette of the dataviz skill (light mode)
SURFACE, INK, INK2, MUTED, GRID, BASE = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
EST_COLOR = dict(zip(ESTIMATORS, ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']))
SEQ = ['#86b6ef', '#6da7ec', '#5598e7', '#3987e5', '#2a78d6', '#256abf', '#1c5cab', '#184f95', '#104281',
       '#0d366b']                        # blue 250 -> 700: |sideways distance|, light = on the line
CRITICAL, WARNING = '#d03b3b', '#fab219'


# ------------------------------------------------------------------------------------------ loading

def find_key(obj, key):
    """First value of `key` anywhere in a nested dict/list (meta.json layouts changed over time)."""
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        obj = list(obj.values())
    if isinstance(obj, list):
        for v in obj:
            r = find_key(v, key)
            if r is not None:
                return r
    return None


def mpc_param(meta, key):
    """A parameter of the MPC node as it ran (meta['mpc']['parameters']); NOT the recorder's own pose_topic."""
    params = (meta.get('mpc') or {}).get('parameters') or {}
    return params.get(key, find_key(meta.get('mpc') or {}, key))


def estimator_of(meta):
    est = meta.get('estimator')
    if est:
        return est, False
    topic = mpc_param(meta, 'pose_topic') or ''
    return ('1A' if 'state_estimate' in str(topic) else 'race'), True   # runs before the package existed


def load_map(track):
    import yaml
    from PIL import Image
    from scipy import ndimage
    from ament_index_python.packages import get_package_share_directory
    maps = Path(get_package_share_directory('particle_filter')) / 'maps'
    name = next((m for prefix, m in MAP_FOR_TRACK if str(track).startswith(prefix)), None)
    if name is None:
        loc = Path(get_package_share_directory('particle_filter')) / 'config' / 'localize.yaml'
        name = yaml.safe_load(loc.read_text())['map_server']['ros__parameters']['map']
    info = yaml.safe_load((maps / f'{name}.yaml').read_text())
    img = np.array(Image.open(maps / info['image']).convert('L'))
    occ = (255 - img) / 255.0 > info['occupied_thresh']
    lab, n = ndimage.label(occ, structure=np.ones((3, 3)))
    sizes = ndimage.sum(occ, lab, range(1, n + 1))
    walls = np.isin(lab, 1 + np.where(sizes >= 20)[0])          # single-pixel specks are not walls
    return {'name': name, 'img': img, 'occ': occ, 'free': img > 250, 'res': float(info['resolution']),
            'origin': [float(v) for v in info['origin'][:2]],
            'dist': ndimage.distance_transform_edt(~walls) * float(info['resolution'])}


def map_lookup(m, x, y, field):
    h, w = m['img'].shape
    c = ((x - m['origin'][0]) / m['res']).astype(int)
    r = (h - 1 - (y - m['origin'][1]) / m['res']).astype(int)
    inside = (c >= 0) & (c < w) & (r >= 0) & (r < h)
    out = np.full(len(x), np.nan if field == 'dist' else False, dtype=float if field == 'dist' else bool)
    out[inside] = m[field][r[inside], c[inside]]
    return out


def read_bag(bag_dir, topics):
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(bag_dir)), rosbag2_py.ConverterOptions('', ''))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    want = [t for t in topics if t in types]
    reader.set_filter(rosbag2_py.StorageFilter(topics=want))
    out = {t: [] for t in want}
    cls = {t: get_message(types[t]) for t in want}
    while reader.has_next():
        topic, data, t_ns = reader.read_next()
        out[topic].append((t_ns * 1e-9, deserialize_message(data, cls[topic])))
    return out


def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


# ------------------------------------------------------------------------------------------ analysis

def analyze(run_dir):
    from run_analysis import build_track, project
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / 'meta.json').read_text())
    raw = np.load(run_dir / 'raw.npz')
    odom, cmd, solve = raw['odom'], raw['cmd'], raw['solve']
    est, inferred = estimator_of(meta)
    track_name = find_key(meta, 'track') or 'unknown'
    wp = np.genfromtxt(run_dir / 'waypoints.csv', delimiter=',', comments='#')
    track = build_track(wp)
    L = track['L']
    seg = {s['name']: s for s in track['segments']}
    s_line = 0.0
    if LAP_LINE_SEGMENT.get(track_name) in seg:
        sg = seg[LAP_LINE_SEGMENT[track_name]]
        s_line = (sg['s0'] + ((sg['s1'] - sg['s0']) % L) / 2) % L

    t, v = odom[:, 0], np.abs(odom[:, 4])
    # rear axle from the particle-filter LiDAR pose (every car run records /pf/pose/odom)
    rx = odom[:, 1] - LIDAR_AHEAD * np.cos(odom[:, 3])
    ry = odom[:, 2] - LIDAR_AHEAD * np.sin(odom[:, 3])
    j, f, e = project(track, np.c_[rx, ry])
    s = (track['s_start'][j] + f * track['seg_len'][j]) % L
    mpc = np.interp(t, cmd[:, 0], cmd[:, 3]) > 0.5 if len(cmd) else np.ones(len(t), bool)

    # laps between forward crossings of the lap line
    p = (s - s_line) % L
    cross = np.where((np.diff(p) < -L / 2) & (v[1:] > 0.5))[0] + 1
    laps = []
    for a, b in zip(cross[:-1], cross[1:]):
        laps.append({'i0': int(a), 'i1': int(b), 'time': float(t[b] - t[a]),
                     'clean': bool(mpc[a:b].mean() >= 0.99),     # <= 1 % joystick (a lap-end overlap)
                     'mean_e': float(np.abs(e[a:b]).mean()), 'max_e': float(np.abs(e[a:b]).max())})
    clean = [lp for lp in laps if lp['clean']]
    in_clean = np.zeros(len(t), bool)
    for lp in clean:
        in_clean[lp['i0']:lp['i1']] = True

    # pose jumps of the particle filter (same rule as run_analysis.pose_jumps), where
    step = np.hypot(np.diff(odom[:, 1]), np.diff(odom[:, 2]))
    allowed = np.maximum(0.3, 3 * v[1:] * np.diff(t) + 0.1)
    jump_idx = np.where(step > allowed)[0] + 1
    jump_idx = jump_idx[mpc[jump_idx]]                                  # while the MPC drove
    # joystick takeovers: the MPC stops driving between its first and last command
    takeovers = []
    if mpc.any():
        first, last = np.argmax(mpc), len(mpc) - 1 - np.argmax(mpc[::-1])
        k = first
        while k < last:
            if not mpc[k]:
                k2 = k
                while k2 < last and not mpc[k2]:
                    k2 += 1
                takeovers.append((int(k), float(t[k2] - t[k])))
                k = k2
            k += 1

    r = {'run': run_dir.name, 'note': meta.get('note', ''), 'estimator': est, 'estimator_inferred': inferred,
         'track': track_name, 'L': L, 's_line': s_line,
         'settings': {'pose_topic': mpc_param(meta, 'pose_topic'), 'delay_comp': mpc_param(meta, 'delay_comp'),
                      'pose_offset': mpc_param(meta, 'pose_offset'),
                      'steer_offset': find_key(meta, 'steering_angle_to_servo_offset'),
                      'steer_gain': find_key(meta, 'steering_angle_to_servo_gain'),
                      'speed_gain': find_key(meta, 'speed_to_erpm_gain')},
         'laps': [{k: lp[k] for k in ('time', 'clean', 'mean_e', 'max_e')} for lp in laps],
         'n_clean': len(clean),
         'lap_time_mean': float(np.mean([lp['time'] for lp in clean])) if clean else None,
         'lap_time_std': float(np.std([lp['time'] for lp in clean])) if len(clean) > 1 else None,
         'mean_e': float(np.abs(e[in_clean]).mean()) if clean else None,
         'max_e': float(np.abs(e[in_clean]).max()) if clean else None,
         'signed_e': float(e[in_clean].mean()) if clean else None,
         'jumps': int(len(jump_idx)), 'takeovers': len(takeovers),
         'mpc_minutes': float(np.sum(np.diff(t)[mpc[1:]]) / 60.0),
         'solve_ms': float(np.mean(solve[:, 1])) if len(solve) else None}
    r['jumps_per_min'] = r['jumps'] / r['mpc_minutes'] if r['mpc_minutes'] > 0 else None
    series = {'t': t, 'rx': rx, 'ry': ry, 'e': e, 's': s, 'mpc': mpc, 'laps': clean, 'jumps': jump_idx,
              'takeovers': takeovers, 'track': track, 'odom': odom}
    r.update(estimation_metrics(run_dir, meta, r, series))
    return meta, r, series


def estimation_metrics(run_dir, meta, r, sr):
    """Scan-to-map alignment and MPC-input jitter; needs the run's bag (keep_bag)."""
    out = {'scan_on_walls': None, 'scan_on_walls_worst5': None, 'person_scans_pct': None,
           'jitter_med_mm': None, 'jitter_p95_mm': None, 'estimation_note': ''}
    bag = run_dir / 'bag'
    if not (bag / 'metadata.yaml').exists():
        out['estimation_note'] = 'no bag kept: scan alignment and jitter not available'
        return out
    pose_topic = str(r['settings']['pose_topic'] or 'pf/pose/odom')
    pose_topic = pose_topic if pose_topic.startswith('/') else '/' + pose_topic
    try:
        m = load_map(r['track'])
    except Exception as exc:                                          # noqa: BLE001
        out['estimation_note'] = f'map not found ({exc})'
        return out
    out['map'] = m['name']
    bag_msgs = read_bag(bag, ['/scan', '/pf/pose/odom', pose_topic])
    pf = bag_msgs.get('/pf/pose/odom', [])
    if not pf:
        out['estimation_note'] = 'no /pf/pose/odom in the bag'
        return out
    # bag time -> run time: the first kept raw pose is the bag's pose with the same position
    odom = sr['odom']
    P = np.array([(tb, msg.pose.pose.position.x, msg.pose.pose.position.y, yaw_of(msg.pose.pose.orientation))
                  for tb, msg in pf])
    k = int(np.argmin(np.hypot(P[:, 1] - odom[0, 1], P[:, 2] - odom[0, 2]) + 1e3 * (P[:, 0] < P[0, 0])))
    t0 = P[k, 0] - odom[0, 0]
    on = lambda tr: (np.interp(tr, sr['t'], sr['mpc'].astype(float)) > 0.5) & \
        (np.interp(tr, sr['t'], np.abs(odom[:, 4])) > 1.0)              # noqa: E731
    al, person = [], []
    th_unwrap = np.unwrap(P[:, 3])
    for tb, sc in bag_msgs.get('/scan', []):
        tr = tb - t0
        if tb < P[0, 0] or tb > P[-1, 0] or not on(tr):
            continue
        x0, y0, th = np.interp(tb, P[:, 0], P[:, 1]), np.interp(tb, P[:, 0], P[:, 2]), np.interp(tb, P[:, 0], th_unwrap)
        rng = np.asarray(sc.ranges, dtype=float)
        a = sc.angle_min + np.arange(len(rng)) * sc.angle_increment
        ok = np.isfinite(rng) & (rng > 0.1) & (rng < 8.0)
        X, Y = x0 + rng[ok] * np.cos(th + a[ok]), y0 + rng[ok] * np.sin(th + a[ok])
        d = map_lookup(m, X, Y, 'dist')
        fr = map_lookup(m, X, Y, 'free')
        person.append(np.sum((np.nan_to_num(d, nan=9.9) > PERSON_GAP) & fr & (rng[ok] < PERSON_RANGE)) >= PERSON_BEAMS)
        al.append(np.mean(np.nan_to_num(d, nan=9.9) < WALL_TOL))
    if al:
        al, person = np.array(al), np.array(person)
        keep = ~person if (~person).sum() >= 20 else np.ones(len(al), bool)
        out['scan_on_walls'] = float(100 * al[keep].mean())
        out['scan_on_walls_worst5'] = float(100 * np.percentile(al[keep], 5))
        out['person_scans_pct'] = float(100 * person.mean())
    mi = bag_msgs.get(pose_topic, [])
    if len(mi) > 10:
        q = np.array([(tb, msg.pose.pose.position.x, msg.pose.pose.position.y) for tb, msg in mi])
        keep = [0]
        for i in range(1, len(q)):                       # one sample per >= 40 ms: same rate for every estimator
            if q[i, 0] - q[keep[-1], 0] >= 0.038:
                keep.append(i)
        q = q[keep]
        dt = np.diff(q[:, 0])
        sd = np.hypot(np.diff(q[:, 1], 2), np.diff(q[:, 2], 2)) * (0.04 / dt[1:]) ** 2
        sel = on(q[2:, 0] - t0)
        if sel.sum() > 20:
            out['jitter_med_mm'] = float(1e3 * np.median(sd[sel]))
            out['jitter_p95_mm'] = float(1e3 * np.percentile(sd[sel], 95))
    else:
        out['estimation_note'] = f'{pose_topic} not in the bag: jitter not available'
    return out


# ------------------------------------------------------------------------------------------ figures

def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(BASE)
    ax.tick_params(colors=MUTED, labelcolor=INK2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def fmt(x, nd=1, unit=''):
    return '—' if x is None else f'{x:.{nd}f}{unit}'


def plot_run(run_dir, meta, r, sr):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.lines import Line2D

    cmap = LinearSegmentedColormap.from_list('dist', SEQ)
    e_abs = np.abs(sr['e'])
    vmax = max(0.10, float(np.percentile(e_abs[sr['mpc']], 99))) if sr['mpc'].any() else 0.15
    norm = Normalize(0.0, vmax)
    track = sr['track']

    fig = plt.figure(figsize=(16, 10.5), dpi=110, facecolor=SURFACE)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.55, 1.0], height_ratios=[1.45, 1.0], hspace=0.28, wspace=0.12,
                          left=0.075, right=0.97, top=0.90, bottom=0.07)
    est = r['estimator']
    fig.text(0.05, 0.955, f"Run {r['run']}  ·  estimator {est}{' (inferred)' if r['estimator_inferred'] else ''}",
             fontsize=17, fontweight='bold', color=INK)
    fig.text(0.05, 0.925, f"track {r['track']}  ·  {r['note'] or 'no note'}", fontsize=10.5, color=INK2)
    fig.patches.append(matplotlib.patches.Rectangle((0.03, 0.925), 0.008, 0.05, transform=fig.transFigure,
                                                    color=EST_COLOR.get(est, MUTED)))

    # --- map: walls, raceline, rear-axle path coloured by |sideways distance|
    ax = fig.add_subplot(gs[0, 0])
    ax.set_facecolor(SURFACE)
    try:
        m = load_map(r['track'])
        h, w = m['img'].shape
        rgb = np.full((h, w, 3), 0.94)                                # unknown
        rgb[m['free']] = (0.988, 0.988, 0.984)
        rgb[m['occ']] = (0.32, 0.32, 0.31)
        ox, oy = m['origin']
        ax.imshow(rgb, extent=[ox, ox + w * m['res'], oy, oy + h * m['res']], origin='upper',
                  interpolation='nearest')
    except Exception:                                                  # noqa: BLE001
        pass
    wpx, wpy = track['A'][:, 0], track['A'][:, 1]
    ax.plot(np.r_[wpx, wpx[0]], np.r_[wpy, wpy[0]], '--', color=MUTED, lw=1.0, zorder=2)
    mp = sr['mpc']
    pts = np.c_[sr['rx'], sr['ry']]
    segs = np.stack([pts[:-1], pts[1:]], axis=1)
    use = mp[1:] & mp[:-1] & (np.hypot(*(pts[1:] - pts[:-1]).T) < 0.5)
    lc = LineCollection(segs[use], cmap=cmap, norm=norm, linewidths=2.0, zorder=3)
    lc.set_array(e_abs[1:][use])
    ax.add_collection(lc)
    joy = ~mp
    ax.plot(np.where(joy, sr['rx'], np.nan), np.where(joy, sr['ry'], np.nan), ':', color=MUTED, lw=1.0, zorder=3)
    if len(sr['jumps']):
        ax.plot(sr['odom'][sr['jumps'], 1], sr['odom'][sr['jumps'], 2], 'x', color=CRITICAL, ms=9, mew=2.2, zorder=5)
    for k, dur in sr['takeovers']:
        ax.plot(sr['rx'][k], sr['ry'][k], '^', color=WARNING, mec=INK, mew=0.8, ms=10, zorder=5)
    pad = 1.5
    ax.set_xlim(wpx.min() - pad, wpx.max() + pad)
    ax.set_ylim(wpy.min() - pad, wpy.max() + pad)
    ax.set_aspect('equal')
    ax.tick_params(colors=MUTED, labelcolor=INK2, labelsize=8)
    for sp in ax.spines.values():
        sp.set_color(BASE)
    ax.set_title('Driven path (rear axle, MPC driving) coloured by distance from the raceline',
                 loc='left', fontsize=11, color=INK)
    handles = [Line2D([], [], ls='--', color=MUTED, label='raceline'),
               Line2D([], [], color=SEQ[5], lw=2, label='MPC driving'),
               Line2D([], [], ls=':', color=MUTED, label='joystick'),
               Line2D([], [], marker='x', ls='', color=CRITICAL, mew=2, label=f"pose jump ({r['jumps']})"),
               Line2D([], [], marker='^', ls='', color=WARNING, mec=INK, label=f"takeover ({r['takeovers']})")]
    ax.legend(handles=handles, loc='lower right', fontsize=8.5, frameon=True, facecolor=SURFACE, edgecolor=GRID)
    cb = fig.colorbar(lc, ax=ax, fraction=0.035, pad=0.01)
    cb.set_label('|distance from raceline| [m]', color=INK2, fontsize=9)
    cb.ax.tick_params(colors=MUTED, labelcolor=INK2, labelsize=8)
    cb.outline.set_edgecolor(BASE)

    # --- metrics panel
    axm = fig.add_subplot(gs[0, 1])
    axm.axis('off')
    st = r['settings']
    lap_t = ', '.join(f"{lp['time']:.2f}" for lp in r['laps'] if lp['clean']) or '—'
    rows = [('CONTROL', None),
            ('clean laps', f"{r['n_clean']} of {len(r['laps'])} timed"),
            ('lap time, mean', fmt(r['lap_time_mean'], 2, ' s') + (f"  (±{r['lap_time_std']:.2f})" if r['lap_time_std'] else '')),
            ('lap times', lap_t + (' s' if lap_t != '—' else '')),
            ('tracking error, mean', fmt(None if r['mean_e'] is None else 100 * r['mean_e'], 1, ' cm')),
            ('tracking error, max', fmt(None if r['max_e'] is None else 100 * r['max_e'], 1, ' cm')),
            ('sideways offset, mean', fmt(None if r['signed_e'] is None else 100 * r['signed_e'], 1, ' cm') + '  (+ = left)'),
            ('takeovers', str(r['takeovers'])),
            ('MPC solve, mean', fmt(r['solve_ms'], 1, ' ms')),
            ('STATE ESTIMATION', None),
            ('scan on map walls', fmt(r['scan_on_walls'], 1, ' %') + '  (person scans excluded)'),
            ('  worst 5 % of scans', fmt(r['scan_on_walls_worst5'], 1, ' %')),
            ('  scans with a person', fmt(r['person_scans_pct'], 0, ' %')),
            ('pose jumps', f"{r['jumps']}  ({fmt(r['jumps_per_min'], 1, ' /min')} of MPC driving)"),
            ('MPC-input jitter, median / p95',
             f"{fmt(r['jitter_med_mm'], 1)} / {fmt(r['jitter_p95_mm'], 1)} mm  (at 25 Hz)"),
            ('SETTINGS', None),
            ('estimator', est + ('  (inferred from pose_topic)' if r['estimator_inferred'] else '')),
            ('pose_topic', str(st['pose_topic'])),
            ('delay_comp / pose_offset', f"{st['delay_comp']} s / {st['pose_offset']} m"),
            ('steering gain / offset', f"{st['steer_gain']} / {st['steer_offset']}"),
            ('speed_to_erpm_gain', str(st['speed_gain']))]
    y = 1.0
    for label, val in rows:
        if val is None:
            y -= 0.025
            axm.text(0.0, y, label, fontsize=10, fontweight='bold', color=EST_COLOR.get(est, INK) if False else INK,
                     transform=axm.transAxes)
            axm.plot([0, 1], [y - 0.012, y - 0.012], color=GRID, lw=0.8, transform=axm.transAxes)
            y -= 0.052
            continue
        axm.text(0.0, y, label, fontsize=9.5, color=INK2, transform=axm.transAxes)
        axm.text(0.47, y, val, fontsize=9.5, color=INK, transform=axm.transAxes)
        y -= 0.043
    if r.get('estimation_note'):
        axm.text(0.0, y - 0.01, r['estimation_note'], fontsize=8.5, color=MUTED, style='italic',
                 transform=axm.transAxes)

    # --- sideways distance along the raceline, every clean lap + mean
    axs = fig.add_subplot(gs[1, :])
    _style(axs)
    L = r['L']
    for sg in track['segments']:
        if sg['kind'] != 'corner':
            continue
        s0, s1 = sg['s0'] % L, sg['s1'] % L
        spans = [(s0, s1)] if s1 > s0 else [(s0, L), (0, s1)]
        for a, b in spans:
            axs.axvspan(a, b, color=GRID, alpha=0.6, lw=0, zorder=0)
        axs.text(((s0 + ((s1 - s0) % L) / 2) % L), 1.0, sg['name'], transform=axs.get_xaxis_transform(),
                 ha='center', va='bottom', fontsize=9, color=INK2)
    axs.axhline(0, color=BASE, lw=1.0, zorder=1)
    bins = np.arange(0, L + 0.1, 0.1)
    lap_means = []
    for lp in sr['laps']:
        sl = slice(lp['i0'], lp['i1'])
        s_l, e_l = sr['s'][sl], sr['e'][sl]
        pts_l = np.c_[s_l, e_l]
        segs_l = np.stack([pts_l[:-1], pts_l[1:]], axis=1)
        ok = np.abs(np.diff(s_l)) < L / 2                      # don't draw across the s = L -> 0 wrap
        lcl = LineCollection(segs_l[ok], cmap=cmap, norm=norm, linewidths=1.1, alpha=0.9, zorder=2)
        lcl.set_array(np.abs(e_l[1:][ok]))
        axs.add_collection(lcl)
        idx = np.digitize(s_l, bins) - 1
        lap_means.append(np.array([e_l[idx == b].mean() if np.any(idx == b) else np.nan for b in range(len(bins) - 1)]))
    if lap_means:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)   # bins no lap passed through
            mean = np.nanmean(np.vstack(lap_means), axis=0)
        axs.plot(bins[:-1] + 0.05, mean, color=INK, lw=2.4, zorder=4)
        lo = np.nanmin(np.vstack([np.nanmin(sr['e'][lp['i0']:lp['i1']]) for lp in sr['laps']] + [-0.05]))
        hi = np.nanmax(np.vstack([np.nanmax(sr['e'][lp['i0']:lp['i1']]) for lp in sr['laps']] + [0.05]))
        lim = max(abs(lo), abs(hi)) * 1.15
        axs.set_ylim(-lim, lim)
    else:
        axs.text(0.5, 0.5, 'no clean lap', transform=axs.transAxes, ha='center', color=MUTED)
    axs.set_xlim(0, L)
    axs.axvline(r['s_line'], color=MUTED, lw=1.0, ls=(0, (2, 3)), zorder=1)
    axs.text(r['s_line'], 0.02, ' lap line', transform=axs.get_xaxis_transform(), fontsize=8.5, color=MUTED)
    axs.set_xlabel('distance along the raceline s [m]   (grey = corners)', color=INK2, fontsize=10)
    axs.set_ylabel('sideways distance [m]\n+ = left of the raceline', color=INK2, fontsize=10)
    axs.set_title(f"Distance from the raceline, every clean lap (thin, coloured by distance) and their mean (bold, "
                  f"{len(sr['laps'])} lap{'s' if len(sr['laps']) != 1 else ''})", loc='left', fontsize=11, color=INK,
                  pad=16)
    axs.legend(handles=[Line2D([], [], color=SEQ[5], lw=1.1, label='one clean lap'),
                        Line2D([], [], color=INK, lw=2.4, label='mean of the clean laps')],
               loc='upper right', fontsize=8.5, frameon=True, facecolor=SURFACE, edgecolor=GRID)

    out = Path(run_dir) / 'report.png'
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    return out


SUMMARY_METRICS = [('mean_e', 'tracking error, mean [cm]', 100), ('max_e', 'tracking error, max [cm]', 100),
                   ('lap_time_mean', 'lap time, mean [s]', 1), ('scan_on_walls', 'scan on map walls [%]', 1),
                   ('scan_on_walls_worst5', 'worst 5 % of scans [%]', 1), ('jumps_per_min', 'pose jumps per minute', 1),
                   ('jitter_p95_mm', 'MPC-input jitter p95 [mm]', 1), ('jitter_med_mm', 'MPC-input jitter median [mm]', 1)]


def summary(data_dir=DATA_DIR, track=None):
    """Runs on one track (default: the newest run's) with a report.json, grouped by estimator:
    data/estimator_summary.png + .csv. Different tracks are never mixed."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    reps = []
    for f in sorted(Path(data_dir).glob('*/report.json')):
        try:
            reps.append(json.loads(f.read_text()))
        except Exception:                                              # noqa: BLE001
            pass
    reps = [r for r in reps if r.get('n_clean')]
    if not reps:
        print('[run_report] no runs with clean laps for the summary')
        return
    track = track or reps[-1]['track']                  # default: the newest run's track (runs sort by date)
    reps = [r for r in reps if r['track'] == track]
    fields = ['run', 'estimator', 'estimator_inferred', 'track', 'note', 'n_clean', 'lap_time_mean', 'mean_e', 'max_e',
              'signed_e', 'scan_on_walls', 'scan_on_walls_worst5', 'person_scans_pct', 'jumps', 'jumps_per_min',
              'jitter_med_mm', 'jitter_p95_mm', 'takeovers', 'solve_ms']
    with open(Path(data_dir) / 'estimator_summary.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(fields + ['delay_comp', 'pose_offset', 'steer_offset', 'speed_gain'])
        for r in reps:
            st = r['settings']
            w.writerow([r.get(k) for k in fields] + [st.get('delay_comp'), st.get('pose_offset'),
                                                     st.get('steer_offset'), st.get('speed_gain')])
    present = [e for e in ESTIMATORS if any(r['estimator'] == e for r in reps)]
    fig, axes = plt.subplots(2, 4, figsize=(16, 8.2), dpi=110, facecolor=SURFACE)
    fig.subplots_adjust(left=0.05, right=0.98, top=0.86, bottom=0.12, hspace=0.45, wspace=0.32)
    fig.text(0.05, 0.94, f'All runs on {track}, grouped by state estimator', fontsize=17, fontweight='bold', color=INK)
    fig.text(0.05, 0.905, f'{len(reps)} runs with clean laps · one dot per run, bar = mean of the estimator\'s runs · '
             'runs differ in day and car settings (steering trim, speed gain): see estimator_summary.csv',
             fontsize=10, color=INK2)
    rng = np.random.default_rng(0)
    for ax, (key, label, scale) in zip(axes.ravel(), SUMMARY_METRICS):
        _style(ax)
        ax.grid(axis='x', visible=False)
        for gi, est in enumerate(present):
            vals = [r[key] * scale for r in reps if r['estimator'] == est and r.get(key) is not None]
            if not vals:
                continue
            x = gi + rng.uniform(-0.12, 0.12, len(vals))
            ax.scatter(x, vals, s=46, color=EST_COLOR[est], edgecolor=SURFACE, linewidth=1.5, zorder=3)
            ax.plot([gi - 0.28, gi + 0.28], [np.mean(vals)] * 2, color=INK, lw=2.2, zorder=4)
            ax.text(gi + 0.31, np.mean(vals), f'{np.mean(vals):.1f}', va='center', fontsize=8.5, color=INK2)
        ax.set_xticks(range(len(present)))
        ax.set_xticklabels([f'{e}\n(n={sum(r["estimator"] == e for r in reps)})' for e in present], color=INK2)
        ax.set_xlim(-0.6, len(present) - 0.4)
        ax.set_title(label, loc='left', fontsize=10.5, color=INK)
    out = Path(data_dir) / 'estimator_summary.png'
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    print(f'[run_report] summary for {track}: {out} and estimator_summary.csv ({len(reps)} runs)', flush=True)


def report(run_dir):
    meta, r, sr = analyze(run_dir)
    (Path(run_dir) / 'report.json').write_text(json.dumps(r, indent=1))
    out = plot_run(run_dir, meta, r, sr)
    print(f'[run_report] {out}', flush=True)
    return r


def start_in_background(run_dir):
    """Called by run_recorder.py: report + summary in a detached process (survives the launch shutting down)."""
    log = open(Path(run_dir) / 'report.log', 'w')
    subprocess.Popen([sys.executable, str(Path(__file__).resolve()), str(run_dir), '--then-summary'],
                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('runs', nargs='*', help='run folders (mpc/data/<run>)')
    ap.add_argument('--summary', action='store_true', help='only the all-runs summary')
    ap.add_argument('--then-summary', action='store_true', help='report the runs, then the summary')
    ap.add_argument('--all', action='store_true', help='report every real run without report.json, then summary')
    a = ap.parse_args()
    runs = [Path(p) for p in a.runs]
    if a.all:
        for d in sorted(DATA_DIR.iterdir()):
            if (d / 'raw.npz').exists() and not (d / 'report.json').exists():
                try:
                    meta = json.loads((d / 'meta.json').read_text())
                except (OSError, ValueError) as exc:
                    print(f'[run_report] {d.name}: skipped, unreadable meta.json ({exc})', flush=True)
                    continue
                if meta.get('platform') == 'real':
                    runs.append(d)
    for d in runs:
        try:
            report(d)
        except Exception as exc:                                       # noqa: BLE001
            print(f'[run_report] {d.name}: failed: {exc!r}', flush=True)
    if a.summary or a.then_summary or a.all:
        summary()


if __name__ == '__main__':
    main()
