#!/usr/bin/env python3
"""Replay a car recording through obstacle detection (offline, no ROS graph needed).

Needs a run recorded with the bag kept (launch/race1_obstacles_launch.py keeps it): /scan
and /tf (the particle filter's map -> laser). For every scan it rebuilds the LiDAR pose at
each beam's time, runs detection + memory exactly like obstacle_avoidance_node.py, and
reports:
  * map fit: distance from every scan point inside the track to the nearest map wall
    (sets detection.wall_margin: clean laps must give no points beyond it);
  * beam timing: the same for several beam_time_offset values (the best one lines the
    walls up tightest, see detection timing in the node);
  * detections: every confirmed obstacle with where it was first seen and confirmed.
Writes replay_detection.png (map, path, kept points, boxes) and replay_detection.csv into
the run folder.

    python3 tools/replay_detection.py mpc/data/<run>
    python3 tools/replay_detection.py mpc/data/<run> --wall-margin 0.12 --sweep-offset
"""
import argparse
import csv
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(HERE, '..')
sys.path[:0] = [os.path.join(PKG, 'scripts')]
from track_model import load_track                                    # noqa: E402
from obstacle_tracker import (DetectionParams, ObstacleTracker,      # noqa: E402
                              ScanView, detect, scan_points)


def read_bag(bag_dir, map_frame='map', laser_frame='laser'):
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(bag_dir), storage_id='mcap'),
                rosbag2_py.ConverterOptions('cdr', 'cdr'))
    types = {t.name: get_message(t.type) for t in reader.get_all_topics_and_types()}
    scans, poses = [], []
    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic == '/scan':
            m = deserialize_message(data, types[topic])
            scans.append((m.header.stamp.sec * 1e9 + m.header.stamp.nanosec,
                          np.asarray(m.ranges, float), m.angle_min, m.angle_increment,
                          m.time_increment))
        elif topic == '/tf':
            m = deserialize_message(data, types[topic])
            for tr in m.transforms:
                if (tr.header.frame_id.lstrip('/') == map_frame
                        and tr.child_frame_id.lstrip('/') == laser_frame):
                    q = tr.transform.rotation
                    yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
                    poses.append((tr.header.stamp.sec * 1e9 + tr.header.stamp.nanosec,
                                  tr.transform.translation.x, tr.transform.translation.y, yaw))
    poses = np.array(sorted(poses))
    poses[:, 3] = np.unwrap(poses[:, 3])
    return scans, poses


def beam_poses(poses, stamp, n, ti, offset, stamp_at_end=True):
    """Per-beam laser pose (x, y, yaw), or None if outside the pose history."""
    if ti > 0:
        first = stamp + offset * 1e9 - ((n - 1) * ti * 1e9 if stamp_at_end else 0.0)
        t = first + np.arange(n) * ti * 1e9
    else:
        t = np.full(n, stamp + offset * 1e9)
    if t[0] < poses[0, 0] or t[-1] > poses[-1, 0]:
        return None
    return (np.interp(t, poses[:, 0], poses[:, 1]), np.interp(t, poses[:, 0], poses[:, 2]),
            np.interp(t, poses[:, 0], poses[:, 3]))


def wall_fit(track, scans, poses, offset, p, every=3):
    """Distances to the map walls of all scan points that land inside the track."""
    out = []
    s_hint = None
    for stamp, r, amin, ainc, ti in scans[::every]:
        bp = beam_poses(poses, stamp, len(r), ti, offset)
        if bp is None:
            continue
        idx, rr, xy = scan_points(r, amin, ainc, *bp, p)
        mid = np.array([[bp[0][len(r) // 2], bp[1][len(r) // 2]]])
        s_l = track.project(mid)[0][0] if s_hint is None else track.project(mid, s_hint, (-2, 3))[0][0]
        s_hint = s_l
        s, d, ok = track.project(xy, s_hint=s_l, window=(-p.lookbehind - 1, p.lookahead + 1))
        lo, hi = track.bounds_at(s)
        inside = ok & (d > lo + p.corridor_tol) & (d < hi - p.corridor_tol) & (rr < p.range_max)
        out.append(track.occ.wall_distance(xy[inside, 0], xy[inside, 1]))
    return np.concatenate(out) if out else np.zeros(0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('run', help='run folder (with bag/) or a bag folder')
    ap.add_argument('--track', default='race1')
    ap.add_argument('--map', default=os.path.join(PKG, 'maps', 'race1_cleaned_blocked.yaml'))
    ap.add_argument('--beam-offset', type=float, default=0.0, help='s, as the node parameter')
    ap.add_argument('--wall-margin', type=float, default=None)
    ap.add_argument('--min-points', type=int, default=None)
    ap.add_argument('--sweep-offset', action='store_true', help='compare beam time offsets')
    ap.add_argument('--no-plot', action='store_true')
    a = ap.parse_args()
    run = os.path.abspath(a.run)
    bag = os.path.join(run, 'bag') if os.path.isdir(os.path.join(run, 'bag')) else run
    track = load_track(os.path.join(PKG, 'waypoints', f'{a.track}.csv'), a.map)
    p = DetectionParams()
    if a.wall_margin is not None:
        p.wall_margin = a.wall_margin
    if a.min_points is not None:
        p.min_points = a.min_points
    scans, poses = read_bag(bag)
    if not scans or len(poses) < 2:
        sys.exit(f'{bag}: needs /scan and /tf map -> laser (got {len(scans)} scans, {len(poses)} poses)')
    dur = (scans[-1][0] - scans[0][0]) * 1e-9
    print(f'{len(scans)} scans over {dur:.1f} s ({len(scans) / max(dur, 1e-6):.1f} Hz), '
          f'{len(poses)} particle filter poses')

    # map fit and beam timing
    offsets = [-0.06, -0.04, -0.02, 0.0, 0.02] if a.sweep_offset else [a.beam_offset]
    print('\nmap fit: distance of in-track scan points to the nearest map wall [m]')
    print('  beam offset |   p50    p90    p99   p99.9  | points beyond wall_margin '
          f'{p.wall_margin:.2f} (+{p.wall_margin_per_m}/m)')
    for off in offsets:
        wd = wall_fit(track, scans, poses, off, p)
        if len(wd) == 0:
            continue
        q = np.percentile(wd, [50, 90, 99, 99.9])
        print(f'  {off:+.3f} s     | {q[0]:.3f}  {q[1]:.3f}  {q[2]:.3f}  {q[3]:.3f}  | '
              f'{np.mean(wd >= p.wall_margin):.3%} of {len(wd)}')

    # detection + memory
    tracker = ObstacleTracker(track, p)
    rows, kept_xy, path = [], [], []
    seen = {}
    s_hint = None
    for k, (stamp, r, amin, ainc, ti) in enumerate(scans):
        bp = beam_poses(poses, stamp, len(r), ti, a.beam_offset)
        if bp is None:
            continue
        m = len(r) // 2
        mid = (bp[0][m], bp[1][m], bp[2][m])
        xy0 = np.array([[mid[0], mid[1]]])
        s_l = track.project(xy0)[0][0] if s_hint is None else track.project(xy0, s_hint, (-2, 3))[0][0]
        s_hint = float(s_l)
        path.append(mid[:2])
        clusters, n_kept = detect(track, r, amin, ainc, *bp, s_hint, p)
        for c in clusters:
            kept_xy.append(c.xy)
        conf = tracker.update(clusters, s_hint, ScanView(np.array(mid[:2]), mid[2], r, amin, ainc))
        t = (stamp - scans[0][0]) * 1e-9
        for ob in conf:
            if ob.id not in seen:
                seen[ob.id] = ob
                print(f'  t {t:6.2f} s: obstacle {ob.id} confirmed at u {ob.confirmed_u:.2f} m '
                      f'(first seen {ob.first_seen_u:.2f} m), d {ob.d_min:+.2f}..{ob.d_max:+.2f}')
        rows.append((t, s_hint, n_kept, len(clusters), len(conf)))
    n_conf = len(seen)
    print(f'\n{n_conf} obstacles confirmed; scans with kept points: '
          f'{np.mean([r_[2] > 0 for r_ in rows]):.1%}, with clusters: {np.mean([r_[3] > 0 for r_ in rows]):.1%}')
    with open(os.path.join(run, 'replay_detection.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['t', 's_laser', 'kept_points', 'clusters', 'confirmed'])
        w.writerows(rows)
    if a.no_plot:
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    occ = track.occ
    fig, ax = plt.subplots(figsize=(9, 9))
    ext = [occ.ox, occ.ox + occ.w * occ.res, occ.oy, occ.oy + occ.h * occ.res]
    ax.imshow(~occ.wall, cmap='gray', origin='lower', extent=ext, alpha=0.6)
    ax.plot(track.xy[:, 0], track.xy[:, 1], '--', color='0.5', lw=0.8, label='raceline')
    P = np.array(path)
    ax.plot(P[:, 0], P[:, 1], color='tab:blue', lw=1, label='LiDAR path')
    if kept_xy:
        K = np.vstack(kept_xy)
        ax.plot(K[:, 0], K[:, 1], '.', ms=2, color='tab:orange', label='obstacle points')
    for ob in seen.values():
        u = np.linspace(0, ob.s_len, 6)
        a1 = track.frenet_to_xy(ob.s_min + u, np.full(6, ob.d_min))
        b1 = track.frenet_to_xy(ob.s_min + u[::-1], np.full(6, ob.d_max))
        B = np.vstack([a1, b1, a1[:1]])
        ax.plot(B[:, 0], B[:, 1], color='tab:red', lw=1.5)
    ax.set_aspect('equal')
    ax.set_xlim(track.xy[:, 0].min() - 1.5, track.xy[:, 0].max() + 1.5)
    ax.set_ylim(track.xy[:, 1].min() - 1.5, track.xy[:, 1].max() + 1.5)
    ax.legend(loc='upper right', fontsize=8)
    ax.set_title(f'{os.path.basename(run)}: {n_conf} obstacles confirmed')
    out = os.path.join(run, 'replay_detection.png')
    fig.savefig(out, dpi=130, bbox_inches='tight')
    print(f'plot: {out}')


if __name__ == '__main__':
    main()
