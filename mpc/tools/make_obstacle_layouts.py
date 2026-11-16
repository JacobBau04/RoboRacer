#!/usr/bin/env python3
"""Random obstacle layouts for the simulator (race1 by default; race2: see the example below).

Each layout is a copy of the race1 map with 1-3 boxes drawn in (walls the sim's LiDAR sees
and its collision check uses), plus a yaml that names the race1 centerline (lap counter).
Boxes: width --width-min..--width-max of the local track width, depth 0.15-0.40 m, slightly
rotated; most of them near the raceline (--near-raceline). With --straights they are only
placed on the straight stretches (centerline curvature below --kappa-straight over at least
--min-straight metres). Placement is "fair": at least one side leaves car width + 2 x fair_gap.
Writes clean.yaml (no boxes), layouts.json with each box's corners (ground truth for
tools/sim_batch.py) and sim_<name>.yaml simulator configs (map + start pose) for each layout.

    # race rules as known on 2026-11-03: boxes only on straights, 10-20 % of the width
    python3 tools/make_obstacle_layouts.py --out ~/sim_ws/race1_straights --n 20 \
        --straights --width-min 0.10 --width-max 0.20
    # race2
    python3 tools/make_obstacle_layouts.py --out ~/sim_ws/race2_straights --n 20 \
        --straights --width-min 0.10 --width-max 0.20 --map maps/race2_blocked.yaml \
        --waypoints waypoints/race2.csv --centerline maps/race2_centerline.csv
"""
import argparse
import json
import math
import os
import shutil
import sys

import numpy as np
import yaml
from PIL import Image
from scipy.ndimage import uniform_filter1d

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(HERE, '..')
sys.path[:0] = [os.path.join(PKG, 'scripts'), HERE]
from track_model import load_track      # noqa: E402
from scan_sim import box_corners        # noqa: E402

CAR_WIDTH = 0.31
SIM_CONFIG = os.path.expanduser('~/sim_ws/install/f1tenth_gym_ros/share/f1tenth_gym_ros/config/sim.yaml')


def straight_intervals(tr, centerline_csv, kappa_max, min_len):
    """Straight stretches as (raceline s of the start, length) from the centerline's curvature."""
    c = np.loadtxt(centerline_csv, delimiter=',', comments='#')
    xy = c[:, :2]
    seg = np.diff(np.vstack([xy, xy[:1]]), axis=0)
    ds = np.hypot(seg[:, 0], seg[:, 1])
    head = np.unwrap(np.arctan2(seg[:, 1], seg[:, 0]))
    kappa = uniform_filter1d(np.gradient(head) / ds, 9, mode='wrap')
    straight = np.abs(kappa) < kappa_max
    if straight.all():
        return [(0.0, tr.length)]
    k0 = int(np.argmin(straight))                 # start the scan on a curved point
    order = np.roll(np.arange(len(xy)), -k0)
    runs, cur = [], []
    for i in order:
        if straight[i]:
            cur.append(i)
        elif cur:
            runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    s_rl = tr.project(xy)[0]
    out = []
    for run in runs:
        if ds[run].sum() < min_len:
            continue
        s0 = float(s_rl[run[0]])
        length = float(tr.rel_s(s_rl[run[-1]], s0)) % tr.length
        out.append((s0, length))
    return out


def draw_box(img, meta, corners):
    """Mark map cells whose centre lies inside the (convex) box as occupied."""
    res, ox, oy = meta['resolution'], meta['origin'][0], meta['origin'][1]
    H, W = img.shape
    xmin, ymin = corners.min(axis=0)
    xmax, ymax = corners.max(axis=0)
    c0, c1 = int((xmin - ox) / res) - 1, int((xmax - ox) / res) + 2
    r0, r1 = int((ymin - oy) / res) - 1, int((ymax - oy) / res) + 2
    cc, rr = np.meshgrid(np.arange(c0, c1), np.arange(r0, r1))
    px, py = ox + (cc + 0.5) * res, oy + (rr + 0.5) * res
    inside = np.ones(cc.shape, bool)
    for k in range(4):
        a, b = corners[k], corners[(k + 1) % 4]
        inside &= ((b[0] - a[0]) * (py - a[1]) - (b[1] - a[1]) * (px - a[0])) >= 0
    rows = H - 1 - rr[inside]          # image row 0 is the top (largest y)
    cols = cc[inside]
    ok = (rows >= 0) & (rows < H) & (cols >= 0) & (cols < W)
    img[rows[ok], cols[ok]] = 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', required=True)
    ap.add_argument('--n', type=int, default=20)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--min-obs', type=int, default=1)
    ap.add_argument('--max-obs', type=int, default=3)
    ap.add_argument('--width-min', type=float, default=0.05, help='box width / track width')
    ap.add_argument('--width-max', type=float, default=0.30)
    ap.add_argument('--near-raceline', type=float, default=0.7,
                    help='share of boxes centred within 0.4 m of the raceline (rest: anywhere)')
    ap.add_argument('--straights', action='store_true', help='boxes only on straight stretches')
    ap.add_argument('--kappa-straight', type=float, default=0.2, help='1/m, centerline curvature limit')
    ap.add_argument('--min-straight', type=float, default=1.5, help='m, shortest straight used')
    ap.add_argument('--fair-gap', type=float, default=0.15,
                    help='m each side of the car in the widest gap at every box')
    ap.add_argument('--start-back', type=float, default=2.0,
                    help='start this far (m) before the lap line, so every counted lap is a flying lap')
    ap.add_argument('--map', default=os.path.join(PKG, 'maps', 'race1_cleaned_blocked.yaml'))
    ap.add_argument('--waypoints', default=os.path.join(PKG, 'waypoints', 'race1.csv'))
    ap.add_argument('--centerline', default=os.path.join(PKG, 'maps', 'race1_cleaned_centerline.csv'))
    a = ap.parse_args()
    out = os.path.abspath(os.path.expanduser(a.out))
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    tr = load_track(a.waypoints, a.map)
    with open(a.map) as f:
        meta = yaml.safe_load(f)
    base = np.array(Image.open(os.path.join(os.path.dirname(a.map), meta['image'])))
    cl_name = os.path.basename(a.centerline)          # lap-counter centerline for the sim
    shutil.copy(a.centerline, os.path.join(out, cl_name))

    s_start = float(tr.wrap_s(-a.start_back))
    x0, y0 = tr.frenet_to_xy(s_start, 0.0)
    start = [float(x0), float(y0), float(tr.yaw_at(s_start))]
    straights = straight_intervals(tr, a.centerline, a.kappa_straight, a.min_straight)
    if a.straights:
        print('straights (raceline s start -> end, length):',
              ', '.join(f'{s0:.1f}->{float(tr.wrap_s(s0 + L)):.1f} ({L:.1f} m)' for s0, L in straights))

    def sample_s():
        if not a.straights:
            return float(rng.uniform(1.5, tr.length - a.start_back - 2.5))
        lengths = np.array([L for _, L in straights])
        k = int(rng.choice(len(straights), p=lengths / lengths.sum()))
        s0, L = straights[k]
        return float(tr.wrap_s(s0 + rng.uniform(0.25, max(L - 0.25, 0.3))))

    def in_start_zone(s):
        return -a.start_back - 0.5 < float(tr.rel_s(s, 0.0)) < 1.5

    with open(SIM_CONFIG) as f:
        sim_cfg = yaml.safe_load(f)

    def write(name, img, boxes):
        Image.fromarray(img).save(os.path.join(out, name + '.pgm'))
        m = dict(meta)
        m['image'] = name + '.pgm'
        m['centerline'] = cl_name
        yaml_path = os.path.join(out, name + '.yaml')
        with open(yaml_path, 'w') as f:
            yaml.safe_dump(m, f, sort_keys=False)
        # simulator config for running this layout by hand (config:=<this file>)
        cfg = json.loads(json.dumps(sim_cfg))
        cfg['bridge']['ros__parameters'].update(map_path=yaml_path, sx=start[0], sy=start[1],
                                                stheta=start[2], num_agents=1, kb_teleop=False)
        with open(os.path.join(out, f'sim_{name}.yaml'), 'w') as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        return {'name': name, 'yaml': yaml_path, 'boxes': boxes}

    layouts = [write('clean', base.copy(), [])]
    for k in range(a.n):
        img = base.copy()
        boxes = []
        n_obs = int(rng.integers(a.min_obs, a.max_obs + 1))
        tries = 0
        while len(boxes) < n_obs and tries < 500:
            tries += 1
            s = sample_s()
            if in_start_zone(s) or any(abs(float(tr.rel_s(s, b['s']))) < 0.8 for b in boxes):
                continue
            lo, hi = tr.bounds_at(s)
            width = hi - lo
            w = float(rng.uniform(a.width_min, a.width_max) * width)
            depth = float(rng.uniform(0.15, 0.40))
            if rng.random() < a.near_raceline:   # most boxes near the raceline (where they matter)
                dc = float(rng.uniform(-0.4, 0.4))
            else:
                dc = float(rng.uniform(lo + w / 2, hi - w / 2))
            gap = max(hi - (dc + w / 2), (dc - w / 2) - lo)
            if dc - w / 2 < lo or dc + w / 2 > hi or gap < CAR_WIDTH + 2 * a.fair_gap:
                continue
            c = tr.frenet_to_xy(s, dc)
            yaw = float(tr.yaw_at(s) + rng.uniform(-0.3, 0.3))
            corners = box_corners(float(c[0]), float(c[1]), yaw, depth, w)
            if tr.occ.wall_distance(corners[:, 0], corners[:, 1]).min() < 0.05:
                continue
            draw_box(img, meta, corners)
            boxes.append({'s': s, 'dc': dc, 'w': w, 'depth': depth, 'yaw': yaw,
                          'corners': corners.tolist()})
        layouts.append(write(f'layout_{k:03d}', img, boxes))
    with open(os.path.join(out, 'layouts.json'), 'w') as f:
        json.dump({'start': start, 'lap_length': tr.length, 'layouts': layouts}, f, indent=1)
    print(f'{len(layouts)} maps in {out} (start pose {np.round(start, 3).tolist()})')


if __name__ == '__main__':
    main()
