#!/usr/bin/env python3
"""
make_waypoints.py: click out the reference trajectory for Lab 7 (MPC).

What it does
  Opens the Levine map, then the Spielberg map. You click waypoints, and a closed
  cubic spline is fitted through them. On Enter, the spline is resampled every
  0.1 m and written to mpc/waypoints/<track>.csv, one row per point, with
  columns  x, y, yaw, v  in the map frame:
    yaw = the spline's tangent direction (so the velocity points along the path)
    v   = speed from curvature (slower in tight turns), limited by accel/brake

Tracks (--track)   map              output
  levine           levine_blocked   waypoints/levine.csv      Lab 7 graded Levine map
  spielberg        Spielberg        waypoints/spielberg.csv   Lab 7 graded Spielberg map
  levine_cl        levine           waypoints/levine_cl.csv   open hallways (only when named)

How to run (from this folder, mpc/tools)
  python3 make_waypoints.py                     # levine, then spielberg
  python3 make_waypoints.py --track levine_cl   # one track only
  python3 make_waypoints.py --help              # speed limits, spacing, ...
  Afterwards, run colcon build so the CSVs are installed with the package.

Controls
  left-click     add a point (near the spline: inserted there; otherwise appended)
  drag           move a point
  right-click    delete a point
  Enter          save the CSV and go to the next track
  close window   skip the track without writing its CSV
  The toolbar's zoom/pan work too; click the tool again to turn it off and edit.

  The car drives the points in the order you click (0, 1, 2, ... back to 0).
  Arrows show that direction; the title says if it disagrees with the start pose.
  Red x's mark rows closer than --wall-margin to a wall.
  Your clicks are kept in tools/<track>_clicks.csv and reloaded on the next run
  (--fresh starts over).
"""
import argparse
import math
import os
from pathlib import Path

import matplotlib
# Use the Tk window backend: the pip-installed PyQt6 crashes on this VM's GNOME
# settings. Set MPLBACKEND (e.g. MPLBACKEND=QtAgg) to choose a different one.
if 'MPLBACKEND' not in os.environ:
    matplotlib.use('TkAgg')
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from scipy.interpolate import CubicSpline
from scipy.ndimage import binary_dilation

TOOLS_DIR = Path(__file__).resolve().parent
PKG_DIR = TOOLS_DIR.parent              # .../mpc
OUT_DIR = PKG_DIR / 'waypoints'         # installed by CMakeLists.txt
SRC_DIR = PKG_DIR.parents[1]            # .../sim_ws/src

# Track name (= output file name) -> map in f1tenth_gym_ros/maps and start pose (x, y, theta).
# The start pose is only used to check the driving direction.
TRACKS = {
    'levine': {'map': 'levine_blocked', 'start': (-12.0, 0.0, 0.0)},     # Lab 7 graded
    'spielberg': {'map': 'Spielberg', 'start': (14.59, 3.92, -2.877)},   # Lab 7 graded
    'levine_cl': {'map': 'levine', 'start': (-12.0, 0.0, 0.0)},          # open hallways
    # Real car: SLAM map in ../waypoints (run with --maps-dir ../waypoints).
    # SLAM puts the origin where mapping started; check 'start' matches your driving direction.
    'real_levine': {'map': 'levine_2nd', 'start': (-15.2, 17.4, - np.pi / 4)},
}


def default_maps_dir():
    local = SRC_DIR / 'f1tenth_gym_ros' / 'maps'
    if local.is_dir():
        return local
    try:
        from ament_index_python.packages import get_package_share_directory
        share = Path(get_package_share_directory('f1tenth_gym_ros')) / 'maps'
        if share.is_dir():
            return share
    except Exception:
        pass
    return local


def overlays_for(track, maps_dir):
    """Guide lines drawn under your clicks: (label, csv, delimiter, x col, y col, style)."""
    if track.startswith('levine'):
        # Lab 5's centerline, found in any sibling package (not this one).
        hits = sorted(p for p in SRC_DIR.glob('*/*/waypoints/levine_centerline.csv')
                      if PKG_DIR not in p.parents)
        return [('Lab 5 centerline', hits[0], ',', 0, 1, '--')] if hits else []
    if track != 'spielberg':
        return []  # e.g. real_levine: no guide lines for a real-car map
    return [('centerline', maps_dir / 'Spielberg_centerline.csv', ',', 0, 1, '--'),
            ('raceline', maps_dir / 'Spielberg_raceline.csv', ';', 1, 2, ':')]


class MapImage:
    """The occupancy-grid PNG plus its yaml: pixel <-> map-frame conversion and wall lookup."""

    def __init__(self, yaml_path, wall_margin):
        meta = yaml.safe_load(yaml_path.read_text())
        img = mpimg.imread(str(yaml_path.parent / meta['image']))
        if img.dtype == np.uint8:
            img = img / 255.0
        if img.ndim == 3:
            img = img[..., :3].mean(axis=2)
        self.gray = img
        self.res = float(meta['resolution'])
        self.ox, self.oy = float(meta['origin'][0]), float(meta['origin'][1])
        self.h, self.w = img.shape

        # Same rule as the ROS map server: dark pixels are occupied.
        occupancy = img if meta.get('negate', 0) else 1.0 - img
        walls = occupancy > float(meta.get('occupied_thresh', 0.65))
        r = int(math.ceil(wall_margin / self.res))
        if r > 0:  # grow the walls by the margin
            yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
            walls = binary_dilation(walls, structure=(xx ** 2 + yy ** 2 <= r * r))
        self.blocked = walls

    @property
    def extent(self):
        # Row 0 of the image is the top (largest y) of the map.
        return [self.ox, self.ox + self.w * self.res, self.oy, self.oy + self.h * self.res]

    def is_blocked(self, xy):
        col = np.floor((xy[:, 0] - self.ox) / self.res).astype(int)
        row = self.h - 1 - np.floor((xy[:, 1] - self.oy) / self.res).astype(int)
        inside = (col >= 0) & (col < self.w) & (row >= 0) & (row < self.h)
        hit = np.ones(len(xy), dtype=bool)          # off the map counts as a wall
        hit[inside] = self.blocked[row[inside], col[inside]]
        return hit


def build_trajectory(points, p):
    """Closed cubic spline through the clicked points, resampled every ~p.ds metres.

    Returns None if there are fewer than 3 distinct points.
    """
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    if len(pts) < 3:
        return None
    closed = np.vstack([pts, pts[:1]])              # loop back to the first point
    chords = np.hypot(*np.diff(closed, axis=0).T)
    if np.any(chords < 1e-3):                       # two points on top of each other
        return None
    # Parameterise by chord length so the spline speed is roughly uniform.
    knots = np.concatenate([[0.0], np.cumsum(chords)])
    spline = CubicSpline(knots, closed, bc_type='periodic')

    # Measure arc length on a dense sampling, then pick evenly spaced rows.
    # The lab's calc_interpolated_ref_trajectory assumes even spacing.
    t_dense = np.linspace(0.0, knots[-1], max(2000, int(knots[-1] / 0.01)))
    xy_dense = spline(t_dense)
    s_dense = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(xy_dense, axis=0).T))])
    length = s_dense[-1]
    n = max(int(round(length / p.ds)), 3)
    ds = length / n                                 # so the last gap (back to row 0) matches too
    t = np.interp(np.arange(n) * ds, s_dense, t_dense)

    x, y = spline(t).T
    dx, dy = spline(t, 1).T
    ddx, ddy = spline(t, 2).T
    # Heading = tangent of the spline, so the velocity vector follows the path.
    # TODO (later): choose the path and headings by optimisation instead, e.g. direct
    #   collocation over (x, y, yaw, v) with the kinematic bicycle model, track-width
    #   constraints and a minimum-lap-time cost.
    yaw = np.arctan2(dy, dx)
    kappa = (dx * ddy - dy * ddx) / np.maximum((dx * dx + dy * dy) ** 1.5, 1e-9)
    v = speed_profile(kappa, ds, p)
    return dict(x=x, y=y, yaw=yaw, v=v, kappa=kappa, ds=ds, length=length,
                knots=knots, t_dense=t_dense, xy_dense=xy_dense)


def speed_profile(kappa, ds, p):
    """Curvature speed limit, then limit how fast speed may rise and fall along the loop."""
    # Lateral acceleration v^2 * |kappa| must stay under a_lat.
    v = np.sqrt(p.a_lat / np.maximum(np.abs(kappa), 1e-6))
    v = np.clip(v, p.v_min, p.v_max)
    n = len(v)
    # v_next^2 <= v^2 + 2 a ds. Go round twice so the limit carries across row 0.
    for _ in range(2):
        for i in range(n):                          # forward: acceleration
            j = (i + 1) % n
            v[j] = min(v[j], math.sqrt(v[i] ** 2 + 2 * p.a_accel * ds))
    for _ in range(2):
        for i in range(n - 1, -1, -1):              # backward: braking
            j = (i - 1) % n
            v[j] = min(v[j], math.sqrt(v[i] ** 2 + 2 * p.a_brake * ds))
    return v


class Editor:
    """One map window: click, drag and delete points; Enter saves."""

    PICK_PX = 10  # how close (screen pixels) a click must be to grab a point or the spline

    def __init__(self, track, p):
        self.track, self.p = track, p
        self.cfg = TRACKS[track]
        self.out_csv = OUT_DIR / f'{track}.csv'
        self.clicks_csv = TOOLS_DIR / f'{track}_clicks.csv'
        self.map = MapImage(Path(p.maps_dir) / f"{self.cfg['map']}.yaml", p.wall_margin)
        self.points = [] if p.fresh else self._load_clicks()
        self.traj = None
        self.drag = None
        self.saved = False
        self.labels = []
        self.quiver = None

        self.fig, self.ax = plt.subplots(figsize=(11, 9))
        self.fig.canvas.manager.set_window_title(
            f"make_waypoints: {track} (map {self.cfg['map']} -> {self.out_csv.name})")
        ax = self.ax
        ax.imshow(self.map.gray, cmap='gray', vmin=0, vmax=1, extent=self.map.extent,
                  origin='upper', interpolation='nearest')
        ax.set_aspect('equal')
        ax.set_xlabel('x [m] (map frame)')
        ax.set_ylabel('y [m]')

        guide_xy = self._draw_overlays()
        sx, sy, sth = self.cfg['start']
        ax.annotate('', xy=(sx + 2 * math.cos(sth), sy + 2 * math.sin(sth)), xytext=(sx, sy),
                    arrowprops=dict(arrowstyle='-|>', color='red', lw=2), zorder=6)
        ax.plot([sx], [sy], 'o', color='red', ms=6, label='start pose', zorder=6)
        ax.legend(loc='upper right', fontsize=8)

        # Start zoomed on the track instead of the whole map image.
        view = np.vstack(guide_xy + [np.asarray(self.points).reshape(-1, 2), [[sx, sy]]])
        lo, hi = view.min(axis=0) - 3.0, view.max(axis=0) + 3.0
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])

        norm = Normalize(p.v_min, p.v_max)
        self.lc = LineCollection([], cmap='viridis', norm=norm, linewidths=3, zorder=3)
        ax.add_collection(self.lc)
        self.fig.colorbar(ScalarMappable(norm=norm, cmap='viridis'), ax=ax,
                          fraction=0.04, label='v [m/s]')
        self.wall_artist, = ax.plot([], [], 'x', color='red', ms=6, mew=2, zorder=4)
        self.pt_artist, = ax.plot([], [], 'o', ms=7, mfc='white', mec='black', zorder=5)
        self.first_artist, = ax.plot([], [], 'o', ms=10, mfc='lime', mec='black', zorder=5)
        self.fig.text(0.01, 0.01,
                      'left-click: add / insert   drag: move   right-click: delete   '
                      'Enter: save & next   close window: skip',
                      fontsize=9)

        c = self.fig.canvas
        c.mpl_connect('button_press_event', self.on_press)
        c.mpl_connect('motion_notify_event', self.on_motion)
        c.mpl_connect('button_release_event', self.on_release)
        c.mpl_connect('key_press_event', self.on_key)
        c.mpl_connect('close_event', self.on_close)
        self.refresh()

    # ---------- setup helpers ----------
    def _load_clicks(self):
        if not self.clicks_csv.exists():
            return []
        pts = np.loadtxt(self.clicks_csv, delimiter=',', comments='#', ndmin=2)
        print(f'{self.track}: reloaded {len(pts)} points from {self.clicks_csv.name}')
        return pts.tolist()

    def _save_clicks(self):
        if self.points:
            np.savetxt(self.clicks_csv, np.asarray(self.points), fmt='%.4f', delimiter=', ',
                       header='clicked waypoints x, y (map frame), in driving order')
        elif self.clicks_csv.exists():
            self.clicks_csv.unlink()

    def _draw_overlays(self):
        xys = []
        for label, path, delim, xc, yc, style in overlays_for(self.track, Path(self.p.maps_dir)):
            try:
                data = np.loadtxt(path, delimiter=delim, comments='#', ndmin=2)
            except OSError:
                print(f'(overlay not found: {path})')
                continue
            xy = data[:, [xc, yc]]
            self.ax.plot(xy[:, 0], xy[:, 1], style, lw=1.2, alpha=0.7, label=label, zorder=2)
            xys.append(xy)
        return xys

    # ---------- mouse / keyboard ----------
    def _toolbar_busy(self):
        """True while the zoom or pan tool is active, so those clicks don't add points."""
        tb = getattr(self.fig.canvas, 'toolbar', None)
        mode = getattr(tb, 'mode', '')
        return bool(getattr(mode, 'value', mode))

    def _nearest_point(self, event):
        if not self.points:
            return None
        disp = self.ax.transData.transform(np.asarray(self.points))
        d = np.hypot(disp[:, 0] - event.x, disp[:, 1] - event.y)
        i = int(np.argmin(d))
        return i if d[i] < self.PICK_PX else None

    def _insert_index(self, event):
        """Index in self.points where a click near the spline belongs, or None if not near."""
        if self.traj is None:
            return None
        disp = self.ax.transData.transform(self.traj['xy_dense'])
        d = np.hypot(disp[:, 0] - event.x, disp[:, 1] - event.y)
        k = int(np.argmin(d))
        if d[k] > self.PICK_PX:
            return None
        # The spline between points i and i+1 covers knots[i] <= t < knots[i+1].
        seg = int(np.searchsorted(self.traj['knots'], self.traj['t_dense'][k], side='right')) - 1
        return min(seg, len(self.points) - 1) + 1

    def on_press(self, event):
        if event.inaxes is not self.ax or event.xdata is None or self._toolbar_busy():
            return
        i = self._nearest_point(event)
        if event.button == 1:
            if i is not None:
                self.drag = i
                return
            pt = [event.xdata, event.ydata]
            j = self._insert_index(event)
            if j is None:
                self.points.append(pt)
            else:
                self.points.insert(j, pt)
            self.refresh()
        elif event.button == 3 and i is not None:
            del self.points[i]
            self.refresh()

    def on_motion(self, event):
        if self.drag is None or event.inaxes is not self.ax or event.xdata is None:
            return
        self.points[self.drag] = [event.xdata, event.ydata]
        self.refresh()

    def on_release(self, event):
        self.drag = None

    def on_key(self, event):
        if event.key != 'enter':
            return
        if self.traj is None:
            print('Need at least 3 separate points before saving.')
            return
        self.save()
        self.saved = True
        plt.close(self.fig)

    def on_close(self, event):
        self._save_clicks()
        if not self.saved:
            print(f'{self.track}: window closed without Enter, {self.out_csv.name} not written.')

    # ---------- drawing and saving ----------
    def direction_check(self, t):
        """Compare the path heading near the start pose with the start pose heading."""
        sx, sy, sth = self.cfg['start']
        d = np.hypot(t['x'] - sx, t['y'] - sy)
        i = int(np.argmin(d))
        if d[i] > 3.0:
            return False, f'path is {d[i]:.1f} m from the start pose'
        err = abs(math.atan2(math.sin(t['yaw'][i] - sth), math.cos(t['yaw'][i] - sth)))
        if err < math.pi / 2:
            return True, 'direction OK'
        return False, 'direction REVERSED vs start pose'

    def refresh(self):
        pts = np.asarray(self.points, dtype=float).reshape(-1, 2)
        self.pt_artist.set_data(pts[:, 0], pts[:, 1])
        self.first_artist.set_data(pts[:1, 0], pts[:1, 1])
        for txt in self.labels:
            txt.remove()
        self.labels = [self.ax.annotate(str(i), (x, y), xytext=(6, 6), textcoords='offset points',
                                        fontsize=8, zorder=6)
                       for i, (x, y) in enumerate(pts)]
        if self.quiver is not None:
            self.quiver.remove()
            self.quiver = None

        self.traj = build_trajectory(pts, self.p)
        t = self.traj
        if t is None:
            self.lc.set_segments([])
            self.wall_artist.set_data([], [])
            self.ax.set_title(f'{self.track}: {len(pts)} points (need at least 3)', color='black')
        else:
            xy = np.column_stack([t['x'], t['y']])
            closed = np.vstack([xy, xy[:1]])
            self.lc.set_segments(np.stack([closed[:-1], closed[1:]], axis=1))
            self.lc.set_array(t['v'])
            hit = self.map.is_blocked(xy)
            self.wall_artist.set_data(xy[hit, 0], xy[hit, 1])
            step = max(1, int(round(3.0 / t['ds'])))   # one arrow every ~3 m
            self.quiver = self.ax.quiver(t['x'][::step], t['y'][::step],
                                         np.cos(t['yaw'][::step]), np.sin(t['yaw'][::step]),
                                         color='black', scale=45, width=0.003, zorder=4)
            ok, msg = self.direction_check(t)
            lap = float(np.sum(t['ds'] / t['v']))
            self.ax.set_title(
                f"{self.track}: {len(pts)} points | {t['length']:.1f} m | lap ~{lap:.1f} s | "
                f"{msg} | {int(hit.sum())} rows near walls",
                color='black' if ok and not hit.any() else 'red')
        self.fig.canvas.draw_idle()

    def save(self):
        t, p = self.traj, self.p
        data = np.column_stack([t['x'], t['y'], t['yaw'], t['v']])
        header = ('x, y, yaw, v\n'
                  f'tools/make_waypoints.py: {len(data)} rows every {t["ds"]:.3f} m, '
                  f'v_max={p.v_max}, v_min={p.v_min}, a_lat={p.a_lat}, '
                  f'a_accel={p.a_accel}, a_brake={p.a_brake}')
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        np.savetxt(self.out_csv, data, fmt='%.4f', delimiter=', ', header=header)
        self._save_clicks()
        lap = float(np.sum(t['ds'] / t['v']))
        print(f'{self.track}: wrote {len(data)} rows to {self.out_csv} '
              f'({t["length"]:.1f} m, v {t["v"].min():.2f}-{t["v"].max():.2f} m/s, lap ~{lap:.1f} s)')
        ok, msg = self.direction_check(t)
        if not ok:
            print(f'  WARNING: {msg}')
        n_hit = int(self.map.is_blocked(data[:, :2]).sum())
        if n_hit:
            print(f'  WARNING: {n_hit} rows are within {p.wall_margin} m of a wall')

# to run: python3 make_waypoints --track levine --v-max 5.5 --v-min 2.0
def main():
    ap = argparse.ArgumentParser(
        description='Click waypoints on a map and save a Lab 7 reference trajectory (x, y, yaw, v).',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--track', choices=['all', *TRACKS], default='all',
                    help='which track to edit (writes waypoints/<track>.csv); '
                         '"all" does levine (levine_blocked map) then spielberg')
    ap.add_argument('--ds', type=float, default=0.1, help='spacing between output rows [m]')
    ap.add_argument('--v-max', type=float, default=10.0, help='top speed [m/s]')
    ap.add_argument('--v-min', type=float, default=1.5, help='lowest speed, even in tight turns [m/s]')
    ap.add_argument('--a-lat', type=float, default=3.0, help='max lateral acceleration in turns [m/s^2]')
    ap.add_argument('--a-accel', type=float, default=6.0, help='max speed-up [m/s^2]')
    ap.add_argument('--a-brake', type=float, default=6.0, help='max slow-down [m/s^2]')
    ap.add_argument('--wall-margin', type=float, default=0.25,
                    help='flag rows closer than this to a wall [m]')
    ap.add_argument('--maps-dir', default=None, help='folder with the map .yaml/.png files '
                    '(if not given: f1tenth_gym_ros/maps next to this repo)')
    ap.add_argument('--fresh', action='store_true', help='ignore previously saved clicks')
    p = ap.parse_args()
    if p.maps_dir is None:
        p.maps_dir = default_maps_dir()

    for track in (['levine', 'spielberg'] if p.track == 'all' else [p.track]):
        editor = Editor(track, p)  # keep a reference: matplotlib holds callbacks weakly
        plt.show()                 # blocks until the window is closed (Enter closes it)
        del editor


if __name__ == '__main__':
    main()
