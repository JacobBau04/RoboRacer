"""Track geometry for obstacle avoidance (pure numpy, no ROS).

The raceline is a closed curve. A point is described by track coordinates (s, d):
s = arc length along the raceline [m], d = signed sideways offset, + = left of the
driving direction. The walls come from the occupancy map (the closed-off race1 map).
"""
import os

import numpy as np
import yaml
from PIL import Image
from scipy.interpolate import CubicSpline
from scipy.ndimage import distance_transform_edt


class OccupancyMap:
    """Walls of a ROS map (yaml + image) and the distance from any point to the nearest wall."""

    def __init__(self, yaml_path):
        with open(yaml_path) as f:
            meta = yaml.safe_load(f)
        img = np.array(Image.open(os.path.join(os.path.dirname(yaml_path), meta['image'])))
        if img.ndim == 3:
            img = img[..., 0]
        img = img.astype(float)
        if meta.get('negate', 0):
            img = 255.0 - img
        occupancy = (255.0 - img) / 255.0
        # occupied or unknown = wall (not drivable); rows flipped so row i grows with +y
        self.wall = (occupancy >= meta['free_thresh'])[::-1].copy()
        self.res = float(meta['resolution'])
        self.ox, self.oy = float(meta['origin'][0]), float(meta['origin'][1])
        self.h, self.w = self.wall.shape
        # distance [m] from each cell centre to the nearest wall cell centre
        self.dist = distance_transform_edt(~self.wall) * self.res

    def wall_distance(self, x, y):
        """Distance [m] from (x, y) to the nearest wall surface (bilinear; 0 outside the map)."""
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        fx = (x - self.ox) / self.res - 0.5
        fy = (y - self.oy) / self.res - 0.5
        c0 = np.floor(fx).astype(int)
        r0 = np.floor(fy).astype(int)
        tx, ty = fx - c0, fy - r0
        inside = (c0 >= 0) & (c0 < self.w - 1) & (r0 >= 0) & (r0 < self.h - 1)
        c0 = np.clip(c0, 0, self.w - 2)
        r0 = np.clip(r0, 0, self.h - 2)
        d = self.dist
        val = ((1 - tx) * (1 - ty) * d[r0, c0] + tx * (1 - ty) * d[r0, c0 + 1]
               + (1 - tx) * ty * d[r0 + 1, c0] + tx * ty * d[r0 + 1, c0 + 1])
        # cell centre -> wall surface is about half a cell closer
        return np.where(inside, np.maximum(val - 0.5 * self.res, 0.0), 0.0)

    def is_wall(self, x, y):
        c = np.floor((np.asarray(x) - self.ox) / self.res).astype(int)
        r = np.floor((np.asarray(y) - self.oy) / self.res).astype(int)
        out = (c < 0) | (c >= self.w) | (r < 0) | (r >= self.h)
        return out | self.wall[np.clip(r, 0, self.h - 1), np.clip(c, 0, self.w - 1)]


def wrap_angle(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


class TrackModel:
    """Raceline (closed loop) + walls.

    waypoints: (N, >=4) array of x, y, yaw, v (the MPC's waypoint csv).
    occ: OccupancyMap of the walls.
    """

    def __init__(self, waypoints, occ, max_wall_ray=3.0):
        wp = np.asarray(waypoints, float)
        self.xy = wp[:, :2].copy()
        self.v = wp[:, 3].copy()
        self.n = len(wp)
        seg = np.roll(self.xy, -1, axis=0) - self.xy           # segment i -> i+1 (closing too)
        self.seg_len = np.hypot(seg[:, 0], seg[:, 1])
        self.s = np.concatenate([[0.0], np.cumsum(self.seg_len[:-1])])
        self.length = float(self.seg_len.sum())
        self.ds = float(np.mean(self.seg_len))
        # direction from the points themselves (consistent with the positions)
        fwd = np.roll(self.xy, -1, axis=0) - np.roll(self.xy, 1, axis=0)
        self.yaw = np.arctan2(fwd[:, 1], fwd[:, 0])
        self.tangent = np.column_stack((np.cos(self.yaw), np.sin(self.yaw)))
        self.normal = np.column_stack((-self.tangent[:, 1], self.tangent[:, 0]))   # left
        dyaw = wrap_angle(np.roll(self.yaw, -1) - np.roll(self.yaw, 1))
        self.kappa = dyaw / (self.seg_len + np.roll(self.seg_len, 1))
        self.occ = occ
        # smooth closed curve through the points (positions and normals between points are
        # then continuous, so curvature of offset paths can be measured numerically)
        self._s_ext = np.concatenate([self.s, [self.length]])
        loop = np.vstack([self.xy, self.xy[:1]])
        self._spx = CubicSpline(self._s_ext, loop[:, 0], bc_type='periodic')
        self._spy = CubicSpline(self._s_ext, loop[:, 1], bc_type='periodic')
        # wall distance along the normal, to the left (+d) and to the right (-d)
        self.d_left = self._ray(self.normal, max_wall_ray)
        self.d_right = self._ray(-self.normal, max_wall_ray)

    def _ray(self, direction, max_r, step=0.01):
        r = np.full(self.n, max_r)
        hit = np.zeros(self.n, bool)
        for t in np.arange(0.0, max_r, step):
            p = self.xy + direction * t
            h = self.occ.is_wall(p[:, 0], p[:, 1]) & ~hit
            r[h] = t
            hit |= h
        return r

    # ---- s helpers -------------------------------------------------------------------
    def wrap_s(self, s):
        return np.mod(s, self.length)

    def rel_s(self, s, s_ref):
        """Signed arc length from s_ref to s, in [-L/2, L/2)."""
        L = self.length
        return np.mod(np.asarray(s) - s_ref + 0.5 * L, L) - 0.5 * L

    def _interp(self, s, values):
        ext = np.concatenate([values, values[:1]])
        return np.interp(self.wrap_s(s), self._s_ext, ext)

    def index_at(self, s):
        """Index of the raceline point at or just before s."""
        return np.searchsorted(self.s, self.wrap_s(s), side='right') - 1

    def v_at(self, s):
        return self._interp(s, self.v)

    def kappa_at(self, s):
        return self._interp(s, self.kappa)

    def bounds_at(self, s):
        """(d_right_wall, d_left_wall) at s: wall offsets along the normal, right one negative."""
        return -self._interp(s, self.d_right), self._interp(s, self.d_left)

    def frenet_to_xy(self, s, d):
        sw = self.wrap_s(np.asarray(s, float))
        d = np.asarray(d, float)
        tx, ty = self._spx(sw, 1), self._spy(sw, 1)
        tn = np.hypot(tx, ty)
        return np.stack([self._spx(sw) - d * ty / tn, self._spy(sw) + d * tx / tn], axis=-1)

    def yaw_at(self, s):
        sw = self.wrap_s(np.asarray(s, float))
        return np.arctan2(self._spy(sw, 1), self._spx(sw, 1))

    # ---- projection ----------------------------------------------------------------
    def project(self, xy, s_hint=None, window=(-3.0, 12.0)):
        """Track coordinates of points.

        xy: (M, 2). With s_hint, only raceline points within s_hint + window are searched,
        so a point is never matched to a different part of the track (e.g. across the inner
        wall). Returns s (M,), d (M,), ok (M,): ok is False where the nearest raceline point
        is at the edge of the search window (the point lies beyond it).
        """
        xy = np.atleast_2d(np.asarray(xy, float))
        if s_hint is None:
            idx = np.arange(self.n)
        else:
            i0 = self.index_at(s_hint + window[0])
            count = int(np.ceil((window[1] - window[0]) / self.ds)) + 1
            idx = (i0 + np.arange(count)) % self.n
        P = self.xy[idx]
        diff = xy[:, None, :] - P[None, :, :]
        d2 = np.einsum('mkj,mkj->mk', diff, diff)
        k = np.argmin(d2, axis=1)
        j = idx[k]
        rel = xy - self.xy[j]
        along = np.einsum('mj,mj->m', rel, self.tangent[j])
        d = np.einsum('mj,mj->m', rel, self.normal[j])
        s = self.wrap_s(self.s[j] + along)
        if s_hint is None:
            ok = np.ones(len(xy), bool)
        else:
            ok = (k > 0) & (k < len(idx) - 1)
        return s, d, ok


def load_track(waypoints_csv, map_yaml):
    wp = np.loadtxt(waypoints_csv, delimiter=',')
    return TrackModel(wp, OccupancyMap(map_yaml))
