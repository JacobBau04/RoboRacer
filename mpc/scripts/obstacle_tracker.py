"""LiDAR obstacle detection and per-lap memory (pure numpy, no ROS).

Idea: race1 is walled all round, so a LiDAR point that lands inside the track and clearly
away from every map wall is an obstacle. Neighbouring such points are grouped into
clusters, each cluster becomes a box in track coordinates (s, d), and boxes seen in 2
scans in a row are confirmed and remembered until the car has passed them.

All positions here are in the map frame; track coordinates come from track_model.TrackModel.
u = arc length relative to the car (s - s_car, wrapped), so the lap start never matters.
"""
from dataclasses import dataclass, field, replace

import numpy as np


@dataclass
class DetectionParams:
    range_min: float = 0.05          # m, shorter returns are invalid (the SICK reports 0 for no echo)
    range_max: float = 10.0          # m, ignore returns further than this
    lookahead: float = 10.0          # m of raceline ahead of the car searched for obstacles
    lookbehind: float = 1.0          # m of raceline behind the car (keeps boxes alive while passing)
    wall_margin: float = 0.15        # m, points closer than this to a map wall count as wall
    wall_margin_per_m: float = 0.01  # m per m of range, for heading error of the pose
    corridor_tol: float = 0.05       # m, points must be this far inside the wall offsets
    cluster_gap: float = 0.10        # m, max gap between neighbouring points of one object ...
    cluster_gap_per_beam: float = 2.5  # ... plus this many times the beam spacing (r * dtheta)
    cluster_max_beam_gap: int = 3    # beams; a bigger index gap always splits a cluster
    min_points: int = 3              # smaller clusters are noise
    occlusion_gap: float = 0.20      # m, a neighbour beam this much shorter hides the cluster's edge
    occlusion_extend: float = 0.30   # m, planning box grows this much on a hidden side
    min_depth: float = 0.30          # m, box length along s at least this (back side is unseen) ...
    pad_min_u: float = 1.0           # m, ... but only for clusters this far ahead (seen from the front)
    # memory
    assoc_gate: float = 0.30         # m, a cluster this close to a box belongs to it
    confirm_hits: int = 2            # scans in a row needed to confirm an obstacle
    see_through_tol: float = 0.35    # m, beam passes this far beyond a box -> nothing there
    see_through_scans: int = 4       # scans of "nothing there" that delete a confirmed box
    rear_extent: float = 0.39        # m, car body behind the pose point
    passed_margin: float = 0.30      # m, drop a box once it is this far behind the car's rear
    max_tracks: int = 30


@dataclass
class Cluster:
    u_min: float
    u_max: float
    d_min: float
    d_max: float
    n: int
    r_min: float
    xy: np.ndarray = field(repr=False)
    hidden_lo: bool = False          # the -d (right) edge is hidden behind something closer
    hidden_hi: bool = False          # the +d (left) edge is hidden


@dataclass
class Obstacle:
    id: int
    s_min: float          # absolute arc length of the near end
    s_len: float          # extent along s (>= min_depth)
    d_min: float
    d_max: float
    hits: int = 1
    confirmed: bool = False
    see_through: int = 0
    seen_this_scan: bool = False
    hidden_lo: bool = False          # right edge hidden in the latest view (planning box grows)
    hidden_hi: bool = False          # left edge hidden in the latest view
    first_seen_u: float = float('nan')   # u of the near end when first detected (diagnostics)
    confirmed_u: float = float('nan')    # u of the near end when confirmed (diagnostics)


@dataclass
class ScanView:
    """What the see-through test needs from one scan."""
    laser_xy: np.ndarray        # (2,) laser position at mid scan
    laser_yaw: float
    ranges: np.ndarray          # all beams
    angle_min: float
    angle_inc: float


def scan_points(ranges, angle_min, angle_inc, beam_x, beam_y, beam_yaw, params):
    """Valid returns -> (beam index, range, map xy). beam_* are per-beam laser poses
    (arrays of len(ranges)) or scalars."""
    r = np.asarray(ranges, float)
    idx = np.arange(len(r))
    valid = np.isfinite(r) & (r >= params.range_min) & (r <= params.range_max)
    th = np.asarray(beam_yaw, float) + angle_min + idx * angle_inc
    x = np.asarray(beam_x, float) + r * np.cos(th)
    y = np.asarray(beam_y, float) + r * np.sin(th)
    xy = np.column_stack((np.broadcast_to(x, r.shape), np.broadcast_to(y, r.shape)))
    return idx[valid], r[valid], xy[valid]


def obstacle_points(track, xy, r, s_car, params):
    """Which points are obstacle candidates. Returns keep mask and (u, d) of all points."""
    if len(xy) == 0:
        e = np.zeros(0)
        return np.zeros(0, bool), e, e
    s, d, ok = track.project(xy, s_hint=s_car,
                             window=(-params.lookbehind - 1.0, params.lookahead + 1.0))
    u = track.rel_s(s, s_car)
    lo, hi = track.bounds_at(s)
    wall = track.occ.wall_distance(xy[:, 0], xy[:, 1])
    margin = params.wall_margin + params.wall_margin_per_m * r
    keep = (ok & (u > -params.lookbehind) & (u < params.lookahead)
            & (d > lo + params.corridor_tol) & (d < hi - params.corridor_tol)
            & (wall >= margin))
    return keep, u, d


def _hidden(all_ranges, beam, step, r_edge, params):
    """Is the cluster edge at `beam` hidden: does the next valid beam outward hit something
    clearly closer?"""
    if all_ranges is None:
        return False
    n = len(all_ranges)
    for k in range(1, 4):
        b = beam + step * k
        if b < 0 or b >= n:
            return False
        rb = all_ranges[b]
        if np.isfinite(rb) and rb >= params.range_min:
            return bool(rb < r_edge - params.occlusion_gap)
    return False


def cluster_points(beam_idx, r, xy, u, d, angle_inc, params, all_ranges=None):
    """Group points of neighbouring beams into objects (input in scan order).
    all_ranges (the whole scan) lets each cluster know whether an edge is hidden."""
    if len(beam_idx) == 0:
        return []
    gaps = np.diff(beam_idx)
    step = np.hypot(*np.diff(xy, axis=0).T)
    thr = params.cluster_gap + params.cluster_gap_per_beam * r[1:] * angle_inc * gaps
    split = (gaps > params.cluster_max_beam_gap) | (step > thr)
    starts = np.concatenate([[0], np.nonzero(split)[0] + 1])
    ends = np.concatenate([starts[1:], [len(beam_idx)]])
    out = []
    for a, b in zip(starts, ends):
        if b - a < params.min_points:
            continue
        uu, dd = u[a:b], d[a:b]
        u_min = float(uu.min())
        u_max = float(uu.max())
        if u_min > params.pad_min_u:
            u_max = max(u_max, u_min + params.min_depth)
        c = Cluster(u_min, u_max, float(dd.min()), float(dd.max()), int(b - a),
                    float(r[a:b].min()), xy[a:b].copy())
        # beams count counter-clockwise: the first beam is the object's right (-d) edge
        # for objects ahead of the car, the last beam its left (+d) edge
        c.hidden_lo = _hidden(all_ranges, int(beam_idx[a]), -1, float(r[a]), params)
        c.hidden_hi = _hidden(all_ranges, int(beam_idx[b - 1]), +1, float(r[b - 1]), params)
        out.append(c)
    return out


def detect(track, ranges, angle_min, angle_inc, beam_x, beam_y, beam_yaw, s_car, params):
    """One scan -> clusters (obstacle candidates) and the number of kept points."""
    idx, r, xy = scan_points(ranges, angle_min, angle_inc, beam_x, beam_y, beam_yaw, params)
    keep, u, d = obstacle_points(track, xy, r, s_car, params)
    clusters = cluster_points(idx[keep], r[keep], xy[keep], u[keep], d[keep], angle_inc, params,
                              np.asarray(ranges, float))
    return clusters, int(keep.sum())


def _overlap(a0, a1, b0, b1, gate):
    return (a0 - gate <= b1) and (b0 - gate <= a1)


class ObstacleTracker:
    """Confirms clusters seen in consecutive scans and remembers them until passed."""

    def __init__(self, track, params):
        self.track = track
        self.p = params
        self.obstacles = []
        self._next_id = 1

    def reset(self):
        self.obstacles = []

    def _u(self, ob, s_car):
        u0 = float(self.track.rel_s(ob.s_min, s_car))
        return u0, u0 + ob.s_len

    def _set_u(self, ob, u_min, u_max, s_car):
        ob.s_min = float(self.track.wrap_s(s_car + u_min))
        ob.s_len = float(u_max - u_min)

    def update(self, clusters, s_car, view=None):
        """Feed one scan's clusters. Returns the confirmed obstacles."""
        p = self.p
        for ob in self.obstacles:
            ob.seen_this_scan = False
        new = []
        for c in clusters:
            best, best_gap = None, None
            for ob in self.obstacles:
                u0, u1 = self._u(ob, s_car)
                if (_overlap(c.u_min, c.u_max, u0, u1, p.assoc_gate)
                        and _overlap(c.d_min, c.d_max, ob.d_min, ob.d_max, p.assoc_gate)):
                    gap = abs(0.5 * (c.u_min + c.u_max) - 0.5 * (u0 + u1)) \
                        + abs(0.5 * (c.d_min + c.d_max) - 0.5 * (ob.d_min + ob.d_max))
                    if best is None or gap < best_gap:
                        best, best_gap = ob, gap
            if best is None:
                ob = Obstacle(self._next_id, 0.0, 0.0, c.d_min, c.d_max,
                              hidden_lo=c.hidden_lo, hidden_hi=c.hidden_hi,
                              first_seen_u=c.u_min)
                self._next_id += 1
                self._set_u(ob, c.u_min, c.u_max, s_car)
                ob.seen_this_scan = True
                new.append(ob)
                continue
            u0, u1 = self._u(best, s_car)
            # the box grows as more of the object is seen (union)
            self._set_u(best, min(u0, c.u_min), max(u1, c.u_max), s_car)
            best.d_min = min(best.d_min, c.d_min)
            best.d_max = max(best.d_max, c.d_max)
            best.hidden_lo, best.hidden_hi = c.hidden_lo, c.hidden_hi    # latest view
            if not best.seen_this_scan:
                best.hits += 1
                best.seen_this_scan = True
            best.see_through = 0
            if not best.confirmed and best.hits >= p.confirm_hits:
                best.confirmed = True
                best.confirmed_u = c.u_min
        kept = []
        for ob in self.obstacles:
            u0, u1 = self._u(ob, s_car)
            if u1 + p.passed_margin < -p.rear_extent:
                continue                      # passed
            if not ob.seen_this_scan:
                if not ob.confirmed:
                    continue                  # tentative: must be seen in consecutive scans
                if view is not None and self._seen_through(ob, s_car, view):
                    ob.see_through += 1
                    if ob.see_through >= p.see_through_scans:
                        continue              # nothing there (moved or a false detection)
            kept.append(ob)
        for ob in new:
            if ob.hits >= p.confirm_hits:     # confirm_hits 1 confirms at once
                ob.confirmed = True
                ob.confirmed_u = ob.first_seen_u
        self.obstacles = self._merge(kept + new, s_car)[:p.max_tracks]
        return self.confirmed()

    def confirmed(self):
        return [ob for ob in self.obstacles if ob.confirmed]

    def planning_boxes(self):
        """Confirmed boxes, grown on edges that are hidden behind something closer (the
        object may continue there). The safety check uses the seen boxes (confirmed())."""
        e = self.p.occlusion_extend
        return [replace(ob, d_min=ob.d_min - e * ob.hidden_lo, d_max=ob.d_max + e * ob.hidden_hi)
                for ob in self.confirmed()]

    def _merge(self, obs, s_car):
        """Merge boxes that overlap (one object seen as two clusters)."""
        obs = sorted(obs, key=lambda o: (not o.confirmed, o.id))
        out = []
        for ob in obs:
            u0, u1 = self._u(ob, s_car)
            target = None
            for m in out:
                m0, m1 = self._u(m, s_car)
                if _overlap(u0, u1, m0, m1, 0.0) and _overlap(ob.d_min, ob.d_max, m.d_min, m.d_max, 0.0):
                    target = m
                    break
            if target is None:
                out.append(ob)
                continue
            m0, m1 = self._u(target, s_car)
            self._set_u(target, min(m0, u0), max(m1, u1), s_car)
            target.d_min = min(target.d_min, ob.d_min)
            target.d_max = max(target.d_max, ob.d_max)
            target.hidden_lo = target.hidden_lo or ob.hidden_lo
            target.hidden_hi = target.hidden_hi or ob.hidden_hi
            target.hits = max(target.hits, ob.hits)
            target.confirmed = target.confirmed or ob.confirmed
            target.seen_this_scan = target.seen_this_scan or ob.seen_this_scan
        return out

    def _seen_through(self, ob, s_car, view):
        """True if the beams toward the box's nearest point reach well beyond it."""
        u0, u1 = self._u(ob, s_car)
        uu, dd = np.meshgrid(np.linspace(u0, u1, 5), np.linspace(ob.d_min, ob.d_max, 3))
        pts = self.track.frenet_to_xy(s_car + uu.ravel(), dd.ravel())
        rel = pts - view.laser_xy
        rng = np.hypot(rel[:, 0], rel[:, 1])
        k = int(np.argmin(rng))
        expected = rng[k]
        bearing = float(np.arctan2(rel[k, 1], rel[k, 0]) - view.laser_yaw)
        bearing = (bearing + np.pi) % (2 * np.pi) - np.pi
        b = int(round((bearing - view.angle_min) / view.angle_inc))
        n = len(view.ranges)
        if b < 1 or b > n - 2 or expected > self.p.range_max - 0.5:
            return False                       # outside the field of view or too far
        meas = np.asarray(view.ranges[b - 1:b + 2], float)
        meas = meas[np.isfinite(meas) & (meas >= self.p.range_min)]
        if len(meas) == 0:
            return False                       # no echo: no information
        return bool(meas.min() > expected + self.p.see_through_tol)
