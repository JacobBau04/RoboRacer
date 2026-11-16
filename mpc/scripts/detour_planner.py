"""Detour planner (pure numpy, no ROS).

When a confirmed obstacle box is in the way, the raceline is replaced over a stretch by a
smooth sideways offset d(s): it leaves the raceline, passes the obstacle at a chosen
side and margin, and rejoins. A handful of candidates (side x margin x swerve length,
plus combinations for obstacles close together) are checked for clearance to the
obstacles and walls and for the steering limit; each feasible one gets a speed profile
and the fastest is used. A plan is kept until it becomes infeasible or a clearly faster
one appears, so the car does not switch sides back and forth.

If nothing fits with the normal margins, a "tight" level is tried: smaller margins, more
steering, slow through the gap (a hit only costs time, getting stuck costs the race).
Only if that fails too, the car brakes to a stop before the obstacle.

Coordinates: u = arc length relative to the car's pose point along the raceline, d =
sideways offset (+ left). The MPC tracks the pose point (the particle filter's pose,
which is the LiDAR position on the car); `front` and `rear` give the car body around it.
"""
from dataclasses import dataclass, field
import itertools
import math

import numpy as np
import osqp
import scipy.sparse as sp

from track_model import wrap_angle


@dataclass
class PlannerParams:
    half_width: float = 0.155        # m, half the car's width
    front: float = 0.19              # m, body ahead of the pose point (car: LiDAR pose)
    rear: float = 0.1               # WAS 0.39 m, body behind the pose point
    # obstacle gaps below: +0.08 m over the originals (2026-11-04; +0.15 made slaloms infeasible)
    margin_min: float = 0.20         # m, smallest normal gap between car side and obstacle
    margin_comfort: float = 0.38     # m, from this gap on there is no speed cap
    v_gap_min: float = 1.5           # m/s, speed cap at margin_min (linear up to comfort)
    wall_margin: float = 0.08        # m, smallest normal gap car side to wall during a detour
    kappa_max: float = 1.17          # 1/m, 0.85 x steering limit tan(0.4189)/0.324
    tight_margin: float = 0.13       # m, tight level: obstacle gap
    tight_wall_margin: float = 0.03  # m, tight level: wall gap
    tight_kappa: float = 1.31        # 1/m, tight level: 0.95 x steering limit
    tight_speed: float = 1.5         # WAS: 1.5 m/s, tight level: speed through the gap / sharp steering
    crawl_margin: float = 0.10       # m, crawl level (last resort before stopping): obstacle gap
    crawl_wall_margin: float = 0.01  # m, crawl level: wall gap
    crawl_kappa: float = 1.35        # 1/m, crawl level: almost the full steering limit (0.985 x)
    crawl_speed: float = 0.6         # m/s, crawl level speed through the gap / sharp steering
    a_lat_transition: float = 3.0    # m/s^2, sideways acceleration used to size a swerve
    a_lat_max: float = 3.43          # WAS: 5.89 m/s^2, cornering limit in the speed profile (mu 0.6)
    a_accel: float = 3.0             # m/s^2
    a_brake: float = 3.0             # m/s^2
    v_max: float = 4.0               # m/s (MPC MAX_SPEED)
    horizon: float = 10.0            # m, obstacles further ahead are not planned for yet
    max_obstacles: int = 3           # planned jointly (up to 2^n side combinations)
    L_min: float = 0.8               # m, shortest swerve
    L_scales: tuple = (0.6, 1.0, 1.6)        # swerve length multipliers tried
    tight_L_scales: tuple = (0.5, 0.8, 1.2)
    margin_levels: int = 3           # margins tried: margin_min .. margin_comfort
    du: float = 0.10                 # m, grid step along s
    behind: float = 1.0              # m of plan kept behind the car (reference continuity)
    mpc_back: float = 0.27           # m, the MPC tracks the rear axle, this far behind the pose point
    v_start_min: float = 0.8         # m/s, a plan never starts slower than this unless braking
                                     # for a stop (below ~0.5 m/s the VESC stutters and stalls)
    replan_gain: float = 0.10        # s, a new plan must be this much faster to replace one
    flip_penalty: float = 1.0        # s, cost of passing a planned obstacle on the other side
    brake_tolerance: float = 1.0     # m/s, start-speed excess over the profile still accepted
    stop_buffer: float = 1.50        # m, stop this far before the obstacle if nothing fits
                                     # (room to swerve from standstill once a crawl path appears)
    hold_extra: float = 0.02         # m, holds aim this much beyond the margin (heading, rounding)
    active_slack: float = 0.05       # m, a plan under way may lose this much margin (box grows)
    qp_weights: tuple = (0.3,)       # corridor QP: weight on |d| (small = early, gentle swerve)
    qp_wall_extra: float = 0.03      # m, extra wall margin in the QP (its walls are approximate)


@dataclass
class Limits:
    level: int
    margin_min: float
    wall_margin: float
    kappa_max: float
    margins: tuple
    scales: tuple


@dataclass
class ObstacleBox:
    id: int
    u0: float      # near end, relative to the car
    u1: float      # far end
    lo: float      # d_min
    hi: float      # d_max


@dataclass
class Candidate:
    key: tuple     # ((obstacle id, side), ..., (level, margin index, length scale))
    u: np.ndarray
    d: np.ndarray
    feasible: bool = False
    reason: str = ''
    time: float = math.inf
    cost: float = math.inf
    margins: dict = field(default_factory=dict)
    xy: np.ndarray = None
    v: np.ndarray = None
    speed_capped: bool = False
    wall_clear: float = math.inf     # smallest gap car side to wall ahead [m]


@dataclass
class Plan:
    s_car: float
    u: np.ndarray        # grid, u[0] = start of the plan (behind the car)
    d: np.ndarray
    xy: np.ndarray       # (len(u), 2) map positions of the pose point
    v: np.ndarray        # planned speed
    stop: bool
    key: tuple
    margins: dict
    time: float
    level: int = 0       # 0 normal, 1 tight, 2 stop
    n_candidates: int = 0
    n_feasible: int = 0


def quintic(d0, p0, q0, d1, L, t):
    """d(t), t in [0, L]: from value/slope/curvature (d0, p0, q0) to (d1, 0, 0)."""
    T = max(L, 1e-6)
    a2 = 0.5 * q0
    e0 = d1 - (d0 + p0 * T + a2 * T * T)
    e1 = -(p0 + 2 * a2 * T)
    e2 = -2 * a2
    # closed form of the 3x3 solve for a3, a4, a5
    a3 = (10 * e0 - 4 * e1 * T + 0.5 * e2 * T * T) / T ** 3
    a4 = (-15 * e0 + 7 * e1 * T - e2 * T * T) / T ** 4
    a5 = (6 * e0 - 3 * e1 * T + 0.5 * e2 * T * T) / T ** 5
    t = np.clip(t, 0.0, T)
    return d0 + t * (p0 + t * (a2 + t * (a3 + t * (a4 + t * a5))))


def eval_segments(u, segs, d_before):
    """segs: list of (u_s, u_e, d_s, p_s, q_s, d_e), in order and touching."""
    d = np.full(len(u), float(d_before))
    for (us, ue, ds, ps, qs, de) in segs:
        m = (u >= us) & (u <= ue)
        if m.any():
            d[m] = quintic(ds, ps, qs, de, ue - us, u[m] - us)
        d[u > ue] = de
    return d


class DetourPlanner:
    def __init__(self, track, params=None):
        self.track = track
        self.p = params or PlannerParams()
        self.active = None            # (s_abs grid, d grid, key)
        self.active_stop = False
        self._signature = None        # obstacles the active plan was made for
        tr = track
        clear = self._footprint_clearance(tr.xy, tr.yaw)
        self._clear_s = np.concatenate([tr.s, [tr.length]])
        self._clear_v = np.concatenate([clear, clear[:1]])

    def reset(self):
        self.active = None
        self.active_stop = False
        self._signature = None

    @staticmethod
    def _sig(obstacles):
        q = lambda x: int(round(x / 0.05))
        return tuple(sorted((o.id, q(o.s_min), q(o.s_len), q(o.d_min), q(o.d_max)) for o in obstacles))

    def limits(self, level):
        p = self.p
        if level == 0:
            m = tuple(np.linspace(p.margin_min, p.margin_comfort, p.margin_levels)) \
                if p.margin_levels > 1 else (p.margin_min,)
            return Limits(0, p.margin_min, p.wall_margin, p.kappa_max, m, p.L_scales)
        if level == 1:
            return Limits(1, p.tight_margin, p.tight_wall_margin, p.tight_kappa,
                          (p.tight_margin,), p.tight_L_scales)
        return Limits(2, p.crawl_margin, p.crawl_wall_margin, p.crawl_kappa,
                      (p.crawl_margin,), p.tight_L_scales)

    # ---- helpers ---------------------------------------------------------------------
    def swerve_length(self, dd, v):
        p = self.p
        return max(p.L_min, v * math.sqrt(5.77 * abs(dd) / p.a_lat_transition))

    def v_gap(self, margin):
        p = self.p
        if margin >= p.margin_comfort:
            return math.inf
        if margin < p.tight_margin - 1e-3:
            return p.crawl_speed
        if margin < p.margin_min - 1e-3:
            return p.tight_speed
        f = (margin - p.margin_min) / max(p.margin_comfort - p.margin_min, 1e-6)
        return p.v_gap_min + (p.v_max - p.v_gap_min) * min(max(f, 0.0), 1.0)

    def _raceline_clear(self, s):
        return np.interp(self.track.wrap_s(s), self._clear_s, self._clear_v)

    def _active_d(self, u, s_car):
        s_abs, d_abs, _ = self.active
        ua = self.track.rel_s(s_abs, s_car)
        # the stored grid is monotonic in u; outside it the plan is on the raceline
        return np.interp(u, ua, d_abs, left=d_abs[0], right=0.0)

    def _start_state(self, s_car):
        if self.active is None:
            return 0.0, 0.0, 0.0
        h = 0.1
        dm, d0, dp = self._active_d(np.array([-h, 0.0, h]), s_car)
        return float(d0), float((dp - dm) / (2 * h)), float((dp - 2 * d0 + dm) / h ** 2)

    def _footprint_clearance(self, xy, psi):
        """Smallest distance from the car's sides to a wall at each grid point."""
        p, occ = self.p, self.track.occ
        c, s = np.cos(psi), np.sin(psi)
        pts = np.concatenate([xy, xy + p.front * np.column_stack((c, s)),
                              xy - p.rear * np.column_stack((c, s))])
        wd = occ.wall_distance(pts[:, 0], pts[:, 1]).reshape(3, -1).min(axis=0)
        return wd - p.half_width

    @staticmethod
    def _geometry(xy):
        seg = np.diff(xy, axis=0)
        dl = np.hypot(seg[:, 0], seg[:, 1])
        head = np.unwrap(np.arctan2(seg[:, 1], seg[:, 0]))
        psi = np.concatenate([head, head[-1:]])
        psi[1:-1] = 0.5 * (head[:-1] + head[1:])
        kappa = np.zeros(len(xy))
        kappa[1:-1] = np.diff(head) / np.maximum(0.5 * (dl[:-1] + dl[1:]), 1e-6)
        return dl, psi, kappa

    # ---- candidates ------------------------------------------------------------------
    def _hold_need(self, margin):
        return self.p.half_width + margin + self.p.hold_extra

    def _side_possible(self, s_car, ob, side, lim):
        """Quick check: does the car fit between the box and the wall on this side?"""
        p = self.p
        s = s_car + np.linspace(ob.u0, ob.u1, 3)
        lo, hi = self.track.bounds_at(s)
        need = self._hold_need(lim.margin_min) + p.half_width + lim.wall_margin - 0.05
        room = (hi - ob.hi) if side > 0 else (ob.lo - lo)
        return bool(np.any(room >= need))

    def _holds(self, obs, sides, margin):
        """Hold intervals [a, b] with offset d for the chosen sides; None if impossible."""
        p = self.p
        raw = []
        need = self._hold_need(margin)
        for ob, side in zip(obs, sides):
            a = ob.u0 - p.front - 0.05
            b = ob.u1 + p.rear + 0.05
            if side > 0:      # pass left: d >= hi + half width + margin
                raw.append([a, b, ob.hi + need, math.inf])
            else:             # pass right: d <= lo - half width - margin
                raw.append([a, b, -math.inf, ob.lo - need])
        raw.sort(key=lambda h: h[0])
        merged = []
        for h in raw:         # overlapping holds must satisfy both obstacles at once
            if merged and h[0] <= merged[-1][1] + 0.05:
                m = merged[-1]
                m[1] = max(m[1], h[1])
                m[2] = max(m[2], h[2])
                m[3] = min(m[3], h[3])
            else:
                merged.append(list(h))
        holds = []
        for a, b, lo_req, hi_req in merged:
            if lo_req > hi_req:
                return None
            holds.append((a, b, float(min(max(0.0, lo_req), hi_req))))
        return holds

    def _segments(self, holds, start, v_plan, scale):
        d0, p0, q0 = start
        moving = abs(p0) > 1e-3 or abs(q0) > 1e-2
        segs = []
        cur_u, cur_d = 0.0, d0
        for j, (a, b, dh) in enumerate(holds):
            L = scale * self.swerve_length(dh - cur_d, v_plan)
            if j == 0:
                if moving:
                    segs.append((0.0, max(a, 0.3), d0, p0, q0, dh))
                else:
                    us = max(0.0, a - L)
                    segs.append((us, max(a, us + 0.3), d0, 0.0, 0.0, dh))
            else:
                gap = a - cur_u
                if abs(dh - cur_d) < 1e-6:
                    segs.append((cur_u, max(a, cur_u), cur_d, 0.0, 0.0, dh))
                else:
                    back = scale * self.swerve_length(cur_d, v_plan)
                    there = scale * self.swerve_length(dh, v_plan)
                    if cur_d != 0.0 and dh != 0.0 and gap > back + there + 0.5:
                        segs.append((cur_u, cur_u + back, cur_d, 0.0, 0.0, 0.0))
                        segs.append((cur_u + back, a - there, 0.0, 0.0, 0.0, 0.0))
                        segs.append((a - there, a, 0.0, 0.0, 0.0, dh))
                    else:
                        segs.append((cur_u, max(a, cur_u + 0.3), cur_d, 0.0, 0.0, dh))
            segs.append((segs[-1][1], max(b, segs[-1][1]), dh, 0.0, 0.0, dh))
            cur_u, cur_d = segs[-1][1], dh
        L = scale * self.swerve_length(cur_d, v_plan)
        segs.append((cur_u, cur_u + L, cur_d, 0.0, 0.0, 0.0))
        return segs

    def evaluate(self, cand, s_car, v_car, obs, lim, force=False):
        """Check a candidate and give it a speed profile. force: compute the speed profile
        even if a check fails (cand.reason is still set, cand.feasible stays False)."""
        p, tr = self.p, self.track
        u, d = cand.u, cand.d
        s = s_car + u
        xy = tr.frenet_to_xy(s, d)
        dl, psi, kappa = self._geometry(xy)
        cand.xy = xy
        ahead = u >= 0.0
        i0 = int(np.argmax(ahead))
        # obstacles (cheapest check first)
        dpsi = wrap_angle(psi - tr.yaw_at(s))
        hw = p.half_width + np.abs(np.sin(dpsi)) * max(p.front, p.rear)
        caps = []
        for ob in obs:
            over = ahead & (u + p.front >= ob.u0) & (u - p.rear <= ob.u1)
            if not over.any():
                cand.margins[ob.id] = math.inf
                continue
            gap = np.maximum(ob.lo - (d + hw), (d - hw) - ob.hi)[over]
            m = float(gap.min())
            cand.margins[ob.id] = m
            if m < lim.margin_min - 1e-3:
                cand.reason = cand.reason or f'obstacle {ob.id}'
                if not force:
                    return cand
            caps.append((over, self.v_gap(m)))
        # steering
        k_abs = np.abs(kappa)
        k_abs[0] = k_abs[-1] = 0.0
        if np.any(k_abs[ahead] > lim.kappa_max):
            cand.reason = cand.reason or 'steering'
            if not force:
                return cand
        # walls (relative to the raceline's own clearance where that is smaller)
        need_normal = np.minimum(p.wall_margin, self._raceline_clear(s) - 0.02)
        need = np.minimum(lim.wall_margin, need_normal)
        wall_clear = self._footprint_clearance(xy, psi)
        cand.wall_clear = float(wall_clear[ahead].min(initial=math.inf))
        if np.any((wall_clear < need)[ahead]):
            cand.reason = cand.reason or 'wall'
            if not force:
                return cand
        # speed profile
        v_lim = np.minimum(tr.v_at(s), p.v_max)
        v_lim = np.minimum(v_lim, np.sqrt(p.a_lat_max / np.maximum(k_abs, 1e-3)))
        sharp = ahead & (k_abs > p.kappa_max)
        if sharp.any():
            v_lim[sharp] = np.minimum(v_lim[sharp], p.tight_speed)
            cand.speed_capped = True
        sharper = ahead & (k_abs > p.tight_kappa)
        if sharper.any():
            v_lim[sharper] = np.minimum(v_lim[sharper], p.crawl_speed)
        close = ahead & (wall_clear < need_normal)       # closer to a wall than normally allowed
        if close.any():
            v_lim[close] = np.minimum(v_lim[close], p.crawl_speed)
            cand.speed_capped = True
        for over, cap in caps:
            if cap < math.inf:
                v_lim[over] = np.minimum(v_lim[over], cap)
                cand.speed_capped = True
        v = self._speed_profile(v_lim, dl, i0, v_car, allow_late=force)
        if v is None:
            cand.reason = cand.reason or 'too late to brake'
            return cand
        cand.v = v
        va = 0.5 * (v[i0:-1] + v[i0 + 1:])
        cand.time = float(np.sum(dl[i0:] / np.maximum(va, 0.05)))
        cand.feasible = not cand.reason
        return cand

    def _speed_profile(self, v_lim, dl, i0, v_car, allow_late=False):
        p = self.p
        # start the profile where the MPC tracks (rear axle), not at the pose point: otherwise
        # the MPC sits in the flat v_car part behind i0 and never speeds up again
        i0 = max(i0 - int(round(p.mpc_back / p.du)), 0)
        v = v_lim.copy()
        n = len(v)
        for i in range(n - 2, i0 - 1, -1):           # braking limit (backward)
            vb = math.sqrt(v[i + 1] * v[i + 1] + 2 * p.a_brake * dl[i])
            if vb < v[i]:
                v[i] = vb
        if v_car > v[i0] + p.brake_tolerance and not allow_late:
            return None
        v[i0] = min(v[i0], max(v_car, p.v_start_min))
        for i in range(i0, n - 1):                    # acceleration limit (forward)
            vf = math.sqrt(v[i] * v[i] + 2 * p.a_accel * dl[i])
            if vf < v[i + 1]:
                v[i + 1] = vf
        v[:i0] = v[i0]
        return v

    def _qp_shape(self, s_car, obs, sides, margin, lim, start, u, w0):
        """Smoothest offset d(s) inside the walls that clears the boxes on the given sides
        (corridor QP, OSQP). Walls and curvature are linearised; evaluate() checks exactly."""
        p, tr = self.p, self.track
        i0 = int(np.argmax(u >= 0.0))
        uu = u[i0:]
        m = len(uu)
        if m < 8:
            return None
        h = float(uu[1] - uu[0])
        s = s_car + uu
        lo_w, hi_w = tr.bounds_at(s)
        side = p.half_width + lim.wall_margin + p.qp_wall_extra
        lo = np.minimum(lo_w + side, 0.0)          # the raceline itself is always allowed
        hi = np.maximum(hi_w - side, 0.0)
        need = self._hold_need(margin)
        for ob, sd in zip(obs, sides):
            over = (uu + p.front >= ob.u0 - 0.05) & (uu - p.rear <= ob.u1 + 0.05)
            if sd > 0:
                lo[over] = np.maximum(lo[over], ob.hi + need)
            else:
                hi[over] = np.minimum(hi[over], ob.lo - need)
        lo[-3:] = np.maximum(lo[-3:], 0.0)         # back on the raceline at the end
        hi[-3:] = np.minimum(hi[-3:], 0.0)
        if np.any(lo > hi + 1e-6):
            return None
        kr = tr.kappa_at(s)
        D2 = sp.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(m - 2, m)) / h ** 2
        D3 = sp.diags([-1.0, 3.0, -3.0, 1.0], [0, 1, 2, 3], shape=(m - 3, m)) / h ** 3
        P = 2.0 * (D2.T @ D2 + 0.05 * D3.T @ D3 + w0 * sp.eye(m))
        d0, p0, _ = start
        start_rows = sp.csr_matrix(([1.0, -1.0, 1.0], ([0, 1, 1], [0, 0, 1])), shape=(2, m))
        # curvature of the offset path ~ kappa_r + kappa_r^2 d + d''
        K = D2 + sp.diags(kr[1:-1] ** 2, 1, shape=(m - 2, m))
        A = sp.vstack([sp.eye(m), start_rows, K]).tocsc()
        lo_all = np.concatenate([lo, [d0, p0 * h], -lim.kappa_max - kr[1:-1]])
        hi_all = np.concatenate([hi, [d0, p0 * h], lim.kappa_max - kr[1:-1]])
        solver = osqp.OSQP()
        solver.setup(sp.csc_matrix(P), np.zeros(m), A, lo_all, hi_all, verbose=False,
                     eps_abs=1e-4, eps_rel=1e-4, max_iter=4000, polishing=True)
        res = solver.solve(raise_error=False)
        if res.info.status not in ('solved', 'solved inaccurate') or res.x is None:
            return None
        d = np.empty(len(u))
        d[:i0] = d0
        d[i0:] = res.x
        return d

    def _candidates(self, s_car, v_car, obs, u, start, v_plan, lim):
        out = []
        options = [[sd for sd in (1, -1) if self._side_possible(s_car, ob, sd, lim)] for ob in obs]
        if any(len(o) == 0 for o in options):
            return out
        combos = list(itertools.product(*options))
        for sides in combos:
            for li, margin in enumerate(lim.margins):
                holds = self._holds(obs, sides, margin)
                if holds is None:
                    continue
                for scale in lim.scales:
                    key = tuple((o.id, sd) for o, sd in zip(obs, sides)) + ((lim.level, li, scale),)
                    d = eval_segments(u, self._segments(holds, start, v_plan, scale), start[0])
                    out.append(self.evaluate(Candidate(key, u, d), s_car, v_car, obs, lim))
        # corridor QP shapes, only where the fixed shapes fall short (none fits)
        if not any(c.feasible for c in out):
            for sides in combos:
                for li in sorted({0, len(lim.margins) - 1}):
                    for w0 in self.p.qp_weights:
                        d = self._qp_shape(s_car, obs, sides, lim.margins[li], lim, start, u, w0)
                        if d is None:
                            continue
                        key = tuple((o.id, sd) for o, sd in zip(obs, sides)) + ((lim.level, li, f'qp{w0}'),)
                        out.append(self.evaluate(Candidate(key, u, d), s_car, v_car, obs, lim))
        return out

    # ---- main entry ------------------------------------------------------------------
    def plan(self, s_car, v_car, obstacles):
        """obstacles: confirmed obstacles with .id .s_min .s_len .d_min .d_max.
        Returns a Plan, or None to follow the raceline."""
        p, tr = self.p, self.track
        obs = []
        for ob in obstacles:
            u0 = float(tr.rel_s(ob.s_min, s_car))
            u1 = u0 + ob.s_len
            if u1 + p.rear + 0.05 >= 0.0 and u0 < p.horizon:
                obs.append(ObstacleBox(ob.id, u0, u1, ob.d_min, ob.d_max))
        obs.sort(key=lambda o: o.u0)
        obs = obs[:p.max_obstacles]
        start = self._start_state(s_car)

        if not obs:
            if self.active is not None and not self.active_stop:
                u_end = float(tr.rel_s(self.active[0][-1], s_car))
                if u_end > 0.0 and (abs(start[0]) > 0.01 or abs(start[1]) > 1e-3):
                    # finish the detour already under way (its obstacle is gone or passed)
                    return self._keep_active(s_car, v_car, u_end)
            self.reset()
            return None

        v_plan = max(v_car, float(tr.v_at(s_car + max(obs[0].u0, 0.0))), 1.0)
        u_end = max(o.u1 for o in obs) + p.rear + 0.05 \
            + max(p.L_scales) * self.swerve_length(1.0, v_plan) + 0.5
        if self.active is not None:
            u_end = max(u_end, float(tr.rel_s(self.active[0][-1], s_car)))
        u = self._grid(s_car, u_end)

        act = None
        if self.active is not None and not self.active_stop:
            level = self.active[2][-1][0]
            lim = self.limits(level)
            lim.margin_min = max(lim.margin_min - p.active_slack, 0.6 * p.tight_margin)
            act = self.evaluate(Candidate(self.active[2], u, self._active_d(u, s_car)),
                                s_car, v_car, obs, lim)
            sig = self._sig(obstacles)
            if act.feasible and sig == self._signature:
                # nothing changed: keep the plan (only its speeds are recomputed)
                return Plan(s_car, u, act.d, act.xy, act.v, False, act.key, act.margins,
                            act.time, level, 1, 1)
        cands = self._candidates(s_car, v_car, obs, u, start, v_plan, self.limits(0))
        pool = [c for c in cands if c.feasible] + ([act] if act is not None and act.feasible else [])
        for level in (1, 2):
            if pool:
                break
            more = self._candidates(s_car, v_car, obs, u, start, v_plan, self.limits(level))
            cands += more
            pool = [c for c in more if c.feasible]
        if not pool and self.active is not None and not self.active_stop:
            # Nothing new fits. Keep the current plan at crawl speed if it still clears every
            # box and wall (even below the margin), or if the car is already alongside
            # (braking avoids nothing then). Stop only if it would actually touch something.
            c = self.evaluate(Candidate(self.active[2], u, self._active_d(u, s_car)),
                              s_car, v_car, obs, self.limits(2), force=True)
            clears = min(c.margins.values(), default=math.inf) > 0.0 and c.wall_clear > 0.0
            if c.v is not None and (clears or obs[0].u0 < p.front + 0.5):
                return Plan(s_car, u, c.d, c.xy, c.v, False, c.key, c.margins, c.time, 2,
                            len(cands) + 1, 0)
        if not pool:
            return self._stop_plan(s_car, v_car, obs, u, start, len(cands))

        act_sides = dict(self.active[2][:-1]) if self.active is not None and not self.active_stop else {}
        for c in pool:
            flips = sum(1 for oid, sd in c.key[:-1] if oid in act_sides and act_sides[oid] != sd)
            c.cost = c.time + p.flip_penalty * flips
        best = min(pool, key=lambda c: c.cost)
        if act is not None and act.feasible and best is not act \
                and not (best.cost < act.cost - p.replan_gain):
            best = act
        # follow the raceline if the best plan does not leave it and needs no speed cap
        if np.abs(best.d).max() < 1e-3 and not best.speed_capped:
            self.reset()
            return None
        self.active = (tr.wrap_s(s_car + u), best.d.copy(), best.key)
        self.active_stop = False
        self._signature = self._sig(obstacles)
        return Plan(s_car, u, best.d, best.xy, best.v, False, best.key, best.margins,
                    best.time, best.key[-1][0], len(cands) + (act is not None),
                    len(pool))

    def _grid(self, s_car, u_end):
        tr, p = self.track, self.p
        u_end = min(u_end, 0.45 * tr.length)       # stay within half a lap (s wraps)
        i_start = int(tr.index_at(s_car - p.behind))
        i_end = int(tr.index_at(s_car + u_end)) + 1
        u_first = float(tr.rel_s(tr.s[i_start % tr.n], s_car))
        u_last = float(tr.rel_s(tr.s[i_end % tr.n], s_car))
        if u_last <= u_first:
            u_last += tr.length
        n = int(math.ceil((u_last - u_first) / p.du)) + 1
        return np.linspace(u_first, u_last, n)

    def _keep_active(self, s_car, v_car, u_end):
        u = self._grid(s_car, max(u_end, 0.5))
        level = self.active[2][-1][0]
        c = self.evaluate(Candidate(self.active[2], u, self._active_d(u, s_car)),
                          s_car, v_car, [], self.limits(level))
        if not c.feasible:
            self.reset()
            return None
        return Plan(s_car, u, c.d, c.xy, c.v, False, c.key, c.margins, c.time, level, 1, 1)

    def _stop_plan(self, s_car, v_car, obs, u, start, n_cands):
        """Nothing fits: keep the current path and brake to a stop before the obstacle."""
        p, tr = self.p, self.track
        d = self._active_d(u, s_car) if self.active is not None else np.full(len(u), start[0])
        xy = tr.frenet_to_xy(s_car + u, d)
        dl, _, _ = self._geometry(xy)
        u_stop = obs[0].u0 - p.front - p.stop_buffer
        v_stop = np.sqrt(2.0 * p.a_brake * np.maximum(u_stop - u, 0.0))
        v_lim = np.minimum(np.minimum(tr.v_at(s_car + u), p.v_max), v_stop)
        i0 = int(np.argmax(u >= 0.0))
        v = self._speed_profile(v_lim, dl, i0, v_car, allow_late=True)
        v[u >= u_stop] = 0.0
        self.active = (tr.wrap_s(s_car + u), d.copy(), ('stop',))
        self.active_stop = True
        return Plan(s_car, u, d, xy, v, True, ('stop',), {}, math.inf, 2, n_cands, 0)
