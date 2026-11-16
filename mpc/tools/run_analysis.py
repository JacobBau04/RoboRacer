#!/usr/bin/env python3
"""Turn a recorded run (mpc/data/<run>/raw.npz + meta.json) into a diagnosis.

run_recorder.py calls this after Ctrl+C. You can also re-run it on any old
run, e.g. after changing a metric here (no ROS needed, only numpy + matplotlib):

    python3 run_analysis.py ../data/2026-10-29_19-02-11

Writes into the run folder:
    summary.txt   read this first (also printed to the terminal)
    laps.csv      one row per complete lap
    segments.csv  per corner / straight: time lost, speed, tracking error, hits
    hits.csv      one row per wall hit
    results.json  every number above, machine readable
    run.png       track map, speed and error along the track, timing
and appends one line to data/runs.csv so runs can be compared side by side.

Definitions
    lateral error e   signed distance from the raceline (+ = car is left of it)
    corner            raceline curvature above CORNER_KAPPA (turn radius < ~7 m),
                      smoothed over ~1 m; everything else is a straight
    ideal lap         time to drive the raceline at exactly the waypoint speeds
    lap               time between two crossings of the raceline's first waypoint
                      (the sim only times laps on maps with a centerline file)
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

CORNER_KAPPA = 0.15      # 1/m, corner if smoothed |curvature| is above this
SMOOTH_M = 1.0           # m, curvature smoothing window
MIN_SEGMENT_M = 1.0      # m, shorter corners/straights get merged into neighbours
DT_MAX = 0.05            # s, cap per-sample time weight (bridges recording gaps)
RTF_WARN = 0.97          # sim slower than real time below this

INDEX_FIELDS = [
    'run', 'note', 'track', 'laps', 'lap_mean_s', 'lap_std_s', 'lap_best_s',
    'best_clean_lap_s', 'ideal_lap_s', 'hits', 'avg_speed_mps', 'e_mean_m', 'e_corner_mean_m', 'e_max_m',
    'solve_mean_ms', 'control_period_mean_ms', 'solver_failures', 'realtime_factor',
    'platform', 'odom_scale', 'kp_accel', 'kp_brake', 'cmd_delay_s', 'battery_v_start',
    'delay_comp_s', 'steer_osc_rms', 'steer_delay_s', 'pose_jumps',
    'waypoints', 'waypoints_header', 'MAX_SPEED', 'TK', 'DTK', 'Qk', 'Qfk', 'Rk', 'Rdk']
# columns before the real-car fields were added (for reading old runs.csv files)
LEGACY_INDEX_FIELDS = [f for f in INDEX_FIELDS if f not in (
    'platform', 'odom_scale', 'kp_accel', 'kp_brake', 'cmd_delay_s', 'battery_v_start',
    'delay_comp_s', 'steer_osc_rms', 'steer_delay_s', 'pose_jumps')]


# ---------------------------------------------------------------- track geometry

def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def circular_runs(mask):
    """[(start, length, value), ...] runs of equal values around a closed loop."""
    n = len(mask)
    change = np.nonzero(mask != np.roll(mask, 1))[0]
    if len(change) == 0:
        return [(0, n, bool(mask[0]))]
    return [(int(st), int((change[(k + 1) % len(change)] - st) % n or n), bool(mask[st]))
            for k, st in enumerate(change)]


def build_track(wp):
    """Raceline as a closed loop of segments i -> i+1, split into corners and straights."""
    xy = wp[:, :2]
    if np.hypot(*(xy[0] - xy[-1])) < 1e-6:          # drop a duplicated closing point
        wp, xy = wp[:-1], xy[:-1]
    n = len(xy)
    A = xy
    D = np.roll(xy, -1, axis=0) - xy
    seg_len = np.hypot(D[:, 0], D[:, 1])
    s_start = np.r_[0.0, np.cumsum(seg_len)[:-1]]
    v = wp[:, 3]
    v_seg = 0.5 * (v + np.roll(v, -1))
    heading = np.arctan2(D[:, 1], D[:, 0])
    kappa = wrap(np.roll(heading, -1) - heading) / np.maximum(seg_len, 1e-6)
    w = max(1, int(round(SMOOTH_M / np.median(seg_len)))) | 1     # odd window
    pad = w // 2
    kappa_s = np.convolve(np.r_[kappa[-pad:], kappa, kappa[:pad]], np.ones(w) / w, 'valid') \
        if pad else kappa

    # corner mask, then remove runs shorter than MIN_SEGMENT_M (shortest first)
    mask = np.abs(kappa_s) > CORNER_KAPPA
    for _ in range(50):
        runs = circular_runs(mask)
        if len(runs) <= 1:
            break
        lengths = [seg_len[(st + np.arange(ln)) % n].sum() for st, ln, _ in runs]
        k = int(np.argmin(lengths))
        if lengths[k] >= MIN_SEGMENT_M:
            break
        st, ln, val = runs[k]
        mask[(st + np.arange(ln)) % n] = not val

    # order segments in driving order starting at the run containing waypoint 0
    runs = circular_runs(mask)
    first = next(i for i, (st, ln, _) in enumerate(runs) if (0 - st) % n < ln)
    runs = runs[first:] + runs[:first]
    seg_id = np.empty(n, dtype=int)
    segments, n_corner, n_straight = [], 0, 0
    for k, (st, ln, is_corner) in enumerate(runs):
        idx = (st + np.arange(ln)) % n
        seg_id[idx] = k
        if is_corner:
            n_corner += 1
            name = f'C{n_corner}'
        else:
            n_straight += 1
            name = f'S{n_straight}'
        turn = float(np.sum(kappa[idx] * seg_len[idx]))       # total heading change
        segments.append({
            'name': name, 'kind': 'corner' if is_corner else 'straight',
            'idx': idx, 's0': float(s_start[idx[0]]),
            's1': float(s_start[idx[-1]] + seg_len[idx[-1]]),
            'length_m': float(seg_len[idx].sum()),
            'ref_time_s': float(np.sum(seg_len[idx] / np.maximum(v_seg[idx], 0.1))),
            'v_ref_min': float(v[idx].min()), 'turn_deg': float(np.degrees(turn))})
    return {'wp': wp, 'n': n, 'A': A, 'D': D, 'seg_len': seg_len, 's_start': s_start,
            'L': float(seg_len.sum()), 'v': v, 'heading': heading, 'kappa': kappa_s,
            'seg_id': seg_id, 'segments': segments, 'corner_wp': mask}


def project(track, P, chunk=1000):
    """Nearest raceline point for each sample: segment index, fraction, signed error."""
    A, D, l2 = track['A'], track['D'], np.maximum(track['seg_len'] ** 2, 1e-12)
    m = len(P)
    j_all = np.empty(m, dtype=int)
    t_all = np.empty(m)
    e_all = np.empty(m)
    for c in range(0, m, chunk):
        rel = P[c:c + chunk, None, :] - A[None, :, :]                 # (k, n, 2)
        t = np.clip((rel * D[None]).sum(-1) / l2[None], 0.0, 1.0)
        d2 = ((rel - t[..., None] * D[None]) ** 2).sum(-1)
        j = np.argmin(d2, axis=1)
        r = np.arange(len(j))
        relj = rel[r, j]
        cross = D[j, 0] * relj[:, 1] - D[j, 1] * relj[:, 0]
        j_all[c:c + chunk] = j
        t_all[c:c + chunk] = t[r, j]
        e_all[c:c + chunk] = np.sign(cross) * np.sqrt(d2[r, j])
    return j_all, t_all, e_all


# ---------------------------------------------------------------- small helpers

def stats(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {'mean': None, 'std': None, 'p95': None, 'max': None, 'n': 0}
    return {'mean': float(x.mean()), 'std': float(x.std()), 'p95': float(np.percentile(x, 95)),
            'max': float(x.max()), 'n': int(len(x))}


def wmean(x, w):
    w = w * np.isfinite(x)
    return float(np.sum(np.nan_to_num(x) * w) / np.sum(w)) if np.sum(w) > 0 else None


def fmt(x, nd=2, unit=''):
    return '-' if x is None else f'{x:.{nd}f}{unit}'


def optional(raw, name, cols):
    """An array that older recordings (or the other platform) don't have -> empty."""
    return raw[name] if name in raw.files else np.zeros((0, cols))


# ---------------------------------------------------------------- real-car checks

def odom_scale(t, x, y, v, mask, step_s=0.2):
    """Distance the particle filter saw / distance the reported speed integrates to.
    On a step_s grid, so particle-filter jitter between messages doesn't add distance.
    1.00 = the wheel speed (speed_to_erpm_gain) matches how far the car really moved."""
    if len(t) < 2:
        return None
    g = np.arange(t[0], t[-1], step_s)
    if len(g) < 3:
        return None
    gx, gy, gv = np.interp(g, t, x), np.interp(g, t, y), np.interp(g, t, v)
    ok = (np.interp(g, t, mask.astype(float)) > 0.99) & (gv > 0.5)
    sel = ok[1:] & ok[:-1]
    if sel.sum() < 50:
        return None
    dist = np.hypot(np.diff(gx), np.diff(gy))[sel].sum()
    reported = (0.5 * (gv[1:] + gv[:-1]) * step_s)[sel].sum()
    return float(dist / reported)


def steering_response(odom, drive, wb, dt=0.02, delays=np.arange(0.0, 0.62, 0.02)):
    """How the car turns for a steering command: fits
        yaw_rate(t) = gain * v * tan(steer_cmd(t - delay)) / WB
    with the yaw rate taken from the pose's yaw. delay = everything between sending
    /drive and the turn showing up in the pose (servo, tires, particle filter): the
    value for mpc_node's delay_comp. gain 1 = the car turns as much as the MPC's model.
    Only samples above 1 m/s are used. None if too little data."""
    if len(odom) < 50 or len(drive) < 50:
        return None
    t, yaw, v = odom[:, 0], np.unwrap(odom[:, 3]), odom[:, 4]
    g = np.arange(t[0] + 1.0, t[-1] - 1.0, dt)
    if len(g) < 200:
        return None
    k = 7                                              # centred 0.14 s smoothing, no lag
    yw = np.convolve(np.interp(g, t, yaw), np.ones(k) / k, 'same')
    rate = np.gradient(yw, dt)
    vg = np.interp(g, t, v)
    m = (vg > 1.0) & (np.abs(rate) < 3.0)              # |rate| < 3: drop pose jumps
    m[:k] = m[-k:] = False
    if m.sum() < 200:
        return None
    best = None
    for d in delays:
        i = np.clip(np.searchsorted(drive[:, 0], g - d, side='right') - 1, 0, len(drive) - 1)
        pred = vg * np.tan(drive[i, 2]) / wb
        den = np.sum(pred[m] ** 2)
        if den <= 0:
            continue
        gain = float(np.sum(rate[m] * pred[m]) / den)
        res = float(np.mean((rate[m] - gain * pred[m]) ** 2))
        if best is None or res < best['_res']:
            best = {'_res': res, 'delay_s': float(d), 'gain': gain,
                    'r2': float(1 - res / np.var(rate[m])), 'n': int(m.sum())}
    if best is None:
        return None
    best.pop('_res')
    # steering trim: mean command while the car drives straight (0 = servo centred)
    i = np.clip(np.searchsorted(drive[:, 0], g - best['delay_s'], side='right') - 1,
                0, len(drive) - 1)
    straight = m & (np.abs(rate) < 0.05)
    best['straight_steer_rad'] = float(drive[i[straight], 2].mean()) if straight.sum() > 50 else None
    return best


def steering_oscillation(odom, drive, dt=0.05, window_s=2.0):
    """Size and frequency of the steering wiggle: the steering command minus its 2 s moving
    average (what's left is oscillation, not the turns of the track), while moving."""
    if len(drive) < 100 or len(odom) < 50:
        return None
    g = np.arange(drive[0, 0], drive[-1, 0], dt)
    sg = np.interp(g, drive[:, 0], drive[:, 2])
    k = int(window_s / dt)
    hp = sg - np.convolve(np.r_[[sg[0]] * (k // 2), sg, [sg[-1]] * (k - 1 - k // 2)],
                          np.ones(k) / k, 'valid')
    m = np.interp(g, odom[:, 0], odom[:, 4]) > 0.3
    if m.sum() < 100:
        return None
    f = np.fft.rfftfreq(len(hp), dt)
    power = np.abs(np.fft.rfft(hp * m)) ** 2
    band = (f > 0.2) & (f < 5.0)
    return {'rms_rad': float(hp[m].std()),
            'freq_hz': float(f[band][np.argmax(power[band])]) if band.any() else None}


def pose_jumps(odom):
    """Pose messages that jump further than the car could have moved (bad localization)."""
    if len(odom) < 2:
        return 0, None
    step = np.hypot(np.diff(odom[:, 1]), np.diff(odom[:, 2]))
    allowed = np.maximum(0.3, 3 * np.abs(odom[1:, 4]) * np.diff(odom[:, 0]) + 0.1)
    return int(np.sum(step > allowed)), float(step.max())


def speed_response(cmd, vel, dt=0.02, delays=np.arange(0.0, 0.32, 0.02), e_max=1.0):
    """How the car's speed follows the speed command: fits
        a(t) = kp * (v_cmd(t - delay) - v(t))
    with one kp for speeding up and one for braking (the same form as the kp in
    mpc_node.py, which are the sim's values), and the delay that fits best.
    Only speed errors below e_max are used: larger ones hit the acceleration/current limit,
    which would bias kp low (checked on sim runs: 4.67 / 18.9 fitted vs the sim's
    4.755 / 19.02 with e_max = 1 m/s; 3.9 / 18.7 without the limit).
    cmd: rows (t, v_cmd, from_mpc); vel: rows (t, measured v). None if too little data."""
    if len(cmd) < 50 or len(vel) < 50:
        return None
    g = np.arange(max(cmd[0, 0], vel[0, 0]) + 0.5, min(cmd[-1, 0], vel[-1, 0]), dt)
    if len(g) < 200:
        return None
    v = np.interp(g, vel[:, 0], vel[:, 1])
    k = 5                                             # 0.1 s moving average before d/dt
    v = np.convolve(np.r_[[v[0]] * (k // 2), v, [v[-1]] * (k // 2)], np.ones(k) / k, 'valid')
    a = np.gradient(v, dt)
    best = None
    for d in delays:
        i = np.searchsorted(cmd[:, 0], g - d, side='right') - 1
        valid = i >= 0
        i = np.clip(i, 0, len(cmd) - 1)
        fresh = (g - d - cmd[i, 0]) < 0.2             # mux drops commands older than 0.2 s
        e = cmd[i, 1] - v
        use = valid & fresh & (cmd[i, 2] > 0) & (v > 0.3) & (np.abs(e) > 0.05) & (np.abs(e) < e_max)
        acc, brk = use & (e > 0), use & (e < 0)
        if acc.sum() < 20 or brk.sum() < 20:
            continue
        kp_a = float(np.sum(a[acc] * e[acc]) / np.sum(e[acc] ** 2))
        kp_b = float(np.sum(a[brk] * e[brk]) / np.sum(e[brk] ** 2))
        res = np.sum((a[acc] - kp_a * e[acc]) ** 2) + np.sum((a[brk] - kp_b * e[brk]) ** 2)
        tot = np.sum((a[use] - a[use].mean()) ** 2)
        if best is None or res < best['_res']:
            best = {'_res': res, 'delay_s': float(d), 'kp_accel': kp_a, 'kp_brake': kp_b,
                    'r2': float(1 - res / tot) if tot > 0 else None,
                    'n_accel': int(acc.sum()), 'n_brake': int(brk.sum())}
    if best is None:
        return None
    best.pop('_res')
    moving = v > 0.3
    best['accel_p95_mps2'] = float(np.percentile(a[moving], 95)) if moving.any() else None
    best['decel_p95_mps2'] = float(-np.percentile(a[moving], 5)) if moving.any() else None
    return best


# ---------------------------------------------------------------- analysis

def analyze(run_dir):
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / 'meta.json').read_text())
    raw = np.load(run_dir / 'raw.npz')
    odom, drive, solve = raw['odom'], raw['drive'], raw['solve']
    collisions, sim_laps = raw['collisions'], raw['sim_laps']
    platform = meta.get('platform', 'sim')     # runs recorded before the real-car mode = sim
    real = platform == 'real'
    # real car: /ackermann_cmd (t, speed, steer, from_mpc), /odom wheel speed, /sensors/core
    cmd = optional(raw, 'cmd', 4)
    wheel = optional(raw, 'wheel', 3)
    vesc = optional(raw, 'vesc', 6)
    if not real and len(cmd) == 0:
        # the sim has no mux: every /drive reaches the car
        cmd = np.c_[drive, np.ones(len(drive))]

    track = build_track(np.loadtxt(run_dir / 'waypoints.csv', delimiter=',', ndmin=2))
    segs = track['segments']
    mpc_cfg = (meta.get('mpc') or {}).get('mpc_config', {})
    dtk = mpc_cfg.get('DTK', 0.1)

    # --- per-sample quantities (odom: t, x, y, yaw, vx, vy)
    t, x, y, yaw = (odom[:, k] for k in range(4))
    v = np.hypot(odom[:, 4], odom[:, 5])
    j, frac, e = project(track, odom[:, 1:3])
    nxt = (j + 1) % track['n']
    s = track['s_start'][j] + frac * track['seg_len'][j]
    v_ref = track['v'][j] + frac * (track['v'][nxt] - track['v'][j])
    head_err = wrap(yaw - track['heading'][j])
    seg_of = track['seg_id'][j]
    is_corner = track['corner_wp'][j]
    dt = np.clip(np.r_[np.diff(t), 0.0], 0.0, DT_MAX)

    # --- laps, timed here: from one crossing of the raceline's first waypoint
    # (s = 0) to the next. The bit before the first crossing is the out lap.
    # Unwrapping s gives total distance driven, so reversing over the line
    # can't count a lap twice.
    ds = np.diff(s)
    ds[ds < -track['L'] / 2] += track['L']
    ds[ds > track['L'] / 2] -= track['L']
    lap_index = np.floor((s[0] + np.r_[0.0, np.cumsum(ds)]) / track['L']).astype(int) \
        if len(s) else np.array([], int)
    best_so_far = np.maximum.accumulate(lap_index) if len(s) else lap_index
    crossings = t[1:][best_so_far[1:] > best_so_far[:-1]]
    laps = []
    for k in range(len(crossings) - 1):
        laps.append({'lap': k + 1, 'lap_time_s': float(crossings[k + 1] - crossings[k]),
                     't_start': float(crossings[k]),
                     't_end': float(crossings[k + 1])})
    valid_laps = laps

    # analysis window: all valid laps, or the whole run when no lap finished
    if valid_laps:
        in_win = np.zeros(len(t), bool)
        for lp in valid_laps:
            in_win |= (t >= lp['t_start']) & (t < lp['t_end'])
        window = f'{len(valid_laps)} complete lap(s)'
    else:
        in_win = np.ones(len(t), bool)
        window = 'whole run (no complete lap)'
    # real car: judge the MPC only while it drove. While the joystick drives, the MPC keeps
    # publishing /drive, but the car isn't following it.
    mpc_drove = np.ones(len(t), bool)
    if real and len(cmd):
        k = np.searchsorted(cmd[:, 0], t, side='right') - 1
        kk = np.clip(k, 0, len(cmd) - 1)
        mpc_drove = (k >= 0) & (cmd[kk, 3] > 0.5) & (t - cmd[kk, 0] < 0.2)
        if not mpc_drove.all():
            in_win &= mpc_drove
            window += ', only while the MPC drove (joystick time left out)'
    w = dt * in_win

    # --- wall hits: rising edge -> falling edge
    hits = []
    for k, (te, state) in enumerate(collisions):
        if state < 0.5:
            continue
        end = next((c[0] for c in collisions[k + 1:] if c[1] < 0.5), None)
        i = int(np.clip(np.searchsorted(t, te), 0, len(t) - 1)) if len(t) else 0
        lap_no = next((lp['lap'] for lp in laps if lp['t_start'] <= te < lp['t_end']), None)
        hits.append({'hit': len(hits) + 1, 't_s': round(float(te), 3),
                     'duration_s': None if end is None else round(float(end - te), 3),
                     'lap': lap_no, 'segment': segs[seg_of[i]]['name'] if len(t) else '',
                     's_m': round(float(s[i]), 2) if len(t) else None,
                     'x': round(float(x[i]), 2) if len(t) else None,
                     'y': round(float(y[i]), 2) if len(t) else None,
                     'speed_mps': round(float(v[i]), 2) if len(t) else None,
                     'lateral_error_m': round(float(e[i]), 3) if len(t) else None})

    # --- who was driving: each /ackermann_cmd message holds for the time until the next
    # one (at most the mux's 0.2 s timeout). from_mpc = 0 -> joystick took over.
    cmd_dt = np.clip(np.r_[np.diff(cmd[:, 0]), 0.0], 0.0, 0.2) if len(cmd) else np.zeros(0)
    joy_dt = cmd_dt * (cmd[:, 3] < 0.5) if len(cmd) else cmd_dt

    def joystick_s(t_a, t_b):
        if not len(cmd):
            return 0.0
        return float(joy_dt[(cmd[:, 0] >= t_a) & (cmd[:, 0] < t_b)].sum())

    # --- per lap
    for lp in laps:
        m = (t >= lp['t_start']) & (t < lp['t_end'])
        wl = dt * m
        lp['avg_speed_mps'] = wmean(v, wl)
        lp['mean_abs_e_m'] = wmean(np.abs(e), wl)
        lp['corner_mean_abs_e_m'] = wmean(np.abs(e), wl * is_corner)
        lp['max_abs_e_m'] = float(np.abs(e[m]).max()) if m.any() else None
        lp['hits'] = sum(1 for h in hits if lp['t_start'] <= h['t_s'] < lp['t_end'])
        lp['joystick_s'] = round(joystick_s(lp['t_start'], lp['t_end']), 2)
        # clean = no wall hit (sim) and the MPC drove the whole lap (real car)
        lp['clean'] = lp['hits'] == 0 and lp['joystick_s'] < 0.1

    # --- per segment (per lap averages inside the window)
    seg_rows = []
    for k, sg in enumerate(segs):
        m = in_win & (seg_of == k)
        ws = dt * m
        if valid_laps:
            passes = len(valid_laps)
        else:
            entries = np.diff(np.r_[0, (seg_of == k).astype(int)]) == 1
            passes = max(1, int(entries.sum()))
        actual = float(ws.sum()) / passes if m.any() else None
        signed = wmean(e, ws)
        side = ''
        if sg['kind'] == 'corner' and signed is not None and abs(signed) > 0.02:
            # turning left (+) and car right of the line (-) = running wide
            side = 'inside' if np.sign(signed) == np.sign(sg['turn_deg']) else 'wide'
        seg_rows.append({
            'segment': sg['name'], 'kind': sg['kind'],
            's0_m': round(sg['s0'], 1), 's1_m': round(sg['s1'], 1),
            'length_m': round(sg['length_m'], 2), 'turn_deg': round(sg['turn_deg'], 0),
            'ref_time_s': round(sg['ref_time_s'], 3),
            'actual_time_s': None if actual is None else round(actual, 3),
            'time_lost_s': None if actual is None else round(actual - sg['ref_time_s'], 3),
            'v_ref_min_mps': round(sg['v_ref_min'], 2),
            'v_min_mps': round(float(v[m].min()), 2) if m.any() else None,
            'speed_shortfall_mps': wmean(v_ref - v, ws),
            'mean_abs_e_m': wmean(np.abs(e), ws), 'max_abs_e_m':
                float(np.abs(e[m]).max()) if m.any() else None,
            'mean_signed_e_m': signed, 'corner_side': side,
            'hits': sum(1 for h in hits if h['segment'] == sg['name'])})

    # --- control loop timing (solve: t, solve_ms, osqp_ms, callback_ms, status)
    period_ms = np.diff(drive[:, 0]) * 1e3 if len(drive) > 1 else np.array([])
    solve_ms = solve[:, 1] if len(solve) else np.array([])
    if len(solve) and len(t):
        si = np.clip(np.searchsorted(t, solve[:, 0]), 0, len(t) - 1)
        solve_corner = stats(solve_ms[is_corner[si]])['mean']
        solve_straight = stats(solve_ms[~is_corner[si]])['mean']
    else:
        solve_corner = solve_straight = None
    timing = {
        'solve_ms': stats(solve_ms),
        'osqp_ms': stats(solve[:, 2]) if len(solve) else stats([]),
        'callback_ms': stats(solve[:, 3]) if len(solve) else stats([]),
        'control_period_ms': stats(period_ms),
        'periods_over_DTK_pct': float(np.mean(period_ms > dtk * 1e3) * 100)
        if len(period_ms) else None,
        'solve_mean_corner_ms': solve_corner, 'solve_mean_straight_ms': solve_straight,
        'solver_failures': int(np.sum(solve[:, 4] == 0)) if len(solve) else None,
        'solver_inaccurate': int(np.sum(solve[:, 4] == 1)) if len(solve) else None,
        'DTK_ms': dtk * 1e3}

    lap_times = [lp['lap_time_s'] for lp in valid_laps]
    # real-time factor without sim time: odom speed is per *sim* second, while
    # receive times are wall seconds. If the sim falls behind real time, the car
    # moves less per wall second than its reported speed says.
    moving = in_win[1:] & (v[1:] > 0.5)
    step = np.hypot(np.diff(x), np.diff(y))[moving]
    rtf = float(step.sum() / np.sum(v[1:][moving] * np.diff(t)[moving])) \
        if moving.sum() > 100 and not real else None
    # real car: the same comparison measures the wheel-speed calibration instead
    scale = odom_scale(t, x, y, v, in_win) if real else None

    # --- speed response: measured speed = VESC wheel speed (/odom) on the car, else odom
    vel = wheel[:, :2] if len(wheel) else odom[:, [0, 4]]
    response = speed_response(cmd[:, [0, 1, 3]] if len(cmd) else np.zeros((0, 3)), vel)
    odom_mpc = odom.copy()
    odom_mpc[~mpc_drove, 4] = 0.0          # 'not moving' = left out of the steering fits
    steer_resp = steering_response(odom_mpc, drive, mpc_cfg.get('WB', 0.33))
    steer_osc = steering_oscillation(odom_mpc, drive)
    n_jumps, max_step = pose_jumps(odom)
    pf_lat = optional(raw, 'pf_latency', 2)
    mpc_params = (meta.get('mpc') or {}).get('parameters') or {}
    steering = {'response': steer_resp, 'oscillation': steer_osc,
                'pose_jumps': n_jumps, 'max_pose_step_m': max_step,
                'pf_latency_ms': stats(pf_lat[:, 1] * 1e3) if len(pf_lat) else stats([]),
                'delay_comp_s': mpc_params.get('delay_comp')}
    mpc_total = float(cmd_dt.sum()) if len(cmd) else 0.0
    car = {
        'mpc_in_control_pct': 100.0 * (1 - joy_dt.sum() / mpc_total) if mpc_total > 0 else None,
        'joystick_s': float(joy_dt.sum()) if len(cmd) else None,
        'battery_v': None, 'motor_current_p95_a': None}
    if len(vesc):
        vt, volt = vesc[:, 0], vesc[:, 4]
        car['battery_v'] = {'start': float(np.median(volt[vt < vt[0] + 2.0])),
                            'end': float(np.median(volt[vt > vt[-1] - 2.0])),
                            'min': float(volt.min())}
        car['motor_current_p95_a'] = float(np.percentile(np.abs(vesc[:, 2]), 95))
    clean = [lp['lap_time_s'] for lp in laps if lp['clean']]
    ideal = float(sum(sg['ref_time_s'] for sg in segs))
    straight = ~is_corner
    results = {
        'run': meta['run'], 'note': meta.get('note', ''), 'track': meta.get('track'),
        'window': window, 'duration_s': float(t[-1]) if len(t) else 0.0,
        'laps': laps,
        'lap_time': {**stats(lap_times), 'best': min(lap_times) if lap_times else None,
                     'valid_laps': len(lap_times)},
        'ideal_lap_s': ideal, 'raceline_length_m': track['L'],
        'speed': {'avg_mps': wmean(v, w), 'avg_ref_mps': track['L'] / ideal,
                  'max_mps': float(v[in_win].max()) if in_win.any() else None,
                  'shortfall_mps': wmean(v_ref - v, w)},
        'tracking': {
            'mean_abs_e_m': wmean(np.abs(e), w),
            'rms_e_m': (wmean(e ** 2, w) or 0.0) ** 0.5 if in_win.any() else None,
            'max_abs_e_m': float(np.abs(e[in_win]).max()) if in_win.any() else None,
            'corner_mean_abs_e_m': wmean(np.abs(e), w * is_corner),
            'corner_max_abs_e_m': float(np.abs(e[in_win & is_corner]).max())
            if (in_win & is_corner).any() else None,
            'straight_mean_abs_e_m': wmean(np.abs(e), w * straight),
            'mean_abs_heading_err_deg': (lambda h: None if h is None else float(np.degrees(h)))(
                wmean(np.abs(head_err), w))},
        'platform': platform, 'hits_recorded': not real,
        'hits': hits, 'segments': seg_rows, 'timing': timing,
        'realtime_factor': rtf, 'odom_scale': scale,
        'speed_response': response, 'car': car, 'steering': steering,
        'best_clean_lap_s': min(clean) if clean else None,
        'sim_lap_times_s': [float(lt_) for lt_ in sim_laps[:, 1]] if len(sim_laps) else [],
        'recorder_cpu_pct': meta.get('recorder_cpu_pct'),
    }
    results['hints'] = make_hints(results, meta, track)
    series = {'t': t, 'x': x, 'y': y, 'v': v, 's': s, 'e': e, 'v_ref': v_ref,
              'in_win': in_win, 'solve': solve, 'period_ms': period_ms}
    return meta, track, results, series


def make_hints(r, meta, track):
    """A few data-driven pointers. They are starting points, not verdicts."""
    hints = []
    cfg = (meta.get('mpc') or {}).get('mpc_config', {})
    lt = r['lap_time']
    if lt['mean'] is not None:
        gap = lt['mean'] - r['ideal_lap_s']
        pct = 100 * gap / r['ideal_lap_s']
        if pct < 3:
            hints.append(
                f'Laps are within {pct:.1f}% of the ideal lap ({r["ideal_lap_s"]:.2f} s), so the '
                'waypoint speed profile sets your lap time. To go faster, raise v_max / a_lat in '
                'make_waypoints.py, then watch corner error and hits.')
        else:
            worst = sorted([sg for sg in r['segments'] if sg['time_lost_s'] is not None],
                           key=lambda sg: -sg['time_lost_s'])[:3]
            where = ', '.join(f"{sg['segment']} (+{sg['time_lost_s']:.2f} s)" for sg in worst)
            hints.append(f'Laps are {gap:.2f} s ({pct:.0f}%) slower than the waypoint speeds '
                         f'allow. Most time lost in: {where}.')
    v_top = float(track['v'].max())
    if cfg.get('MAX_SPEED') is not None and cfg['MAX_SPEED'] < v_top - 0.05:
        hints.append(f'MAX_SPEED ({cfg["MAX_SPEED"]}) is below the waypoints\' top speed '
                     f'({v_top:.2f}), so the MPC can never reach the reference on straights.')
    if r['hits']:
        by_seg = {}
        for h in r['hits']:
            by_seg[h['segment']] = by_seg.get(h['segment'], 0) + 1
        hints.append(f'{len(r["hits"])} wall hit(s): ' + ', '.join(
            f'{k} x{n}' for k, n in by_seg.items()) +
            '. Check lateral error and corner side (wide/inside) for those segments.')
    tr = r['tracking']
    if tr['corner_mean_abs_e_m'] and tr['straight_mean_abs_e_m'] and \
            tr['corner_mean_abs_e_m'] > 2 * tr['straight_mean_abs_e_m']:
        wide = [sg['segment'] for sg in r['segments'] if sg['corner_side'] == 'wide']
        hints.append(f'Tracking error is mostly in corners ({tr["corner_mean_abs_e_m"]:.3f} m vs '
                     f'{tr["straight_mean_abs_e_m"]:.3f} m on straights)' +
                     (f'; running wide in {", ".join(wide)}.' if wide else '.'))
    tm = r['timing']
    per, sol, osqp = tm['control_period_ms'], tm['solve_ms'], tm['osqp_ms']
    if per['mean'] is not None:
        rate = 1000 / per['mean']
        if per['p95'] > 0.5 * tm['DTK_ms']:
            hints.append(f'Control loop is slow: p95 period {per["p95"]:.0f} ms vs DTK '
                         f'{tm["DTK_ms"]:.0f} ms. Solve time may be limiting.')
        else:
            hints.append(f'Solve time is not the bottleneck: the MPC sends commands at '
                         f'~{rate:.0f} Hz (p95 period {per["p95"]:.0f} ms).')
        if sol['mean'] and osqp['mean'] and osqp['mean'] < 0.3 * sol['mean']:
            hints.append(f'OSQP itself takes {osqp["mean"]:.1f} of {sol["mean"]:.1f} ms per '
                         'solve; the rest is cvxpy overhead (only matters if the loop is slow).')
    if tm['solver_failures']:
        hints.append(f'{tm["solver_failures"]} solver failure(s): no drive command was sent '
                     'for those odom messages.')
    if r['realtime_factor'] is not None and r['realtime_factor'] < RTF_WARN:
        hints.append(f'The sim ran at {r["realtime_factor"]:.2f}x real time: the computer was '
                     'overloaded, so these lap times may not match the autograder.')
    sc = r['odom_scale']
    if sc is not None and abs(sc - 1) > 0.05:
        gain = ((meta.get('car') or {}).get('vesc') or {}).get('speed_to_erpm_gain')
        fix = f' (try speed_to_erpm_gain ~{gain / sc:.0f} instead of {gain})' if gain else ''
        hints.append(f'The car moved {sc:.2f}x the distance its wheel speed says: '
                     + ('the wheels spin faster than the car moves (wheel slip, or '
                        'speed_to_erpm_gain too low)' if sc < 1 else
                        'the wheel speed reads low (speed_to_erpm_gain too high)')
                     + f'{fix}. The MPC uses this speed as its state. A bad particle '
                     'filter pose also affects this number.')
    sr = r['speed_response']
    if sr is not None and r['platform'] == 'real':
        quality = ('good fit' if (sr['r2'] or 0) > 0.6 else
                   'rough fit, needs more speed changes in the run' if (sr['r2'] or 0) > 0.3
                   else 'poor fit, treat as a guess')
        hints.append(f'Speed response measured on the car: kp speeding up {sr["kp_accel"]:.2f}, '
                     f'braking {sr["kp_brake"]:.2f} 1/s, delay {sr["delay_s"]:.2f} s '
                     f'(R² {fmt(sr["r2"])}, {quality}). mpc_node.py still uses the sim\'s '
                     'kp 4.755 / 19.02 to turn acceleration into a speed command.')
    sg_ = r['steering']
    rs, osc = sg_['response'], sg_['oscillation']
    dc = float(sg_['delay_comp_s'] or 0.0)
    if rs and rs['r2'] > 0.5 and rs['delay_s'] >= 0.1 and abs(dc - rs['delay_s']) > 0.05:
        hints.append(f'Steering acts {rs["delay_s"]:.2f} s after the command (MPC step DTK is '
                     f'{cfg.get("DTK", 0.1)} s) and delay_comp is {dc:g}: try '
                     f'delay_comp:={rs["delay_s"]:.2f}. An unmodelled delay makes the MPC '
                     'over-correct and oscillate.')
    if osc and osc['rms_rad'] > 0.03:
        hints.append(f'Steering oscillates ({osc["rms_rad"]:.3f} rad rms at {fmt(osc["freq_hz"])} Hz). '
                     'Causes, most likely first: delay (see above), low steering cost Rk/Rdk, '
                     'pose jumps.')
    if sg_['pose_jumps'] > 20:
        hints.append(f'{sg_["pose_jumps"]} particle-filter pose jumps (largest '
                     f'{sg_["max_pose_step_m"]:.2f} m): localization was poor; each jump yanks the '
                     'steering. Set a good initial pose and check the map.')
    car = r['car']
    if r['platform'] == 'real' and car['joystick_s']:
        hints.append(f'The joystick drove for {car["joystick_s"]:.1f} s; laps with a takeover '
                     'are marked not clean.')
    bv = car['battery_v']
    if bv and bv['start'] - bv['end'] > 0.3:
        hints.append(f'Battery dropped {bv["start"]:.2f} -> {bv["end"]:.2f} V during the run '
                     '(min {:.2f} V): later laps get less acceleration.'.format(bv['min']))
    hints.extend(meta.get('warnings', []))
    return hints


# ---------------------------------------------------------------- outputs

def summary_text(meta, r):
    L = []
    note = f'  "{r["note"]}"' if r['note'] else ''
    L.append(f'=== MPC run {r["run"]}{note} ===')
    wph = '; '.join(h.lstrip('# ') for h in meta.get('waypoints', {}).get('header', [])[1:])
    L.append(f'track {r["track"]} | {Path(meta["waypoints"]["installed"]).name} '
             f'({r["raceline_length_m"]:.1f} m) {wph}')
    cfg = (meta.get('mpc') or {}).get('mpc_config')
    if cfg:
        model = ((meta.get('mpc') or {}).get('parameters') or {}).get('model', 'kinematic')
        L.append(f'MPC  model={model} ' + ' '.join(f'{k}={cfg[k]}' for k in
                                    ('TK', 'DTK', 'MAX_SPEED', 'MAX_ACCEL', 'Qk', 'Qfk', 'Rk', 'Rdk')
                                    if k in cfg))
    L.append(f'{r["platform"]} run, pose from {meta.get("pose_topic", "/ego_racecar/odom")}; '
             f'recorded {r["duration_s"]:.1f} s; stats below use: {r["window"]}')

    L.append('\nLAP TIMES (timed here, line = first raceline waypoint)')
    for lp in r['laps']:
        if r['hits_recorded']:
            extra = '' if lp['clean'] else '  (hit a wall)'
            last = f'hits {lp["hits"]}{extra}'
        else:
            last = (f'joystick {lp["joystick_s"]:.1f} s  (not clean)'
                    if lp['joystick_s'] >= 0.1 else 'MPC drove the whole lap')
        L.append(f'  lap {lp["lap"]:>2}  {lp["lap_time_s"]:7.2f} s   avg {fmt(lp["avg_speed_mps"])} '
                 f'm/s  mean|e| {fmt(lp["mean_abs_e_m"], 3)} m  {last}')
    lt = r['lap_time']
    if lt['n']:
        L.append(f'  mean {lt["mean"]:.2f} s   std {lt["std"]:.3f} s   best {lt["best"]:.2f} s'
                 f'   best clean {fmt(r["best_clean_lap_s"])} s   ({lt["n"]} laps)')
    else:
        L.append('  no complete lap (a lap needs two crossings of the line)')
    if r['sim_lap_times_s']:
        L.append('  sim lap times (cross-check): '
                 + ', '.join(f'{x:.2f}' for x in r['sim_lap_times_s']))
    L.append(f'  ideal lap at waypoint speeds: {r["ideal_lap_s"]:.2f} s')

    sp = r['speed']
    L.append(f'\nSPEED  avg {fmt(sp["avg_mps"])} m/s (reference avg {sp["avg_ref_mps"]:.2f}), '
             f'max {fmt(sp["max_mps"])}, avg shortfall vs reference {fmt(sp["shortfall_mps"])} m/s')

    sr = r['speed_response']
    L.append('\nSPEED RESPONSE  a = kp * (v_cmd(t - delay) - v), fit from the run')
    if sr:
        L.append(f'  kp speeding up {sr["kp_accel"]:.2f} 1/s ({sr["n_accel"]} samples), '
                 f'braking {sr["kp_brake"]:.2f} 1/s ({sr["n_brake"]}), delay {sr["delay_s"]:.2f} s, '
                 f'R² {fmt(sr["r2"])}')
        L.append(f'  measured accel p95 {fmt(sr["accel_p95_mps2"])} m/s², '
                 f'decel p95 {fmt(sr["decel_p95_mps2"])} m/s²'
                 + (f'   (MAX_ACCEL {cfg["MAX_ACCEL"]})' if cfg and 'MAX_ACCEL' in cfg else ''))
    else:
        L.append('  not enough speed changes (needs both speeding up and braking above 0.3 m/s)')

    sg_ = r['steering']
    L.append(f'\nSTEERING  (delay_comp in the node: {sg_["delay_comp_s"] if sg_["delay_comp_s"] is not None else "not set"})')
    osc = sg_['oscillation']
    if osc:
        L.append(f'  oscillation {osc["rms_rad"]:.3f} rad rms at {fmt(osc["freq_hz"])} Hz '
                 '(steering minus its 2 s average, while moving; lower = smoother)')
    rs = sg_['response']
    if rs:
        L.append(f'  command -> turn in pose: delay {rs["delay_s"]:.2f} s, gain {rs["gain"]:.2f} '
                 f'(1 = turns as the MPC model says), R² {rs["r2"]:.2f}')
        if rs['straight_steer_rad'] is not None:
            L.append(f'  steering command while driving straight {rs["straight_steer_rad"]:+.3f} rad '
                     '(servo trim; 0 = centred)')
    else:
        L.append('  command -> turn: not enough driving above 1 m/s to fit')
    pl = sg_['pf_latency_ms']
    L.append(f'  particle filter delay (scan -> pose) '
             + (f'mean {pl["mean"]:.0f} ms, p95 {pl["p95"]:.0f} ms' if pl['n'] else 'not recorded (needs /tf)'))
    L.append(f'  pose jumps (> possible motion): {sg_["pose_jumps"]}, largest step '
             f'{fmt(sg_["max_pose_step_m"])} m')

    if not r['hits_recorded']:
        L.append('\nWALL HITS: not recorded on the real car')
    else:
        L.append(f'\nWALL HITS: {len(r["hits"])}')
    for h in r['hits']:
        dur = f'{h["duration_s"]:.2f} s' if h['duration_s'] is not None else 'ongoing'
        L.append(f'  #{h["hit"]} t={h["t_s"]:.2f} s  lap {h["lap"] or "-"}  {h["segment"]} '
                 f's={h["s_m"]} m  ({h["x"]}, {h["y"]})  v={h["speed_mps"]} m/s  '
                 f'e={h["lateral_error_m"]} m  contact {dur}')

    tr = r['tracking']
    L.append('\nTRACKING (lateral distance from the raceline)')
    L.append(f'  all        mean|e| {fmt(tr["mean_abs_e_m"], 3)} m   rms {fmt(tr["rms_e_m"], 3)} m'
             f'   max {fmt(tr["max_abs_e_m"], 3)} m')
    L.append(f'  corners    mean|e| {fmt(tr["corner_mean_abs_e_m"], 3)} m'
             f'   max {fmt(tr["corner_max_abs_e_m"], 3)} m')
    L.append(f'  straights  mean|e| {fmt(tr["straight_mean_abs_e_m"], 3)} m')
    L.append(f'  heading error mean {fmt(tr["mean_abs_heading_err_deg"], 1)} deg')

    L.append('\nSEGMENTS (per lap; lost = actual - ideal time; short = v_ref - v)')
    L.append('  seg  turn   len   ideal  actual   lost  vref_min  v_min  short  mean|e|  '
             'max|e|  side    hits')
    for sg in r['segments']:
        L.append(f'  {sg["segment"]:<4}{sg["turn_deg"]:+5.0f}° {sg["length_m"]:5.1f}  '
                 f'{sg["ref_time_s"]:5.2f}  {fmt(sg["actual_time_s"]):>6} {fmt(sg["time_lost_s"]):>6}'
                 f'  {sg["v_ref_min_mps"]:7.2f}  {fmt(sg["v_min_mps"]):>5}  '
                 f'{fmt(sg["speed_shortfall_mps"]):>5}  {fmt(sg["mean_abs_e_m"], 3):>7}  '
                 f'{fmt(sg["max_abs_e_m"], 3):>6}  {sg["corner_side"]:<6}  {sg["hits"]}')

    tm = r['timing']
    L.append('\nCONTROL LOOP')
    for key, label in (('solve_ms', 'solve() total'), ('osqp_ms', 'OSQP only'),
                       ('callback_ms', 'odom callback'), ('control_period_ms', '/drive period')):
        st = tm[key]
        if st['n']:
            L.append(f'  {label:<14} mean {st["mean"]:6.1f} ms  std {st["std"]:5.1f}  '
                     f'p95 {st["p95"]:6.1f}  max {st["max"]:6.1f}')
        else:
            L.append(f'  {label:<14} no data' + (' (MPC not publishing /mpc/solve_stats)'
                                                 if key != 'control_period_ms' else ''))
    if tm['control_period_ms']['n']:
        L.append(f'  periods longer than DTK ({tm["DTK_ms"]:.0f} ms): '
                 f'{tm["periods_over_DTK_pct"]:.1f}%')
    if tm['solve_mean_corner_ms'] is not None:
        L.append(f'  solve mean in corners {tm["solve_mean_corner_ms"]:.1f} ms, '
                 f'straights {fmt(tm["solve_mean_straight_ms"], 1)} ms')
    if tm['solver_failures'] is not None:
        L.append(f'  solver failures {tm["solver_failures"]}, inaccurate {tm["solver_inaccurate"]}')

    L.append('\nHEALTH')
    if r['platform'] == 'real':
        car = r['car']
        L.append(f'  odometry scale {fmt(r["odom_scale"], 3)} (particle-filter distance / '
                 'wheel speed x time; 1.00 = speed_to_erpm_gain is right)')
        L.append(f'  MPC in control {fmt(car["mpc_in_control_pct"], 1)}% of the commands that '
                 f'reached the motor (joystick {fmt(car["joystick_s"], 1)} s)')
        bv = car['battery_v']
        if bv:
            L.append(f'  battery {bv["start"]:.2f} -> {bv["end"]:.2f} V (min {bv["min"]:.2f}), '
                     f'motor current p95 {fmt(car["motor_current_p95_a"], 1)} A')
        else:
            L.append('  battery: no /sensors/core data')
    else:
        L.append(f'  sim real-time factor {fmt(r["realtime_factor"], 3)} '
                 '(distance moved / reported speed x time; 1.00 = sim kept up with real time)')
    L.append(f'  recorder CPU {fmt(r["recorder_cpu_pct"], 1)}% of one core (ros2 bag record)')

    L.append('\nHINTS')
    L.extend(f'  - {h}' for h in r['hints'] or ['none'])
    return '\n'.join(L) + '\n'


def write_csv(path, rows, fields=None):
    rows = list(rows)
    fields = fields or (list(rows[0].keys()) if rows else [])
    with open(path, 'w', newline='') as f:
        wr = csv.DictWriter(f, fields, extrasaction='ignore')
        wr.writeheader()
        for row in rows:
            wr.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})


def append_index(data_dir, meta, r):
    """One line per run in data/runs.csv (replaces the line if the run is re-analysed)."""
    cfg = (meta.get('mpc') or {}).get('mpc_config', {})
    hdr = meta.get('waypoints', {}).get('header', [])
    row = {
        'run': r['run'], 'note': r['note'], 'track': r['track'],
        'laps': r['lap_time']['valid_laps'], 'lap_mean_s': r['lap_time']['mean'],
        'lap_std_s': r['lap_time']['std'], 'lap_best_s': r['lap_time']['best'],
        'best_clean_lap_s': r['best_clean_lap_s'],
        'ideal_lap_s': r['ideal_lap_s'], 'hits': len(r['hits']),
        'avg_speed_mps': r['speed']['avg_mps'], 'e_mean_m': r['tracking']['mean_abs_e_m'],
        'e_corner_mean_m': r['tracking']['corner_mean_abs_e_m'],
        'e_max_m': r['tracking']['max_abs_e_m'], 'solve_mean_ms': r['timing']['solve_ms']['mean'],
        'control_period_mean_ms': r['timing']['control_period_ms']['mean'],
        'solver_failures': r['timing']['solver_failures'], 'realtime_factor': r['realtime_factor'],
        'platform': r['platform'], 'odom_scale': r['odom_scale'],
        **{k: (r['speed_response'] or {}).get(src) for k, src in
           (('kp_accel', 'kp_accel'), ('kp_brake', 'kp_brake'), ('cmd_delay_s', 'delay_s'))},
        'battery_v_start': ((r['car'] or {}).get('battery_v') or {}).get('start'),
        'delay_comp_s': r['steering']['delay_comp_s'],
        'steer_osc_rms': (r['steering']['oscillation'] or {}).get('rms_rad'),
        'steer_delay_s': (r['steering']['response'] or {}).get('delay_s'),
        'pose_jumps': r['steering']['pose_jumps'],
        'waypoints': Path(meta['waypoints']['installed']).name,
        'waypoints_header': hdr[1].lstrip('# ') if len(hdr) > 1 else '',
        **{k: json.dumps(cfg.get(k)) if isinstance(cfg.get(k), list) else cfg.get(k)
           for k in ('MAX_SPEED', 'TK', 'DTK', 'Qk', 'Qfk', 'Rk', 'Rdk')}}
    path = Path(data_dir) / 'runs.csv'
    rows = []
    if path.exists():
        with open(path, newline='') as f:
            lines = [ln for ln in csv.reader(f) if ln]
        if lines and lines[0][0] == 'run':                 # normal file with a header
            header, lines = lines[0], lines[1:]
        else:
            # header row lost (e.g. edited by hand): guess the columns from the row length
            header = None
        for ln in lines:
            fields = header or (LEGACY_INDEX_FIELDS if len(ln) == len(LEGACY_INDEX_FIELDS)
                                else INDEX_FIELDS)
            old = dict(zip(fields, ln))
            if old.get('run') != r['run']:
                rows.append(old)
    rows.append(row)
    rows.sort(key=lambda x: x['run'])
    write_csv(path, rows, INDEX_FIELDS)


# ---------------------------------------------------------------- plot

BLUE, ORANGE, CRITICAL = '#2a78d6', '#eb6834', '#d03b3b'
INK, INK2, MUTED, GRID = '#0b0b0b', '#52514e', '#8a8984', '#e4e3df'
BLUE_RAMP = ['#86b6ef', '#5598e7', '#2a78d6', '#1c5cab', '#104281', '#0d366b']


def plot(run_dir, meta, track, r, sr):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import LinearSegmentedColormap

    plt.rcParams.update({
        'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK2,
        'xtick.color': INK2, 'ytick.color': INK2, 'axes.titlesize': 10,
        'axes.titleweight': 'bold', 'axes.titlecolor': INK, 'axes.titlelocation': 'left',
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
        'grid.color': GRID, 'grid.linewidth': 0.6, 'legend.frameon': False})
    fig = plt.figure(figsize=(15, 14))
    gs = fig.add_gridspec(4, 2, height_ratios=[1.5, 1, 1, 1], hspace=0.42, wspace=0.18,
                          top=0.93)
    corners = [sg for sg in track['segments'] if sg['kind'] == 'corner']
    win = sr['in_win']
    lt = r['lap_time']
    title = f'MPC run {r["run"]}' + (f'  "{r["note"]}"' if r['note'] else '')
    sub = (f'mean lap {lt["mean"]:.2f} s ± {lt["std"]:.3f} ({lt["n"]} laps)' if lt['n']
           else 'no valid laps') + f'   ideal {r["ideal_lap_s"]:.2f} s   ' + (
        f'hits {len(r["hits"])}' if r['hits_recorded'] else f'{r["platform"]} car, hits not recorded')
    fig.suptitle(f'{title}\n{sub}', x=0.06, ha='left', fontsize=12, color=INK)

    # 1) track map, path colored by speed, hits
    ax = fig.add_subplot(gs[0, 0])
    ax.grid(False)
    wp = track['wp']
    pad = 1.5
    xmin, xmax = wp[:, 0].min() - pad, wp[:, 0].max() + pad
    ymin, ymax = wp[:, 1].min() - pad, wp[:, 1].max() + pad
    mp = meta.get('map') or {}
    try:
        img = plt.imread(mp['image'])
        if img.ndim == 3:
            img = img[..., :3].mean(-1)
        h_px, w_px = img.shape
        ox, oy = mp['origin'][:2]
        res = mp['resolution']
        ax.imshow(img, cmap='gray', vmin=0, vmax=1, alpha=0.35, origin='upper',
                  extent=[ox, ox + w_px * res, oy, oy + h_px * res])
    except Exception:
        pass
    ax.plot(np.r_[wp[:, 0], wp[0, 0]], np.r_[wp[:, 1], wp[0, 1]], color=MUTED, lw=1, ls='--',
            label='raceline')
    if win.any():
        pts = np.c_[sr['x'], sr['y']][win]
        speeds = sr['v'][win]
        segs_xy = np.stack([pts[:-1], pts[1:]], axis=1)
        jump = np.hypot(*(pts[1:] - pts[:-1]).T) < 0.5          # no lines across lap gaps
        cmap = LinearSegmentedColormap.from_list('speed', BLUE_RAMP)
        lc = LineCollection(segs_xy[jump], cmap=cmap, linewidths=2)
        lc.set_array(speeds[:-1][jump])
        ax.add_collection(lc)
        cb = fig.colorbar(lc, ax=ax, fraction=0.035, pad=0.01)
        cb.set_label('speed (m/s)')
        cb.outline.set_visible(False)
    for sg in corners:
        i = sg['idx'][len(sg['idx']) // 2]
        ax.annotate(sg['name'], wp[i, :2], color=INK, fontsize=9, fontweight='bold',
                    xytext=(0, 0), textcoords='offset points', ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=MUTED, lw=0.5))
    clusters = []                       # hits within 0.5 m share one label
    for h in r['hits']:
        ax.plot(h['x'], h['y'], marker='X', ms=10, color=CRITICAL, mec='white', mew=1.5,
                ls='none', label='wall hit' if h['hit'] == 1 else None)
        for c in clusters:
            if np.hypot(c['x'] - h['x'], c['y'] - h['y']) < 0.5:
                c['ids'].append(h['hit'])
                break
        else:
            clusters.append({'x': h['x'], 'y': h['y'], 'ids': [h['hit']]})
    for c in clusters:
        text = f'#{c["ids"][0]}' if len(c['ids']) == 1 else f'{len(c["ids"])} hits'
        ax.annotate(text, (c['x'], c['y']), xytext=(-8, 0), textcoords='offset points',
                    ha='right', va='center', color=CRITICAL, fontsize=8, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.8))
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal')
    ax.set_title('Driven path (complete laps), colored by speed')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.legend(loc='best', fontsize=8)

    # 2) time lost per segment
    ax = fig.add_subplot(gs[0, 1])
    rows = [sg for sg in r['segments'] if sg['time_lost_s'] is not None]
    names = [sg['segment'] for sg in rows]
    lost = [sg['time_lost_s'] for sg in rows]
    colors = [ORANGE if sg['kind'] == 'corner' else BLUE for sg in rows]
    ypos = np.arange(len(rows))[::-1]
    ax.barh(ypos, lost, color=colors, height=0.6, edgecolor='white', linewidth=2)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(ypos, names)
    for yy, val in zip(ypos, lost):
        ax.annotate(f'{val:+.2f}s', (val, yy), xytext=(4 if val >= 0 else -4, 0),
                    textcoords='offset points', va='center', ha='left' if val >= 0 else 'right',
                    color=INK2, fontsize=8)
    ax.grid(axis='y', visible=False)
    ax.set_xlabel('time lost per lap vs ideal (s)   (+ = slower than the waypoint speeds)')
    ax.set_title('Where the lap time goes')
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=ORANGE, label='corner'), Patch(color=BLUE, label='straight')],
              loc='lower right', fontsize=8)

    def shade_corners(axis):
        for sg in corners:
            if sg['s1'] >= sg['s0']:
                axis.axvspan(sg['s0'], sg['s1'], color=GRID, alpha=0.6, lw=0)
            else:                                                  # wraps past s = 0
                axis.axvspan(sg['s0'], track['L'], color=GRID, alpha=0.6, lw=0)
                axis.axvspan(0, sg['s1'], color=GRID, alpha=0.6, lw=0)
            axis.text((sg['s0'] + (sg['s1'] if sg['s1'] >= sg['s0'] else track['L'])) / 2,
                      1.0, sg['name'], transform=axis.get_xaxis_transform(), ha='center',
                      va='bottom', color=INK2, fontsize=8, fontweight='bold')

    def binned(values, nbins=300):
        """mean / min / max of values vs s over the valid window."""
        edges = np.linspace(0, track['L'], nbins + 1)
        b = np.clip(np.digitize(sr['s'][win], edges) - 1, 0, nbins - 1)
        val = values[win]
        mean = np.full(nbins, np.nan)
        lo, hi = mean.copy(), mean.copy()
        for k in range(nbins):
            sel = val[b == k]
            if len(sel):
                mean[k], lo[k], hi[k] = sel.mean(), sel.min(), sel.max()
        return 0.5 * (edges[:-1] + edges[1:]), mean, lo, hi

    # 3) speed vs s
    ax = fig.add_subplot(gs[1, :])
    shade_corners(ax)
    sline = np.r_[track['s_start'], track['L']]
    ax.plot(sline, np.r_[track['v'], track['v'][0]], color=INK2, lw=1.5, ls='--',
            label='reference (waypoints)')
    if win.any():
        c, m, lo, hi = binned(sr['v'])
        ax.fill_between(c, lo, hi, color=BLUE, alpha=0.18, lw=0, label='actual, min–max over laps')
        ax.plot(c, m, color=BLUE, lw=2, label='actual, mean over laps')
    cfg = (meta.get('mpc') or {}).get('mpc_config', {})
    if cfg.get('MAX_SPEED') is not None:
        ax.axhline(cfg['MAX_SPEED'], color=MUTED, lw=0.8, ls=':')
        ax.annotate(f'MAX_SPEED {cfg["MAX_SPEED"]}', (0, cfg['MAX_SPEED']),
                    xytext=(2, 2), textcoords='offset points', ha='left', va='bottom',
                    color=INK2, fontsize=8)
    for h in r['hits']:
        ax.axvline(h['s_m'], color=CRITICAL, lw=1.2)
    ax.set_xlim(0, track['L'])
    ax.set_ylabel('speed (m/s)')
    ax.set_xlabel('distance along raceline s (m)   (gray = corners, red lines = wall hits)')
    ax.set_title('Speed along the track', pad=14)
    ax.legend(loc='upper center', fontsize=8, ncol=3)

    # 4) lateral error vs s
    ax = fig.add_subplot(gs[2, :])
    shade_corners(ax)
    ax.axhline(0, color=INK2, lw=0.8)
    if win.any():
        c, m, lo, hi = binned(sr['e'])
        ax.fill_between(c, lo, hi, color=BLUE, alpha=0.18, lw=0, label='min–max over laps')
        ax.plot(c, m, color=BLUE, lw=2, label='mean over laps')
    for h in r['hits']:
        ax.axvline(h['s_m'], color=CRITICAL, lw=1.2)
    ax.set_xlim(0, track['L'])
    ax.set_ylabel('lateral error (m)\n+ = left of raceline')
    ax.set_xlabel('distance along raceline s (m)')
    ax.set_title('Distance from the raceline', pad=14)
    ax.legend(loc='upper left', fontsize=8, ncol=2)

    # 5) control loop timing
    ax = fig.add_subplot(gs[3, 0])
    tm = r['timing']
    data = []
    if len(sr['solve']):
        data.append(('solve() time', sr['solve'][:, 1], ORANGE))
    if len(sr['period_ms']):
        data.append(('/drive period', sr['period_ms'], BLUE))
    if data:
        top = 1.3 * max(np.nanpercentile(d, 99.5) for _, d, _ in data)
        bins = np.linspace(0, top, 50)
        for label, d, col in data:
            st = stats(d)
            ax.hist(np.clip(d, 0, top), bins=bins, color=col, histtype='step', lw=2,
                    label=f'{label}  (mean {st["mean"]:.1f}, p95 {st["p95"]:.1f} ms)')
        if tm['DTK_ms'] <= top:
            ax.axvline(tm['DTK_ms'], color=INK2, lw=1, ls='--')
        ax.text(0.99, 0.97, f'DTK (MPC model step) = {tm["DTK_ms"]:.0f} ms'
                + ('' if tm['DTK_ms'] <= top else ', off the right edge'),
                transform=ax.transAxes, ha='right', va='top', color=INK2, fontsize=8)
        ax.legend(loc='center right', fontsize=8)
        ax.set_xlim(0, top)
    else:
        ax.text(0.5, 0.5, 'no timing data', transform=ax.transAxes, ha='center', color=MUTED)
    ax.set_xlabel('ms')
    ax.set_ylabel('count')
    ax.set_title('Control loop timing')

    # 6) lap times
    ax = fig.add_subplot(gs[3, 1])
    if r['laps']:
        labels = [str(lp['lap']) for lp in r['laps']]
        vals = [lp['lap_time_s'] for lp in r['laps']]
        cols = [BLUE if lp['clean'] else CRITICAL for lp in r['laps']]
        ax.bar(labels, vals, color=cols, width=0.6, edgecolor='white', linewidth=2)
        for i, val in enumerate(vals):
            ax.annotate(f'{val:.2f}', (i, val), xytext=(0, 3), textcoords='offset points',
                        ha='center', color=INK2, fontsize=8)
        ax.axhline(r['ideal_lap_s'], color=INK2, lw=1, ls='--',
                   label=f'ideal lap {r["ideal_lap_s"]:.2f} s')
        lo = min(vals + [r['ideal_lap_s']])
        ax.set_ylim(lo * 0.9, max(vals) * 1.08)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(axis='x', visible=False)
    else:
        ax.text(0.5, 0.5, 'no laps completed', transform=ax.transAxes, ha='center', color=MUTED)
    ax.set_xlabel('lap (red = hit a wall)' if r['hits_recorded'] else 'lap (red = joystick took over)')
    ax.set_ylabel('lap time (s)')
    ax.set_title('Lap times')

    fig.savefig(Path(run_dir) / 'run.png', dpi=110, bbox_inches='tight', facecolor='white')
    plt.close(fig)


# ---------------------------------------------------------------- entry points

def report(run_dir):
    """Analyse a run folder, write all outputs, print the summary."""
    run_dir = Path(run_dir)
    meta, track, results, series = analyze(run_dir)
    text = summary_text(meta, results)
    (run_dir / 'summary.txt').write_text(text)
    (run_dir / 'results.json').write_text(json.dumps(results, indent=1, default=float))
    write_csv(run_dir / 'laps.csv', results['laps'])
    write_csv(run_dir / 'segments.csv', results['segments'])
    write_csv(run_dir / 'hits.csv', results['hits'],
              ['hit', 't_s', 'duration_s', 'lap', 'segment', 's_m', 'x', 'y', 'speed_mps',
               'lateral_error_m'])
    append_index(run_dir.parent, meta, results)
    print(text, flush=True)
    try:
        plot(run_dir, meta, track, results, series)
        print(f'plot: {run_dir / "run.png"}', flush=True)
    except Exception as exc:          # a broken matplotlib must not lose the numbers
        print(f'plot failed ({exc!r}); numbers are saved. With the venv active, retry:\n'
              f'  python3 {Path(__file__).resolve()} {run_dir}', flush=True)
    return results


if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit(f'usage: {Path(sys.argv[0]).name} <run folder, e.g. ../data/2026-10-29_19-02-11>')
    report(sys.argv[1])
