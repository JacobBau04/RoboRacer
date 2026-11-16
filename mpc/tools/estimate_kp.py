#!/usr/bin/env python3
"""Estimate how fast the car's speed follows a speed command, from a joystick step test.

Model (the one mpc_node.py's commented-out speed conversion assumes, plus a limit):
    dv/dt = kp * (v_cmd(t - delay) - v),   but at most a_max in size
kp and a_max are fitted separately for speeding up and slowing down, with one delay
shared by both. kp = how hard the VESC pushes per m/s of gap (small steps show it);
a_max = the most it gives (big steps show it: the speed then changes in a straight line,
the VESC's current or brake limit). With these, the MPC's acceleration a becomes
    speed_output = v + a / kp           (valid while |a| < a_max)

Record while driving with the joystick (car stack running):
    ros2 bag record /ackermann_cmd /odom /commands/motor/speed /sensors/core -o ~/kp_test
Then:
    python3 estimate_kp.py ~/kp_test
    python3 estimate_kp.py ~/kp_test --show        # also open the plot window
    python3 estimate_kp.py --selftest              # check the fit on made-up data

Writes <bag>/kp_fit.png: every step with the command, the measured speed and the model.

Speeds: command = /commands/motor/speed (what the VESC was told, erpm) / gain, or
/ackermann_cmd if that is missing. Measured = /sensors/core erpm / gain (50 Hz, no
deadband), or /odom. The same gain is used for both, so a slightly wrong gain does not
change kp.
"""
import argparse
import os
import sys

import numpy as np

DT = 0.01                              # s, analysis grid
KP_GRID = np.geomspace(0.3, 40.0, 110)  # 1/s
LIM_GRID = np.array([1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 7, 8, 10, 13, np.inf])  # m/s^2
KPF, LIMF = (g.ravel() for g in np.meshgrid(KP_GRID, LIM_GRID))   # every (kp, a_max) pair
DELAY_GRID = np.arange(0.0, 0.61, 0.01)  # s
MAX_DELAY = DELAY_GRID[-1]
VESC_YAML = os.path.expanduser(
    '~/f1tenth_ws/src/f1tenth_system/f1tenth_stack/config/vesc.yaml')


# ---------------------------------------------------------------- reading the bag
def read_gain(path):
    import yaml
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return float(cfg['/**']['ros__parameters']['speed_to_erpm_gain'])


def read_bag(bag):
    """Return dict of arrays: motor_cmd (t, erpm), ack (t, speed, steer),
    core (t, erpm), odom (t, v). Times = receive time [s]."""
    import yaml
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    if os.path.isfile(bag):
        bag = os.path.dirname(bag)
    storage = 'mcap'
    meta_path = os.path.join(bag, 'metadata.yaml')
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            storage = yaml.safe_load(f)['rosbag2_bagfile_information'].get(
                'storage_identifier', 'mcap')
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=bag, storage_id=storage),
                rosbag2_py.ConverterOptions('cdr', 'cdr'))
    types = {t.name: get_message(t.type) for t in reader.get_all_topics_and_types()}
    out = {'motor_cmd': [], 'ack': [], 'core': [], 'odom': []}
    while reader.has_next():
        topic, data, t_ns = reader.read_next()
        if topic not in ('/commands/motor/speed', '/ackermann_cmd', '/sensors/core', '/odom'):
            continue
        t = t_ns * 1e-9
        msg = deserialize_message(data, types[topic])
        if topic == '/commands/motor/speed':
            out['motor_cmd'].append((t, msg.data))
        elif topic == '/ackermann_cmd':
            out['ack'].append((t, msg.drive.speed, msg.drive.steering_angle))
        elif topic == '/sensors/core':
            out['core'].append((t, msg.state.speed))
        else:
            out['odom'].append((t, msg.twist.twist.linear.x))
    cols = {'motor_cmd': 2, 'ack': 3, 'core': 2, 'odom': 2}
    return {k: np.array(v, float).reshape(-1, cols[k]) for k, v in out.items()}


def bag_to_signals(raw, gain):
    """Pick the best command / measurement topics and convert to m/s."""
    notes = []
    if len(raw['motor_cmd']) > 10:
        cmd_t, cmd_v = raw['motor_cmd'][:, 0], raw['motor_cmd'][:, 1] / gain
        notes.append('command: /commands/motor/speed')
    elif len(raw['ack']) > 10:
        cmd_t, cmd_v = raw['ack'][:, 0], raw['ack'][:, 1]
        notes.append('command: /ackermann_cmd (/commands/motor/speed not in the bag)')
    else:
        sys.exit('no speed commands in the bag (/commands/motor/speed or /ackermann_cmd)')
    if len(raw['core']) > 10:
        meas_t, meas_v = raw['core'][:, 0], raw['core'][:, 1] / gain
        notes.append('measured: /sensors/core erpm / gain')
    elif len(raw['odom']) > 10:
        meas_t, meas_v = raw['odom'][:, 0], raw['odom'][:, 1]
        notes.append('measured: /odom (|v| < 0.05 m/s reads as 0)')
    else:
        sys.exit('no measured speed in the bag (/sensors/core or /odom)')
    steer = raw['ack'][:, [0, 2]] if len(raw['ack']) else None
    return cmd_t, cmd_v, meas_t, meas_v, steer, notes


# ---------------------------------------------------------------- the fit
def to_grid(cmd_t, cmd_v, meas_t, meas_v, steer):
    t0 = max(cmd_t[0], meas_t[0])
    t1 = min(cmd_t[-1], meas_t[-1])
    t = np.arange(t0, t1, DT)
    i = np.clip(np.searchsorted(cmd_t, t, side='right') - 1, 0, len(cmd_t) - 1)
    cmd = cmd_v[i]                                     # zero-order hold, like the VESC
    # a gap in the command stream = the VESC was not being told anything: not usable
    gap = (t - cmd_t[i]) > 0.15
    v = np.interp(t, meas_t, meas_v)   # not smoothed: smoothing rounds the step's start,
                                       # which the fit would read as an accel limit
    st = np.zeros_like(t)
    if steer is not None and len(steer):
        j = np.clip(np.searchsorted(steer[:, 0], t, side='right') - 1, 0, len(steer) - 1)
        st = steer[j, 1]
    return t - t0, cmd, v, st, gap


def find_steps(cmd, v, gap, min_step, min_hold, max_win):
    """Joystick snaps: the command moves by >= min_step within 0.2 s and then stays
    (within 0.1 m/s) for >= min_hold. Returns (start, end, target) index triples."""
    W = int(0.2 / DT)
    pre = int(MAX_DELAY / DT) + 1
    steps, i, n = [], pre, len(cmd)
    while i < n - W - 1:
        if gap[i] or abs(cmd[i + W] - cmd[i]) < min_step:
            i += 1
            continue
        base = np.median(cmd[max(i - 5, 0):i + 1])
        delta = cmd[i + W] - base
        seg = np.abs(cmd[i:i + W + 1] - base)
        # start = last sample before the command has moved 10% of the step,
        # end of the move = first sample past 90%
        s = i + max(int(np.argmax(seg > max(0.1 * abs(delta), 0.03))) - 1, 0)
        j = i + int(np.argmax(seg >= 0.9 * abs(delta)))
        target = float(np.median(cmd[j:j + int(0.1 / DT) + 1]))
        e = j
        while e + 1 < n and abs(cmd[e + 1] - target) < 0.1 and not gap[e + 1] \
                and (e - s) * DT < max_win:
            e += 1
        if (e - s) * DT >= min_hold and abs(target - base) >= min_step:
            steps.append((s, e, target))
            i = e + 1
        else:
            i = j + 1          # not a clean step (stick not held): look again after the move
    return steps


def simulate(cmd_hist, v0, kp, lim, delay, n, pre):
    """dv/dt = kp (cmd(t-delay) - v) with |dv/dt| <= lim, zero-order hold cmd (exact
    step when not limited). kp, lim may be arrays (vectorised). Returns v[0..n-1]."""
    lag = int(round(delay / DT))
    c = cmd_hist[pre - lag: pre - lag + n]
    a = np.exp(-np.asarray(kp, float) * DT)
    dmax = np.asarray(lim, float) * DT
    x = np.full(np.broadcast(a, dmax).shape, v0, float)
    out = np.empty((n,) + x.shape)
    out[0] = x
    for k in range(n - 1):
        x = x + np.clip((c[k] - x) * (1.0 - a), -dmax, dmax)
        out[k + 1] = x
    return out


def best_pair(sse_row):
    """Index of the best (kp, a_max) pair. A limit the data never reaches fits just as
    well as no limit, so among fits within 1% of the best take the highest a_max."""
    m = sse_row.min()
    ok = np.where(sse_row <= m * 1.01 + 1e-12)[0]
    top = ok[LIMF[ok] == LIMF[ok].max()]
    return int(top[np.argmin(sse_row[top])])


def sse_grid(cmd, v, s, e):
    """Sum of squared errors over (delay, (kp, a_max) pair) for one step window [s, e]."""
    pre = int(MAX_DELAY / DT) + 1
    hist = cmd[s - pre: e + 1]
    meas = v[s: e + 1]
    n = len(meas)
    out = np.empty((len(DELAY_GRID), len(KPF)))
    for a, d in enumerate(DELAY_GRID):
        sim = simulate(hist, meas[0], KPF, LIMF, d, n, pre)
        out[a] = ((sim - meas[:, None]) ** 2).sum(axis=0)
    return out, n


def analyze(t, cmd, v, steer, gap, args):
    steps = find_steps(cmd, v, gap, args.min_step, args.min_hold, args.max_window)
    rows = []
    skipped = {'reverse': 0, 'not settled': 0}
    for s, e, target in steps:
        v0 = v[s]
        base = np.median(cmd[max(s - 5, 0):s + 1])
        if min(target, base, v0) < -0.1:            # the MPC never reverses
            skipped['reverse'] += 1
            continue
        # the car must be at the old command when the step starts (stick held long enough),
        # or the step mixes in the response to the previous one
        if abs(v0 - base) > max(0.15, 0.3 * abs(target - base)):
            skipped['not settled'] += 1
            continue
        kind = 'up' if abs(target) > abs(v0) else 'down'
        from_stop = abs(v0) < 0.3
        to_stop = abs(target) < 0.15
        if to_stop:                      # stop using the window once the car has stopped:
            stop = np.where(np.abs(v[s:e + 1]) < 0.2)[0]   # near 0 the VESC holds a brake
            if len(stop):
                e = s + max(stop[0], int(0.3 / DT))
        if (e - s) * DT < 0.3:
            continue
        sse, n = sse_grid(cmd, v, s, e)
        a = int(np.argmin(sse.min(axis=1)))
        b = best_pair(sse[a])
        k = 5                                          # smoothed only for the peak column
        vs = np.convolve(np.pad(v[s:e + 1], k, mode='edge'), np.ones(2 * k + 1) / (2 * k + 1),
                         'valid')
        acc = np.gradient(vs, DT)
        group = ('from stop' if from_stop else 'to stop' if to_stop else kind)
        rows.append(dict(s=s, e=e, t=t[s], v0=v0, target=target, gap=target - v0, kind=kind,
                         group=group, sse=sse, n=n, kp=KPF[b], lim=LIMF[b],
                         delay=DELAY_GRID[a],
                         rmse=np.sqrt(sse[a, b] / n),
                         peak_acc=acc.max() if kind == 'up' else acc.min(),
                         steer=np.mean(np.abs(steer[s:e + 1]))))
    analyze.skipped = skipped
    return rows


def fit_groups(rows, groups_up, groups_down):
    """Shared delay; kp and a_max for up and for down: minimise total SSE."""
    up = [r for r in rows if r['group'] in groups_up]
    down = [r for r in rows if r['group'] in groups_down]
    if not up and not down:
        return None
    G_up = sum((r['sse'] for r in up), np.zeros((len(DELAY_GRID), len(KPF))))
    G_dn = sum((r['sse'] for r in down), np.zeros((len(DELAY_GRID), len(KPF))))
    total = (G_up.min(axis=1) if up else 0) + (G_dn.min(axis=1) if down else 0)
    a = int(np.argmin(total))
    bu, bd = best_pair(G_up[a]), best_pair(G_dn[a])
    res = dict(delay=DELAY_GRID[a], n_up=len(up), n_down=len(down),
               kp_up=KPF[bu] if up else np.nan, lim_up=LIMF[bu] if up else np.nan,
               kp_down=KPF[bd] if down else np.nan, lim_down=LIMF[bd] if down else np.nan)
    n = sum(r['n'] for r in up + down)
    res['rmse'] = np.sqrt(((G_up[a, bu] if up else 0) + (G_dn[a, bd] if down else 0)) / n)
    return res


def tau63_kp(rows, t, v, delay):
    """Textbook check: 1 / (time to cover 63% of the step, after the delay)."""
    out = {}
    for r in rows:
        s, e, v0, tg = r['s'], r['e'], r['v0'], r['target']
        if abs(tg - v0) < 0.15:
            continue
        frac = (v[s:e + 1] - v0) / (tg - v0)
        hit = np.where(frac >= 0.632)[0]
        tau = hit[0] * DT - delay if len(hit) else np.nan
        r['kp63'] = 1.0 / tau if tau > 0.02 else np.nan
        out.setdefault(r['group'], []).append(r['kp63'])
    return out


# ---------------------------------------------------------------- report
def lim_str(x):
    return 'none' if not np.isfinite(x) else f'{x:.1f}'


def report(rows, t, cmd, v, args, notes, bag):
    for nline in notes:
        print(nline)
    if not rows:
        print('\nNo usable steps found. Snap the stick between speeds and hold each for '
              f'>= {args.min_hold} s (see the procedure in the script docstring).')
        return
    sk = getattr(analyze, 'skipped', {})
    print(f'\n{len(rows)} steps used' + (f'  (skipped: {sk["reverse"]} reversing, '
          f'{sk["not settled"]} where the car had not settled at the old speed)' if sk else '')
          + '\n')
    print('   t[s]  group      v0 -> target   gap   kp[1/s] a_max delay[s] rmse  peak_acc |steer|')
    for r in rows:
        print(f'{r["t"]:7.1f}  {r["group"]:9s} {r["v0"]:5.2f} -> {r["target"]:5.2f} '
              f'{r["gap"]:+5.2f}  {r["kp"]:6.2f}  {lim_str(r["lim"]):>5s}  {r["delay"]:4.2f}  '
              f'{r["rmse"]:.3f}  {r["peak_acc"]:+5.2f}   {r["steer"]:.2f}')

    main = fit_groups(rows, {'up'}, {'down'})
    used = 'moving-to-moving steps'
    if main is None or main['n_up'] < 2 or main['n_down'] < 2:
        main = fit_groups(rows, {'up', 'from stop'}, {'down', 'to stop'})
        used = 'all steps (not enough moving-to-moving steps: results less reliable)'
    k63 = tau63_kp(rows, t, v, main['delay'])

    print('\nGROUPS (per-step kp: median [min..max])')
    for g in ('up', 'down', 'from stop', 'to stop'):
        ks = [r['kp'] for r in rows if r['group'] == g]
        if ks:
            k6 = [x for x in k63.get(g, []) if np.isfinite(x)]
            print(f'  {g:9s} n={len(ks):2d}  kp {np.median(ks):5.2f} [{min(ks):.2f}..{max(ks):.2f}]'
                  + (f'   63%-rule kp {np.median(k6):5.2f}' if k6 else ''))

    print(f'\nFIT ({used}, one shared delay)')
    print(f'  speeding up: kp {main["kp_up"]:.2f} 1/s, max accel {lim_str(main["lim_up"])} m/s^2'
          f'   ({main["n_up"]} steps)')
    print(f'  slowing:     kp {main["kp_down"]:.2f} 1/s, max decel {lim_str(main["lim_down"])} m/s^2'
          f'   ({main["n_down"]} steps)')
    print(f'  delay {main["delay"]:.2f} s')
    print('  (max = "none" means no limit showed up: steps were too small to reach it)')
    print(f'  model error {main["rmse"]:.3f} m/s rms')

    straight = [r for r in rows if r['steer'] < 0.15]
    if len(straight) < len(rows):
        fs = fit_groups(straight, {'up'}, {'down'})
        if fs and fs['n_up'] >= 2 and fs['n_down'] >= 2:
            print(f'  straight-only steps: kp_up {fs["kp_up"]:.2f}, kp_down {fs["kp_down"]:.2f}, '
                  f'delay {fs["delay"]:.2f}')

    hints = []
    for kind in ('up', 'down'):
        gaps = [abs(r['gap']) for r in rows if r['kind'] == kind]
        if gaps and max(gaps) < 0.6:
            hints.append(f'{kind}: all steps are under 0.6 m/s, too small to show the max '
                         f'{"accel" if kind == "up" else "decel"}. Add some bigger steps.')
    spread = [r['kp'] for r in rows if r['group'] in ('up', 'down')]
    if len(spread) >= 3 and np.max(spread) > 3 * np.min(spread):
        hints.append('per-step kp varies more than 3x: check the plot for steps where the stick '
                     'was not held still, wheel spin, or corners.')
    if main['rmse'] > 0.15:
        hints.append('model error is large: the first-order model fits poorly (see the plot).')
    if hints:
        print('\nHINTS')
        for h in hints:
            print('  - ' + h)

    print('\nIN mpc_node.py (speed conversion):')
    print(f'    kp = {main["kp_up"]:.2f} if self.oa[0] >= 0 else {main["kp_down"]:.2f}')
    print('    speed_output = vehicle_state.v + self.oa[0] / kp')
    lims = [x for x in (main['lim_up'], main['lim_down']) if np.isfinite(x)]
    if lims:
        print(f'  MAX_ACCEL should stay under {min(lims):.1f} m/s^2 (the car cannot do more, '
              'so the MPC would plan with braking it never gets)')
    print(f'  speed delay about {main["delay"]:.2f} s (delay_comp is 0.26 s for pose/steering)')

    if bag is not None or args.show:
        plot(rows, t, cmd, v, main, bag, args.show)


def plot(rows, t, cmd, v, fit, bag, show):
    import matplotlib
    if not show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = rows[:30]
    ncol = 5
    nrow = int(np.ceil(len(rows) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 2.4 * nrow), squeeze=False)
    pre = int(MAX_DELAY / DT) + 1
    for ax, r in zip(axs.flat, rows):
        s, e = r['s'], r['e']
        a0 = max(s - int(0.3 / DT), 0)
        tt = t[a0:e + 1] - t[s]
        ax.plot(tt, cmd[a0:e + 1], color='0.5', ls='--', lw=1, label='command')
        ax.plot(tt, v[a0:e + 1], color='tab:blue', lw=1.5, label='measured')
        kp, lim = ((fit['kp_up'], fit['lim_up']) if r['kind'] == 'up'
                   else (fit['kp_down'], fit['lim_down']))
        if np.isfinite(kp):
            sim = simulate(cmd[s - pre:e + 1], v[s], kp, lim, fit['delay'], e - s + 1, pre)
            ax.plot(t[s:e + 1] - t[s], sim, color='tab:orange', lw=1.5, label='model')
        ax.set_title(f'{r["t"]:.1f}s {r["group"]}  step kp {r["kp"]:.1f}', fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)
    for ax in list(axs.flat)[len(rows):]:
        ax.axis('off')
    axs.flat[0].legend(fontsize=7)
    fig.suptitle(f'speed response: up kp {fit["kp_up"]:.2f} max {lim_str(fit["lim_up"])}, '
                 f'down kp {fit["kp_down"]:.2f} max {lim_str(fit["lim_down"])} m/s^2, '
                 f'delay {fit["delay"]:.2f} s   (x: s from step, y: m/s)')
    fig.tight_layout()
    if bag is not None:
        path = os.path.join(bag if os.path.isdir(bag) else os.path.dirname(bag), 'kp_fit.png')
        fig.savefig(path, dpi=110)
        print(f'\nplot: {path}')
    if show:
        plt.show()


# ---------------------------------------------------------------- self test
def selftest(args):
    """Joystick-like command with known kp_up=3, kp_down=6, braking limited to 4 m/s^2,
    delay 0.25 s, plus noise."""
    rng = np.random.default_rng(0)
    kp_up, kp_dn, delay, brake_lim = 3.0, 6.0, 0.25, 4.0
    levels = [0, 1.0, 0.5, 1.0, 2.0, 1.0, 2.0, 0.6, 1.5, 0.0, 1.2, 0.5, 1.8, 0.8, 0.0]
    cmd_t = np.arange(0, 2.0 * len(levels), 0.02)
    cmd_v = np.zeros_like(cmd_t)
    for i, lv in enumerate(levels):                     # 80 ms snaps between levels
        prev = levels[i - 1] if i else 0.0
        m = (cmd_t >= 2.0 * i)
        ramp = np.clip((cmd_t - 2.0 * i) / 0.08, 0, 1)
        cmd_v[m] = (prev + (lv - prev) * ramp)[m]
    cmd_v += rng.normal(0, 0.01, len(cmd_v))           # stick jitter
    ts = np.arange(0, cmd_t[-1], 0.001)
    c = np.interp(ts - delay, cmd_t, cmd_v, left=0)
    x = np.zeros_like(ts)
    for k in range(len(ts) - 1):
        kp = kp_up if c[k] > x[k] else kp_dn
        x[k + 1] = x[k] + 0.001 * max(kp * (c[k] - x[k]), -brake_lim)
    meas_t = np.arange(0, cmd_t[-1], 0.02)
    meas_v = np.interp(meas_t, ts, x) + rng.normal(0, 0.02, len(meas_t))
    t, cmd, v, st, gap = to_grid(cmd_t, cmd_v, meas_t, meas_v, None)
    rows = analyze(t, cmd, v, st, gap, args)
    print(f'self test: true kp_up {kp_up} (no limit), kp_down {kp_dn} (max decel '
          f'{brake_lim}), delay {delay}\n')
    report(rows, t, cmd, v, args, ['synthetic data'], None)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('bag', nargs='?', help='bag folder (from ros2 bag record -o ...)')
    ap.add_argument('--gain', type=float, help=f'speed_to_erpm_gain (default: from {VESC_YAML})')
    ap.add_argument('--min-step', type=float, default=0.25, help='m/s, smallest step used')
    ap.add_argument('--min-hold', type=float, default=0.6, help='s, stick held at least this')
    ap.add_argument('--max-window', type=float, default=2.0, help='s, fit at most this per step')
    ap.add_argument('--show', action='store_true', help='open the plot window')
    ap.add_argument('--selftest', action='store_true', help='run on made-up data')
    args = ap.parse_args()
    if args.selftest:
        selftest(args)
        return
    if not args.bag:
        ap.error('give the bag folder, e.g. ~/kp_test')
    bag = os.path.expanduser(args.bag)
    gain = args.gain if args.gain is not None else read_gain(VESC_YAML)
    raw = read_bag(bag)
    cmd_t, cmd_v, meas_t, meas_v, steer, notes = bag_to_signals(raw, gain)
    notes.insert(0, f'speed_to_erpm_gain {gain}')
    t, cmd, v, st, gap = to_grid(cmd_t, cmd_v, meas_t, meas_v, steer)
    moving = np.abs(cmd) > 0.3
    if moving.sum() > 100 and np.corrcoef(cmd[moving], v[moving])[0, 1] < 0:
        notes.append('WARNING: measured speed has the opposite sign to the command: '
                     'wrong --gain sign?')
    rows = analyze(t, cmd, v, st, gap, args)
    report(rows, t, cmd, v, args, notes, bag)


if __name__ == '__main__':
    main()
