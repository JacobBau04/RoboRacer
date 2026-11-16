#!/usr/bin/env python3
"""Steering and acceleration tests for the real car.

Three modes, plus an analysis step:

  circle   Drives a slow arc (default 0.5 m/s) at each steering angle and records the
           particle-filter pose. At low speed the tires don't slide, so the arc's radius
           gives the real (effective) wheel angle: angle = atan(wheelbase / radius).
           Needs the car stack + particle filter running and localized, and room for
           each arc (the script prints how much before each one).
  stand    Car on a stand, wheels off the ground. Holds each angle and asks you whether
           the wheels moved further than at the previous step. Saves your answers.
  accel    Straight hard launches and hard stops: commands a speed step from rest to each
           target speed, holds it briefly, then commands 0. Reports the acceleration and
           braking reached, motor and battery current, speed overshoot, and wheel spin or
           skid (wheel speed against the particle-filter speed).
           Needs a straight run; the script prints how much room each target needs.
  analyze  Reads a circle- or accel-mode folder and prints the results.

Start the car stack (and for circle mode the particle filter) first. The MPC must NOT be
running. Hold the autonomous button (the one you hold for the MPC) while the script
drives: the mux only passes /drive while it is held, so letting go stops the car.

    python3 ~/f1tenth_ws/src/jacob_opt_obstacles/mpc/tools/steering_test.py circle
    python3 ~/f1tenth_ws/src/jacob_opt_obstacles/mpc/tools/steering_test.py circle --angles 0.3 0.3875 --speed 0.4
    python3 ~/f1tenth_ws/src/jacob_opt_obstacles/mpc/tools/steering_test.py stand
    python3 ~/f1tenth_ws/src/jacob_opt_obstacles/mpc/tools/steering_test.py accel --speeds 1.5 2.0 2.5 --repeats 2
    python3 ~/f1tenth_ws/src/jacob_opt_obstacles/mpc/tools/steering_test.py analyze ~/steering_tests/<folder>

Output: ~/steering_tests/<date>_<mode>/ with poses.csv, servo.csv, cmd.csv, wheel_odom.csv,
vesc.csv, segments.csv (circle) / accel_segments.csv (accel) or stand_results.csv (stand),
plus meta.json and a copy of vesc.yaml.
"""
import argparse
import csv
import datetime
import json
import math
import os
import shutil
import sys
import threading
import time
from pathlib import Path

WB = 0.324         # wheelbase [m], same as mpc_node.py
LIDAR_AHEAD = 0.27  # particle-filter pose (LiDAR) ahead of the rear axle [m] (bringup tf)
# <ws>/src/jacob_opt_obstacles/mpc/tools/steering_test.py -> <ws>/src/f1tenth_system/...
VESC_YAML = Path(__file__).resolve().parents[3] / 'f1tenth_system/f1tenth_stack/config/vesc.yaml'
OUT_ROOT = Path.home() / 'steering_tests'

CIRCLE_ANGLES = [0.20, 0.30, 0.35, 0.3875, -0.20, -0.30, -0.35, -0.40]
STAND_ANGLES = [0.0, 0.20, 0.30, 0.34, 0.3875, 0.45, 0.0, -0.20, -0.30, -0.40, -0.4875, 0.0]


# ----------------------------------------------------------------------------- ROS part
def make_node(out_dir, record):
    import rclpy
    from rclpy.node import Node
    from ackermann_msgs.msg import AckermannDriveStamped
    from nav_msgs.msg import Odometry
    from std_msgs.msg import Float64

    class SteeringTest(Node):
        def __init__(self):
            super().__init__('steering_test')
            self.pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)
            self.target = None              # (speed, steer) to publish, None = publish nothing
            self.lock = threading.Lock()
            self.create_timer(0.05, self.tick)   # 20 Hz, well inside the mux's 0.2 s timeout
            self.files = {}
            self.last_servo = float('nan')
            self.last_pose_t = None
            self.last_wheel_v = 0.0
            self.last_wheel_t = None
            if record:
                self.files['poses'] = self.open_csv('poses.csv', ['t', 'x', 'y', 'yaw', 'vx'])
                self.files['servo'] = self.open_csv('servo.csv', ['t', 'servo'])
                self.files['cmd'] = self.open_csv('cmd.csv', ['t', 'speed', 'steer'])
                self.files['odom'] = self.open_csv('wheel_odom.csv', ['t', 'v'])
                self.create_subscription(Odometry, 'pf/pose/odom', self.on_pose, 10)
                self.create_subscription(AckermannDriveStamped, '/ackermann_cmd', self.on_cmd, 10)
                self.create_subscription(Odometry, '/odom', self.on_odom, 10)
                try:   # VESC state: motor speed (erpm), motor/battery current, voltage, duty
                    from vesc_msgs.msg import VescStateStamped
                    self.files['vesc'] = self.open_csv(
                        'vesc.csv', ['t', 'erpm', 'current_motor', 'current_input', 'voltage', 'duty'])
                    self.create_subscription(VescStateStamped, '/sensors/core', self.on_vesc, 10)
                except ImportError:
                    self.get_logger().warn('vesc_msgs not found: motor current will not be recorded')
            # the servo value after vesc.yaml's servo_min/servo_max clamp (vesc_driver)
            self.create_subscription(Float64, '/sensors/servo_position_command', self.on_servo, 10)

        def open_csv(self, name, header):
            f = open(out_dir / name, 'w', newline='')
            w = csv.writer(f)
            w.writerow(header)
            return f, w

        def write(self, key, row):
            if key in self.files:
                self.files[key][1].writerow(row)

        def tick(self):
            with self.lock:
                tgt = self.target
            if tgt is None:
                return
            m = AckermannDriveStamped()
            m.header.stamp = self.get_clock().now().to_msg()
            m.drive.speed = float(tgt[0])
            m.drive.steering_angle = float(tgt[1])
            self.pub.publish(m)

        def set_target(self, speed, steer):
            with self.lock:
                self.target = None if speed is None else (speed, steer)

        def on_pose(self, m):
            q = m.pose.pose.orientation
            yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
            self.last_pose_t = time.monotonic()
            self.write('poses', [time.monotonic(), m.pose.pose.position.x, m.pose.pose.position.y,
                                 yaw, m.twist.twist.linear.x])

        def on_cmd(self, m):
            self.write('cmd', [time.monotonic(), m.drive.speed, m.drive.steering_angle])

        def on_odom(self, m):
            self.last_wheel_v = m.twist.twist.linear.x
            self.last_wheel_t = time.monotonic()
            self.write('odom', [time.monotonic(), m.twist.twist.linear.x])

        def on_vesc(self, m):
            s = m.state
            self.write('vesc', [time.monotonic(), s.speed, s.current_motor, s.current_input,
                                s.voltage_input, s.duty_cycle])

        def on_servo(self, m):
            self.last_servo = m.data
            self.write('servo', [time.monotonic(), m.data])

        def close(self):
            for f, _ in self.files.values():
                f.close()

    from rclpy.executors import SingleThreadedExecutor
    rclpy.init()
    node = SteeringTest()
    ex = SingleThreadedExecutor()
    ex.add_node(node)
    th = threading.Thread(target=ex.spin, daemon=True)
    th.start()

    def shutdown():
        node.close()
        ex.shutdown()
        th.join(timeout=2.0)
        node.destroy_node()
        rclpy.try_shutdown()
    return node, shutdown


def stop_car(node, seconds=1.5):
    node.set_target(0.0, 0.0)
    time.sleep(seconds)
    node.set_target(None, 0.0)


def read_vesc_yaml():
    vals = {}
    try:
        for line in VESC_YAML.read_text().splitlines():
            line = line.split('#')[0].strip()
            if ':' in line:
                k, v = (s.strip() for s in line.split(':', 1))
                try:
                    vals[k] = float(v)
                except ValueError:
                    pass
    except OSError:
        pass
    return vals


def new_out_dir(mode):
    d = OUT_ROOT / f"{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}_{mode}"
    d.mkdir(parents=True, exist_ok=False)
    if VESC_YAML.exists():
        shutil.copy(VESC_YAML, d / 'vesc.yaml')
    return d


def expected_servo(angle, v):
    g = v.get('steering_angle_to_servo_gain', float('nan'))
    o = v.get('steering_angle_to_servo_offset', float('nan'))
    lo, hi = v.get('servo_min', 0.0), v.get('servo_max', 1.0)
    return min(max(g * angle + o, lo), hi)


# ----------------------------------------------------------------------------- circle mode
def run_circle(args):
    out = new_out_dir('circle')
    vesc = read_vesc_yaml()
    node, shutdown = make_node(out, record=True)
    seg_f = open(out / 'segments.csv', 'w', newline='')
    seg = csv.writer(seg_f)
    seg.writerow(['index', 'steer_cmd', 'speed_cmd', 't_drive_start', 't_drive_end', 'servo_seen'])
    json.dump({'mode': 'circle', 'args': vars(args), 'wheelbase': WB, 'lidar_ahead': LIDAR_AHEAD,
               'vesc': vesc, 'started': datetime.datetime.now().isoformat()},
              open(out / 'meta.json', 'w'), indent=1)
    print(f'\nRecording to {out}')
    time.sleep(1.0)
    if node.last_pose_t is None:
        print('WARNING: no particle-filter pose on pf/pose/odom yet. Start the particle filter and '
              'set the pose, or the arcs cannot be measured.')
    try:
        for i, ang in enumerate(args.angles):
            r = WB / math.tan(abs(ang)) if ang != 0 else float('inf')
            arc = math.radians(args.arc_deg)
            t_drive = min(args.max_time, arc * r / args.speed) if ang != 0 else args.max_time
            side = 'LEFT' if ang > 0 else 'RIGHT'
            print(f'\n[{i + 1}/{len(args.angles)}] steering {ang:+.4f} rad ({side}): expected radius '
                  f'~{r:.2f} m, needs a clear area about {2 * r + 0.6:.1f} m across on the {side.lower()} '
                  f'side; drives {t_drive:.1f} s at {args.speed} m/s.')
            ans = input('  Place the car, CHECK THE LOCALIZATION (set the pose again if needed), hold the '
                        'autonomous button, then press Enter '
                        '(s = skip, q = quit): ').strip().lower()
            if ans == 'q':
                break
            if ans == 's':
                continue
            for k in (3, 2, 1):
                print(f'  {k}...', flush=True)
                time.sleep(0.7)
            node.set_target(0.0, ang)            # turn the wheels first, standing still
            time.sleep(0.8)
            t0 = time.monotonic()
            node.set_target(args.speed, ang)
            time.sleep(t_drive)
            t1 = time.monotonic()
            servo = node.last_servo
            stop_car(node)
            seg.writerow([i, ang, args.speed, t0, t1, servo])
            seg_f.flush()
            print(f'  done. servo value sent: {servo:.3f} (expected from vesc.yaml '
                  f'{expected_servo(ang, vesc):.3f})')
            if node.last_pose_t is None or time.monotonic() - node.last_pose_t > 1.0:
                print('  WARNING: particle-filter pose is not arriving; this arc cannot be measured.')
    except KeyboardInterrupt:
        print('\nstopped')
    finally:
        stop_car(node, 1.0)
        seg_f.close()
        shutdown()
    print(f'\nSaved in {out}\nAnalyze with:\n  python3 {Path(__file__).resolve()} analyze {out}')
    analyze(out)


# ----------------------------------------------------------------------------- stand mode
def run_stand(args):
    out = new_out_dir('stand')
    vesc = read_vesc_yaml()
    node, shutdown = make_node(out, record=False)
    res_f = open(out / 'stand_results.csv', 'w', newline='')
    res = csv.writer(res_f)
    res.writerow(['step', 'steer_cmd', 'servo_expected', 'servo_sent', 'moved_further', 'note'])
    json.dump({'mode': 'stand', 'args': vars(args), 'vesc': vesc,
               'started': datetime.datetime.now().isoformat()}, open(out / 'meta.json', 'w'), indent=1)
    print(f'\nSaving answers to {out}\nCar on the stand, wheels off the ground. Hold the autonomous '
          'button for the whole test. Speed is 0 at every step.')
    try:
        for i, ang in enumerate(args.angles):
            node.set_target(0.0, ang)
            time.sleep(args.hold)
            servo = node.last_servo
            exp = expected_servo(ang, vesc)
            print(f'\n[{i + 1}/{len(args.angles)}] steering {ang:+.4f} rad -> servo sent {servo:.3f} '
                  f'(expected {exp:.3f}). Holding this angle while you look.')
            a = ''
            while a not in ('y', 'n', 'b', '?'):
                a = input('  Compared with the previous step, did the wheels move further in this '
                          'direction?  y = yes, n = no (stopped), b = buzzing/straining, ? = unsure: '
                          ).strip().lower()[:1]
            note = input('  Optional note (Enter to skip): ').strip()
            res.writerow([i, ang, round(exp, 4), servo, a, note])
            res_f.flush()
            if a == 'b':
                print('  Servo straining: going back to straight for 2 s to let it rest.')
                node.set_target(0.0, 0.0)
                time.sleep(2.0)
    except KeyboardInterrupt:
        print('\nstopped')
    finally:
        stop_car(node, 1.0)
        res_f.close()
        shutdown()
    print(f'\nSaved in {out / "stand_results.csv"}')


# ----------------------------------------------------------------------------- accel mode
ACCEL_GUESS, BRAKE_GUESS = 3.0, 1.5   # m/s^2, only to estimate the room each run needs


def run_accel(args):
    out = new_out_dir('accel')
    vesc = read_vesc_yaml()
    node, shutdown = make_node(out, record=True)
    seg_f = open(out / 'accel_segments.csv', 'w', newline='')
    seg = csv.writer(seg_f)
    seg.writerow(['index', 'target', 'steer', 't_go', 't_stop', 't_end', 'reached'])
    json.dump({'mode': 'accel', 'args': vars(args), 'vesc': vesc,
               'started': datetime.datetime.now().isoformat()}, open(out / 'meta.json', 'w'), indent=1)
    print(f'\nRecording to {out}')
    time.sleep(1.0)
    if node.last_wheel_t is None:
        print('WARNING: no wheel odometry on /odom: is the car stack running?')
    if node.last_pose_t is None:
        print('WARNING: no particle-filter pose: acceleration is still measured, but wheel spin '
              'and skid (wheel vs real speed) cannot be checked.')
    runs = [v for v in args.speeds for _ in range(args.repeats)]
    i = 0
    try:
        for i, target in enumerate(runs):
            room = (target ** 2 / (2 * ACCEL_GUESS) + target * args.hold
                    + target ** 2 / (2 * BRAKE_GUESS) + 0.5)
            print(f'\n[{i + 1}/{len(runs)}] launch to {target:.2f} m/s, hold {args.hold:.1f} s, then '
                  f'hard stop. Needs about {room:.1f} m of straight, clear track ahead.')
            ans = input('  Point the car straight down the straight, CHECK THE LOCALIZATION, hold the '
                        'autonomous button, then press Enter (s = skip, q = quit): ').strip().lower()
            if ans == 'q':
                break
            if ans == 's':
                continue
            for k in (3, 2, 1):
                print(f'  {k}...', flush=True)
                time.sleep(0.7)
            node.set_target(0.0, args.steer)        # wheels straight, standing still
            time.sleep(0.5)
            t_go = time.monotonic()
            node.set_target(target, args.steer)     # speed STEP: the VESC decides how fast
            reached, t_hit = False, None
            while time.monotonic() - t_go < args.max_time:
                if node.last_wheel_v >= 0.95 * target and t_hit is None:
                    t_hit = time.monotonic()
                    reached = True
                if t_hit is not None and time.monotonic() - t_hit >= args.hold:
                    break
                time.sleep(0.01)
            t_stop = time.monotonic()
            node.set_target(0.0, args.steer)        # speed step to 0: hard stop
            time.sleep(args.stop_time)
            t_end = time.monotonic()
            node.set_target(None, 0.0)
            seg.writerow([i, target, args.steer, t_go, t_stop, t_end, int(reached)])
            seg_f.flush()
            print(f'  done ({"reached" if reached else "did NOT reach"} {0.95 * target:.2f} m/s '
                  f'within {args.max_time:.1f} s).')
    except KeyboardInterrupt:
        print('\nstopped')
    finally:
        stop_car(node, 1.0)
        seg_f.close()
        shutdown()
    print(f'\nSaved in {out}\nAnalyze again with:\n  python3 {Path(__file__).resolve()} analyze {out}')
    analyze_accel(out, vesc)


def analyze_accel(out, vesc=None):
    """Per launch / stop: acceleration from wheel speed, current, overshoot, spin/skid."""
    import numpy as np
    out = Path(out)
    if vesc is None:
        try:
            vesc = json.load(open(out / 'meta.json')).get('vesc', {})
        except (OSError, ValueError):
            vesc = {}
    load = lambda n, c: (np.loadtxt(out / n, delimiter=',', skiprows=1, ndmin=2)
                         if (out / n).exists() and (out / n).stat().st_size > 60 else np.zeros((0, c)))
    G = load('accel_segments.csv', 7)
    W = load('wheel_odom.csv', 2)
    V = load('vesc.csv', 6)
    P = load('poses.csv', 5)
    C = load('cmd.csv', 3)
    if len(G) == 0 or len(W) < 10:
        print('no launches or no wheel speed recorded')
        return
    tw, vw = W[:, 0], W[:, 1]

    def slope(t0, t1, win=0.10):
        """wheel acceleration on a 10 ms grid, from a +-win/2 central difference"""
        tg = np.arange(t0, t1, 0.01)
        vg = np.interp(tg, tw, vw)
        k = max(1, int(round(win / 0.02)))
        a = (vg[2 * k:] - vg[:-2 * k]) / (2 * k * 0.01)
        return tg[k:-k], vg[k:-k], a

    def pf_dist(t0, t1):
        if len(P) < 5:
            return float('nan')
        m = (P[:, 0] >= t0) & (P[:, 0] <= t1)
        if m.sum() < 4:
            return float('nan')
        x = P[m, 1] - LIDAR_AHEAD * np.cos(P[m, 3])
        y = P[m, 2] - LIDAR_AHEAD * np.sin(P[m, 3])
        every = max(1, int(round(0.25 / max(float(np.median(np.diff(P[m, 0]))), 1e-3))))
        xs, ys = np.r_[x[::every], x[-1]], np.r_[y[::every], y[-1]]
        return float(np.hypot(np.diff(xs), np.diff(ys)).sum())

    def wheel_dist(t0, t1):
        tg = np.arange(t0, t1, 0.01)
        return float(np.sum(np.interp(tg, tw, vw)) * 0.01)

    def crossing(tg, vg, level, rising):
        idx = np.flatnonzero(vg >= level) if rising else np.flatnonzero(vg <= level)
        return tg[idx[0]] if len(idx) else float('nan')

    rows = []
    ramp = None
    print('\nWheel-speed numbers (the VESC\'s own speed). The particle filter has measured the car '
          'moving about 0.95x the wheel distance in normal driving, so real acceleration is ~5 % lower.')
    print('\n target  | LAUNCH: avg 10-90%  peak   t to 90%  overshoot  peak motor A  peak batt A  pf/wheel'
          ' | STOP: avg 90-10%  peak   min motor A  min batt A  pf/wheel | notes')
    for idx_, target, steer, t_go, t_stop, t_end, reached in G:
        notes = []
        mc = (C[:, 0] >= t_go) & (C[:, 0] <= t_stop) if len(C) else np.zeros(0, bool)
        if len(C) and mc.sum() and np.mean(np.abs(C[mc, 1] - target) < 1e-3) < 0.9:
            notes.append('button let go during launch?')
        # launch
        tg, vg, a = slope(t_go - 0.2, t_stop)
        t10 = crossing(tg, vg, 0.1 * target, True)
        t90 = crossing(tg, vg, 0.9 * target, True)
        avg_up = 0.8 * target / (t90 - t10) if t90 > t10 else float('nan')
        peak_up = float(a.max()) if len(a) else float('nan')
        over = float(np.interp(np.arange(t_go, t_stop + 0.6, 0.01), tw, vw).max() - target)
        mv = (V[:, 0] >= t_go) & (V[:, 0] <= t_stop) if len(V) else np.zeros(0, bool)
        cur_up = float(np.abs(V[mv, 2]).max()) if mv.sum() else float('nan')
        bat_up = float(V[mv, 3].max()) if mv.sum() else float('nan')
        r_up = pf_dist(t_go, t_stop) / max(wheel_dist(t_go, t_stop), 1e-3)
        if not reached:
            notes.append('did not reach target')
        # stop
        v0 = float(np.interp(t_stop, tw, vw))
        tg2, vg2, a2 = slope(t_stop - 0.1, t_end)
        s90 = crossing(tg2, vg2, 0.9 * v0, False)
        s10 = crossing(tg2, vg2, 0.1 * v0, False)
        avg_dn = 0.8 * v0 / (s10 - s90) if s10 > s90 else float('nan')
        peak_dn = float(-a2.min()) if len(a2) else float('nan')
        ms = (V[:, 0] >= t_stop) & (V[:, 0] <= t_end) if len(V) else np.zeros(0, bool)
        cur_dn = float(V[ms, 2].min()) if ms.sum() else float('nan')
        bat_dn = float(V[ms, 3].min()) if ms.sum() else float('nan')
        r_dn = pf_dist(t_stop, t_end) / max(wheel_dist(t_stop, t_end), 1e-3)
        if r_up < 0.85:
            notes.append('wheel spin on launch?')
        if r_dn > 1.10:
            notes.append('wheels locked / skidding on stop?')
        if over > 0.15 * target:
            notes.append('big overshoot')
        rows.append([target, avg_up, peak_up, t90 - t_go, over, cur_up, bat_up, r_up,
                     v0, avg_dn, peak_dn, cur_dn, bat_dn, r_dn, '; '.join(notes)])
        print(f' {target:4.2f}    |   {avg_up:5.2f}      {peak_up:5.2f}   {t90 - t_go:5.2f} s   {over:+5.2f}'
              f'       {cur_up:5.1f}        {bat_up:5.1f}      {r_up:4.2f}  |   {avg_dn:5.2f}     {peak_dn:5.2f}'
              f'     {cur_dn:6.1f}      {bat_dn:6.1f}     {r_dn:4.2f}  | {"; ".join(notes)}')
    with open(out / 'accel_results.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['target', 'accel_avg', 'accel_peak', 't90', 'overshoot', 'motor_A_peak', 'batt_A_peak',
                    'pf_over_wheel_launch', 'v_at_stop', 'brake_avg', 'brake_peak', 'motor_A_min',
                    'batt_A_min', 'pf_over_wheel_stop', 'notes'])
        w.writerows(rows)
    g = abs(vesc.get('speed_to_erpm_gain', float('nan')))
    print(f'\nwritten {out / "accel_results.csv"}')
    print(f'For reference: a VESC speed ramp of R ERPM/s limits the setpoint to R / {g:.0f} m/s^2 '
          f'(14000 -> {14000 / g:.2f}, 20000 -> {20000 / g:.2f}, 25000 -> {25000 / g:.2f}). '
          'If "avg" sits near that, the ramp is the limit; if motor current sits at the VESC current '
          'limit, the motor is; if pf/wheel drops well below ~0.95, the tires are spinning.')


# ----------------------------------------------------------------------------- analysis
def analyze(out):
    """Effective wheel angle per arc, from the clean part of each arc only.

    Uses only samples where the test command actually reached the car (/ackermann_cmd
    equals it, so not after the autonomous button was let go) and the wheels were turning.
    Takes the longest such stretch, measures turn / distance of the rear axle, and marks
    the arc BAD if the particle filter jumped, the stretch is too short, the turn rate
    changed a lot along the arc, or the pose distance disagrees with wheel odometry."""
    import numpy as np
    out = Path(out)
    try:
        P = np.loadtxt(out / 'poses.csv', delimiter=',', skiprows=1, ndmin=2)
        G = np.loadtxt(out / 'segments.csv', delimiter=',', skiprows=1, ndmin=2)
        C = np.loadtxt(out / 'cmd.csv', delimiter=',', skiprows=1, ndmin=2)
        W = np.loadtxt(out / 'wheel_odom.csv', delimiter=',', skiprows=1, ndmin=2)
        S = np.loadtxt(out / 'servo.csv', delimiter=',', skiprows=1, ndmin=2)
    except (OSError, ValueError) as e:
        print(f'nothing to analyze in {out}: {e}')
        return
    if len(P) == 0 or len(G) == 0 or len(C) == 0 or len(W) == 0:
        print('missing poses, arcs, commands or wheel odometry')
        return
    rows = []
    print('\n cmd [rad]  servo  used[s]  turn[deg]  effective [rad]  eff/cmd  quality')
    for _, ang, spd, t0, t1, _ in G:
        m = (P[:, 0] > t0) & (P[:, 0] < t1 + 0.5)
        t, x, y, yaw = P[m, 0], P[m, 1], P[m, 2], np.unwrap(P[m, 3])
        if len(t) < 10:
            print(f' {ang:+.4f}   no poses')
            continue
        xr, yr = x - LIDAR_AHEAD * np.cos(yaw), y - LIDAR_AHEAD * np.sin(yaw)   # rear axle
        jc = np.searchsorted(C[:, 0], t).clip(1, len(C) - 1) - 1
        vw = np.interp(t, W[:, 0], W[:, 1])
        good = (np.abs(C[jc, 2] - ang) < 1e-3) & (C[jc, 1] > 0.3) & (vw > 0.35) & (t > t0 + 1.0)
        best, start = None, None                     # longest run of good samples
        for k in range(len(good) + 1):
            g = k < len(good) and good[k]
            if g and start is None:
                start = k
            if not g and start is not None:
                if best is None or t[k - 1] - t[start] > t[best[1]] - t[best[0]]:
                    best = (start, k - 1)
                start = None
        if best is None:
            print(f' {ang:+.4f}   test command never reached the car while moving')
            continue
        a_, b_ = best
        k = slice(a_, b_ + 1)
        step = np.hypot(np.diff(xr[k]), np.diff(yr[k]))   # raw steps: only for jump detection
        # distance along a path sampled every ~0.25 s: summing raw 26 Hz steps adds the pose's
        # cm-level jitter and made arcs look 25-35 % longer than they were
        every = max(1, int(round(0.25 / max(float(np.median(np.diff(t[k]))), 1e-3))))
        xs, ys = xr[k][::every], yr[k][::every]
        if (len(xr[k]) - 1) % every:
            xs, ys = np.r_[xs, xr[k][-1]], np.r_[ys, yr[k][-1]]
        dist = float(np.hypot(np.diff(xs), np.diff(ys)).sum())
        dist_wh = 0.95 * float(np.sum(0.5 * (vw[k][1:] + vw[k][:-1]) * np.diff(t[k])))  # 0.95 = odom scale
        turn = yaw[b_] - yaw[a_]
        used = t[b_] - t[a_]
        eff = math.atan(WB * turn / dist) if dist > 0.2 else float('nan')
        thirds = []
        for q in range(3):
            s0, s1 = a_ + (b_ - a_) * q // 3, a_ + (b_ - a_) * (q + 1) // 3
            dd = np.hypot(np.diff(xr[s0:s1 + 1:every]), np.diff(yr[s0:s1 + 1:every])).sum()
            if dd > 0.1:
                thirds.append(math.atan(WB * (yaw[s1] - yaw[s0]) / dd))
        problems = []
        if step.size and step.max() > 0.15:
            problems.append(f'pose jump {step.max():.2f} m')
        if used < 2.5 or abs(turn) < math.radians(45):
            problems.append('too short')
        if len(thirds) == 3 and max(thirds) - min(thirds) > 0.12:   # thirds are short: ~0.05 rad noise
            problems.append('turn rate not steady')
        if dist_wh > 0.2 and abs(dist / dist_wh - 1) > 0.15:
            problems.append(f'pose vs wheel distance {dist / dist_wh:.2f}')
        q = 'OK' if not problems else 'BAD: ' + ', '.join(problems)
        sm = (S[:, 0] > t[a_]) & (S[:, 0] < t[b_]) if len(S) else np.zeros(0, bool)
        sv = float(np.median(S[sm, 1])) if sm.any() else float('nan')
        rows.append((ang, sv, round(used, 2), round(math.degrees(turn), 1), round(eff, 4), q))
        print(f' {ang:+.4f}   {sv:.3f}   {used:5.1f}    {math.degrees(turn):+6.0f}       {eff:+.3f}'
              f'          {eff / ang:.2f}    {q}')
    if rows:
        with open(out / 'results.csv', 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['steer_cmd', 'servo', 'used_s', 'turn_deg', 'effective_rad', 'quality'])
            w.writerows(rows)
        print(f'\nwritten {out / "results.csv"}  (only trust rows marked OK)')


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='mode', required=True)
    c = sub.add_parser('circle', help='drive slow arcs and record poses')
    c.add_argument('--angles', type=float, nargs='+', default=CIRCLE_ANGLES)
    c.add_argument('--speed', type=float, default=0.5, help='m/s (keep low: no tire slip)')
    c.add_argument('--arc-deg', type=float, default=160.0, help='how much of a circle to drive')
    c.add_argument('--max-time', type=float, default=15.0, help='longest drive per angle [s]')
    s = sub.add_parser('stand', help='car on a stand: step through angles, record what you see')
    s.add_argument('--angles', type=float, nargs='+', default=STAND_ANGLES)
    s.add_argument('--hold', type=float, default=1.5, help='seconds before asking at each angle')
    x = sub.add_parser('accel', help='straight hard launches and stops: max acceleration/braking')
    x.add_argument('--speeds', type=float, nargs='+', default=[1.5, 2.0, 2.5], help='target speeds [m/s]')
    x.add_argument('--repeats', type=int, default=2, help='launches per target speed')
    x.add_argument('--hold', type=float, default=0.4, help='seconds at target speed before stopping')
    x.add_argument('--stop-time', type=float, default=2.0, help='seconds of 0 m/s command after the stop')
    x.add_argument('--max-time', type=float, default=3.0, help='give up launching after this [s]')
    x.add_argument('--steer', type=float, default=0.0, help='steering during the run [rad]')
    a = sub.add_parser('analyze', help='analyze a circle- or accel-mode folder')
    a.add_argument('folder')
    args = ap.parse_args()
    if args.mode == 'analyze':
        if (Path(args.folder) / 'accel_segments.csv').exists():
            analyze_accel(args.folder)
        else:
            analyze(args.folder)
    elif args.mode == 'accel':
        if max(args.speeds) > 4.0:
            sys.exit('target above 4.0 m/s: too fast for a first test in this space')
        run_accel(args)
    elif args.mode == 'circle':
        if args.speed > 1.0:
            sys.exit('speed > 1.0 m/s: the tires may slide and spoil the measurement')
        run_circle(args)
    else:
        run_stand(args)


if __name__ == '__main__':
    main()
