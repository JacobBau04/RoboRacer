#!/usr/bin/env python3
"""Run race1 obstacle layouts in the simulator and score them.

For each layout (tools/make_obstacle_layouts.py): start f1tenth_gym_ros with that map,
start launch/race1_sim_launch.py, and watch until N laps are done, the car hits something,
gets stuck or the time runs out. Scores lap times, hits, the true clearance between the
car's outline and each box, and how the planner behaved. Everything runs on ROS domain 42
(SIM_DOMAIN), never the car's.

    python3 tools/sim_batch.py --layouts ~/sim_ws/race1_layouts --laps 3
    python3 tools/sim_batch.py --layouts ~/sim_ws/race1_layouts --only clean,layout_004
    python3 tools/sim_batch.py --layouts ~/sim_ws/race1_layouts --avoidance false   # baseline
    python3 tools/sim_batch.py --layouts ~/sim_ws/race2_straights --launch race2_sim_launch.py
"""
import os
os.environ['ROS_DOMAIN_ID'] = os.environ.get('SIM_DOMAIN', '42')    # before rclpy
import argparse                     # noqa: E402
import csv                          # noqa: E402
import json                         # noqa: E402
import math                         # noqa: E402
import signal                       # noqa: E402
import subprocess                   # noqa: E402
import sys                          # noqa: E402
import time                         # noqa: E402

import numpy as np                  # noqa: E402
import rclpy                        # noqa: E402
import yaml                         # noqa: E402
from nav_msgs.msg import Odometry   # noqa: E402
from rclpy.node import Node         # noqa: E402
from std_msgs.msg import Bool, Float32, Float64MultiArray, Int32   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE]
from offline_driveby import polygon_distance   # noqa: E402

HOME = os.path.expanduser('~')
SIM_WS = os.path.join(HOME, 'sim_ws')
CAR_WS = os.path.join(HOME, 'f1tenth_ws')
FRONT, REAR, HW = 0.46, 0.12, 0.155          # sim car body around base_link (rear axle)
# stats fields (obstacle_avoidance_node.STATS_FIELDS)
ST_PROC, ST_PLAN, ST_LEVEL, ST_STOP = 0, 1, 6, 7


def car_outline(x, y, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    pts = [(FRONT, HW), (-REAR, HW), (-REAR, -HW), (FRONT, -HW)]
    return np.array([(x + c * a - s * b, y + s * a + c * b) for a, b in pts])


class Monitor(Node):
    def __init__(self, boxes):
        super().__init__('sim_batch_monitor')
        self.boxes = [np.array(b['corners']) for b in boxes]
        self.min_clear = [math.inf] * len(self.boxes)
        self.t_odom = None
        self.t_move = None
        self.t_still = None
        self.v = 0.0
        self.pose = None
        self.collided = None
        self.lap_count = 0
        self.lap_times = []
        self.lap_wall = []
        self._last_lap_wall = None
        self._last_lap_time = None
        self._last_clear_t = 0.0
        self.levels = []
        self.stops = 0
        self._stop_prev = False
        self.proc_ms, self.plan_ms = [], []
        self.create_subscription(Odometry, '/ego_racecar/odom', self.odom_cb, 10)
        self.create_subscription(Bool, '/ego_racecar/collision', self.coll_cb, 10)
        self.create_subscription(Int32, '/ego_racecar/lap_count', self.lap_cb, 10)
        self.create_subscription(Float32, '/ego_racecar/lap_time', self.lap_time_cb, 10)
        self.create_subscription(Float64MultiArray, '/obstacle_avoidance/stats', self.stats_cb, 10)

    def odom_cb(self, m):
        now = time.monotonic()
        self.t_odom = self.t_odom or now
        p, q = m.pose.pose.position, m.pose.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        self.pose = (p.x, p.y, yaw)
        self.v = m.twist.twist.linear.x
        if self.v > 0.2 and self.t_move is None:
            self.t_move = now
            self._last_lap_wall = now
        self.t_still = None if self.v > 0.05 else (self.t_still or now)
        if self.boxes and now - self._last_clear_t > 0.02:
            self._last_clear_t = now
            car = car_outline(*self.pose)
            for k, B in enumerate(self.boxes):
                if abs(B[0, 0] - p.x) + abs(B[0, 1] - p.y) < 3.0:
                    self.min_clear[k] = min(self.min_clear[k], polygon_distance(car, B))

    def coll_cb(self, m):
        if m.data and self.collided is None and self.pose is not None:
            self.collided = self.pose

    def lap_time_cb(self, m):
        # published every tick; changes only when a lap is completed
        v = float(m.data)
        if self._last_lap_time is not None and v != self._last_lap_time and v > 0:
            now = time.monotonic()
            self.lap_times.append(v)
            if self._last_lap_wall is not None:
                self.lap_wall.append(now - self._last_lap_wall)
            self._last_lap_wall = now
        self._last_lap_time = v

    def lap_cb(self, m):
        self.lap_count = max(self.lap_count, int(m.data))

    def stats_cb(self, m):
        d = list(m.data)
        self.proc_ms.append(d[ST_PROC])
        self.plan_ms.append(d[ST_PLAN])
        self.levels.append(int(d[ST_LEVEL]))
        stop = d[ST_STOP] > 0.5
        if stop and not self._stop_prev:
            self.stops += 1
        self._stop_prev = stop


def start(cmd, log_path):
    log = open(log_path, 'w')
    return subprocess.Popen(['bash', '-c', cmd], stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True, env=dict(os.environ))


def stop(proc, wait=15.0):
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGINT)
        proc.wait(timeout=wait)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
    except ProcessLookupError:
        pass


def sim_config(layout, start_pose, out_dir):
    base = os.path.join(SIM_WS, 'install', 'f1tenth_gym_ros', 'share', 'f1tenth_gym_ros',
                        'config', 'sim.yaml')
    with open(base) as f:
        cfg = yaml.safe_load(f)
    b = cfg['bridge']['ros__parameters']
    b.update(map_path=layout['yaml'], sx=start_pose[0], sy=start_pose[1], stheta=start_pose[2],
             num_agents=1, kb_teleop=False)
    path = os.path.join(out_dir, f'sim_{layout["name"]}.yaml')
    with open(path, 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return path


def other_sim_running(node, wait=10.0):
    """True if something already publishes the sim's odometry on this ROS domain."""
    t0 = time.monotonic()
    while time.monotonic() - t0 < wait:
        rclpy.spin_once(node, timeout_sec=0.2)
        if not node.get_publishers_info_by_topic('/ego_racecar/odom'):
            return False
    return True


def run_layout(layout, start_pose, args, out_dir):
    guard = rclpy.create_node('sim_batch_guard')
    busy = other_sim_running(guard)
    guard.destroy_node()
    if busy:
        return {'layout': layout['name'], 'boxes': len(layout['boxes']),
                'end': 'another sim is running on this ROS domain'}
    cfg = sim_config(layout, start_pose, out_dir)
    sim = start(f'source {SIM_WS}/.venv/bin/activate && source /opt/ros/jazzy/setup.bash && '
                f'source {SIM_WS}/install/setup.bash && export OPENBLAS_NUM_THREADS=1 && '
                f'exec ros2 launch f1tenth_gym_ros gym_bridge_launch.py config:={cfg} '
                f'open_foxglove:=false', os.path.join(out_dir, f'{layout["name"]}_sim.log'))
    mon = Monitor(layout['boxes'])
    nodes = None
    result = {'layout': layout['name'], 'boxes': len(layout['boxes'])}
    try:
        t0 = time.monotonic()
        while mon.t_odom is None and time.monotonic() - t0 < 90:
            rclpy.spin_once(mon, timeout_sec=0.1)
        if mon.t_odom is None:
            result['end'] = 'sim did not start'
            return result
        nodes = start(f'source /opt/ros/jazzy/setup.bash && source {CAR_WS}/install/setup.bash && '
                      f'exec ros2 launch mpc {args.launch} avoidance:={args.avoidance} '
                      f'record:={args.record}', os.path.join(out_dir, f'{layout["name"]}_nodes.log'))
        t_nodes = time.monotonic()
        end = 'timeout'
        while True:
            rclpy.spin_once(mon, timeout_sec=0.05)
            now = time.monotonic()
            if mon.lap_count >= args.laps:
                end = 'laps done'
                break
            if mon.collided is not None:
                end = 'hit'
                break
            if mon.t_move is None and now - t_nodes > 60:
                end = 'never moved'
                break
            if mon.t_move is not None and mon.t_still is not None and now - mon.t_still > args.stuck:
                end = 'stuck'
                break
            if mon.t_move is not None and now - mon.t_move > args.laps * args.lap_timeout + 10:
                break
        result['end'] = end
    finally:
        stop(nodes)
        stop(sim)
        mon.destroy_node()
    lv = np.array(mon.levels) if mon.levels else np.array([-1])
    result.update(
        laps=mon.lap_count,
        lap_times=' '.join(f'{x:.2f}' for x in mon.lap_times if x is not None),
        best_lap=min([x for x in mon.lap_times if x], default=float('nan')),
        wall_per_sim=(np.mean(mon.lap_wall) / np.mean([x for x in mon.lap_times if x])
                      if mon.lap_wall and any(mon.lap_times) else float('nan')),
        hit_at=('' if mon.collided is None else f'({mon.collided[0]:.2f}, {mon.collided[1]:.2f})'),
        min_clear=min(mon.min_clear, default=float('nan')),
        clear_each=' '.join(f'{c:.2f}' for c in mon.min_clear),
        planned_frac=float(np.mean(lv >= 0)), tight_frac=float(np.mean(lv >= 1)),
        crawl_frac=float(np.mean(lv >= 2)), stops=mon.stops,
        proc_ms_p95=float(np.percentile(mon.proc_ms, 95)) if mon.proc_ms else float('nan'),
        plan_ms_max=float(np.max(mon.plan_ms)) if mon.plan_ms else float('nan'))
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--layouts', required=True, help='folder from make_obstacle_layouts.py')
    ap.add_argument('--only', default='', help='comma-separated layout names')
    ap.add_argument('--laps', type=int, default=3)
    ap.add_argument('--lap-timeout', type=float, default=20.0, help='s per lap before giving up')
    ap.add_argument('--stuck', type=float, default=6.0, help='s standing still = stuck')
    ap.add_argument('--avoidance', default='true')
    ap.add_argument('--launch', default='race1_sim_launch.py', help='race2: race2_sim_launch.py')
    ap.add_argument('--record', default='false', help='run_recorder per layout (mpc/data)')
    ap.add_argument('--out', default='', help='results folder (default <layouts>/results_<time>)')
    args = ap.parse_args()
    folder = os.path.abspath(os.path.expanduser(args.layouts))
    with open(os.path.join(folder, 'layouts.json')) as f:
        spec = json.load(f)
    layouts = spec['layouts']
    if args.only:
        names = set(args.only.split(','))
        layouts = [l for l in layouts if l['name'] in names]
    out_dir = args.out or os.path.join(folder, time.strftime('results_%Y-%m-%d_%H-%M-%S'))
    os.makedirs(out_dir, exist_ok=True)
    rclpy.init()
    rows = []
    try:
        for layout in layouts:
            t0 = time.monotonic()
            r = run_layout(layout, spec['start'], args, out_dir)
            r['wall_s'] = round(time.monotonic() - t0, 1)
            rows.append(r)
            print(f"{r['layout']:11s} boxes {r['boxes']} | {r['end']:11s} laps {r.get('laps', 0)} "
                  f"[{r.get('lap_times', '')}] min clear {r.get('min_clear', float('nan')):.3f} "
                  f"planned {r.get('planned_frac', 0):.0%} tight {r.get('tight_frac', 0):.0%} "
                  f"stops {r.get('stops', 0)} {r.get('hit_at', '')}", flush=True)
            with open(os.path.join(out_dir, 'results.csv'), 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=sorted({k for row in rows for k in row}))
                w.writeheader()
                w.writerows(rows)
            time.sleep(2.0)
    finally:
        rclpy.try_shutdown()
    hits = sum(r['end'] == 'hit' for r in rows)
    stuck = sum(r['end'] in ('stuck', 'timeout', 'never moved') for r in rows)
    print(f'\n{len(rows)} layouts: hits {hits}, stuck/timeout {stuck}, results in {out_dir}')


if __name__ == '__main__':
    main()
