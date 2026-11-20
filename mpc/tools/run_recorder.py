#!/usr/bin/env python3
"""Record one MPC run (real car or sim), then save a diagnosis to mpc/data/<date_time>/.

Normally started by launch/levine_debug_launch.py. By hand, while the car stack
and particle filter run (start it before the MPC so the recording starts with the run):
    python3 run_recorder.py --track levine_real_mincurv --note "what I changed"
    python3 run_recorder.py --platform sim --track levine   # in the simulator

What is recorded (--platform real, the default):
    /pf/pose/odom      particle filter: x, y, yaw, forward speed (the PF copies it from /odom)
    /drive             what the MPC commanded (speed, steering)
    /ackermann_cmd     what the mux sent to the VESC (its output topic; bringup's remap to
                       /ackermann_drive does not match): tells MPC control from joystick takeover
    /odom              VESC wheel speed at the VESC's rate (for the speed-response fit)
    /sensors/core      VESC motor speed (erpm), motor/battery current, battery voltage, duty
    /tf                only map->laser is kept: the particle filter stamps it with the scan's
                       time, so receive time - stamp = the particle filter's delay
    /mpc/solve_stats, /mpc/config
No collision flag (the car has none). Laps are timed by run_analysis.py from the
particle-filter pose crossing the raceline's first waypoint.
With --platform sim it records the sim's /ego_racecar/odom, collision flag and lap
timer instead of the car-only topics.

How it stays out of the way: during the run this script only sleeps. The
recording is done by `ros2 bag record` (C++, a few % of one core). Python reads
1000 msgs/s far too slowly (it measured ~30% of a core here). After Ctrl+C it
converts the bag to raw.npz, deletes the bag (unless --keep-bag), and runs
run_analysis.py.

Testing aid only: in the sim it records the collision flag and the lap timer,
which the autograder forbids for graded nodes. It must never go into
levine_launch.py or spielberg_launch.py.
"""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import sys
import time
import traceback

from ament_index_python.packages import get_package_share_directory
import numpy as np

PKG_DIR = Path(__file__).resolve().parents[1]          # .../mpc (source)
sys.path.insert(0, str(PKG_DIR / 'scripts'))            # avoidance_msgs (plan/obstacle decoding)
DATA_DIR = PKG_DIR / 'data'
COMMON_TOPICS = ['/drive', '/mpc/solve_stats', '/mpc/config',
                 # obstacle avoidance (obstacle_avoidance_node.py, mpc_node.py safety check)
                 '/obstacle_avoidance/plan', '/obstacle_avoidance/obstacles',
                 '/obstacle_avoidance/stats', '/mpc/safety',
                 # avoidance debugging: what each node drew (replay the bag in Foxglove/RViz),
                 # and every node's log lines (plan changes, STOP, TF and stale-plan warnings)
                 '/obstacle_avoidance/markers', '/mpc/active_path', '/mpc/predicted',
                 '/mpc/ref_traj', '/rosout']
PLATFORMS = {
    # pose = where x, y, yaw, speed come from; extra = platform-only topics
    # /scan + /tf_static on the car: lets tools/replay_detection.py rerun detection offline
    'real': {'pose': '/pf/pose/odom',
             'extra': ['/ackermann_cmd', '/odom', '/sensors/core', '/tf', '/tf_static', '/scan']},
    'sim': {'pose': '/ego_racecar/odom',
            'extra': ['/ego_racecar/collision', '/ego_racecar/lap_time']},
}
# Car raw inputs for offline estimator replays (1A/1B/1C can be re-run on any run's bag)
RAW_ESTIMATION_TOPICS = ['/sensors/imu/raw', '/sensors/servo_position_command', '/commands/motor/speed']
# Extra topics recorded when the MPC drives on a stage 1 estimator (state_estimation package: 1A, 1B)
STAGE1_TOPICS = ['/state_estimate', '/state_estimate/latency', '/ekf1/odometry', '/prep/wheel_twist', '/prep/imu',
                 '/prep/steering', '/shadow_ekf1/odometry', '/shadow_ekf1/diagnostics']
SOLVE_FIELDS = 17   # numbers per /mpc/solve_stats message (mpc_node.publish_stats)
AVOID_FIELDS = 13   # numbers per /obstacle_avoidance/stats message (obstacle_avoidance_node.py)
VESC_FIELDS = ('erpm', 'current_motor', 'current_input', 'voltage_input', 'duty_cycle')
# /ackermann_cmd counts as "from the MPC" if it equals a /drive command received this
# close in time (the mux forwards the message unchanged)
MPC_MATCH_S = 0.3


def md5(path):
    path = Path(path)
    return hashlib.md5(path.read_bytes()).hexdigest() if path.is_file() else None


def git(*args):
    try:
        return subprocess.run(['git', *args], cwd=PKG_DIR, capture_output=True,
                              text=True, timeout=5).stdout
    except Exception:
        return ''


def map_info(map_yaml):
    """Image path, resolution and origin of a map_server yaml (for the plot background)."""
    import yaml
    map_yaml = Path(map_yaml)
    m = yaml.safe_load(map_yaml.read_text())
    img = Path(m['image'])
    return {'yaml': str(map_yaml), 'image': str(img if img.is_absolute() else map_yaml.parent / img),
            'resolution': m['resolution'], 'origin': m['origin']}


def sim_metadata(meta):
    """Sim settings + map, from f1tenth_gym_ros/config/sim.yaml."""
    import yaml
    sim_share = Path(get_package_share_directory('f1tenth_gym_ros'))
    sim = yaml.safe_load((sim_share / 'config' / 'sim.yaml').read_text())
    p = sim['bridge']['ros__parameters']
    meta['sim'] = {k: p.get(k) for k in ('map_path', 'sx', 'sy', 'stheta', 'vehicle_params',
                                         'scale', 'num_agents', 'async_mode', 'use_sim_time')}
    map_yaml = Path(p['map_path'] + '.yaml')
    if not map_yaml.is_absolute():
        map_yaml = sim_share / map_yaml
    meta['map'] = map_info(map_yaml)


def car_metadata(meta):
    """Car settings that change how a run behaves, from the installed config files the car
    stack and particle filter were launched with (edit + rebuild = new values here)."""
    import yaml
    car = {}
    try:
        cfg = Path(get_package_share_directory('f1tenth_stack')) / 'config'
        vesc = yaml.safe_load((cfg / 'vesc.yaml').read_text())
        p = vesc.get('/**', {}).get('ros__parameters', {})
        car['vesc'] = {k: p.get(k) for k in (
            'speed_to_erpm_gain', 'speed_to_erpm_offset', 'steering_angle_to_servo_gain',
            'steering_angle_to_servo_offset', 'servo_min', 'servo_max', 'speed_min', 'speed_max',
            'current_max', 'brake_max')}
        car['vesc']['odom_wheelbase'] = (vesc.get('vesc_to_odom_node', {})
                                         .get('ros__parameters', {}).get('wheelbase'))
        mux = yaml.safe_load((cfg / 'mux.yaml').read_text())
        car['mux'] = mux['ackermann_mux']['ros__parameters']['topics']
    except Exception as exc:
        print(f'[run_recorder] could not read f1tenth_stack configs ({exc!r}); continuing',
              flush=True)
    try:
        pf_share = Path(get_package_share_directory('particle_filter'))
        loc = yaml.safe_load((pf_share / 'config' / 'localize.yaml').read_text())
        car['particle_filter'] = loc['particle_filter']['ros__parameters']
        # same lookup as particle_filter/launch/localize_launch.py
        map_name = loc['map_server']['ros__parameters']['map']
        meta['map'] = map_info(pf_share / 'maps' / f'{map_name}.yaml')
    except Exception as exc:
        print(f'[run_recorder] could not read the particle filter config/map ({exc!r}); '
              'trying the map copy in waypoints/', flush=True)
        try:
            meta['map'] = map_info(PKG_DIR / 'waypoints' / 'levine_2nd.yaml')
        except Exception:
            pass
    meta['car'] = car


def collect_metadata(track, note, platform):
    """Everything that describes *what* ran, gathered once before driving starts."""
    started = datetime.now()
    share = Path(get_package_share_directory('mpc'))
    wp_installed = share / 'waypoints' / f'{track}.csv'
    wp_source = PKG_DIR / 'waypoints' / f'{track}.csv'
    node_installed = share.parents[1] / 'lib' / 'mpc' / 'mpc_node.py'
    node_source = PKG_DIR / 'scripts' / 'mpc_node.py'
    header = []
    if wp_installed.is_file():
        with open(wp_installed) as f:
            header = [ln.rstrip() for ln in f if ln.startswith('#')]

    meta = {
        'run': started.strftime('%Y-%m-%d_%H-%M-%S'), 'note': note, 'track': track,
        'platform': platform, 'pose_topic': PLATFORMS[platform]['pose'],
        'started': started.isoformat(timespec='seconds'),
        'waypoints': {'installed': str(wp_installed), 'header': header,
                      'md5': md5(wp_installed), 'source_md5': md5(wp_source)},
        'code': {'git_commit': git('rev-parse', '--short', 'HEAD').strip(),
                 'git_changed_files': git('status', '--porcelain', '--', '.').splitlines(),
                 'mpc_node_installed': str(node_installed),
                 'mpc_node_installed_md5': md5(node_installed),
                 'mpc_node_source_md5': md5(node_source)},
        'mpc': None,          # filled from /mpc/config in the bag
        'map': None, 'sim': None, 'car': None,
        'warnings': [],
    }
    if not wp_installed.is_file():
        meta['warnings'].append(
            f'waypoints/{track}.csv is not installed: wrong --track, or run '
            'colcon build --packages-select mpc')
    if meta['waypoints']['md5'] != meta['waypoints']['source_md5']:
        meta['warnings'].append(
            f'waypoints/{track}.csv in src differs from the installed copy the MPC loaded: '
            'run colcon build --packages-select mpc')
    if meta['code']['mpc_node_installed_md5'] != meta['code']['mpc_node_source_md5']:
        meta['warnings'].append(
            'scripts/mpc_node.py in src differs from the installed copy that ran: '
            'run colcon build --packages-select mpc')

    # platform settings + map (for the plot background)
    try:
        if platform == 'sim':
            sim_metadata(meta)
        else:
            car_metadata(meta)
    except Exception as exc:
        print(f'[run_recorder] could not read the {platform} settings / map ({exc!r}); '
              'continuing', flush=True)
    return meta


def mark_mpc_commands(cmd, drive):
    """For each /ackermann_cmd row (t, speed, steer), 1.0 if it is a /drive command the
    MPC sent within MPC_MATCH_S before it (the mux forwarded it), else 0.0 (joystick)."""
    from_mpc = np.zeros(len(cmd))
    if len(cmd) == 0 or len(drive) == 0:
        return from_mpc
    td = drive[:, 0]
    for i, (t, speed, steer) in enumerate(cmd):
        lo, hi = np.searchsorted(td, [t - MPC_MATCH_S, t + 0.05])   # +0.05: bag receive order
        d = drive[lo:hi]
        from_mpc[i] = float(np.any((np.abs(d[:, 1] - speed) < 1e-4) &
                                   (np.abs(d[:, 2] - steer) < 1e-4)))
    return from_mpc


def _plan(t, msg, plan, plan_pts, boxes, box_pts, pred):
    import avoidance_msgs as am
    p = am.decode_plan(msg.data)
    k = len(plan)
    n = 0 if p['pts'] is None else len(p['pts'])
    plan.append((t, p['plan_id'], p['i_start'], p['i_end'], float(p['stop']), n))
    if n:
        plan_pts.extend(np.c_[np.full(n, k), p['pts']].tolist())


def _obstacles(t, msg, plan, plan_pts, boxes, box_pts, pred):
    import avoidance_msgs as am
    b, pts = am.decode_obstacles(msg.data)
    k = _obstacles.count
    _obstacles.count += 1
    boxes.extend((t, k, *row) for row in b.tolist())
    box_pts.extend((k, x, y) for x, y in pts.tolist())


_obstacles.count = 0


def _predicted(t, msg, plan, plan_pts, boxes, box_pts, pred):
    pred.append([t] + [p.x for p in msg.points] + [p.y for p in msg.points])


AVOID_DECODERS = {'/obstacle_avoidance/plan': _plan,
                  '/obstacle_avoidance/obstacles': _obstacles,
                  '/mpc/predicted': _predicted}


def pad_rows(rows):
    """Rows [t, x0..xn, y0..yn] of different n -> equal length (NaN padded per half)."""
    if not rows:
        return []
    m = max((len(r) - 1) // 2 for r in rows)
    out = []
    for r in rows:
        n = (len(r) - 1) // 2
        nan = [np.nan] * (m - n)
        out.append([r[0], *r[1:1 + n], *nan, *r[1 + n:], *nan])
    return out


def bag_to_arrays(bag_dir, meta):
    """Read the bag into plain arrays. Times = receive time, seconds since the start of the
    run: the first command of the MPC that reached the motor (real car: the first MPC
    message on /ackermann_cmd), else the first /drive."""
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(bag_dir), storage_id='mcap'),
                rosbag2_py.ConverterOptions('cdr', 'cdr'))
    types = {t.name: get_message(t.type) for t in reader.get_all_topics_and_types()}
    pose_topic = meta['pose_topic']
    used = {pose_topic, '/drive', '/ackermann_cmd', '/odom', '/sensors/core', '/tf',
            '/mpc/solve_stats', '/ego_racecar/collision', '/ego_racecar/lap_time', '/mpc/config',
            '/obstacle_avoidance/stats', '/mpc/safety',
            '/obstacle_avoidance/plan', '/obstacle_avoidance/obstacles', '/mpc/predicted',
            '/rosout'}
    odom, drive, cmd, wheel, vesc, solve, collisions, sim_laps, pf_lat = ([] for _ in range(9))
    avoid, safety = [], []
    # avoidance details (one message index k per plan / obstacles / predicted message)
    plan, plan_pts, boxes, box_pts, pred, log = [], [], [], [], [], []
    in_contact, last_lap_time = False, None
    while reader.has_next():
        topic, data, t_ns = reader.read_next()
        if topic not in used:
            continue                       # /scan etc. stay in the bag (--keep-bag)
        t = t_ns * 1e-9
        msg = deserialize_message(data, types[topic])
        if topic == pose_topic:
            # x, y, yaw, forward speed, sideways speed (always 0 from the particle filter)
            q, p, tw = msg.pose.pose.orientation, msg.pose.pose.position, msg.twist.twist.linear
            yaw = np.arctan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
            odom.append((t, p.x, p.y, yaw, tw.x, tw.y))
        elif topic == '/drive':
            drive.append((t, msg.drive.speed, msg.drive.steering_angle))
        elif topic == '/ackermann_cmd':
            cmd.append((t, msg.drive.speed, msg.drive.steering_angle))
        elif topic == '/odom':
            wheel.append((t, msg.twist.twist.linear.x, msg.twist.twist.angular.z))
        elif topic == '/sensors/core':
            s = msg.state
            vesc.append((t, s.speed, s.current_motor, s.current_input, s.voltage_input,
                         s.duty_cycle))
        elif topic == '/tf':
            for tr in msg.transforms:
                if (tr.header.frame_id.lstrip('/') == 'map'
                        and tr.child_frame_id.lstrip('/') == 'laser'):
                    st = tr.header.stamp
                    pf_lat.append((t, t - (st.sec + st.nanosec * 1e-9)))
        elif topic == '/mpc/solve_stats':
            # solve_ms, osqp_ms, callback_ms, status, then 13 cost terms (see
            # mpc_node.cost_terms); older nodes send only the first 4 -> NaN
            row = list(msg.data)[:SOLVE_FIELDS]
            solve.append((t, *row, *[np.nan] * (SOLVE_FIELDS - len(row))))
        elif topic == '/ego_racecar/collision':
            if msg.data != in_contact:
                in_contact = msg.data
                collisions.append((t, 1.0 if msg.data else 0.0))
        elif topic == '/ego_racecar/lap_time':
            # published every tick; it only changes when the sim finishes a lap
            if last_lap_time is not None and msg.data != last_lap_time and msg.data > 0:
                sim_laps.append((t, msg.data))
            last_lap_time = msg.data
        elif topic == '/mpc/config':
            meta['mpc'] = json.loads(msg.data)
        elif topic == '/obstacle_avoidance/stats':
            # obstacle_avoidance_node.STATS_FIELDS
            avoid.append((t, *list(msg.data)[:AVOID_FIELDS],
                          *[np.nan] * max(0, AVOID_FIELDS - len(msg.data))))
        elif topic == '/mpc/safety':
            safety.append((t, *list(msg.data)[1:4]))   # v_safe, prediction step, distance
        elif topic in AVOID_DECODERS:
            try:
                AVOID_DECODERS[topic](t, msg, plan, plan_pts, boxes, box_pts, pred)
            except Exception as exc:                   # a bad message must not lose the run
                log.append((t, 'run_recorder', 30, f'could not decode {topic}: {exc!r}'))
        elif topic == '/rosout':
            log.append((t, msg.name, msg.level, msg.msg))

    if not drive:
        return None
    drive = np.array(drive, dtype=float)
    cmd = np.array(cmd, dtype=float).reshape(-1, 3)
    cmd = np.c_[cmd, mark_mpc_commands(cmd, drive)]       # t, speed, steer, from_mpc
    if cmd[:, 3].any():
        t0 = cmd[cmd[:, 3] > 0, 0][0]
    else:
        t0 = drive[0, 0]
        if meta['platform'] == 'real':
            meta['warnings'].append(
                'no MPC command reached /ackermann_cmd (was the car stack running, or the '
                'joystick held the whole time?); the run starts at the first /drive instead')
    meta['t0_source'] = 'first MPC /ackermann_cmd' if cmd[:, 3].any() else 'first /drive'

    def arr(rows, cols):
        a = np.array(rows, dtype=float).reshape(-1, cols)
        a[:, 0] -= t0
        return a[a[:, 0] >= 0]

    def rel(rows, cols):
        # like arr() but keeps messages from before the run starts (detections while the
        # car still stands matter); rows reference these by message index k
        a = np.array(rows, dtype=float).reshape(-1, cols)
        a[:, 0] -= t0
        return a

    meta['log'] = [(round(tl - t0, 3), name, level, text) for tl, name, level, text in log]

    return {'odom': arr(odom, 6), 'drive': arr(drive, 3), 'solve': arr(solve, 1 + SOLVE_FIELDS),
            'cmd': arr(cmd, 4), 'wheel': arr(wheel, 3), 'vesc': arr(vesc, 1 + len(VESC_FIELDS)),
            'collisions': arr(collisions, 2), 'sim_laps': arr(sim_laps, 2),
            'pf_latency': arr(pf_lat, 2),                # t, particle filter delay [s]
            'avoid': arr(avoid, 1 + AVOID_FIELDS),      # t, obstacle_avoidance_node.STATS_FIELDS
            'safety': arr(safety, 4),                   # t, v_safe, step, distance
            # /obstacle_avoidance/plan, one row per message k: t, plan_id, i_start, i_end,
            # stop, n_pts (0 = raceline); plan_pts rows: k, x, y, yaw, v
            'plan': rel(plan, 6), 'plan_pts': np.array(plan_pts, float).reshape(-1, 5),
            # /obstacle_avoidance/obstacles (confirmed boxes), one row per box: t, k, id,
            # s_min, s_len, d_min, d_max; box_pts (the map points the MPC checks): k, x, y
            'boxes': rel(boxes, 7), 'box_pts': np.array(box_pts, float).reshape(-1, 3),
            # /mpc/predicted (MPC predicted rear-axle path): t, then x0..xK, y0..yK (NaN pad)
            'predicted': rel(pad_rows(pred), 1 + 2 * max((len(r) - 1) // 2 for r in pred) if pred else 1)}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--track', default='levine_real_mincurv')
    parser.add_argument('--platform', default='real', choices=sorted(PLATFORMS),
                        help='real = F1TENTH car with the particle filter, sim = f1tenth_gym_ros')
    parser.add_argument('--note', default='', help='free text saved with the run')
    parser.add_argument('--keep-bag', default='false',
                        help='true = keep the rosbag (~20 MB per 100 s) next to the results')
    parser.add_argument('--estimator', default='race',
                        help='state estimator the MPC drives on (state_estimation package); not race = also '
                             'record the stage 1 topics')
    args = parser.parse_args()

    meta = collect_metadata(args.track, args.note, args.platform)
    topics = [PLATFORMS[args.platform]['pose'], *COMMON_TOPICS, *PLATFORMS[args.platform]['extra']]
    meta['estimator'] = args.estimator
    if args.platform == 'real':
        topics += RAW_ESTIMATION_TOPICS
        if args.estimator != 'race':
            topics += STAGE1_TOPICS
    for w in meta['warnings']:
        print(f'[run_recorder] WARNING: {w}', flush=True)
    run_dir = DATA_DIR / meta['run']
    run_dir.mkdir(parents=True, exist_ok=True)
    bag_dir = run_dir / 'bag'

    # Ctrl+C reaches us twice in a terminal (from the terminal and from ros2 launch);
    # a flag makes that harmless instead of killing us halfway through saving.
    stop = []
    signal.signal(signal.SIGINT, lambda *_: stop.append(1))
    signal.signal(signal.SIGTERM, lambda *_: stop.append(1))

    bag_log = open(run_dir / 'bag.log', 'w')   # a file, not a pipe: a full pipe would stall the bag
    bag = subprocess.Popen(['ros2', 'bag', 'record', '-s', 'mcap', '-o', str(bag_dir), *topics],
                           stdout=bag_log, stderr=subprocess.STDOUT)
    wall0 = time.monotonic()
    print(f'[run_recorder] recording ({args.platform}: {" ".join(topics)}) to {run_dir}; '
          'press Ctrl+C to stop and save', flush=True)
    while not stop and bag.poll() is None:
        time.sleep(0.2)
    if bag.poll() is None:
        bag.send_signal(signal.SIGINT)         # it usually got the terminal's Ctrl+C already
    try:
        bag.wait(timeout=30)
    except subprocess.TimeoutExpired:
        bag.kill()
        bag.wait()
    bag_log.close()
    wall = time.monotonic() - wall0
    ru = resource.getrusage(resource.RUSAGE_CHILDREN)
    meta['recorder_cpu_pct'] = 100.0 * (ru.ru_utime + ru.ru_stime) / max(wall, 1e-6)

    if not (bag_dir / 'metadata.yaml').exists():
        print('[run_recorder] the bag was not written:\n'
              + (run_dir / 'bag.log').read_text()[-2000:], flush=True)
        return
    try:
        raw = bag_to_arrays(bag_dir, meta)
    except Exception:
        traceback.print_exc()
        print(f'[run_recorder] could not read the bag; it is kept in {bag_dir}', flush=True)
        return
    if raw is None or len(raw['odom']) == 0:
        what = ('the MPC never sent a /drive command' if raw is None else
                f'no {meta["pose_topic"]} messages after the run started (is the particle '
                'filter running?)')
        print(f'[run_recorder] {what}; no run saved.', flush=True)
        shutil.rmtree(run_dir, ignore_errors=True)
        return
    if meta['mpc'] is None:
        meta['warnings'].append(
            'no /mpc/config from the MPC node, so run settings and solve times are missing. '
            'Rebuild (colcon build --packages-select mpc) and use levine_debug_launch.py.')

    np.savez_compressed(run_dir / 'raw.npz', **raw)
    # every node's log lines, t = seconds since the run start (negative = before it)
    log = meta.pop('log', [])
    level = {10: 'DEBUG', 20: 'INFO', 30: 'WARN', 40: 'ERROR', 50: 'FATAL'}
    (run_dir / 'rosout.log').write_text(''.join(
        f'{t:9.3f} {level.get(lv, lv):5} [{name}] {text}\n' for t, name, lv, text in log))
    meta['duration_s'] = float(raw['odom'][-1, 0])
    (run_dir / 'meta.json').write_text(json.dumps(meta, indent=1))
    # snapshots of exactly what ran, so old runs stay explainable
    shutil.copy(meta['waypoints']['installed'], run_dir / 'waypoints.csv')
    node_file = Path(meta['code']['mpc_node_installed'])
    if node_file.is_file():
        shutil.copy(node_file, run_dir / 'mpc_node_ran.py')
    patch = git('diff', 'HEAD', '--', '.')
    if patch:
        (run_dir / 'changes_since_commit.patch').write_text(patch)
    if args.keep_bag.lower() not in ('1', 'true', 'yes'):
        shutil.rmtree(bag_dir, ignore_errors=True)
        (run_dir / 'bag.log').unlink(missing_ok=True)

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        import run_analysis
        run_analysis.report(run_dir)
    except Exception:
        traceback.print_exc()
        print(f'[run_recorder] analysis failed; raw data is safe. Retry with:\n'
              f'  python3 {Path(__file__).with_name("run_analysis.py")} {run_dir}', flush=True)


if __name__ == '__main__':
    os.environ.setdefault('PYTHONUNBUFFERED', '1')
    main()
