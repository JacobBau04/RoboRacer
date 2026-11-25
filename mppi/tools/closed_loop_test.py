"""Closed-loop, real-time test of the edited MPPI node against a simulated car with the real car's delays.
Car model (from the car reference / TUNING_LOG): kinematic bicycle WB 0.324, wheels turn 0.9x the command,
steering acts d_act = 0.14 s after the command, pose shows the car d_pose = 0.06 s late (total 0.20 s),
speed: first-order, kp 5 1/s speeding up, 1 1/s braking. Pose = LiDAR point 0.27 m ahead of the rear axle.
Usage: python3 mppi_closed_loop.py <seconds per run> <delay_comp> [<delay_comp> ...]"""
import os, sys, time, math, io, contextlib, bisect
import numpy as np
WS = '/home/nvidia/f1tenth_ws'; sys.path.insert(0, os.environ.get('SRC', f'{WS}/src/mppi_car/scripts')); SHARE = f'{WS}/install/mpc/share/mpc/waypoints'
import rclpy
rclpy.init(args=['--ros-args', '-p', 'wall_map_yaml:=' + os.environ.get('MAP', f'{SHARE}/levine_2nd.yaml'), '-p', 'waypoint_path:=' + os.environ.get('CSV', f'{SHARE}/levine_race.csv'),
                 ] + ([] if os.environ.get('ORIG') else ['-p', 'pose_offset:=0.27', '-p', 'delay_comp:=0.20']))
from nav_msgs.msg import Odometry
import mppi_node as M
if os.environ.get('ORIG'):
    M.simulate = M.jax.jit(M.simulate, static_argnames=('cfg',))   # speed only, same result
print(f'MAX_SPEED {M.MPPIConfig().MAX_SPEED} MAX_STEER {M.MPPIConfig().MAX_STEER}', file=sys.stderr)
OUT = sys.stderr
E = lambda k, d: float(os.environ.get(k, d))
WB, GAIN, D_ACT, D_POSE, KP_UP, KP_DN, LIDAR = E('WB',0.324), E('GAIN',0.9), E('D_ACT',0.14), E('D_POSE',0.06), E('KP_UP',5.0), E('KP_DN',1.0), E('LIDAR',0.27)
print(f'src {os.environ.get("SRC","mppi_car")}, config MAX_SPEED/STEER from module; car: WB {WB} gain {GAIN} d_act {D_ACT} d_pose {D_POSE} kp {KP_UP}/{KP_DN} lidar {LIDAR}', file=sys.stderr)

with contextlib.redirect_stdout(io.StringIO()):
    node = M.MPPI()
wp = node.waypoints
import dataclasses
_ov = {k: float(os.environ[e]) for k, e in (('MAX_SPEED','MAXV'), ('MAX_STEER','STEER')) if os.environ.get(e)}
if os.environ.get('USTD'): _ov['U_STD'] = tuple(float(v) for v in os.environ['USTD'].split(','))
if 'MAX_STEER' in _ov: _ov['MIN_STEER'] = -_ov['MAX_STEER']
if _ov: node.config = dataclasses.replace(node.config, **_ov)
if os.environ.get('VSCALE'): node.ref_v = node.ref_v * float(os.environ['VSCALE'])
print(f'run config: MAX_SPEED {node.config.MAX_SPEED} STEER +-{node.config.MAX_STEER} ref_v {np.min(node.ref_v):.2f}-{np.max(node.ref_v):.2f}', file=sys.stderr)
rx, ry, ryaw = np.asarray(node.ref_x), np.asarray(node.ref_y), np.asarray(node.ref_yaw)
seg = np.hypot(np.diff(rx, append=rx[0]), np.diff(ry, append=ry[0])); s_cum = np.concatenate(([0], np.cumsum(seg)))
lap_len = s_cum[-1]
sdf, sdf_o, sdf_r = node.wall_sdf_np, node.wall_origin_np, node.wall_resolution

def wall_dist(x, y):
    c = int((x - sdf_o[0]) / sdf_r); r = int((y - sdf_o[1]) / sdf_r)
    return float(sdf[r, c]) if 0 <= r < sdf.shape[0] and 0 <= c < sdf.shape[1] else float('nan')

def lat_err(x, y):
    d = np.hypot(rx - x, ry - y); i = int(np.argmin(d)); j = (i + 1) % len(rx)
    ex, ey = rx[j] - rx[i], ry[j] - ry[i]; L = math.hypot(ex, ey) or 1.0
    return abs((x - rx[i]) * ey - (y - ry[i]) * ex) / L, i

def odom(x, y, yaw, v):
    m = Odometry()
    m.pose.pose.position.x, m.pose.pose.position.y = float(x), float(y)
    m.pose.pose.orientation.z, m.pose.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
    m.twist.twist.linear.x = float(v); return m

DIAG = {'ess': [], 'hit': []}
_orig_step = M.mppi_step
def _diag_step(*a, **k):
    r = _orig_step(*a, **k)
    U, paths, c0, _ = r
    c0 = np.asarray(c0); cfg = node.config
    nrm = (c0 - c0.min()) / (c0.max() - c0.min() + 1e-6); w = np.exp(-(nrm - nrm.min()) / cfg.LAMBDA); w /= w.sum()
    DIAG['ess'].append(1.0 / np.sum(w ** 2) / len(w))
    P = np.asarray(paths)[:, 1:, :2]
    cc = ((P[..., 0] - sdf_o[0]) / sdf_r).astype(int).clip(0, sdf.shape[1] - 1); rr = ((P[..., 1] - sdf_o[1]) / sdf_r).astype(int).clip(0, sdf.shape[0] - 1)
    DIAG['hit'].append(np.mean(sdf[rr, cc].min(axis=1) < cfg.CAR_RADIUS + cfg.WALL_BUBBLE))
    return r
M.mppi_step = _diag_step

def run(duration, delay_comp):
    DIAG['ess'].clear(); DIAG['hit'].clear()
    if hasattr(node, 'delay_comp'): node.delay_comp = delay_comp
    if hasattr(node, 'cmd_hist'): node.cmd_hist.clear()
    node.U = M.jnp.zeros_like(node.U)
    if False: node.cmd_hist.clear(); node.U = M.jnp.zeros_like(node.U)
    cmds = [(-1e9, 0.0, 0.0)]; ctimes = [-1e9]          # (sent time, speed, steer)
    def pub(m):
        tn = time.monotonic(); cmds.append((tn, m.drive.speed, m.drive.steering_angle)); ctimes.append(tn)
    node.drive_pub_.publish = pub
    x, y, yaw, v = rx[0], ry[0], ryaw[0], 0.0
    t = time.monotonic(); t_end = t + duration
    hist_t, hist_s = [t], [(x, y, yaw, v)]
    errs, steers, steer_t, walls, solves = [], [], [], [], []
    prog, last_i = 0.0, 0
    while True:
        now = time.monotonic()
        while t < now:                                      # integrate the car to now, 5 ms steps
            dt = min(0.005, now - t)
            k = bisect.bisect_right(ctimes, t - D_ACT) - 1
            _, v_cmd, d_cmd = cmds[k]
            v += (KP_UP if v_cmd > v else KP_DN) * (v_cmd - v) * dt
            x += v * math.cos(yaw) * dt; y += v * math.sin(yaw) * dt
            yaw += v * math.tan(GAIN * d_cmd) / WB * dt; t += dt
            hist_t.append(t); hist_s.append((x, y, yaw, v))
        if now >= t_end: break
        j = max(0, bisect.bisect_right(hist_t, now - D_POSE) - 1)   # pose is D_POSE old
        px, py, pyaw, pv = hist_s[j]
        t0 = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):
            node.odom_callback(odom(px + LIDAR * math.cos(pyaw), py + LIDAR * math.sin(pyaw), pyaw, pv))
        solves.append(time.perf_counter() - t0)
        e, i = lat_err(x, y); errs.append(e); walls.append(wall_dist(x, y))
        if os.environ.get('DEBUG') and len(errs) % 4 == 0:
            r0 = None
            print(f't {t-(t_end-duration):5.2f} car ({x:6.2f},{y:6.2f}) yaw {yaw:+.2f} v {v:.2f} | wp {i:3d} ({rx[i]:6.2f},{ry[i]:6.2f}) yaw {ryaw[i]:+.2f} | err {e*100:5.1f} cm wall {wall_dist(x,y):.2f} | cmd v {cmds[-1][1]:.2f} steer {cmds[-1][2]:+.3f}', file=OUT)
        steers.append(cmds[-1][2]); steer_t.append(cmds[-1][0])
        di = (i - last_i) % len(rx)
        if di < len(rx) // 2: prog += (s_cum[i] - s_cum[last_i]) % lap_len; last_i = i
        if e > 1.5: break                                   # left the track
    st = np.array(steers); tt = np.array(steer_t)
    osc = st - np.array([st[(tt > a - 1) & (tt < a + 1)].mean() for a in tt])   # minus 2 s average
    m = len(errs) // 5                                      # skip the start-up fifth
    e = np.array(errs[m:]); w = np.array(walls[m:])
    print(f'delay_comp {delay_comp:.2f} s: drove {prog:5.1f} m in {t - (t_end - duration):4.1f} s'
          f'{"  LEFT THE TRACK (>1.5 m off)" if errs[-1] > 1.5 else ""}; '
          f'|lateral error| mean {e.mean()*100:4.1f} cm, max {e.max()*100:4.1f} cm; '
          f'steering wobble {np.sqrt(np.mean(osc[m:]**2)):.3f} rad rms; closest wall {np.nanmin(w)*100:3.0f} cm (rear axle); '
          f'solve {np.mean(solves)*1000:.0f} ms ({1/np.mean(solves):.0f} Hz); speed {np.mean([h[3] for h in hist_s[len(hist_s)//5:]]):.2f} m/s',
          file=OUT)
    q = len(DIAG['ess']) // 5
    print(f'    samples hitting a wall: {np.mean(DIAG["hit"][q:])*100:.0f}% (mean); effective samples (weights spread): {np.mean(DIAG["ess"][q:])*100:.1f}% of {node.config.T} (100% = all equal = steering averages noise)', file=OUT)

dur = float(sys.argv[1])
for dc in sys.argv[2:]:
    run(dur, float(dc))
node.destroy_node(); rclpy.shutdown()
