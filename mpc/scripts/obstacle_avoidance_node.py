#!/usr/bin/env python3
"""Obstacle detection + detour planning for the MPC (ROS 2 node).

Per LiDAR scan (at most max_scan_rate Hz):
  1. Pose of the LiDAR at each beam's time, from TF map -> scan frame (the particle filter
     publishes map -> laser on the car; the sim publishes map -> ego_racecar/... ).
  2. Detection (obstacle_tracker.py): points inside the track and away from the map walls,
     clustered into boxes in track coordinates; confirmed after 2 scans; remembered until passed.
  3. Planning (detour_planner.py): a smooth detour around the boxes, or the raceline.
Publishes the plan for mpc_node.py (avoidance_msgs.py format), the obstacles (for the MPC's
safety check and the recorder), stats and RViz/Foxglove markers.
"""
import os
for _var in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_var, '1')   # numpy's OpenBLAS otherwise spins idle threads at 100 %
import collections
import dataclasses
import math
import time

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float64MultiArray
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

import avoidance_msgs as am
from detour_planner import DetourPlanner, PlannerParams
from obstacle_tracker import DetectionParams, ObstacleTracker, ScanView, detect
from track_model import load_track

PLAN_COLORS = {0: (0.1, 0.5, 1.0), 1: (1.0, 0.6, 0.0), 2: (1.0, 0.0, 1.0)}   # level -> rgb
# stats message layout (Float64MultiArray), also documented for tools/run_recorder.py
STATS_FIELDS = ('proc_ms', 'plan_ms', 'n_points', 'n_clusters', 'n_tracks', 'n_confirmed',
                'plan_level', 'stop', 'min_margin', 's_car', 'v', 'skipped_pose_jump',
                'tf_wait_ms')


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class ObstacleAvoidance(Node):
    def __init__(self):
        super().__init__('obstacle_avoidance')
        P = self.declare_parameter
        track = P('track', 'race1').value
        share = get_package_share_directory('mpc')
        wp = P('waypoints_file', '').value or os.path.join(share, 'waypoints', f'{track}.csv')
        map_yaml = P('map_yaml', 'race1_cleaned_blocked.yaml').value
        if not os.path.isabs(map_yaml):
            map_yaml = os.path.join(share, 'maps', map_yaml)
        self.map_frame = P('map_frame', 'map').value
        self.pose_frame = P('pose_frame', '').value          # '' = the scan's frame
        # where poses come from: 'tf' (car: particle filter map -> laser) or 'odom' (sim: the
        # exact /ego_racecar/odom; its ~1000 TF msgs/s are too many for a Python TF listener)
        self.pose_source = P('pose_source', 'tf').value
        self.laser_offset = [float(x) for x in P('laser_offset', [0.0, 0.0, 0.0]).value]  # odom mode
        self.max_rate = float(P('max_scan_rate', 0.0).value)  # Hz, 0 = every scan
        self.stamp_at_end = bool(P('scan_stamp_at_end', True).value)
        self.beam_offset = float(P('beam_time_offset', 0.0).value)
        self.tf_timeout = float(P('tf_timeout', 0.15).value)
        self.jump_dist = float(P('pose_jump_dist', 0.30).value)
        self.jump_yaw = float(P('pose_jump_yaw', 0.35).value)
        self.markers_on = bool(P('publish_markers', True).value)
        self.dp = self._declare_dataclass('detection', DetectionParams())
        self.pp = self._declare_dataclass('planner', PlannerParams())
        self.dp.rear_extent = self.pp.rear

        t0 = time.perf_counter()
        self.track = load_track(wp, map_yaml)
        self.tracker = ObstacleTracker(self.track, self.dp)
        self.planner = DetourPlanner(self.track, self.pp)
        self.get_logger().info(
            f'track {wp} ({self.track.length:.1f} m), walls {map_yaml}, '
            f'loaded in {time.perf_counter() - t0:.2f} s')

        if self.pose_source == 'tf':
            self.tf_buffer = Buffer(cache_time=Duration(seconds=5.0))
            # None = the listener gets its own node; passing self would let rclpy.spin(self) take
            # the node from the listener's thread, and TF would then stall while scan_cb waits
            self.tf_listener = TransformListener(self.tf_buffer, None, spin_thread=True)
        elif self.pose_source != 'odom':
            raise ValueError("pose_source must be 'tf' or 'odom'")
        self.odom_hist = collections.deque(maxlen=400)     # (t_ns, x, y, yaw) for odom mode
        self.v = 0.0
        self.s_laser = None
        self.prev_pose = None          # (t, x, y, yaw) of the last used scan
        self.last_scan_t = None
        self.plan_id = 0
        self.had_plan = False
        self.last_key = None

        self.create_subscription(LaserScan, P('scan_topic', '/scan').value, self.scan_cb,
                                 qos_profile_sensor_data)
        self.create_subscription(Odometry, P('pose_topic', 'pf/pose/odom').value, self.odom_cb, 1)
        self.plan_pub = self.create_publisher(Float64MultiArray, '/obstacle_avoidance/plan', 1)
        self.obs_pub = self.create_publisher(Float64MultiArray, '/obstacle_avoidance/obstacles', 1)
        self.stats_pub = self.create_publisher(Float64MultiArray, '/obstacle_avoidance/stats', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/obstacle_avoidance/markers', 1)

    def _declare_dataclass(self, prefix, obj):
        """Every field of a params dataclass becomes a ROS parameter <prefix>.<field>."""
        for f in dataclasses.fields(obj):
            default = getattr(obj, f.name)
            if isinstance(default, tuple):
                value = self.declare_parameter(f'{prefix}.{f.name}', [float(x) for x in default]).value
                value = tuple(value)
            else:
                value = self.declare_parameter(f'{prefix}.{f.name}', default).value
                value = type(default)(value)
            setattr(obj, f.name, value)
        return obj

    # ---- callbacks -------------------------------------------------------------------
    def odom_cb(self, msg):
        self.v = float(msg.twist.twist.linear.x)
        if self.pose_source == 'odom':
            p = msg.pose.pose.position
            self.odom_hist.append((Time.from_msg(msg.header.stamp).nanoseconds, p.x, p.y,
                                   yaw_of(msg.pose.pose.orientation)))

    def _pose_at(self, frame, t_ns, laser=True):
        """Pose (x, y, yaw) in the map frame at time t_ns (0 = latest)."""
        if self.pose_source == 'odom':
            if not self.odom_hist:
                raise LookupError('no odometry yet')
            h = np.array(self.odom_hist)
            if t_ns == 0:
                x, y, yaw = h[-1, 1:]
            else:
                if t_ns < h[0, 0] or t_ns > h[-1, 0] + 0.05e9:
                    raise LookupError('scan time outside the odometry history')
                yaw_u = np.unwrap(h[:, 3])
                x, y, yaw = (np.interp(t_ns, h[:, 0], h[:, 1]), np.interp(t_ns, h[:, 0], h[:, 2]),
                             np.interp(t_ns, h[:, 0], yaw_u))
            if laser:
                ox, oy, oyaw = self.laser_offset
                c, s = math.cos(yaw), math.sin(yaw)
                x, y, yaw = x + c * ox - s * oy, y + s * ox + c * oy, yaw + oyaw
            return float(x), float(y), float(yaw)
        tf = self.tf_buffer.lookup_transform(self.map_frame, frame, Time(nanoseconds=int(t_ns)),
                                             timeout=Duration(seconds=self.tf_timeout))
        tr = tf.transform
        return tr.translation.x, tr.translation.y, yaw_of(tr.rotation)

    def scan_cb(self, scan):
        t_start = time.perf_counter()
        t_scan = Time.from_msg(scan.header.stamp).nanoseconds
        if self.max_rate > 0 and self.last_scan_t is not None \
                and (t_scan - self.last_scan_t) < 0.9e9 / self.max_rate:
            return
        self.last_scan_t = t_scan
        n = len(scan.ranges)
        frame = scan.header.frame_id.lstrip('/')
        # beam times: the SICK stamps the end of the sweep; beams are time_increment apart
        ti = float(scan.time_increment) if scan.time_increment > 0 else 0.0
        off = self.beam_offset * 1e9
        if ti > 0:
            first = t_scan + off - (n - 1) * ti * 1e9 if self.stamp_at_end else t_scan + off
            times = first + np.array([0.0, 0.5, 1.0]) * (n - 1) * ti * 1e9
        else:
            times = np.array([t_scan + off])
        t_tf = time.perf_counter()
        try:
            poses = np.array([self._pose_at(frame, t) for t in times])
        except Exception as e:                                   # noqa: BLE001 (any TF error)
            self.get_logger().warn(f'no TF {self.map_frame} -> {frame} at scan time: {e}',
                                   throttle_duration_sec=2.0)
            return
        tf_wait_ms = (time.perf_counter() - t_tf) * 1e3
        poses[:, 2] = np.unwrap(poses[:, 2])
        if len(times) > 1:
            bt = times[0] + np.arange(n) * ti * 1e9
            bx = np.interp(bt, times, poses[:, 0])
            by = np.interp(bt, times, poses[:, 1])
            byaw = np.interp(bt, times, poses[:, 2])
        else:
            bx, by, byaw = poses[0]
        mid = poses[len(poses) // 2]

        # particle filter pose jump: skip this scan (its points would land in wrong places)
        skipped = 0
        if self.prev_pose is not None:
            dt = max((times[len(times) // 2] - self.prev_pose[0]) * 1e-9, 1e-3)
            step = math.hypot(mid[0] - self.prev_pose[1], mid[1] - self.prev_pose[2])
            dyaw = abs((mid[2] - self.prev_pose[3] + math.pi) % (2 * math.pi) - math.pi)
            if step > self.jump_dist + abs(self.v) * dt * 1.5 + 0.05 or dyaw > self.jump_yaw + 3.0 * dt:
                skipped = 1
        self.prev_pose = (times[len(times) // 2], mid[0], mid[1], mid[2])

        # where the LiDAR is along the raceline (search near the last value)
        xy = np.array([[mid[0], mid[1]]])
        if self.s_laser is None:
            s_l = self.track.project(xy)[0][0]
        else:
            s_l, _, ok = self.track.project(xy, s_hint=self.s_laser, window=(-2.0, 3.0))
            s_l = s_l[0] if ok[0] else self.track.project(xy)[0][0]
        self.s_laser = float(s_l)

        n_points, clusters = 0, []
        if not skipped:
            clusters, n_points = detect(self.track, np.asarray(scan.ranges, float), scan.angle_min,
                                        scan.angle_increment, bx, by, byaw, self.s_laser, self.dp)
            view = ScanView(np.array(mid[:2]), float(mid[2]), np.asarray(scan.ranges, float),
                            scan.angle_min, scan.angle_increment)
            self.tracker.update(clusters, self.s_laser, view)
        confirmed = self.tracker.confirmed()

        # plan from the latest pose of the point the MPC tracks
        t_plan = time.perf_counter()
        s_car = self.s_laser
        try:
            px, py, _ = self._pose_at(self.pose_frame or frame, 0, laser=False)   # 0 = latest
            s_c, _, ok = self.track.project(np.array([[px, py]]), s_hint=self.s_laser,
                                            window=(-2.0, 2.0))
            if ok[0]:
                s_car = float(s_c[0])
        except Exception:                                        # noqa: BLE001 (keep scan s)
            pass
        plan = self.planner.plan(s_car, self.v, self.tracker.planning_boxes())
        plan_ms = (time.perf_counter() - t_plan) * 1e3
        key = None if plan is None else (plan.key, plan.stop)
        if key != self.last_key:
            # one line per plan change: what was planned and around which boxes
            boxes = ' '.join(f'[{o.id}: u {float(self.track.rel_s(o.s_min, s_car)):+.2f}+{o.s_len:.2f} '
                             f'd {o.d_min:+.2f}..{o.d_max:+.2f}]' for o in confirmed)
            what = 'raceline' if plan is None else (
                'STOP' if plan.stop else f'level {plan.level} {plan.key} margins '
                + ' '.join(f'{m:.2f}' for m in plan.margins.values()))
            self.get_logger().info(f's {s_car:.2f} v {self.v:.2f}: {what} | {boxes}')
            self.last_key = key
        self._publish(plan, confirmed, clusters, s_car)

        min_margin = min(plan.margins.values(), default=math.inf) if plan is not None else math.inf
        stats = [(time.perf_counter() - t_start) * 1e3, plan_ms, n_points, len(clusters),
                 len(self.tracker.obstacles), len(confirmed),
                 -1 if plan is None else plan.level, float(plan is not None and plan.stop),
                 min_margin if math.isfinite(min_margin) else -1.0, s_car, self.v, skipped,
                 tf_wait_ms]
        self.stats_pub.publish(Float64MultiArray(data=[float(x) for x in stats]))
        if plan is not None and plan.stop:
            self.get_logger().warn('no way past the obstacle: braking to a stop',
                                   throttle_duration_sec=1.0)

    # ---- output ----------------------------------------------------------------------
    def _publish(self, plan, confirmed, clusters, s_car):
        if plan is None:
            self.plan_pub.publish(Float64MultiArray(data=am.encode_plan(self.plan_id)))
            if self.had_plan:
                self.plan_id += 1
            self.had_plan = False
        else:
            if not self.had_plan:
                self.plan_id += 1
            self.had_plan = True
            i0, i1, pts = am.resample_plan(self.track, plan)
            self.plan_pub.publish(Float64MultiArray(
                data=am.encode_plan(self.plan_id, i0, i1, pts, plan.stop)))
        self.obs_pub.publish(Float64MultiArray(data=am.encode_obstacles(self.track, confirmed)))
        if self.markers_on:
            self._markers(plan, clusters, s_car)

    def _marker(self, ns, mid, mtype, rgb, scale, pts, life=0.3):
        m = Marker()
        m.header.frame_id = self.map_frame
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns, m.id, m.type, m.action = ns, mid, mtype, Marker.ADD
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = scale
        m.color.r, m.color.g, m.color.b = rgb
        m.color.a = 1.0
        m.lifetime = Duration(seconds=life).to_msg()
        m.points = [Point(x=float(x), y=float(y), z=0.1) for x, y in pts]
        return m

    def _box_outline(self, s_min, s_len, d_min, d_max):
        u = np.linspace(0.0, s_len, 6)
        a = self.track.frenet_to_xy(s_min + u, np.full(6, d_min))
        b = self.track.frenet_to_xy(s_min + u[::-1], np.full(6, d_max))
        return np.vstack([a, b, a[:1]])

    def _markers(self, plan, clusters, s_car):
        arr = MarkerArray()
        for k, ob in enumerate(self.tracker.obstacles[:20]):
            rgb = (1.0, 0.1, 0.1) if ob.confirmed else (1.0, 0.8, 0.0)
            arr.markers.append(self._marker('obstacles', k, Marker.LINE_STRIP, rgb, 0.04,
                                            self._box_outline(ob.s_min, ob.s_len, ob.d_min, ob.d_max)))
        pts = np.vstack([c.xy for c in clusters]) if clusters else np.zeros((0, 2))
        arr.markers.append(self._marker('points', 0, Marker.POINTS, (1.0, 1.0, 0.0), 0.05, pts))
        if plan is not None:
            rgb = (1.0, 0.0, 0.0) if plan.stop else PLAN_COLORS.get(plan.level, (1, 1, 1))
            arr.markers.append(self._marker('plan', 0, Marker.LINE_STRIP, rgb, 0.06, plan.xy))
        self.marker_pub.publish(arr)


def main(args=None):
    rclpy.init(args=args)
    node = ObstacleAvoidance()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
