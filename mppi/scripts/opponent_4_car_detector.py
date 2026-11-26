#!/usr/bin/env python3
"""Python port of opponent_lidar_detector_multi.cpp. ROS interfaces and algorithm retained.
NumPy geometry and Python stable sorting may differ in rounding/ties. Not ROS runtime tested.
"""
from dataclasses import dataclass, field
from collections import deque
import math
from typing import Optional
import numpy as np


@dataclass
class Candidate:
    x: float                         # estimated opponent center in MAP frame
    y: float
    yaw: float                       # approximate heading from raceline tangent
    s: float                         # projected raceline progress
    score: float
    distance_from_raceline: float
    range_from_ego: float
    extent_along_track: float
    extent_across_track: float
    correction_norm: float = 0.0


@dataclass
class Track:
    candidate: Optional[Candidate] = None
    last_time: float = 0.0
    vx: float = 0.0                   # estimated map-frame velocity
    vy: float = 0.0
    hits: int = 0                     # successful matches since this ID was assigned
    observed: bool = False           # matched in THIS scan

    @property
    def active(self):
        return self.candidate is not None


class ThreeOpponentDetector:
    def make_inflated_wall_mask(self):
        """Precompute a wall mask so each LiDAR point is checked in O(1)."""
        blocked = (self.grid >= self.occupied_threshold)
        if self.treat_unknown_as_static:
            blocked = blocked | (self.grid < 0)
        k = math.ceil(self.wall_inflation_radius / self.resolution)
        inflated = blocked.copy()
        ys, xs = np.where(blocked)
        for y, x in zip(ys, xs):
            inflated[max(0, y-k):y+k+1, max(0, x-k):x+k+1] = True
        return inflated

    def is_static_wall(self, xy):
        """Anything inside inflated walls or outside map bounds is rejected."""
        if not self.map_received or self.blocked.size == 0:
            return self.require_static_map
        mx, my = ((np.asarray(xy) - self.origin) / self.resolution).astype(int)
        if my < 0 or mx < 0 or my >= self.blocked.shape[0] or mx >= self.blocked.shape[1]:
            return True
        return bool(self.blocked[my, mx])

    def project_to_raceline(self, xy):
        """Find closest point on ANY raceline segment; return distance, s, yaw."""
        if len(self.raceline) < 2:
            return 0.0, 0.0, self.ego_yaw
        lengths = np.linalg.norm(np.roll(self.raceline, -1, axis=0) - self.raceline, axis=1)
        starts = self.raceline
        ends = np.roll(starts, -1, axis=0)
        vec = ends - starts
        a = np.clip(np.sum((xy - starts) * vec, axis=1) /
                    np.maximum(np.sum(vec*vec, axis=1), 1e-9), 0.0, 1.0)
        projected = starts + a[:, None] * vec
        errors = np.linalg.norm(projected - xy, axis=1)
        i = int(np.argmin(errors))
        yaw = math.atan2(vec[i, 1], vec[i, 0])
        total = self.raceline_s[-1] + lengths[-1]
        end_s = self.raceline_s[i+1] if i+1 < len(starts) else total
        s = float((self.raceline_s[i] + a[i]*(end_s-self.raceline_s[i])) % total)
        return float(errors[i]), s, yaw

    def scan_to_dynamic_points(self, ranges, angle_min, angle_increment,
                               ego_x, ego_y, ego_yaw, scan_range_max=float('inf')):
        """Convert rays to MAP coordinates and remove points near static walls.

        Each output row is [map_x,map_y,base_x,base_y], just like C++ ScanPoint.
        """
        cy, sy = math.cos(ego_yaw), math.sin(ego_yaw)
        cl, sl = math.cos(self.laser_yaw_offset), math.sin(self.laser_yaw_offset)
        points = []
        for i, r in enumerate(ranges):
            if not math.isfinite(r) or r < self.min_range or r > min(self.max_range, scan_range_max):
                continue
            angle = angle_min + i * angle_increment
            if self.front_fov_only and abs(angle) > math.pi/2:
                continue
            lx, ly = r * math.cos(angle), r * math.sin(angle)
            bx = self.laser_x_offset + cl*lx - sl*ly
            by = self.laser_y_offset + sl*lx + cl*ly
            if not (self.min_detection_range <= math.hypot(bx, by) <= self.max_detection_range):
                continue
            if bx < self.min_base_x:
                continue
            mx = ego_x + cy*bx - sy*by
            my = ego_y + sy*bx + cy*by
            if not self.is_static_wall((mx, my)):
                points.append((mx, my, bx, by))
        return np.array(points, dtype=float).reshape(-1, 4)

    def cluster_points(self, points):
        """Grid-bucketed BFS: same nine neighboring cells as C++."""
        xy = points[:, :2]
        cell = max(self.cluster_tolerance, 1e-3)
        cells = np.floor(xy / cell).astype(int)
        grid = {}
        for i, c in enumerate(cells):
            grid.setdefault(tuple(c), []).append(i)
        visited = set()
        clusters = []
        for seed in range(len(points)):
            if seed in visited:
                continue
            q = deque([seed]); visited.add(seed); group = []
            while q:
                i = q.popleft(); group.append(i)
                cx, cy = cells[i]
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        for j in grid.get((cx+dx, cy+dy), []):
                            delta = xy[i] - xy[j]
                            if j not in visited and delta @ delta <= self.cluster_tolerance**2:
                                visited.add(j); q.append(j)
            clusters.append(points[group])
        return clusters

    def cluster_to_candidate(self, points, ego_xy):
        """Apply exactly the geometric checks and center correction from C++."""
        if not self.min_cluster_points <= len(points) <= self.max_cluster_points:
            self.tally["size"] += 1
            return None
        centroid_map = points[:, :2].mean(axis=0)
        centroid_base = points[:, 2:].mean(axis=0)
        if centroid_base[0] < self.min_base_x:
            self.tally["base_x"] += 1
            return None
        distance, _, yaw = self.project_to_raceline(centroid_map)
        if distance > self.max_raceline_projection_dist:
            self.tally["proj"] += 1
            self.best_proj = min(self.best_proj, distance)
            return None
        tangent = np.array([math.cos(yaw), math.sin(yaw)])
        normal = np.array([-math.sin(yaw), math.cos(yaw)])
        centered = points[:, :2] - centroid_map
        along = centered @ tangent
        across = centered @ normal
        extent_t = float(along.max() - along.min())
        extent_n = float(across.max() - across.min())
        if not self.min_cluster_extent <= max(extent_t, extent_n) <= self.max_cluster_extent:
            self.tally["extent"] += 1
            return None

        # LiDAR generally sees only the facing surface, not the car's center.
        # Shift the observed centroid towards an estimated vehicle center.
        view = centroid_map - np.asarray(ego_xy)
        view /= max(np.linalg.norm(view), 1e-9)
        missing_t = max(0.0, 0.5*self.opponent_length - 0.5*extent_t)
        missing_n = max(0.0, 0.5*self.opponent_width - 0.5*extent_n)
        correction = tangent*np.dot(view, tangent)*missing_t + normal*np.dot(view, normal)*missing_n
        correction_length = np.linalg.norm(correction)
        if correction_length > self.max_center_correction:
            correction *= self.max_center_correction / correction_length
        center = centroid_map + correction
        distance, s, yaw = self.project_to_raceline(center)
        self.best_proj = min(self.best_proj, distance)
        if distance > self.max_raceline_projection_dist:
            self.tally["proj_post"] += 1
            return None
        self.tally["accepted"] += 1
        range_from_ego = float(np.linalg.norm(centroid_base))
        score = self.projection_weight*distance + self.range_weight*range_from_ego
        return Candidate(float(center[0]), float(center[1]), yaw, s, score,
                         distance, range_from_ego, extent_t, extent_n, float(np.linalg.norm(correction)))

    def collect_candidates(self, clusters, ego_xy):
        """Retain ALL valid clusters (unlike the old single-opponent detector)."""
        self.tally = dict.fromkeys(("accepted", "size", "base_x", "proj", "extent", "proj_post", "speed", "jump"), 0)
        self.best_proj = math.inf
        candidates = [self.cluster_to_candidate(c, ego_xy) for c in clusters]
        candidates = sorted((c for c in candidates if c is not None), key=lambda c: c.score)
        kept = []
        for candidate in candidates:
            if all(math.hypot(candidate.x - old.x, candidate.y - old.y)
                   >= self.track_dedup_distance for old in kept):
                kept.append(candidate)
        return kept

    def update_tracks(self, candidates, now):
        """Associate candidates to up to 3 persistent opponent IDs.

        1. Expire old tracks; 2. predict their current positions;
        3. gather feasible (track,candidate) pairs; 4. match nearest first;
        5. initialize fresh unmatched candidates in vacant slots.
        """
        for i, track in enumerate(self.tracks):
            track.observed = False
            if track.active and (now < track.last_time-0.001 or now-track.last_time > self.track_timeout):
                self.tracks[i] = Track()

        pairs = []  # (distance, slot_id, candidate_id)
        for slot, track in enumerate(self.tracks):
            if not track.active:
                continue
            dt = max(0.0, now - track.last_time)
            predicted = np.array([track.candidate.x + track.vx*dt,
                                  track.candidate.y + track.vy*dt])
            gate = min(self.track_match_distance,
                       self.max_detection_speed * max(dt, 0.05) + 0.20)
            for j, candidate in enumerate(candidates):
                d = float(np.linalg.norm(predicted - [candidate.x, candidate.y]))
                if d <= gate:
                    pairs.append((d, slot, j))
        pairs.sort(key=lambda pair: pair[0])  # nearest-first greedy matching (not Hungarian)
        used_candidates, used_slots = set(), set()
        for _, slot, j in pairs:
            if slot in used_slots or j in used_candidates:
                continue
            track = self.tracks[slot]
            c = candidates[j]
            dt = now - track.last_time
            if dt > 0.001:
                vx = (c.x - track.candidate.x) / dt
                vy = (c.y - track.candidate.y) / dt
                scale = min(1.0, self.max_detection_speed / max(math.hypot(vx, vy), 1e-9))
                track.vx = 0.5*track.vx + 0.5*vx*scale
                track.vy = 0.5*track.vy + 0.5*vy*scale
            track.candidate = c
            track.last_time = now
            track.observed = True
            track.hits += 1
            used_candidates.add(j)
            used_slots.add(slot)

        for j, candidate in enumerate(candidates):
            if j in used_candidates:
                continue
            for slot, track in enumerate(self.tracks):
                if not track.active:
                    self.tracks[slot] = Track(candidate=candidate, last_time=now,
                                              hits=1, observed=True)
                    break

    def confirmed_observations(self):
        """Only return confirmed tracks seen in this scan; don't publish stale poses."""
        return {slot: track for slot, track in enumerate(self.tracks)
                if track.active and track.observed and track.hits >= self.track_min_hits}

# ROS 2 integration: corresponds to the C++ node constructor and callbacks.
import os
import time
import rclpy
from rclpy.node import Node
from rclpy.clock import Clock, ClockType
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Pose, PoseArray, Point, Quaternion
from nav_msgs.msg import Odometry, OccupancyGrid
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32MultiArray
from visualization_msgs.msg import Marker, MarkerArray


def yaw_to_quaternion(yaw):
    return Quaternion(z=math.sin(yaw/2), w=math.cos(yaw/2))


class OpponentLidarDetectorNode(Node, ThreeOpponentDetector):
    def __init__(self):
        Node.__init__(self, 'opponent_lidar_detector_node')
        defaults = dict(scan_topic='/scan', ego_odom_topic='/ego_racecar/odom',
            map_topic='/map', detected_odom_topic='/opponent/detection_odom',
            detector_marker_topic='/opponent/detection_markers',
            detector_debug_topic='/opponent/detection_debug',
            multi_detection_topic='/opponents/detections',
            waypoint_path='waypoints/old/lev_testing/lev_blocked.csv',
            waypoint_path_absolute=False, reverse_waypoints=False,
            frame_id='map', base_frame_id='ego_racecar/base_link')
        for name, default in defaults.items():
            setattr(self, name, self.declare_parameter(name, default).value)
        self.tracks = [Track() for _ in range(3)]
        self.have_ego = self.map_received = False
        self.ego_x = self.ego_y = self.ego_yaw = 0.0
        self.resolution = 0.0
        self.blocked = np.empty((0,0), dtype=bool)
        self.mask_settings = None
        self.scan_skip_counter = 0
        self.last_detection_time = None
        self.last_detection_s = 0.0
        self.refresh_live_params()
        self.load_waypoints()
        sensor = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        map_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.scan_sub = self.create_subscription(LaserScan, self.scan_topic, self.scan_callback, sensor)
        self.ego_sub = self.create_subscription(Odometry, self.ego_odom_topic, self.ego_odom_callback, sensor)
        self.map_sub = self.create_subscription(OccupancyGrid, self.map_topic, self.map_callback, map_qos)
        self.detection_pub = self.create_publisher(Odometry, self.detected_odom_topic, sensor)
        self.marker_pub = self.create_publisher(MarkerArray, self.detector_marker_topic, sensor)
        self.debug_pub = self.create_publisher(Float32MultiArray, self.detector_debug_topic, sensor)
        self.multi_pub = self.create_publisher(PoseArray, self.multi_detection_topic, sensor)
        self.track_pubs = [self.create_publisher(Odometry, f'/opponents/detection_odom_{i}', sensor) for i in range(3)]
        self.params_timer = self.create_timer(0.5, self.on_params_timer, clock=Clock(clock_type=ClockType.STEADY_TIME))
        self.get_logger().info(f'Opponent LiDAR detector ready: scan={self.scan_topic} ego={self.ego_odom_topic} map={self.map_topic} waypoints={len(self.raceline)}')

    def param(self, name, default):
        if not self.has_parameter(name):
            self.declare_parameter(name, default)
        return self.get_parameter(name).value

    def refresh_live_params(self):
        # Same defaults and clamps as refreshLiveParams in C++.
        self.laser_x_offset = self.param("laser_x_offset", 0.0)
        self.laser_y_offset = self.param("laser_y_offset", 0.0)
        self.laser_yaw_offset = self.param("laser_yaw_offset", 0.0)
        self.opponent_length = max(0.05, self.param("opponent_length", 0.58))
        self.opponent_width = max(0.05, self.param("opponent_width", 0.31))
        self.max_center_correction = max(0.0, self.param("max_center_correction", 0.35))
        self.min_range = max(0.0, self.param("min_range", 0.15))
        self.max_range = max(self.min_range, self.param("max_range", 8.0))
        self.min_detection_range = max(0.0, self.param("min_detection_range", 0.35))
        self.max_detection_range = max(self.min_detection_range, self.param("max_detection_range", 6.0))
        self.front_fov_only = self.param("front_fov_only", False)
        self.min_base_x = self.param("min_base_x", -0.5)
        self.require_static_map = self.param("require_static_map", True)
        self.occupied_threshold = max(0, min(100, int(self.param("occupied_threshold", 50))))
        self.wall_inflation_radius = max(0.0, self.param("wall_inflation_radius", 0.18))
        self.treat_unknown_as_static = self.param("treat_unknown_as_static", True)
        self.cluster_tolerance = max(0.01, self.param("cluster_tolerance", 0.18))
        self.min_cluster_points = max(1, int(self.param("min_cluster_points", 4)))
        self.max_cluster_points = max(self.min_cluster_points, int(self.param("max_cluster_points", 80)))
        self.min_cluster_extent = max(0.0, self.param("min_cluster_extent", 0.08))
        self.max_cluster_extent = max(self.min_cluster_extent, self.param("max_cluster_extent", 0.9))
        self.max_raceline_projection_dist = max(0.0, self.param("max_raceline_projection_dist", 1.2))
        self.max_candidate_jump = max(0.0, self.param("max_candidate_jump", 1.5))
        self.continuity_gate_timeout = max(0.0, self.param("continuity_gate_timeout", 0.4))
        self.continuity_weight = max(0.0, self.param("continuity_weight", 2.0))
        self.projection_weight = max(0.0, self.param("projection_weight", 1.0))
        self.range_weight = max(0.0, self.param("range_weight", 0.2))
        self.max_detection_speed = max(0.1, self.param("max_detection_speed", 12.0))
        self.detection_publish_rate_hz = max(1.0, self.param("detection_publish_rate_hz", 20.0))
        self.scan_throttle_n = max(1, int(self.param("scan_throttle_n", 2)))
        self.track_match_distance = max(0.05, self.param("track_match_distance", 0.75))
        self.track_timeout = max(0.05, self.param("track_timeout", 0.8))
        self.track_dedup_distance = max(0.01, self.param("track_dedup_distance", 0.32))
        self.track_min_hits = max(1, int(self.param("track_min_hits", 2)))

    def load_waypoints(self):
        path = self.waypoint_path
        if path and not self.waypoint_path_absolute and not os.path.isabs(path):
            path = os.path.join(get_package_share_directory('mppi_bringup'), path)
        rows = []
        try:
            with open(path) as f:
                for line in f:
                    if not line.strip() or line.startswith('#'): continue
                    try: rows.append([float(v) for v in line.split(';')[:4]]) if len(line.split(';')) >= 4 else None
                    except ValueError: continue
        except OSError as e:
            self.get_logger().warning(str(e))
        if len(rows) < 2: rows = []
        arr = np.asarray(rows).reshape(-1,4)
        self.raceline = arr[:,1:3]
        self.raceline_s = arr[:,0]
        if self.reverse_waypoints and len(arr):
            self.raceline = self.raceline[::-1]
            self.raceline_s = np.r_[0., np.cumsum(np.linalg.norm(np.diff(self.raceline,axis=0),axis=1))]
        self.track_length = float(self.raceline_s[-1] + np.linalg.norm(self.raceline[0]-self.raceline[-1])) if len(arr) else 0.
        if self.track_length <= 0: self.raceline = np.empty((0,2))

    def on_params_timer(self):
        self.refresh_live_params()
        if self.map_received and (self.blocked.size == 0 or self.mask_settings != self.current_mask_settings()):
            self.rebuild_inflated_map()

    def current_mask_settings(self):
        return (math.ceil(self.wall_inflation_radius/self.resolution) if self.resolution > 0 else 0,
                self.occupied_threshold, self.treat_unknown_as_static)

    def rebuild_inflated_map(self):
        self.blocked = self.make_inflated_wall_mask() if self.resolution > 0 and self.grid.size else np.empty((0,0),dtype=bool)
        self.mask_settings = self.current_mask_settings()
        self.get_logger().info(f'Inflated static-map mask rebuilt: {self.blocked.shape}')

    def map_callback(self, msg):
        self.grid = np.asarray(msg.data).reshape(msg.info.height,msg.info.width)
        self.resolution = msg.info.resolution
        self.origin = np.array([msg.info.origin.position.x,msg.info.origin.position.y])
        self.map_received = True
        self.rebuild_inflated_map()

    def ego_odom_callback(self, msg):
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        yaw = math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))
        self.have_ego = all(map(math.isfinite,(p.x,p.y,yaw)))
        if self.have_ego:
            self.ego_x,self.ego_y,self.ego_yaw = p.x,p.y,yaw
            self.ego_stamp = msg.header.stamp
        else: self.get_logger().warning('Ignoring non-finite ego odom measurement.', throttle_duration_sec=1.)

    def unwrap_s(self, s, ref):
        if self.track_length <= 0: return s
        while s-ref > self.track_length/2: s -= self.track_length
        while s-ref < -self.track_length/2: s += self.track_length
        return s

    def make_pose(self,c):
        pose = Pose()
        pose.position.x,pose.position.y = c.x,c.y
        pose.orientation = yaw_to_quaternion(c.yaw)
        return pose

    def make_odom(self,c,stamp,child,speed):
        msg = Odometry()
        msg.header.stamp, msg.header.frame_id = stamp,self.frame_id
        msg.child_frame_id = child
        msg.pose.pose = self.make_pose(c)
        msg.twist.twist.linear.x = float(speed)
        return msg

    def publish_multi_detections(self,stamp):
        poses = PoseArray()
        poses.header.stamp,poses.header.frame_id = stamp,self.frame_id
        for i,t in self.confirmed_observations().items():
            c = t.candidate
            poses.poses.append(self.make_pose(c))
            speed = t.vx*math.cos(c.yaw)+t.vy*math.sin(c.yaw)
            self.track_pubs[i].publish(self.make_odom(c,stamp,f'opponent_detection_{i}',speed))
        self.multi_pub.publish(poses)

    def publish_markers(self,points,stamp):
        markers = MarkerArray()
        clear = Marker(action=Marker.DELETEALL)
        markers.markers.append(clear)
        def marker(i,kind):
            m = Marker(id=i,type=kind,action=Marker.ADD,ns='opponent_lidar_detection')
            m.header.stamp,m.header.frame_id = stamp,self.frame_id
            return m
        dyn = marker(0,Marker.SPHERE_LIST)
        dyn.scale.x = dyn.scale.y = dyn.scale.z = 0.06
        dyn.color.r,dyn.color.g,dyn.color.b,dyn.color.a = 1.,.1,.1,.75
        dyn.points = [Point(x=float(p[0]),y=float(p[1]),z=.08) for p in points]
        markers.markers.append(dyn)
        for i,t in self.confirmed_observations().items():
            c = t.candidate
            m = marker(1+2*i,Marker.ARROW)
            m.pose = self.make_pose(c); m.pose.position.z = .2
            m.scale.x,m.scale.y,m.scale.z = self.opponent_length,self.opponent_width,.12
            m.color.r,m.color.g,m.color.b,m.color.a = .1,1.,.25,.9
            markers.markers.append(m)
            m = marker(2+2*i,Marker.TEXT_VIEW_FACING)
            m.pose.position.x,m.pose.position.y,m.pose.position.z = c.x,c.y,.65
            m.scale.z = .2
            m.color.r=m.color.g=m.color.b=1.; m.color.a=.95
            m.text = f'opponent {i} score={c.score:.2f} d={c.distance_from_raceline:.2f}'
            markers.markers.append(m)
        self.marker_pub.publish(markers)

    def scan_callback(self,msg):
        self.scan_skip_counter += 1
        if self.scan_skip_counter % self.scan_throttle_n: return
        start = time.monotonic()
        if not self.have_ego or (self.require_static_map and not self.map_received):
            self.get_logger().warning('Waiting for ego odom or static map.', throttle_duration_sec=1.)
            return
        stamp = msg.header.stamp
        if stamp.sec == 0 and stamp.nanosec == 0: stamp = self.get_clock().now().to_msg()
        now = stamp.sec + stamp.nanosec*1e-9
        points = self.scan_to_dynamic_points(msg.ranges,msg.angle_min,msg.angle_increment,self.ego_x,self.ego_y,self.ego_yaw,msg.range_max)
        clusters = self.cluster_points(points)
        candidates = self.collect_candidates(clusters,(self.ego_x,self.ego_y))
        self.update_tracks(candidates,now)
        selected = min((t.candidate for t in self.confirmed_observations().values()),key=lambda c:c.score,default=None)
        speed = 0.
        if selected is not None:
            dt = now-self.last_detection_time if self.last_detection_time is not None else 0.
            s = self.unwrap_s(selected.s,self.last_detection_s) if self.last_detection_time is not None else selected.s
            if dt > .001: speed = min(self.max_detection_speed,max(0.,(s-self.last_detection_s)/dt))
            if self.last_detection_time is None or dt >= 1/self.detection_publish_rate_hz:
                self.detection_pub.publish(self.make_odom(selected,stamp,'opponent_lidar_detection',speed))
                self.last_detection_time,self.last_detection_s = now,s
        self.publish_multi_detections(stamp)
        self.publish_markers(points,stamp)
        values = [selected.score,selected.range_from_ego,selected.distance_from_raceline,selected.correction_norm,selected.extent_along_track,selected.extent_across_track] if selected else [-1.]*6
        self.debug_pub.publish(Float32MultiArray(data=[float(v) for v in [len(points),len(clusters),*values,speed,self.map_received]]))
        self.get_logger().info(f'scanCallback {(time.monotonic()-start)*1000:.2f} ms | dyn_pts={len(points)} clusters={len(clusters)} | {self.tally} best_proj_d={self.best_proj if math.isfinite(self.best_proj) else -1:.2f} (thresh={self.max_raceline_projection_dist:.2f})', throttle_duration_sec=2.)


def main():
    rclpy.init()
    node = OpponentLidarDetectorNode()
    try: rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()