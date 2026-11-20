#!/usr/bin/env python3
"""Python port of Cedric's ROS 2 opponent_predictor_node.cpp.

Predicts a single opponent's future (x, y, yaw) positions along a closed
raceline. Designed to subscribe to /opponent/detection_odom from the LiDAR
detector; it does not detect opponents from LaserScan itself.
"""
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import Odometry, Path as PathMsg
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from std_msgs.msg import Float32MultiArray
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

EPS = 1e-9


def clamp(value, low, high):
    return max(low, min(value, high))


def wrap_angle(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def yaw_quaternion(yaw):
    from geometry_msgs.msg import Quaternion
    return Quaternion(x=0.0, y=0.0, z=math.sin(yaw / 2.0), w=math.cos(yaw / 2.0))


@dataclass
class Waypoint:
    s: float #Distance Along The Raceline
    x: float 
    y: float 
    yaw: float 
    speed: float


@dataclass
class TrackPose:
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    speed: float = 0.0


@dataclass
class Projection:
    s: float = 0.0 #How far around the track is the opponent
    lateral_error: float = 0.0 #How far to the left or right of the raceline is the opponent
    distance: float = math.inf
    yaw: float = 0.0
    profile_speed: float = 0.0


class OpponentPredictorNode(Node):
    def __init__(self):
        super().__init__('opponent_predictor_node')

        # Defaults adapted for your working detector and its Levine CSV.
        defaults = {
            'input_odom_topic': '/opponent/detection_odom', #Input topic that comes from the opponent detector
            'estimated_odom_topic': '/opponent/odom', #Output topic, filtered opponent position and velocity
            'predicted_path_topic': '/opponent/predicted_path', #Output Topic Future Predicted Opponent Positions
            'marker_topic': '/opponent/markers', #Output Topic - Visualization
            'debug_topic': '/opponent/debug',#Output Topic Debug Topic
            'waypoint_path': 'waypoints/old/lev_testing/lev_blocked.csv',
            'waypoint_path_absolute': False,
            'reverse_waypoints': False,
            'frame_id': 'map',
            'pose_source': 'odom_pose',
            'tf_lookup_timeout': 0.02,
            'prediction_steps': 5,
            'prediction_dt': 0.1,
            'publish_rate_hz': 20.0,
            'stale_timeout': 0.5,
            'max_stale_prediction_time': 2.0,  # Kept for compatibility; original timer doesn't use it.
            'speed_profile_scale': 1.0,
            'speed_profile_min_speed': 0.0,
            'speed_profile_max_speed': 20.0,
            'profile_speed_blend': 0.7,
            'out_of_sight_profile_speed_blend': 1.0,
            'use_profile_speed_fallback': False,
            'stationary_speed_threshold': 0.15,
            'hold_stationary_when_stale': True,
            'lateral_offset_alpha': 0.35,
            'lateral_offset_decay_time': 0.3,
            'position_snap_alpha': 1.0,
            'initial_speed': 1.0,
            'max_progress_speed': 12.0,
            'max_progress_accel': 8.0,
            'kf_process_var_s': 0.05,
            'kf_process_var_v': 0.5,
            'kf_measurement_var_s': 0.15,
            'kf_measurement_var_v': 0.6,
        }
        for name, default in defaults.items():
            self.declare_parameter(name, default)
        self.refresh_live_params()

        self.input_odom_topic = self.p('input_odom_topic')
        self.estimated_odom_topic = self.p('estimated_odom_topic')
        self.predicted_path_topic = self.p('predicted_path_topic')
        self.marker_topic = self.p('marker_topic')
        self.debug_topic = self.p('debug_topic')
        self.frame_id = self.p('frame_id')

        self.waypoints = []
        self.track_length = 0.0
        self.load_waypoints(self.resolve_waypoint_path(
            self.p('waypoint_path'), self.p('waypoint_path_absolute')))

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        sensor_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
                                history=HistoryPolicy.KEEP_LAST)
        self.odom_sub = self.create_subscription(
            Odometry, self.input_odom_topic, self.odom_callback, sensor_qos)
        self.estimated_pub = self.create_publisher(Odometry, self.estimated_odom_topic, sensor_qos)
        self.path_pub = self.create_publisher(PathMsg, self.predicted_path_topic, sensor_qos)
        self.marker_pub = self.create_publisher(MarkerArray, self.marker_topic, sensor_qos)
        self.debug_pub = self.create_publisher(Float32MultiArray, self.debug_topic, sensor_qos)

        # 2-state filter: [unwrapped track distance s, forward progress speed v].
        self.initialized = False
        self.visualization_cleared = True
        self.s_hat = 0.0
        self.v_hat = 0.0
        self.p00, self.p01, self.p10, self.p11 = 0.5, 0.0, 0.0, 1.0
        self.last_filter_time = 0.0
        self.last_measurement_time = -math.inf
        self.last_projection_time = -math.inf
        self.last_projected_s = 0.0
        self.last_measured_speed = math.nan
        self.last_progress_speed = math.nan
        self.last_twist_speed = math.nan
        self.last_profile_speed = math.nan
        self.last_projection_distance = 0.0
        self.last_lateral_error = 0.0
        self.lateral_offset_hat = 0.0

        self.timer = self.create_timer(1.0 / self.publish_rate_hz, self.timer_callback)
        self.params_timer = self.create_timer(0.5, self.refresh_live_params)
        self.get_logger().info(
            f'Opponent predictor ready: topic={self.input_odom_topic}, '
            f'waypoints={len(self.waypoints)}, '
            f'horizon={self.prediction_steps} x {self.prediction_dt:.3f}s')

    def p(self, name):
        return self.get_parameter(name).value

    def refresh_live_params(self):
        # Same tunable parameters as C++ (refreshed every 0.5 s).
        self.pose_source = self.p('pose_source')
        self.tf_lookup_timeout = max(0.0, float(self.p('tf_lookup_timeout')))
        self.prediction_steps = max(1, int(self.p('prediction_steps')))
        self.prediction_dt = max(0.001, float(self.p('prediction_dt')))
        self.publish_rate_hz = max(1.0, float(self.p('publish_rate_hz')))
        self.stale_timeout = max(0.0, float(self.p('stale_timeout')))
        self.max_stale_prediction_time = max(
            self.stale_timeout, float(self.p('max_stale_prediction_time')))
        self.speed_profile_scale = max(0.0, float(self.p('speed_profile_scale')))
        self.speed_profile_min_speed = max(0.0, float(self.p('speed_profile_min_speed')))
        self.speed_profile_max_speed = max(
            self.speed_profile_min_speed, float(self.p('speed_profile_max_speed')))
        self.profile_speed_blend = clamp(float(self.p('profile_speed_blend')), 0, 1)
        self.out_of_sight_profile_speed_blend = clamp(
            float(self.p('out_of_sight_profile_speed_blend')), 0, 1)
        self.use_profile_speed_fallback = bool(self.p('use_profile_speed_fallback'))
        self.stationary_speed_threshold = max(0.0, float(self.p('stationary_speed_threshold')))
        self.hold_stationary_when_stale = bool(self.p('hold_stationary_when_stale'))
        self.lateral_offset_alpha = clamp(float(self.p('lateral_offset_alpha')), 0, 1)
        self.lateral_offset_decay_time = max(0.001, float(self.p('lateral_offset_decay_time')))
        self.position_snap_alpha = clamp(float(self.p('position_snap_alpha')), 0, 1)
        self.initial_speed = max(0.0, float(self.p('initial_speed')))
        self.max_progress_speed = max(0.1, float(self.p('max_progress_speed')))
        self.max_progress_accel = max(0.0, float(self.p('max_progress_accel')))
        self.q_s = max(0.0, float(self.p('kf_process_var_s')))
        self.q_v = max(0.0, float(self.p('kf_process_var_v')))
        self.r_s = max(1e-6, float(self.p('kf_measurement_var_s')))
        self.r_v = max(1e-6, float(self.p('kf_measurement_var_v')))

    @staticmethod
    def resolve_waypoint_path(path, absolute):
        if not path or absolute or Path(path).is_absolute():
            return path
        return str(Path(get_package_share_directory('mppi_bringup')) / path)

    def load_waypoints(self, path):
        if not path:
            self.get_logger().warning('No waypoint_path supplied; predictor will not publish.')
            return
        try:
            with open(path, newline='') as f:
                for fields in csv.reader(f, delimiter=';'):
                    if not fields or fields[0].startswith('#') or len(fields) < 6:
                        continue
                    try:
                        self.waypoints.append(Waypoint(
                            float(fields[0]), float(fields[1]), float(fields[2]),
                            float(fields[3]), float(fields[5])))
                    except ValueError:
                        continue
        except OSError as exc:
            self.get_logger().error(f'Cannot open waypoint CSV {path}: {exc}')
            return
        if len(self.waypoints) < 2:
            self.get_logger().error(f'Fewer than two usable waypoints: {path}')
            self.waypoints.clear()
            return
        if self.p('reverse_waypoints'):
            self.waypoints.reverse()
            self.waypoints[0].s = 0.0
            self.waypoints[0].yaw = wrap_angle(self.waypoints[0].yaw + math.pi)
            for i in range(1, len(self.waypoints)):
                a, b = self.waypoints[i - 1], self.waypoints[i]
                b.s = a.s + math.hypot(b.x - a.x, b.y - a.y)
                b.yaw = wrap_angle(b.yaw + math.pi)
        first, last = self.waypoints[0], self.waypoints[-1]
        self.track_length = last.s + math.hypot(first.x - last.x, first.y - last.y)
        if self.track_length <= 0:
            self.get_logger().error(f'Invalid track length: {path}')
            self.waypoints.clear()

    def normalize_s(self, s):
        """ 
        #Convert opponent total progres to current lap progress - Track is 20 meters, opponent has traveled 23 meters, current lap progress = 3 meters
        """
        return s % self.track_length if self.track_length > 0 else s

    def unwrap_s(self, measured_s, reference_s):
        """
        s_unrapped = s_measured + number of laps completed * track length
        """
        if self.track_length <= 0:
            return measured_s
        return measured_s + round((reference_s - measured_s) / self.track_length) * self.track_length

    def interpolate_track(self, s):
        """
        Takes in S and returns interpolated values corresponding to values between waypoints

        
                    s x y yaw speed
        waypoint 0  
        waypoint 1
        waypoint 2

        Given an s we return a new row 
                        s x y yaw speed
        interpolated s 

        That has values somehwere between waypoints
        """
        if not self.waypoints:
            return TrackPose()
        s = self.normalize_s(s)
        idx = len(self.waypoints) - 1
        for i in range(len(self.waypoints) - 1):
            if self.waypoints[i].s <= s < self.waypoints[i + 1].s:
                idx = i
                break
        a = self.waypoints[idx]
        b = self.waypoints[idx + 1] if idx + 1 < len(self.waypoints) else self.waypoints[0]
        b_s = b.s if idx + 1 < len(self.waypoints) else self.track_length
        ratio = clamp((s - a.s) / max(EPS, b_s - a.s), 0, 1)
        speed = a.speed + ratio * (b.speed - a.speed)
        return TrackPose(
            x=a.x + ratio * (b.x - a.x),
            y=a.y + ratio * (b.y - a.y),
            yaw=wrap_angle(a.yaw + ratio * wrap_angle(b.yaw - a.yaw)),
            speed=clamp(speed * self.speed_profile_scale,
                        self.speed_profile_min_speed, self.speed_profile_max_speed))

    @staticmethod
    def apply_lateral_offset(pose, lateral_offset):
        """
        Takes a pose  x, y, yaw, speed

        Outputs a new pose at a 90 degree offset lateral offset meters away from the original pose
        
        
        """
        return TrackPose(
            x=pose.x - math.sin(pose.yaw) * lateral_offset,
            y=pose.y + math.cos(pose.yaw) * lateral_offset,
            yaw=pose.yaw, speed=pose.speed)

    def project_to_track(self, x, y):
        """
        Input = Opponent X, Y

        Function:
        
        Break the entire waypoint raceline into segments, then project the opponent onto the nearest segment
        
        We return:
        S: Opponent distance along the raceline
        distance: distance of opponent from the raceline
        yaw: yaw
        profile speed: opponent estimated speed
        lateral error: distance from the raceline but taking into account sign

        
        """
        best = Projection()
        for i, a in enumerate(self.waypoints):
            b = self.waypoints[(i + 1) % len(self.waypoints)]
            b_s = b.s if i + 1 < len(self.waypoints) else self.track_length
            dx, dy = b.x - a.x, b.y - a.y
            t = clamp(((x - a.x) * dx + (y - a.y) * dy) / max(EPS, dx * dx + dy * dy), 0, 1)
            px, py = a.x + t * dx, a.y + t * dy
            ex, ey = x - px, y - py
            distance = math.hypot(ex, ey)
            if distance < best.distance:
                yaw = math.atan2(dy, dx)
                s = self.normalize_s(a.s + t * (b_s - a.s))
                best = Projection(
                    s=s, distance=distance, yaw=yaw,
                    profile_speed=self.interpolate_track(s).speed,
                    lateral_error=ex * (-math.sin(yaw)) + ey * math.cos(yaw))
        return best

    def predict_kalman_to(self, now_sec):
        """
        Kalman Filter Covariance Error Calculations. Use the previous error covariances to update
        New error covariances

        p00: Variance in estimated position error
        p01: Covariance between position error and velocity error
        p10: Covariance between velocity error and position error
        p11: Variance in estimated velocity error
        """

        #If we haven't initialized return
        if not self.initialized:
            return
        #How much time has passed since we did our last filter
        dt = max(0.0, now_sec - self.last_filter_time)

        #If no time has elapsed nothing to predict
        if dt <= 0.0:
            return

        #distance along the track  = velocity along the track * time
        self.s_hat += self.v_hat * dt
        # Use old error covariances to calculate new error covariances
        #If you want to work these out by hand take E(es + ev * delta t)**2 to find p values

        #Add prediction uncertainty since we made a new prediction
        p00, p01, p10, p11 = self.p00, self.p01, self.p10, self.p11
        self.p00 = p00 + dt * (p10 + p01) + dt * dt * p11 + self.q_s
        self.p01 = p01 + dt * p11
        self.p10 = p10 + dt * p11
        self.p11 = p11 + self.q_v
        self.last_filter_time = now_sec

    def update_kalman(self, s_meas, v_meas):
        """
        Update our Kalman filter using actual measurements
        """

        #Measurement error
        y_s = s_meas - self.s_hat
        #prediction error variance + measurement error variance
        S_s = self.p00 + self.r_s
        #Calculate Kalman gains.
        K0_s, K1_s = self.p00 / S_s, self.p10 / S_s
        #Update progress on track with error times gain
        self.s_hat += K0_s * y_s
        #Update velocity with error times gain
        self.v_hat += K1_s * y_s
        #Reduce prediction variance since we made a measurement
        old_p00, old_p01 = self.p00, self.p01
        self.p00 -= K0_s * old_p00
        self.p01 -= K0_s * old_p01
        self.p10 -= K1_s * old_p00
        self.p11 -= K1_s * old_p01

        #Do same thing for velocity
        y_v = v_meas - self.v_hat
        S_v = self.p11 + self.r_v
        K0_v, K1_v = self.p01 / S_v, self.p11 / S_v
        self.s_hat += K0_v * y_v
        self.v_hat += K1_v * y_v
        old_p10, old_p11 = self.p10, self.p11
        self.p00 -= K0_v * old_p10
        self.p01 -= K0_v * old_p11
        self.p10 -= K1_v * old_p10
        self.p11 -= K1_v * old_p11
        self.v_hat = max(0.0, self.v_hat)

    def transform_opponent_pose(self, msg):
        """
        Convert opponent pose into our coordinate frame
        
        """
        pose = PoseStamped()
        pose.header = msg.header
        pose.pose = msg.pose.pose
        if not pose.header.frame_id:
            pose.header.frame_id = self.frame_id
        if self.pose_source == 'tf' and msg.child_frame_id:
            try:
                transform = self.tf_buffer.lookup_transform(
                    self.frame_id, msg.child_frame_id, Time(),
                    timeout=Duration(seconds=self.tf_lookup_timeout))
                result = PoseStamped()
                result.header = transform.header
                result.pose.position.x = transform.transform.translation.x
                result.pose.position.y = transform.transform.translation.y
                result.pose.position.z = transform.transform.translation.z
                result.pose.orientation = transform.transform.rotation
                return result
            except TransformException as exc:
                self.get_logger().warning(f'Opponent TF lookup failed: {exc}')
                return None
        if self.pose_source == 'odom_pose' and pose.header.frame_id == self.frame_id:
            return pose
        try:
            # Imports the type conversion registration needed by tf2_ros.Buffer.transform.
            import tf2_geometry_msgs  # noqa: F401
            return self.tf_buffer.transform(
                pose, self.frame_id, timeout=Duration(seconds=self.tf_lookup_timeout))
        except (TransformException, ImportError, TypeError) as exc:
            self.get_logger().warning(f'Opponent pose transform failed: {exc}')
            return None

    def odom_callback(self, msg):
        if not self.waypoints:
            return
        #Convert opponent pose to our coordinate frame and get their pose
        opponent_pose = self.transform_opponent_pose(msg)
        if opponent_pose is None:
            return
        
        stamp_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        now_sec = stamp_sec if stamp_sec > 0 else self.get_clock().now().nanoseconds * 1e-9
        #Get x, y opponent pose in our frame
        x, y = opponent_pose.pose.position.x, opponent_pose.pose.position.y
        if not all(map(math.isfinite, (now_sec, x, y))):
            self.get_logger().warning('Ignoring non-finite opponent odometry measurement')
            return
        #Project opponent to track returning, s,distance,yaw,profile speed,lateral error
        proj = self.project_to_track(x, y)
        #Get their x,y and z velocity
        twist = msg.twist.twist.linear
        vx, vy, vz = twist.x, twist.y, twist.z
        self.last_twist_speed = (math.sqrt(vx * vx + vy * vy + vz * vz)
                                 if all(map(math.isfinite, (vx, vy, vz))) else math.nan)
        #S measurement is the projected s, last twist speed is their speed
        s_meas, v_meas = proj.s, self.last_twist_speed
        self.last_progress_speed = math.nan
        progress_speed_valid = False
        if self.initialized:
            #unrap measurements so they align with our kalman filter
            s_meas = self.unwrap_s(proj.s, self.s_hat)
            s_for_delta = self.unwrap_s(proj.s, self.last_projected_s)
            dt = now_sec - self.last_projection_time
            #If enough time has passed since last measurement
            if dt > 1e-3:
                raw_progress_speed = (s_for_delta - self.last_projected_s) / dt
                self.last_progress_speed = raw_progress_speed
                progress_speed_valid = (math.isfinite(raw_progress_speed)
                                        and -0.25 <= raw_progress_speed <= self.max_progress_speed)
                if progress_speed_valid:
                    v_meas = clamp(raw_progress_speed, 0.0, self.max_progress_speed)
                    if math.isfinite(self.last_measured_speed) and self.max_progress_accel > 0:
                        max_step = self.max_progress_accel * dt
                        v_meas = clamp(v_meas,
                                       max(0.0, self.last_measured_speed - max_step),
                                       self.last_measured_speed + max_step)
        if (self.use_profile_speed_fallback and not progress_speed_valid
                and (not math.isfinite(v_meas) or abs(v_meas) < 0.05)):
            v_meas = proj.profile_speed
        # C++ std::clamp preserves NaN in some circumstances; reject it here.
        if not math.isfinite(v_meas):
            v_meas = self.initial_speed
        v_meas = clamp(v_meas, 0.0, self.max_progress_speed)
        if not self.initialized:
            self.initialized = True
            self.s_hat = proj.s
            self.v_hat = v_meas
            self.lateral_offset_hat = proj.lateral_error
            self.last_filter_time = now_sec
        else:
            #Make a prediction and add uncertainty
            self.predict_kalman_to(now_sec)
            #Adjust prediction and update uncertainty
            self.update_kalman(s_meas, v_meas)
            #adjust kalman filter after the fact (potentially unneccessary)
            self.s_hat = ((1.0 - self.position_snap_alpha) * self.s_hat
                          + self.position_snap_alpha * s_meas)
            #Comptue lateral offset
            self.lateral_offset_hat = ((1.0 - self.lateral_offset_alpha) * self.lateral_offset_hat
                                       + self.lateral_offset_alpha * proj.lateral_error)
        self.last_measurement_time = now_sec
        self.last_projection_time = now_sec
        self.last_projected_s = self.unwrap_s(proj.s, self.s_hat)
        self.last_measured_speed = v_meas
        self.last_projection_distance = proj.distance
        self.last_lateral_error = proj.lateral_error
        self.last_profile_speed = proj.profile_speed

    def make_prediction(self, stale=False):
        """
        The bulk of our work is here. The kalman filter just gives predicted locations
        and velocities, we then need to project those forward.
        
        """
        prediction = []
        pred_s = self.s_hat
        pred_v = max(0.0, self.v_hat)
        if pred_v < self.stationary_speed_threshold:
            pred_v = 0.0
        blend = self.out_of_sight_profile_speed_blend if stale else self.profile_speed_blend
        stationary = pred_v <= self.stationary_speed_threshold
        if stationary and (not stale or self.hold_stationary_when_stale):
            blend = 0.0
        for k in range(self.prediction_steps + 1):
            pose = self.interpolate_track(pred_s)
            lateral_decay = math.exp(-k * self.prediction_dt / self.lateral_offset_decay_time)
            pose = self.apply_lateral_offset(pose, self.lateral_offset_hat * lateral_decay)
            pose.speed = pred_v
            prediction.append(pose)
            profile_v = self.interpolate_track(pred_s).speed
            pred_v = (1.0 - blend) * pred_v + blend * profile_v
            if stationary and (not stale or self.hold_stationary_when_stale):
                pred_v = 0.0
            pred_s += pred_v * self.prediction_dt
        return prediction

    def _marker(self, marker_type, marker_id, stamp):
        marker = Marker()
        marker.header.stamp = stamp
        marker.header.frame_id = self.frame_id
        marker.ns = 'opponent_prediction'
        marker.id = marker_id
        marker.type = marker_type
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        return marker

    @staticmethod
    def _point(pose, z):
        return Point(x=pose.x, y=pose.y, z=z)

    def publish_prediction(self, prediction, stale, age):
        if not prediction:
            return
        stamp = self.get_clock().now().to_msg()
        current = prediction[0]
        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self.frame_id
        odom.child_frame_id = 'opponent_estimate'
        odom.pose.pose.position.x = current.x
        odom.pose.pose.position.y = current.y
        odom.pose.pose.orientation = yaw_quaternion(current.yaw)
        odom.twist.twist.linear.x = current.speed
        self.estimated_pub.publish(odom)

        path = PathMsg()
        path.header.stamp = stamp
        path.header.frame_id = self.frame_id
        for p in prediction:
            ps = PoseStamped()
            ps.header = path.header
            ps.pose.position.x = p.x
            ps.pose.position.y = p.y
            ps.pose.orientation = yaw_quaternion(p.yaw)
            path.poses.append(ps)
        self.path_pub.publish(path)

        markers = MarkerArray()
        erase = Marker()
        erase.action = Marker.DELETEALL
        markers.markers.append(erase)
        line = self._marker(Marker.LINE_STRIP, 0, stamp)
        line.scale.x = 0.06
        line.color.a = 0.9
        line.color.r = 1.0 if stale else 0.1
        line.color.g = 0.55 if stale else 0.8
        line.color.b = 0.1 if stale else 1.0
        line.points = [self._point(p, 0.08) for p in prediction]
        markers.markers.append(line)
        arrow = self._marker(Marker.ARROW, 1, stamp)
        arrow.pose.position = self._point(current, 0.12)
        arrow.pose.orientation = yaw_quaternion(current.yaw)
        arrow.scale.x, arrow.scale.y, arrow.scale.z = 0.4, 0.12, 0.12
        arrow.color.a = 0.95
        arrow.color.r = 1.0 if stale else 0.05
        arrow.color.g = 0.45 if stale else 0.85
        arrow.color.b = 0.05 if stale else 0.25
        markers.markers.append(arrow)
        dots = self._marker(Marker.SPHERE_LIST, 2, stamp)
        dots.scale.x = dots.scale.y = dots.scale.z = 0.16
        dots.color.a = 0.85
        dots.color.r = 1.0 if stale else 0.0
        dots.color.g = 0.8 if stale else 0.3
        dots.color.b = 0.0 if stale else 1.0
        dots.points = [self._point(p, 0.1) for p in prediction]
        markers.markers.append(dots)
        label = self._marker(Marker.TEXT_VIEW_FACING, 3, stamp)
        label.pose.position = self._point(current, 0.7)
        label.scale.z = 0.22
        label.color.a = label.color.r = label.color.g = label.color.b = 1.0
        label.text = f'v={self.v_hat:.1f} age={age:.2f}' + (' stale' if stale else '')
        markers.markers.append(label)
        self.marker_pub.publish(markers)
        self.visualization_cleared = False

        debug = Float32MultiArray()
        debug.data = [float(self.normalize_s(self.last_projected_s)),
                      float(self.v_hat), float(self.last_progress_speed),
                      float(self.last_twist_speed), float(self.last_profile_speed),
                      float(self.last_projection_distance), float(self.lateral_offset_hat),
                      1.0 if stale else 0.0]
        self.debug_pub.publish(debug)

    def clear_prediction_visualization(self):
        if self.visualization_cleared:
            return
        path = PathMsg()
        path.header.stamp = self.get_clock().now().to_msg()
        path.header.frame_id = self.frame_id
        self.path_pub.publish(path)
        erase = Marker()
        erase.action = Marker.DELETEALL
        arr = MarkerArray()
        arr.markers.append(erase)
        self.marker_pub.publish(arr)
        self.visualization_cleared = True

    def timer_callback(self):
        if not self.initialized or not self.waypoints:
            return
        now_sec = self.get_clock().now().nanoseconds * 1e-9
        self.predict_kalman_to(now_sec)
        age = now_sec - self.last_measurement_time
        if age > self.stale_timeout:
            self.clear_prediction_visualization()
            return
        self.publish_prediction(self.make_prediction(stale=False), False, age)


def main(args=None):
    rclpy.init(args=args)
    node = OpponentPredictorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()