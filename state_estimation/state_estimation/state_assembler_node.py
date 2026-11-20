"""state_assembler: PF pose (scan-stamped) + EKF1 odometry -> /state_estimate for the MPC, TF map -> odom.

Publishes on every /ekf1/odometry message: rear-axle pose in `map` (compensated to the EKF message's
stamp in config C, the latest PF pose unmoved in config B), v and yaw rate from EKF1, vy = 0.
"""
import signal

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Float32
from tf2_ros import TransformBroadcaster

from . import se2
from .assembler_core import Assembler


def _t(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


def _pose(msg):
    p = msg.pose.pose
    return (p.position.x, p.position.y, se2.yaw_from_quat(p.orientation))


class StateAssembler(Node):

    def __init__(self):
        super().__init__('state_assembler')
        self.declare_parameter('compensation', True)
        self.declare_parameter('lidar_offset', 0.275)
        self.declare_parameter('buffer_seconds', 1.0)
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'ego_racecar/base_link')
        self.declare_parameter('publish_tf', True)
        self.declare_parameter('ekf_topic', '/ekf1/odometry')
        self.declare_parameter('pf_topic', '/pf/pose/odom')
        # compensation only: also predict from the EKF1 stamp to now and stamp the output now
        self.declare_parameter('predict_to_now', True)
        self.declare_parameter('max_predict', 0.1)          # never extrapolate further than this [s]
        self.declare_parameter('correction_tau', 0.0)       # >0: low-pass the PF correction of map->odom [s]
        # Twist convention of the output. 'cog' = like the simulator's odometry, which the slip MPC was
        # tuned on: pose at the rear axle, twist at the centre of gravity. Rigid body:
        #   vy_cog = vy_ra + lr * yaw_rate
        # Wheel speed + gyro cannot observe vy_ra (EKF1 holds it near 0), so a steady-state single-track
        # relation supplies it: the rear axle slides outward as vy_ra = -k * v^2 * yaw_rate, giving
        #   vy_cog = yaw_rate * (lr - k * v^2)
        # k fitted on sim truth (2026-11-14): 0.0182 s^2/m, same at 9, 11 and 13 m/s (residual 0.08 m/s
        # vs 0.31 m/s for vy = 0). Re-fit on the car. Without it the slip MPC gets beta = 0 and chatters.
        self.declare_parameter('twist_point', 'cog')        # 'cog' or 'rear_axle'
        self.declare_parameter('lr', 0.17145)               # rear axle -> CoG [m]
        self.declare_parameter('sideslip_k', 0.0182)        # [s^2/m], 0 = kinematic (no tyre slip)
        self.declare_parameter('publish_every', 1)          # publish every Nth EKF1 update (car: 2 -> 25 Hz = race PF rate)
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.core = Assembler(g('lidar_offset'), g('compensation'), g('buffer_seconds'), g('correction_tau'))
        self.map_frame, self.odom_frame, self.base_frame = g('map_frame'), g('odom_frame'), g('base_frame')
        self.publish_tf = bool(g('publish_tf'))
        self.predict_to_now = bool(g('predict_to_now')) and bool(g('compensation'))
        self.max_predict = float(g('max_predict'))
        self.cog_twist = g('twist_point') == 'cog'
        self.lr, self.sideslip_k = float(g('lr')), float(g('sideslip_k'))
        self.publish_every, self._n_ekf = max(1, int(g('publish_every'))), 0
        self.pf_cov = [0.0] * 36
        self.get_logger().info(f"compensation {'ON (config C)' if g('compensation') else 'OFF (config B)'}")
        self.pub = self.create_publisher(Odometry, '/state_estimate', 10)
        self.pub_lat = self.create_publisher(Float32, '/state_estimate/latency', 10)
        self.br = TransformBroadcaster(self) if self.publish_tf else None
        self.create_subscription(Odometry, g('ekf_topic'), self.on_ekf, 10)
        self.create_subscription(Odometry, g('pf_topic'), self.on_pf, 10)

    def on_pf(self, msg):
        t_s = _t(msg.header.stamp)
        status = self.core.add_pf(t_s, _pose(msg))
        if status in ('older', 'empty'):
            self.get_logger().warn(f'PF stamp outside the odometry buffer ({status})', throttle_duration_sec=2.0)
        # PF writes its 3x3 (x, y, theta) particle covariance into the first 9 entries; map to 6x6
        c3 = np.array(msg.pose.covariance[:9]).reshape(3, 3)
        cov = [0.0] * 36
        for i, a in enumerate((0, 1, 5)):
            for j, b in enumerate((0, 1, 5)):
                cov[a * 6 + b] = float(c3[i, j])
        cov[14] = cov[21] = cov[28] = 1e3
        self.pf_cov = cov

    def on_ekf(self, msg):
        t = _t(msg.header.stamp)
        pose_odom = _pose(msg)
        self.core.add_odom(t, pose_odom)
        now_msg = self.get_clock().now()
        now = now_msg.nanoseconds * 1e-9
        dt_fwd = min(max(now - t, 0.0), self.max_predict) if self.predict_to_now else 0.0
        est = self.core.estimate(pose_odom, msg.twist.twist.linear.x, msg.twist.twist.angular.z, dt_fwd)
        if est is None:
            return
        self._n_ekf += 1
        if self._n_ekf % self.publish_every:
            return          # every update still feeds the buffer (add_odom above); only publishing is thinned
        out = Odometry()
        out.header.stamp = now_msg.to_msg() if self.predict_to_now else msg.header.stamp
        out.header.frame_id = self.map_frame
        out.child_frame_id = self.base_frame
        out.pose.pose.position.x, out.pose.pose.position.y = float(est[0]), float(est[1])
        q = se2.quat_from_yaw(est[2])
        o = out.pose.pose.orientation
        o.x, o.y, o.z, o.w = q
        out.pose.covariance = self.pf_cov
        out.twist.twist.linear.x = msg.twist.twist.linear.x
        out.twist.twist.angular.z = msg.twist.twist.angular.z
        v, w = msg.twist.twist.linear.x, msg.twist.twist.angular.z
        vy_ra = msg.twist.twist.linear.y - self.sideslip_k * v * v * w
        out.twist.twist.linear.y = vy_ra + (self.lr * w if self.cog_twist else 0.0)
        out.twist.covariance = msg.twist.covariance
        self.pub.publish(out)
        self.pub_lat.publish(Float32(data=float(now - self.core.pf_stamp)))
        if self.br is not None:
            T = self.core.T_map_odom
            tf = TransformStamped()
            tf.header.stamp = msg.header.stamp
            tf.header.frame_id = self.map_frame
            tf.child_frame_id = self.odom_frame
            tf.transform.translation.x, tf.transform.translation.y = float(T[0]), float(T[1])
            r = tf.transform.rotation
            r.x, r.y, r.z, r.w = se2.quat_from_yaw(T[2])
            self.br.sendTransform(tf)


def main(args=None):
    rclpy.init(args=args)
    node = StateAssembler()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            node.destroy_node()
        except Exception:  # noqa: BLE001
            pass
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
