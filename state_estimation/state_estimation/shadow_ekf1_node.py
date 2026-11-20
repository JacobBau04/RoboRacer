"""shadow_ekf1: FILTER 1B, the shadow EKF1 with a = 0, run live next to robot_localization (1A).
Comparison only: it never drives the car.

Files: model math student_ekf.py (Jacob's), filter logic shadow_core.py (ShadowFilter: dt guards,
steering hold), this ROS node, parameters params.py (ModelParams L, tau; ShadowNoise Q, Sigma0).

Dynamics (explicit Euler; sub-steps of 0.02 s when dt > 0.1 s):
    x' = x + v cos(theta) dt        y' = y + v sin(theta) dt        theta' = theta + omega dt
    v' = v                          (a = 0: constant speed, Q_v covers real changes)
    omega' = omega + (dt / tau) (v tan(delta) / L - omega)    tau = 0.1 s, L = 0.3302 m
States: 5: x, y, theta, v, omega (frame odom, rear axle).
Sensors (event-driven, no timer: each message predicts from the filter time to its stamp, then
updates; processed in RECEIVE order, messages older than the filter time are dropped):
    /prep/wheel_twist   50 Hz   v = wheel speed (vy entry not used)
    /prep/imu          100 Hz   omega = gyro z
Inputs: steering delta from /prep/steering (~45 Hz; servo command converted to rad), zero-order hold,
    no delay modelled. Acceleration a = 0.
Output: /shadow_ekf1/odometry (frame odom, no TF) after every update (~150 Hz),
    /shadow_ekf1/diagnostics (state_estimation_msgs/FilterDiag: innovation, S, NIS).
R: each message's covariance (sensor_prep). Q: ShadowNoise.q_diag (1e-6, 1e-6, 1e-6, 0.5, 0.3)
    x the q_scale parameter (1 on the car).
"""
# TODO (Jacob): think about implementing the commanded steering angle into the dynamics model or measurement for EKF_1. Would this be helpful?
# Now: 1B already predicts yaw rate from the steering command (bicycle model). On 2026-11-19 its yaw-rate
# bias doubled (+0.014 rad/s) when the car's steering response dropped to 0.77x. Keep it, calibrate it, or drop it?
import signal

import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import TwistWithCovarianceStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Imu
from state_estimation_msgs.msg import FilterDiag

from . import se2
from .params import ModelParams, ShadowNoise
from .shadow_core import ShadowFilter


def _t(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


class ShadowEkf1(Node):

    def __init__(self):
        super().__init__('shadow_ekf1')
        d = ShadowNoise()
        self.declare_parameter('L', ModelParams.L)
        self.declare_parameter('tau', ModelParams.tau)
        self.declare_parameter('q_diag', list(d.q_diag))
        self.declare_parameter('sigma0_diag', list(d.sigma0_diag))
        self.declare_parameter('q_scale', 1.0)
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'ego_racecar/base_link')
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        q_scale = float(g('q_scale'))
        self.f = ShadowFilter(ModelParams(L=g('L'), tau=g('tau')),
                              ShadowNoise(tuple(g('q_diag')), tuple(g('sigma0_diag'))), q_scale)
        self.odom_frame, self.base_frame = g('odom_frame'), g('base_frame')
        self.not_implemented = False
        self.pub = self.create_publisher(Odometry, '/shadow_ekf1/odometry', 10)
        self.pub_diag = self.create_publisher(FilterDiag, '/shadow_ekf1/diagnostics', 10)
        self.create_subscription(TwistWithCovarianceStamped, '/prep/wheel_twist', self.on_speed, 10)
        self.create_subscription(Imu, '/prep/imu', self.on_gyro, 10)
        self.create_subscription(AckermannDriveStamped, '/prep/steering', self.on_steer, 10)
        self.get_logger().info(f'shadow EKF1: {self.f.p}, Q scale {q_scale}')

    def on_steer(self, msg):
        self.f.add_steering(_t(msg.header.stamp), msg.drive.steering_angle)

    def on_speed(self, msg):
        self._measure(msg.header.stamp, 'speed', msg.twist.twist.linear.x, msg.twist.covariance[0])

    def on_gyro(self, msg):
        self._measure(msg.header.stamp, 'gyro', msg.angular_velocity.z, msg.angular_velocity_covariance[8])

    def _measure(self, stamp, sensor, y, R):
        if self.not_implemented:
            return
        try:
            r = self.f.measure(_t(stamp), sensor, float(y), float(R))
        except NotImplementedError:
            self.not_implemented = True
            self.get_logger().warn('student functions not implemented (student_ekf.py): shadow EKF publishes nothing')
            return
        if r is None:
            self.get_logger().warn('dropped a message older than the filter time', throttle_duration_sec=2.0)
            return
        if r['dt'] > self.f.max_step:
            self.get_logger().warn(f"long gap {r['dt']:.3f} s: predicted in sub-steps", throttle_duration_sec=2.0)
        self._publish(stamp, sensor, r)

    def _publish(self, stamp, sensor, r):
        mu, S = self.f.mu, self.f.Sigma
        o = Odometry()
        o.header.stamp = stamp
        o.header.frame_id = self.odom_frame
        o.child_frame_id = self.base_frame
        o.pose.pose.position.x, o.pose.pose.position.y = float(mu[0]), float(mu[1])
        q = o.pose.pose.orientation
        q.x, q.y, q.z, q.w = se2.quat_from_yaw(float(mu[2]))
        pc = [0.0] * 36
        for i, a in enumerate((0, 1, 5)):
            for j, b in enumerate((0, 1, 5)):
                pc[a * 6 + b] = float(S[i, j])
        pc[14] = pc[21] = pc[28] = 1e3
        o.pose.covariance = pc
        o.twist.twist.linear.x = float(mu[3])
        o.twist.twist.angular.z = float(mu[4])
        tc = [0.0] * 36
        tc[0], tc[35], tc[5], tc[30] = float(S[3, 3]), float(S[4, 4]), float(S[3, 4]), float(S[4, 3])
        tc[7] = tc[14] = tc[21] = tc[28] = 1e3
        o.twist.covariance = tc
        self.pub.publish(o)
        dmsg = FilterDiag()
        dmsg.header.stamp = stamp
        dmsg.sensor = sensor
        dmsg.measurement = float(r['y'])
        dmsg.innovation = r['innovation']
        dmsg.innovation_variance = r['S']
        dmsg.nis = r['nis']
        dmsg.dt = float(r['dt'])
        dmsg.sigma_diag = [float(v) for v in np.diag(S)]
        c = self.f.c
        dmsg.dropped_old, dmsg.substeps, dmsg.resets, dmsg.stale_steering = (
            c.dropped_old, c.substeps, c.resets, c.stale_steering)
        self.pub_diag.publish(dmsg)


def main(args=None):
    rclpy.init(args=args)
    node = ShadowEkf1()
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
