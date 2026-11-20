"""sensor_prep: raw car sensors -> filter inputs with covariances. Same node in sim and on the car.

  /odom (VESC speed in twist.linear.x)       -> /prep/wheel_twist  TwistWithCovarianceStamped (vx, vy = 0)
  /sensors/imu/raw (angular_velocity.z)      -> /prep/imu          Imu (yaw rate only; frame = base_link)
  /sensors/servo_position_command (Float64)  -> /prep/steering     AckermannDriveStamped (delta, stamped on arrival)

Every output keeps the input stamp (the servo topic has none, so it is stamped on arrival).
"""
import dataclasses
import signal

import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import TwistWithCovarianceStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Imu
from std_msgs.msg import Float64

from .params import PrepParams
from .prep_core import GyroPrep, SpeedPrep, steering_from_servo


class SensorPrep(Node):

    def __init__(self):
        super().__init__('sensor_prep')
        defaults = PrepParams()
        for f in dataclasses.fields(PrepParams):
            self.declare_parameter(f.name, getattr(defaults, f.name))
        self.declare_parameter('r_scale', 1.0)
        self.declare_parameter('sigma_vy', 0.2)         # std of the vy = 0 pseudo-measurement [m/s]
        self.declare_parameter('base_frame', 'ego_racecar/base_link')
        self.declare_parameter('odom_in', '/odom')
        self.declare_parameter('imu_in', '/sensors/imu/raw')
        self.declare_parameter('servo_in', '/sensors/servo_position_command')
        g = lambda n: self.get_parameter(n).value  # noqa: E731

        p = PrepParams(**{f.name: g(f.name) for f in dataclasses.fields(PrepParams)})
        r_scale = float(g('r_scale'))
        self.p = p
        self.get_logger().info(f'prep params: {p}, r_scale {r_scale}, sigma_vy {g("sigma_vy")}')
        self.speed = SpeedPrep(p, r_scale)
        self.gyro = GyroPrep(p, r_scale)
        self.var_vy = float(g('sigma_vy')) ** 2
        self.frame = g('base_frame')
        self.last_speed = 0.0

        self.pub_twist = self.create_publisher(TwistWithCovarianceStamped, '/prep/wheel_twist', 10)
        self.pub_imu = self.create_publisher(Imu, '/prep/imu', 10)
        self.pub_steer = self.create_publisher(AckermannDriveStamped, '/prep/steering', 10)
        self.create_subscription(Odometry, g('odom_in'), self.on_odom, 10)
        self.create_subscription(Imu, g('imu_in'), self.on_imu, 10)
        self.create_subscription(Float64, g('servo_in'), self.on_servo, 10)

    @staticmethod
    def _t(stamp):
        return stamp.sec + stamp.nanosec * 1e-9

    def on_odom(self, msg):
        v, var = self.speed(self._t(msg.header.stamp), msg.twist.twist.linear.x)
        self.last_speed = v
        out = TwistWithCovarianceStamped()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = self.frame
        out.twist.twist.linear.x = v
        cov = [0.0] * 36
        cov[0] = var
        cov[7] = self.var_vy
        for i in (14, 21, 28, 35):       # unused axes: large but finite
            cov[i] = 1e3
        out.twist.covariance = cov
        self.pub_twist.publish(out)

    def on_imu(self, msg):
        was = self.gyro.calibrating
        r = self.gyro(self._t(msg.header.stamp), msg.angular_velocity.z, self.last_speed)
        if was and not self.gyro.calibrating:
            self.get_logger().info(f'gyro offset calibrated: {self.gyro.offset:+.5f} rad/s')
        if r is None:
            return
        w, var = r
        out = Imu()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = self.frame      # yaw rate is the same everywhere on a rigid body
        out.orientation_covariance[0] = -1.0
        out.linear_acceleration_covariance[0] = -1.0
        out.angular_velocity.z = w
        out.angular_velocity_covariance = [1e3, 0.0, 0.0, 0.0, 1e3, 0.0, 0.0, 0.0, var]
        self.pub_imu.publish(out)

    def on_servo(self, msg):
        out = AckermannDriveStamped()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = self.frame
        out.drive.steering_angle = steering_from_servo(msg.data, self.p)
        self.pub_steer.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = SensorPrep()
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
