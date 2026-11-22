#!/usr/bin/env python3

import os
import numpy as np
import rclpy

from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from ament_index_python.packages import get_package_share_directory


class OpponentPurePursuit(Node):

    def __init__(self):
        super().__init__('opponent_pure_pursuit')

        self.lookahead = 0.8
        self.wheelbase = 0.33
        self.speed = 1.5

        bringup = get_package_share_directory('mppi_bringup')
        csv_path = os.path.join(
            bringup, 'waypoints', 'old', 'lev_testing', 'lev_blocked.csv'
        )

        data = np.loadtxt(csv_path, delimiter=';', comments='#')
        self.waypoints = data[:, 1:3]

        self.create_subscription(
            Odometry,
            '/opp_racecar/odom',
            self.pose_callback,
            qos_profile_sensor_data
        )

        self.drive_pub = self.create_publisher(
            AckermannDriveStamped, '/opp_drive', 10
        )

        self.get_logger().info('Opponent Pure Pursuit initialized')

    def pose_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        pos = np.array([x, y])
        distances = np.linalg.norm(self.waypoints - pos, axis=1)

        nearest = int(np.argmin(distances))

        # Search forward around the closed raceline
        indices = (nearest + np.arange(1, len(self.waypoints) + 1)) % len(self.waypoints)
        candidates = indices[distances[indices] >= self.lookahead]

        if len(candidates) == 0:
            return

        goal = self.waypoints[candidates[0]]

        q = msg.pose.pose.orientation
        yaw = np.arctan2(
            2 * (q.w * q.z + q.x * q.y),
            1 - 2 * (q.y**2 + q.z**2)
        )

        dx = goal[0] - x
        dy = goal[1] - y

        local_y = -np.sin(yaw) * dx + np.cos(yaw) * dy

        curvature = 2 * local_y / max(dx**2 + dy**2, 1e-6)
        steering = np.clip(
            np.arctan(self.wheelbase * curvature),
            -0.4189, 0.4189
        )

        speed = 1.0 if abs(steering) > 0.2 else self.speed

        command = AckermannDriveStamped()
        command.drive.speed = float(speed)
        command.drive.steering_angle = float(steering)

        self.drive_pub.publish(command)


def main():
    rclpy.init()
    node = OpponentPurePursuit()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
