#!/usr/bin/env python3
"""Latch and summarise ego-car collisions reported by f1tenth_gym_ros.

The sim publishes /ego_racecar/collision as an *instantaneous* Bool (True only
while the body overlaps a wall/obstacle), so a short graze is easy to miss
with `ros2 topic echo`.  This node counts each False->True edge as one hit,
records where the car was (from odom) and how long contact lasted, and prints
a summary on Ctrl+C.

Usage (in a sourced terminal, from mpc/tools, while the sim + your MPC run):
    python3 collision_monitor.py                           # print hits live
    python3 collision_monitor.py --csv spielberg_hits.csv  # also append them to a CSV

Testing aid only. It reads the collision flag and the lap counter, which the
Lab 7 autograder forbids for graded nodes, so it lives in tools/ (not installed
by CMakeLists.txt) and must never be started from a launch file.

Each hit is written to the CSV (and flushed to disk) as soon as its contact
ends, so the file grows during the run and survives the script being killed.
Rows from earlier runs are kept; the `run` column tells runs apart.
"""

import argparse
import csv
from datetime import datetime
import math
import os
import time

from nav_msgs.msg import Odometry
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Bool, Int32

CSV_FIELDS = [
    'run', 'hit', 't_start_s', 'duration_s', 'lap', 'x', 'y', 'speed_mps']


class CollisionMonitor(Node):

    def __init__(self, namespace, csv_path=None):
        super().__init__('collision_monitor')
        self.start = time.monotonic()
        self.run_id = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.in_contact = False
        self.hits = []          # one dict per contact episode
        self.x = self.y = self.speed = float('nan')
        self.lap = 0

        self.csv_file = None
        if csv_path:
            write_header = (
                not os.path.exists(csv_path)
                or os.path.getsize(csv_path) == 0)
            self.csv_file = open(csv_path, 'a', newline='')
            self.csv_writer = csv.DictWriter(self.csv_file, CSV_FIELDS)
            if write_header:
                self.csv_writer.writeheader()
                self.csv_file.flush()

        self.create_subscription(
            Bool, f'/{namespace}/collision', self.collision_callback, 50)
        self.create_subscription(
            Odometry, f'/{namespace}/odom', self.odom_callback, 10)
        self.create_subscription(
            Int32, f'/{namespace}/lap_count', self.lap_callback, 10)
        self.get_logger().info(
            f'Watching /{namespace}/collision - Ctrl+C for a summary')

    def elapsed(self):
        return time.monotonic() - self.start

    def odom_callback(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        self.speed = math.hypot(
            msg.twist.twist.linear.x, msg.twist.twist.linear.y)

    def lap_callback(self, msg):
        self.lap = msg.data

    def collision_callback(self, msg):
        if msg.data and not self.in_contact:
            hit = {
                'run': self.run_id,
                'hit': len(self.hits) + 1,
                't_start_s': round(self.elapsed(), 3),
                'duration_s': None,
                'lap': self.lap,
                'x': round(self.x, 2),
                'y': round(self.y, 2),
                'speed_mps': round(self.speed, 2),
            }
            self.hits.append(hit)
            self.get_logger().error(
                f"HIT #{hit['hit']} at t={hit['t_start_s']:.2f}s, "
                f"base_link=({hit['x']:.2f}, {hit['y']:.2f}), "
                f"speed={hit['speed_mps']:.2f} m/s, lap {hit['lap']}")
        elif not msg.data and self.in_contact and self.hits:
            hit = self.hits[-1]
            hit['duration_s'] = round(self.elapsed() - hit['t_start_s'], 3)
            self.get_logger().info(
                f"  contact #{hit['hit']} ended after {hit['duration_s']:.2f}s")
            self.write_row(hit)
        self.in_contact = msg.data

    def write_row(self, hit):
        if self.csv_file is None:
            return
        self.csv_writer.writerow(hit)
        self.csv_file.flush()   # push to disk now, not at exit

    def close(self):
        """Record a contact still in progress, then close the CSV."""
        if self.in_contact and self.hits:
            self.write_row(self.hits[-1])   # duration_s left blank
        if self.csv_file is not None:
            self.csv_file.close()
            self.csv_file = None

    def summary(self):
        print(f'\n=== {len(self.hits)} collision(s) in '
              f'{self.elapsed():.1f} s ===')
        for hit in self.hits:
            duration = (f"{hit['duration_s']:.2f}s"
                        if hit['duration_s'] is not None else 'ongoing')
            print(f"  #{hit['hit']:>2}  t={hit['t_start_s']:7.2f}s  "
                  f"({hit['x']:6.2f}, {hit['y']:6.2f})  "
                  f"v={hit['speed_mps']:.2f}  contact={duration}  "
                  f"lap={hit['lap']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--namespace', default='ego_racecar')
    parser.add_argument(
        '--csv', help='append each hit to this CSV file as it happens')
    args = parser.parse_args()

    rclpy.init()
    node = CollisionMonitor(args.namespace, args.csv)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.close()
        node.summary()
        if args.csv:
            print(f'  saved to {os.path.abspath(args.csv)}')
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
