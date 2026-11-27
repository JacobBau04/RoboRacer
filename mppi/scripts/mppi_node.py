#!/usr/bin/env python3
from dataclasses import dataclass
from functools import partial
from typing import final

import jax
import jax.numpy as jnp
import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from mppi_utils import calc_interpolated_ref_trajectory
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from nav_msgs.msg import Path
import os
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped
from wall_map import WallEnv
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Float32MultiArray
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
import time
import math
from collections import deque
from rcl_interfaces.msg import ParameterDescriptor
import threading
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

# Re-export the original functions for existing scripts and visualizations.
from mppi_core import (MPPIConfig, dynamics, step_euler, step_rk4, step,
                       simulate, simulate_jit, rollout, cost, mppi_step)
from physics_overtaking import (OvertakingPlanner, PlannerConfig, OpponentPrediction,
                                Track, SCENARIO_NAMES, predict_opponent_scenarios)
from std_msgs.msg import String

def save_trajectory_snapshot(
    paths, costs, selected, ref,
    opponent_0, opponent_1, opponent_2,
    wall_sdf, wall_origin, wall_resolution, filename,
    best_breakdown, selected_breakdown
):
    fig, ax = plt.subplots(figsize=(10, 9))

    costs = np.asarray(costs)
    paths = np.asarray(paths)

    # ----------------------------------------
    # 1. WALLS FROM THE ACTUAL MPPI DISTANCE FIELD
    # ----------------------------------------

    sdf = np.asarray(wall_sdf)
    origin = np.asarray(wall_origin)
    resolution = float(wall_resolution)

    # Show a region around the ego car
    cx, cy = selected[0, :2]
    radius = 3.5

    xmin, xmax = cx - radius, cx + radius
    ymin, ymax = cy - radius, cy + radius

    # Convert world coordinates to SDF grid indices
    c0 = max(0, int((xmin - origin[0]) / resolution))
    c1 = min(sdf.shape[1], int((xmax - origin[0]) / resolution) + 1)

    r0 = max(0, int((ymin - origin[1]) / resolution))
    r1 = min(sdf.shape[0], int((ymax - origin[1]) / resolution) + 1)

    local_sdf = sdf[r0:r1, c0:c1]

    if local_sdf.shape[0] > 1 and local_sdf.shape[1] > 1:

        xs = origin[0] + (
            np.arange(c0, c1) + 0.5
        ) * resolution

        ys = origin[1] + (
            np.arange(r0, r1) + 0.5
        ) * resolution

        # Wall boundary, approximated from the distance field
        if np.min(local_sdf) <= resolution <= np.max(local_sdf):
            ax.contour(
                xs,
                ys,
                local_sdf,
                levels=[resolution],
                colors="black",
                linewidths=2.5,
                zorder=3
            )

    # ----------------------------------------
    # 2. COST-WEIGHTED TRAJECTORIES
    # ----------------------------------------

    low = np.percentile(costs, 5)
    high = np.percentile(costs, 95)

    normalized = np.clip(
        (costs - low) / (high - low + 1e-6),
        0.0,
        1.0
    )

    cmap = plt.get_cmap("RdYlGn_r")

    colors = cmap(normalized)

    # All samples visible, but faint
    colors[:, 3] = 0.10

    widths = 0.2 + 0.8 * (1.0 - normalized)

    lines = LineCollection(
        paths[:, :, :2],
        colors=colors,
        linewidths=widths,
        zorder=2
    )

    ax.add_collection(lines)

    # Highlight the lowest-cost 5%
    best_mask = costs <= low

    best_colors = cmap(normalized[best_mask])
    best_colors[:, 3] = 0.75

    best_lines = LineCollection(
        paths[best_mask, :, :2],
        colors=best_colors,
        linewidths=2.0,
        zorder=4
    )

    ax.add_collection(best_lines)

    # Colorbar showing actual cost values
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    norm = Normalize(vmin=low, vmax=high)

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    fig.colorbar(
        sm,
        ax=ax,
        label="Total MPPI trajectory cost (5th–95th percentile)",
        shrink=0.75
    )

    # ----------------------------------------
    # 3. REFERENCE AND SELECTED TRAJECTORIES
    # ----------------------------------------

    ax.plot(
        ref[:, 0],
        ref[:, 1],
        "k--",
        linewidth=2,
        label="Reference",
        zorder=5
    )

    ax.plot(
        selected[:, 0],
        selected[:, 1],
        color="blue",
        linewidth=3,
        label="Selected MPPI",
        zorder=6
    )

    # ----------------------------------------
    # 4. OPPONENT
    # ----------------------------------------

    opponents = [opponent_0, opponent_1, opponent_2]
    colors = ["magenta", "orange", "cyan"]

    for idx, (opponent, color) in enumerate(zip(opponents, colors)):

        if not np.isfinite(opponent).all() or np.max(np.abs(opponent)) >= 100:
            continue

        ax.plot(
        opponent[:, 0],
        opponent[:, 1],
        color=color,
        linestyle="--",
        linewidth=2.5,
        label=f"Opponent {idx} prediction",
        zorder=7
        )

        ax.scatter(
        opponent[0, 0],
        opponent[0, 1],
        color=color,
        s=150,
        marker="X",
        zorder=8
        )

        ax.add_patch(
        plt.Circle(
            opponent[0, :2],
            0.4,
            color=color,
            alpha=0.15,
            zorder=3
            )
        )

    # ----------------------------------------
    # 5. EGO POSITION
    # ----------------------------------------

    ax.scatter(
        cx,
        cy,
        color="black",
        s=120,
        zorder=9,
        label="Ego"
    )

    # ----------------------------------------
    # 6. PLOT SETTINGS
    # ----------------------------------------

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    ax.set_aspect("equal")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")

    ax.set_title(
        f"MPPI Trajectory Costs\n"
        f"Minimum: {np.min(costs):.2f} | "
        f"Median: {np.median(costs):.2f} | "
        f"Maximum: {np.max(costs):.2f}"
    )

    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.2)

    cost_text = "COST BREAKDOWN\n"
    cost_text += f"{'Component':<14} {'Best':>9} {'Selected':>10}\n"
    cost_text += "-" * 37 + "\n"

    for name in best_breakdown:
        cost_text += (
            f"{name:<14} "
            f"{best_breakdown[name]:>9.2f} "
            f"{selected_breakdown[name]:>10.2f}\n"
        )

    fig.text(
    0.76, 0.55,
    cost_text,
    fontsize=10,
    family="monospace",
    verticalalignment="center",
    bbox=dict(facecolor="white", edgecolor="black", alpha=0.95)
)

    fig.set_size_inches(16, 9)
    fig.subplots_adjust(left=0.06, right=0.57)

        # Numbered timesteps for selected MPPI trajectory
    for i, (x, y) in enumerate(selected[1:, :2], start=1):
        ax.annotate(
            str(i),
            (x, y),
            xytext=(0, -14),
            textcoords="offset points",
            ha="center",
            fontsize=7,
            color="blue",
            bbox=dict(
                boxstyle="circle,pad=0.2",
                facecolor="white",
                edgecolor="blue",
                alpha=0.9
            ),
            zorder=15
        )

    # Numbered timesteps for opponent prediction
    for idx, (opponent, color) in enumerate(zip(opponents, colors)):

        if not np.isfinite(opponent).all() or np.max(np.abs(opponent)) >= 100:
            continue

        for i, (x, y) in enumerate(opponent, start=1):
            ax.annotate(
                str(i),
                (x, y),
                xytext=(0, 14),
                textcoords="offset points",
                ha="center",
                fontsize=7,
                color=color,
                bbox=dict(
                    boxstyle="circle,pad=0.2",
                    facecolor="white",
                    edgecolor=color,
                    alpha=0.9
                ),
                zorder=15
            )

    fig.savefig(filename, dpi=150)
    plt.close(fig)

class MPPI(Node):
    """
    Implement Kinematic MPPI on the car
    This is just a template, you are free to implement your own node!
    """
    def __init__(self):
        super().__init__('mppi_node')
        # launch/levine_launch.py sets track:=levine, launch/spielberg_launch.py track:=spielberg
        self.declare_parameter('wall_map_yaml', '')
        self.declare_parameter('waypoint_path', '')
        self.declare_parameter('physics_overtaking', True)
        self.physics_enabled = bool(self.get_parameter('physics_overtaking').value)
        self.declare_parameter('prediction_start_delay', 0.0)
        self.prediction_start_delay = float(self.get_parameter('prediction_start_delay').value)
        self.declare_parameter('pose_timeout', 0.5)
        self.pose_timeout = float(self.get_parameter('pose_timeout').value)
        self.last_pose_arrival = None
        self.physics_predictions = [None, None, None]
        self.physics_invalid = set()
        self.previous_steer = 0.0
        self.behavior_pub = self.create_publisher(String, '/mppi/behavior', 10)
        self.last_physics_status = None
        self.create_timer(0.1, self._pose_watchdog)
        # Every physics planner setting can be tuned from a launch file.
        from dataclasses import fields
        defaults = PlannerConfig()
        for field in fields(defaults):
            self.declare_parameter('overtaking.' + field.name, getattr(defaults, field.name))
        self.physics_settings = PlannerConfig(**{
            field.name: self.get_parameter('overtaking.' + field.name).value
            for field in fields(defaults)})

        # ---- car settings, ported from the RoboRacer MPC (defaults 0 = sim behaviour unchanged) ----
        # dynamic_typing: a 0 from a launch file arrives as an int
        # The particle-filter pose is the LiDAR, pose_offset m ahead of the rear axle (0.27 on the car);
        # the bicycle model is for the rear axle.
        self.declare_parameter('pose_offset', 0.0, ParameterDescriptor(dynamic_typing=True))
        self.pose_offset = float(self.get_parameter('pose_offset').value)
        # Steering command -> turn in the pose takes ~0.20 s on the car (TUNING_LOG R2: without
        # compensation the MPC weaved at 0.6 Hz). Each solve starts from the measured state moved
        # forward by delay_comp s, replaying the commands already sent in that window.
        self.declare_parameter('delay_comp', 0.0, ParameterDescriptor(dynamic_typing=True))
        self.delay_comp = float(self.get_parameter('delay_comp').value)
        self.cmd_hist = deque()   # (time.monotonic(), speed, steer) of each sent /drive
        self.get_logger().info(f'pose_offset {self.pose_offset:.3f} m, delay_comp {self.delay_comp:.3f} s'
                               + (' (off)' if self.delay_comp <= 0 else ''))

        # TODO: create ROS subscribers and publishers
        #       use MPPI as a tracker (similar to pure pursuit)

        self.last_snapshot_time = 0.0
        self.snapshot_busy = False

        self.snapshot_dir = os.path.join(os.path.expanduser("~/roboracer_ws/src/RoboRacer"),"mppi_debug")

        # Create debug output only when snapshots are explicitly enabled.

        self.pose_sub_ = self.create_subscription(Odometry, '/ego_racecar/odom', self.odom_callback, 1)
        self.drive_pub_ = self.create_publisher(AckermannDriveStamped, '/drive', 1)
        self.path_pub_ = self.create_publisher(Path, '/mppi/selected_path', 10)
        self.opponent_sub_0 = self.create_subscription(Path,'/opponents/predicted_path_0',self.opponent_callback_0,qos_profile_sensor_data)
        self.opponent_sub_1 = self.create_subscription(Path,'/opponents/predicted_path_1',self.opponent_callback_1,qos_profile_sensor_data)
        self.opponent_sub_2 = self.create_subscription(Path,'/opponents/predicted_path_2',self.opponent_callback_2,qos_profile_sensor_data)
        self.wall_distance_pub_ = self.create_publisher(
            Float32MultiArray,
            '/mppi/wall_distances',
            10
        )

        self.wall_debug_pub_ = self.create_publisher(
            MarkerArray,
            '/mppi/wall_debug',
            10
        )
        # TODO: load the waypoints of `track` from the package's installed waypoints/ folder:
        #       os.path.join(get_package_share_directory('mppi'), 'waypoints', <your csv>)
        #       (ament_index_python.packages). A path like /home/you/... only exists on your laptop.

        self.declare_parameter('track', 'levine')
        track = self.get_parameter('track').value

        
        map_yaml = self.get_parameter('wall_map_yaml').value
        csv_path = self.get_parameter('waypoint_path').value

        if not map_yaml:
            map_yaml = os.path.join(
                get_package_share_directory('mppi_bringup'),
                'maps',
                track + '.yaml'
            )

        if not csv_path:
            csv_path = os.path.join(
                get_package_share_directory('mppi'),
                'waypoints',
                track + '.csv'
            )

        self.wall_env = WallEnv(map_yaml)

        self.wall_sdf_np = np.asarray(self.wall_env.wall_sdf)
        self.wall_origin_np = np.asarray(self.wall_env.wall_origin)
        self.wall_resolution = self.wall_env.wall_resolution

        if os.path.basename(csv_path) == 'lev_blocked.csv':

            self.waypoints = np.loadtxt(
                csv_path,
                delimiter=';',
                comments='#'
            )

            self.ref_x = self.waypoints[:, 1].copy()
            self.ref_y = self.waypoints[:, 2].copy()
            self.ref_yaw = self.waypoints[:, 3].copy()
            self.ref_v = self.waypoints[:, 5].copy()

        else:

            self.waypoints = np.loadtxt(
                csv_path,
                delimiter=',',
                skiprows=1
            )

            self.ref_x = self.waypoints[:, 0].copy()
            self.ref_y = self.waypoints[:, 1].copy()
            self.ref_yaw = self.waypoints[:, 2].copy()
            self.ref_v = self.waypoints[:, 3].copy()
       

        self.yaw = None
        self.car_x = None
        self.car_y = None

        self.take_snapshots = False
        self.config = MPPIConfig()
        self.physics_planner = OvertakingPlanner(
            Track(np.column_stack((self.ref_x, self.ref_y)), self.ref_v),
            self.wall_env, self.config, self.physics_settings) if self.physics_enabled else None
        self.U = jnp.zeros((self.config.H, self.config.NU))  # mean control sequence
        self.key = jax.random.PRNGKey(0)
        self.opponent_xy_0 = jnp.full((self.config.H, 2),1000.0)
        self.opponent_xy_1 = jnp.full((self.config.H, 2),1000.0)
        self.opponent_xy_2 = jnp.full((self.config.H, 2),1000.0)

        self.opponent_path_time_0 = None
        self.opponent_path_time_1 = None
        self.opponent_path_time_2 = None

        self.opponent_timeout = 2.0

        if not self.physics_enabled:
            self._warm_legacy()

    def _warm_legacy(self):
        mppi_step(
            self.key,
            self.U,
            jnp.zeros(self.config.NX),
            jnp.zeros((self.config.H + 1, self.config.NX)),
            self.config,
            self.opponent_xy_0,
            self.opponent_xy_1,
            self.opponent_xy_2,
            self.wall_env
        )

        
    def opponent_callback_0(self, msg):
        self._receive_physics_prediction(0, msg)
        if self.physics_enabled:
            return
        if len(msg.poses) == 0:
            self.opponent_path_time_0 = None
            return

        points = np.array([
            [pose.pose.position.x, pose.pose.position.y]
            for pose in msg.poses
        ], dtype=np.float32).reshape(-1, 2)

        if len(points) < 2:
            self.opponent_path_time_0= None
            return

        if not np.isfinite(points).all():
            self.opponent_path_time_0 = None
            return

        predicted = points[1:]

        if len(predicted) >= self.config.H:
            predicted = predicted[:self.config.H]

        else:
            padding = np.repeat(
                predicted[-1:],
                self.config.H - len(predicted),
                axis=0
            )

            predicted = np.vstack((predicted, padding))

        self.opponent_xy_0 = jnp.asarray(predicted)

        self.opponent_path_time_0 = (
            self.get_clock().now().nanoseconds * 1e-9
        )  

    def opponent_callback_1(self,msg):
        self._receive_physics_prediction(1, msg)
        if self.physics_enabled:
            return
        if len(msg.poses) ==0:
            self.opponent_path_time_1 = None
            return
        points = np.array([
                    [pose.pose.position.x, pose.pose.position.y]
                    for pose in msg.poses
                ], dtype=np.float32).reshape(-1, 2)
        
        if len(points) < 2:
                    self.opponent_path_time_1= None
                    return
        
        if not np.isfinite(points).all():
                    self.opponent_path_time_1 = None
                    return
        
        predicted = points[1:]
        
        if len(predicted) >= self.config.H:
                    predicted = predicted[:self.config.H]
        
        else:
            padding = np.repeat(
                        predicted[-1:],
                        self.config.H - len(predicted),
                        axis=0
                    )
        
            predicted = np.vstack((predicted, padding))
        
        self.opponent_xy_1 = jnp.asarray(predicted)
        
        self.opponent_path_time_1 = (
                    self.get_clock().now().nanoseconds * 1e-9
                )  
    def opponent_callback_2(self,msg):
        self._receive_physics_prediction(2, msg)
        if self.physics_enabled:
            return
        if len(msg.poses) ==0:
            self.opponent_path_time_2 = None
            return
        points = np.array([
                            [pose.pose.position.x, pose.pose.position.y]
                            for pose in msg.poses
                        ], dtype=np.float32).reshape(-1, 2)
                
        if len(points) < 2:
            self.opponent_path_time_2= None
            return
                
        if not np.isfinite(points).all():
            self.opponent_path_time_2 = None
            return
                
        predicted = points[1:]
                
        if len(predicted) >= self.config.H:
                            predicted = predicted[:self.config.H]
                
        else:
            padding = np.repeat(predicted[-1:],self.config.H - len(predicted),
                                axis=0)
                
            predicted = np.vstack((predicted, padding))
                
        self.opponent_xy_2 = jnp.asarray(predicted)
                
        self.opponent_path_time_2 = (self.get_clock().now().nanoseconds * 1e-9) 

        

    def publish_drive(self, speed, steering):
        msg = AckermannDriveStamped()
        msg.drive.speed = float(speed)
        msg.drive.steering_angle = float(steering)
        self.drive_pub_.publish(msg)
        self.previous_steer = float(steering)
        # every command the car actually got, for predict_delay
        self.cmd_hist.append((time.monotonic(), float(speed), float(steering)))

    def _pose_watchdog(self):
        if self.physics_enabled and (self.last_pose_arrival is None
                                     or time.monotonic() - self.last_pose_arrival > self.pose_timeout):
            self.publish_drive(0.0, self.previous_steer)

    def _receive_physics_prediction(self, index, msg):
        if not self.physics_enabled:
            return
        points = np.asarray([[pose.pose.position.x, pose.pose.position.y]
                             for pose in msg.poses], dtype=float)
        # Empty/stale predictions must not silently make a known opponent disappear.
        if len(points) < 2 or not np.isfinite(points).all() or msg.header.frame_id != 'map':
            if self.physics_predictions[index] is not None or len(points):
                self.physics_invalid.add(index)
            return
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.physics_predictions[index] = OpponentPrediction(
            points, stamp, self.physics_settings.opponent_dt)
        self.physics_invalid.discard(index)

    def _physics_control(self, vehicle_state):
        import json
        now = self.get_clock().now().nanoseconds * 1e-9
        cfg = self.physics_planner.cfg
        paths = np.full((3, cfg.H + 1, 2), 1000.0)
        scenarios = np.repeat(paths[:, None], len(SCENARIO_NAMES), axis=1)
        active = np.zeros(3, dtype=bool)
        stale = bool(self.physics_invalid)
        for i, prediction in enumerate(self.physics_predictions):
            if prediction is None:
                continue
            scenarios[i], covered = predict_opponent_scenarios(
                self.physics_planner.track, prediction, now, cfg.H, cfg.DT,
                self.physics_settings, self.prediction_start_delay)
            paths[i] = scenarios[i, 0]
            active[i] = True
            stale |= not covered or now - prediction.stamp > self.physics_settings.stale_timeout
        result = self.physics_planner.plan(np.asarray(vehicle_state), paths, active,
                                          self.previous_steer, stale, scenarios=scenarios)
        status = json.dumps({'behavior': result.behavior, 'valid': result.valid,
                             'reason': result.reason, 'valid_samples': result.valid_samples,
                             'opponent_scenarios': list(SCENARIO_NAMES)})
        self.behavior_pub.publish(String(data=status))
        if (result.behavior, result.valid, result.reason) != self.last_physics_status:
            self.get_logger().info(status)
            self.last_physics_status = (result.behavior, result.valid, result.reason)
        self.U = jnp.asarray(result.controls)
        path = Path()
        path.header.stamp = self.get_clock().now().to_msg()
        path.header.frame_id = 'map'
        for state in result.trajectory:
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x, pose.pose.position.y = float(state[0]), float(state[1])
            pose.pose.orientation.z = float(np.sin(state[2] / 2))
            pose.pose.orientation.w = float(np.cos(state[2] / 2))
            path.poses.append(pose)
        self.path_pub_.publish(path)
        if not result.valid:
            # No certified finite-horizon plan: demand a stop and report the failure.
            self.publish_drive(0.0, float(result.controls[0, 1]))
            return
        speed = np.clip(float(vehicle_state[3]) + result.controls[0, 0] * cfg.DT,
                        cfg.MIN_SPEED, cfg.MAX_SPEED)
        self.publish_drive(float(speed), float(result.controls[0, 1]))

    def predict_delay(self, x, y, yaw, v, now):
        """Move the measured state forward by delay_comp seconds (kinematic bicycle, speed
        held constant). Ported unchanged from the RoboRacer MPC (mpc_node.py predict_delay).

        delay_comp = pose delay d_pose + actuation delay d_act. The pose shows the car at
        now - d_pose, and the command sent now acts from now + d_act. In between, the car
        is steered by the commands sent during the last delay_comp seconds (a command sent
        at time c acts from c + d_act), so replaying those in order gives the state at the
        moment the new command starts acting. Only the total delay matters."""
        tau = self.delay_comp
        hist = self.cmd_hist
        # keep the newest command sent at or before now - tau (active at the start), drop older
        while len(hist) > 1 and hist[1][0] <= now - tau:
            hist.popleft()
        if tau <= 0 or not hist:
            return x, y, yaw
        n = max(1, int(round(tau / 0.01)))           # 10 ms Euler steps
        dt = tau / n
        i = 0
        for k in range(n):
            ts = now - tau + k * dt
            while i + 1 < len(hist) and hist[i + 1][0] <= ts:
                i += 1
            steer = hist[i][2] if hist[i][0] <= ts else 0.0   # nothing sent yet: straight
            x += v * math.cos(yaw) * dt
            y += v * math.sin(yaw) * dt
            yaw += v * math.tan(steer) / self.config.WB * dt
        return x, y, yaw

    def odom_callback(self, odom_msg):
        self.last_pose_arrival = time.monotonic()
        if self.physics_enabled:
            stamp = odom_msg.header.stamp.sec + odom_msg.header.stamp.nanosec * 1e-9
            age = self.get_clock().now().nanoseconds * 1e-9 - stamp
            if not np.isfinite(age) or age > self.pose_timeout or age < -0.05:
                self.publish_drive(0.0, self.previous_steer)
                return
        # TODO: extract the state x0 = [p_x, p_y, yaw, v] from the ROS msg

        self.car_x = odom_msg.pose.pose.position.x
        self.car_y = odom_msg.pose.pose.position.y
                
    

        quaternion = np.array([odom_msg.pose.pose.orientation.x, 
                                            odom_msg.pose.pose.orientation.y, 
                                            odom_msg.pose.pose.orientation.z, 
                                            odom_msg.pose.pose.orientation.w])
        self.yaw = np.arctan2(2 * (quaternion[3] * quaternion[2] + quaternion[0] * quaternion[1]),
                                    1 - 2 * (quaternion[1]**2 + quaternion[2]**2))

        # car: particle-filter pose is the LiDAR, pose_offset m ahead of the rear axle (no-op if 0)
        self.car_x -= self.pose_offset * np.cos(self.yaw)
        self.car_y -= self.pose_offset * np.sin(self.yaw)
        v_meas = odom_msg.twist.twist.linear.x
        # car: where the car will be when this command takes effect (no-op if delay_comp = 0).
        # The reference path, the MPPI start state and the speed command all use this state;
        # speed is held at the measured value.
        x_pred, y_pred, yaw_pred = self.predict_delay(
            float(self.car_x), float(self.car_y), float(self.yaw), v_meas, time.monotonic())

        vehicle_state = jnp.array([x_pred, y_pred, yaw_pred, v_meas])

        if not np.isfinite(np.asarray(vehicle_state)).all():
            self.publish_drive(0.0, 0.0)
            return

        if self.physics_enabled:
            self._physics_control(vehicle_state)
            return

        # TODO: Calculate the reference trajectory for the next H steps
                #       with the current vehicle state.
                #       ref_x, ref_y, ref_yaw, ref_v are columns of self.waypoints
        
                
        ref_path = np.asarray(
        self.calc_ref_trajectory(
            vehicle_state,
            self.ref_x,
            self.ref_y,
            self.ref_yaw,
            self.ref_v
        )).copy()

        assert ref_path.shape == (self.config.H + 1, self.config.NX)

        ref_path[:, 2] = np.unwrap(ref_path[:, 2])

        ref_path[:, 2] += 2 * np.pi * np.round(
            (float(vehicle_state[2]) - ref_path[0, 2]) / (2 * np.pi)
        )

        ref_path = jnp.asarray(ref_path)



        # TODO: warm start: the first input of self.U was applied at the last
        #       step, so shift self.U one step forward (drop the first input
        #       and repeat the last one)

        self.U = jnp.vstack((self.U[1:], self.U[-1]))

        # a new random key at every step, so MPPI draws new samples
        self.key, subkey = jax.random.split(self.key)
        now = self.get_clock().now().nanoseconds * 1e-9

        inactive = jnp.full((self.config.H, 2), 1000.0)

        opponent_xy_0 = (
            self.opponent_xy_0
            if self.opponent_path_time_0 is not None
            and 0 <= now - self.opponent_path_time_0 <= self.opponent_timeout
            else inactive
        )

        opponent_xy_1 = (
            self.opponent_xy_1
            if self.opponent_path_time_1 is not None
            and 0 <= now - self.opponent_path_time_1 <= self.opponent_timeout
            else inactive
        )

        opponent_xy_2 = (
            self.opponent_xy_2
            if self.opponent_path_time_2 is not None
            and 0 <= now - self.opponent_path_time_2 <= self.opponent_timeout
            else inactive
        )

        # TODO: run MPPI, keep the new mean control sequence in self.U


        self.U, sampled_paths, sampled_costs, sampled_controls = mppi_step(
            subkey,
            self.U,
            vehicle_state,
            ref_path,
            self.config,
            opponent_xy_0,
            opponent_xy_1,
            opponent_xy_2,
            self.wall_env
        )

        selected_trajectory = simulate_jit(
            vehicle_state,
            self.U,
            self.config
        )

        now = time.monotonic()

        if self.take_snapshots and now - self.last_snapshot_time >= 1.0 and not self.snapshot_busy:

            os.makedirs(self.snapshot_dir, exist_ok=True)
            self.last_snapshot_time = now
            self.snapshot_busy = True

            paths = np.asarray(sampled_paths)
            costs = np.asarray(sampled_costs)
            selected = np.asarray(selected_trajectory)
            reference = np.asarray(ref_path)
            opponent_0 = np.asarray(opponent_xy_0)
            opponent_1 = np.asarray(opponent_xy_1)
            opponent_2 = np.asarray(opponent_xy_2)

            best_idx = int(np.argmin(costs))

            best_breakdown = cost(
                sampled_paths[best_idx],
                sampled_controls[best_idx],
                ref_path,
                self.config,
                opponent_xy_0,
                opponent_xy_1,
                opponent_xy_2,
                self.wall_env,
                return_components=True
            )

            selected_breakdown = cost(
                selected_trajectory,
                self.U,
                ref_path,
                self.config,
                opponent_xy_0,
                opponent_xy_1,
                opponent_xy_2,
                self.wall_env,
                return_components=True
            )

            best_breakdown = {
                k: float(v) for k, v in best_breakdown.items()
            }

            selected_breakdown = {
                k: float(v) for k, v in selected_breakdown.items()
            }

            filename = os.path.join(
                self.snapshot_dir,
                f"snapshot_{int(time.time() * 1000)}.png"
            )

            def save_image():
                try:
                    save_trajectory_snapshot(
                    paths,
                    costs,
                    selected,
                    reference,
                    opponent_0,
                    opponent_1,
                    opponent_2,
                    self.wall_sdf_np,
                    self.wall_origin_np,
                    self.wall_resolution,
                    filename,
                    best_breakdown,
                    selected_breakdown
                )
                    print(f"SNAPSHOT SAVED: {filename}")
                except Exception as e:
                    print(f"SNAPSHOT ERROR: {e}")
                finally:
                    self.snapshot_busy = False

            threading.Thread(
                target=save_image,
                daemon=True
            ).start()

        wall_distances = np.asarray(self.wall_env.sample_wall_distance(selected_trajectory[1:, :2]))

        msg = Float32MultiArray()
        msg.data = wall_distances.tolist()

        self.wall_distance_pub_.publish(msg)

        ego_xy = np.asarray(selected_trajectory[1:, :2])        

        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = "map"

        for state in np.asarray(selected_trajectory):
            pose = PoseStamped()
            pose.header = path_msg.header

            pose.pose.position.x = float(state[0])
            pose.pose.position.y = float(state[1])
            pose.pose.position.z = 0.0

            yaw = float(state[2])
            pose.pose.orientation.z = float(np.sin(yaw / 2))
            pose.pose.orientation.w = float(np.cos(yaw / 2))

            path_msg.poses.append(pose)

        self.path_pub_.publish(path_msg)

        # car: commented out, the result was never used and the uncompiled call took ~21 ms per pose
        # selected_cost = cost(
        #         selected_trajectory,
        #         self.U,
        #         ref_path,
        #         self.config,
        #         opponent_xy,
        #         self.wall_env,
        #         debug=True)
      
    

        steer_output = self.U[0,1]
        
        speed_output = jnp.clip(
            vehicle_state[3] + self.U[0, 0] * self.config.DT,
            self.config.MIN_SPEED,
            self.config.MAX_SPEED
        )

        self.publish_drive(speed_output, steer_output)

    def calc_ref_trajectory(self, x0, cx, cy, cyaw, cv):
        """
        Calculate interpolated reference trajectory for H steps: [x, y, yaw, v]

        :param x0: current state [p_x, p_y, yaw, v]
        :param cx: Course X-Position
        :param cy: Course y-Position
        :param cyaw: Course Heading
        :param cv: speed profile
        :return: reference trajectory ref (H+1 x NX)
        """
        return calc_interpolated_ref_trajectory(
            x0[0], x0[1],
            cx, cy, cyaw, cv,
            self.config.DT,
            self.config.H
        )


def main(args=None):
    rclpy.init(args=args)
    print("MPPI Initialized")
    mppi_node = MPPI()
    try:
        rclpy.spin(mppi_node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass  # Ctrl-C: stop quietly
    finally:
        mppi_node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()



"""" 
RACE 3

- At each step test if it is better than the last implementation - Ablation study
1. Validate basic obstacle avoidance with arclength progress - DONE
2. Grab cedric's opponent locator code and opponent predictor code - DONE
3. Implement Cedric's v1 opponent passing code with just bubble cost - DONE
4. Need to adjust all of this code to be able to deal with multiple cars - TODO
--------------------------------------------------------------------------------------

PROJECT EXTENSION
5. Use gaussian sampling to generate many opponent trajectores (can use EKF or not but need some kind of sampling)
   ---Kalman filter can be used to make better predicted trajectories, gaussian will probably work fine though.

   OR

   Create an observed opp raceline which we build after one lap for a better estimation of opp raceline. We then converge to it.

6. Bias MPPI samples based on opponent trajectory using saved last opponent trajectory. Start with generating
MPPI samples and then for every future car state. Look at previous calculate opponent trajectorie(s) and then bias
our MPPI sampling to avoid. 

7. RL Passing Decision (maybe)
"""
