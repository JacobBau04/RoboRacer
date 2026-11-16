#!/usr/bin/env python3
import os
# One BLAS thread. Must run before numpy is imported. The matrices here are tiny (9x9 expm
# in the slip model), and multithreaded OpenBLAS makes them ~100x slower when other processes
# (the sim, rviz) compete for the CPUs: measured 84 ms vs 0.05 ms for 8 expm calls.

'''
T1 (FOR FOXGLOVE: ws://192.168.137.136:8765)
export ROS_DOMAIN_ID=42
source ~/sim_ws/.venv/bin/activate
source ~/sim_ws/install/setup.bash
export OPENBLAS_NUM_THREADS=1
ros2 launch f1tenth_gym_ros gym_bridge_launch.py config:=$HOME/sim_ws/race1_straights/sim_clean.yaml open_foxglove:=false


T2
export ROS_DOMAIN_ID=42
ros2 launch mpc race1_sim_launch.py record:=true note:="layout_000"





POSSIBLE THINGS LEFT TO TUNE:
- physical car parameter (wheelbase length, etc)
- maximize turning angle (currently about .34 rad, can be up to about .42 rad)

'''

for _var in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_var, '1')
import math
from collections import deque
from dataclasses import dataclass, field

import cvxpy
import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDrive, AckermannDriveStamped
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from scipy.linalg import block_diag
from scipy.sparse import block_diag, csc_matrix, diags
from sensor_msgs.msg import LaserScan
from utils import nearest_point, calc_interpolated_ref_trajectory
import slip_model
import avoidance_msgs
import os                                                      # top of file
from ament_index_python.packages import get_package_share_directory   # top of file
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
import json
import time
from rclpy.qos import DurabilityPolicy, QoSProfile
from rcl_interfaces.msg import ParameterDescriptor
from std_msgs.msg import Float32MultiArray, Float64MultiArray as _F64, String
import warnings

# SciPy warns that block_diag of plain numpy arrays will return coo_array instead of
# coo_matrix. Both have the .row/.col/.data/.nnz we use, so the change doesn't affect us.
# Don't "fix" it by wrapping a block in coo_matrix: that drops zero entries, so A_block.nnz
# would vary between solves and break the fixed-size Annz_k/Bnnz_k parameters.
warnings.filterwarnings("ignore", message="`block_diag` is switching", category=DeprecationWarning)

# CHECK: include needed ROS msg type headers and libraries

'''
TODO

fix MPC model error by adding friction

fffdd

LAST CHANGED: MPC & WAYPOITNS VELOCITY AND ACCLERATION UNTIL WALLS ARE IT
NEXT: 
- INCREASE SPEED OF SIM BY ADDING RESOURCES
- add sim turning dynamics to waypoint maker



now
- make the sim stop clipping corners (decrease turning cost)



later
- understand how good raceline package works
- how to change mine
- maybe plot reference errors (position & velocity on corners vs straights?)
'''


'''
0. rebuild mpc package
1. Test current config performance (with debug)
2. Test performance from each of the following changes
  - use sim's cornering dynamics



  - change the v-max on the make_waypoints file & generate new waypoints
  - acceleration gain?
  - friction dynamics on corners?

'''


@dataclass
class mpc_config:
    NXK: int = 4  # length of kinematic state vector: z = [x, y, v, yaw]
    NU: int = 2  # length of input vector: u = = [steering speed, acceleration]
    TK: int = 16  # finite time horizon length kinematic (was 8; 16 x DTK 0.05 = same 0.8 s look-ahead)

    # ---------------------------------------------------
    # TODO: you may need to tune the following matrices
    Rk: list = field(
        default_factory=lambda: np.diag([0.01, 5.0]) # 0.01, 100.0
    )  # input cost matrix, penalty for inputs - [accel, steering_speed]
    Rdk: list = field(
        default_factory=lambda: np.diag([0.01,30.0]) # 0.01, 100.0
    )  # input difference cost matrix, penalty for change of inputs - [accel, steering_speed]
    Qk: list = field(
        default_factory=lambda: np.diag([13.5, 13.5, 5.5, 13.0])
    )  # state error cost matrix, for the the next (T) prediction time steps [x, y, v, yaw]
    Qfk: list = field(
        default_factory=lambda: np.diag([13.5, 13.5, 5.5, 13.0])
    )  # final state error matrix, penalty  for the final state constraints: [x, y, v, yaw]
    # ---------------------------------------------------

    N_IND_SEARCH: int = 20  # Search index number
    DTK: float = 0.05  # time step [s] kinematic (was 0.1: the coarser step made the car turn in early)
    dlk: float = 0.03  # dist step [m] kinematic
    LENGTH: float = 0.58  # Length of the vehicle [m]
    WIDTH: float = 0.31  # Width of the vehicle [m]
    WB: float = 0.324  # Wheelbase [m] (measured on the car)
    MIN_STEER: float = -0.35  # -0.4189 maximum steering angle [rad]
    MAX_STEER: float = 0.35  # 0.4189 maximum steering angle [rad]
    MAX_DSTEER: float = np.deg2rad(180.0)  # maximum steering speed [rad/s]
    MIN_SPEED: float = 0.0  # minimum backward speed [m/s]

    
    
    
    MAX_SPEED: float = 3.3  # TRY: 3.3, MAYBE: 3.5? CH maximum speed [m/s]
    MAX_ACCEL: float = 3.1  # TRY: 4.0       maximum acceleration [m/ss]

    # slip model only (ROS parameter model:=slip). State becomes [x, y, v, yaw, yaw_rate, beta]
    # and Qk/Qfk get two extra zero weights (see MPC.__init__).
    NXK_SLIP: int = 6
    N_SUB: int = 20  # rollout Euler sub-steps per DTK (0.1 s / 20 = 5 ms, stable down to 0.5 m/s)


@dataclass
class State:
    x: float = 0.0
    y: float = 0.0
    v: float = 0.0
    yaw: float = 0.0

class MPC(Node):
    """ 
    Implement Kinematic MPC on the car
    This is just a template, you are free to implement your own node!
    """
    def __init__(self):
        super().__init__('mpc_node')
        # launch/levine_launch.py sets track:=levine, launch/spielberg_launch.py track:=spielberg
        self.declare_parameter('track', 'levine')
        track = self.get_parameter('track').value

        # create ROS subscribers and publishers
        #       use the MPC as a tracker (similar to pure pursuit)
        # pose source: the particle filter on the car, /ego_racecar/odom in the sim
        self.declare_parameter('pose_topic', 'pf/pose/odom')
        self.odom_sub = self.create_subscription(
            Odometry, self.get_parameter('pose_topic').value, self.odom_callback, 1)
        self.drive_pub = self.create_publisher(AckermannDriveStamped,'/drive', 10)
        # load the waypoints of `track` from the package's installed waypoints/ folder:
        #       os.path.join(get_package_share_directory('mpc'), 'waypoints', <your csv>)
        #       (ament_index_python.packages). A path like /home/you/... only exists on your laptop.
        waypoints_path = os.path.join(get_package_share_directory('mpc'), 'waypoints', f'{track}.csv')
        self.waypoints = np.loadtxt(waypoints_path, delimiter=',')
        self.init_avoidance()

        self.wp_pub = self.create_publisher(Marker, '/mpc/waypoints', 1)
        self.ref_pub = self.create_publisher(Marker, '/mpc/ref_traj', 1)
        self.pred_pub = self.create_publisher(Marker, '/mpc/predicted', 1)
        self.create_timer(1.0, lambda: self.wp_pub.publish(
        self.make_marker(self.waypoints[:, 0], self.waypoints[:, 1], Marker.LINE_STRIP, (0.6, 0.6, 0.6), 0.05)))
        self.CMD_PERIOD = 0.027 # time between two drive commands. roughly 27 ms in sim (TODO: WILL BE DIFFERENT ON REAL CAR)




        self.config = mpc_config()
        self.odelta = None
        self.oa = None
        self.init_flag = 0

        # Prediction model: 'kinematic' (default, the original MPC) or 'slip' (the sim's
        # dynamic single-track model, see slip_model.py). Must be set before mpc_prob_init().
        self.declare_parameter('model', 'kinematic')
        self.model = self.get_parameter('model').value
        if self.model not in ('kinematic', 'slip'):
            raise ValueError(f"model must be 'kinematic' or 'slip', got {self.model!r}")
        if self.model == 'slip':
            self.slip_params = slip_model.SlipParams()
            self.config.NXK = self.config.NXK_SLIP
            # Extra states yaw_rate and beta are not tracked (weight 0).
            # TODO: track yaw_rate with reference yaw_rate_ref = v_ref * kappa_ref from the raceline
            self.config.Qk = block_diag((self.config.Qk, np.zeros((2, 2)))).toarray()
            self.config.Qfk = block_diag((self.config.Qfk, np.zeros((2, 2)))).toarray()
        self.get_logger().info(f'MPC prediction model: {self.model}')

        # Delay compensation [s], 0 = off. On the car the pose arrives late (particle filter)
        # and a command acts late (servo, motor): run_analysis.py measured ~0.26 s from a
        # /drive command to the turn showing up in the pose ("STEERING" in summary.txt).
        # The MPC assumes no delay, which made the car oscillate (~0.6 Hz). With this on,
        # each solve starts from the measured state moved forward by delay_comp seconds,
        # using the commands already sent in that window (see predict_delay).
        # dynamic_typing: delay_comp:=0 from a launch file arrives as an int
        self.declare_parameter('delay_comp', 0.0, ParameterDescriptor(dynamic_typing=True))
        self.delay_comp = float(self.get_parameter('delay_comp').value)
        self.cmd_hist = deque()   # (time.monotonic(), speed, steer) of each sent /drive
        self.get_logger().info(f'MPC delay compensation: {self.delay_comp:.3f} s'
                               + (' (off)' if self.delay_comp <= 0 else ''))

        # How far the pose point is ahead of the rear axle [m]. The kinematic model is for the
        # rear axle (the sim's odom is already there: 0). On the car the particle filter gives
        # the LiDAR, 0.27 m ahead (base_link -> laser in bringup_launch.py). Treating it as the
        # rear axle made the car cut inside every corner (it swings into the turn first).
        self.declare_parameter('pose_offset', 0.0, ParameterDescriptor(dynamic_typing=True))
        self.pose_offset = float(self.get_parameter('pose_offset').value)
        self.get_logger().info(f'MPC pose offset (pose point ahead of rear axle): '
                               f'{self.pose_offset:.3f} m')

        # initialize MPC problem
        self.mpc_prob_init()

        # Debug output for tools/run_recorder.py. Off by default, so graded runs
        # (levine_launch.py / spielberg_launch.py) publish nothing extra;
        # levine_debug_launch.py turns it on.
        self.declare_parameter('publish_debug', False)
        self.publish_debug = self.get_parameter('publish_debug').value
        self.last_solve_ms = float('nan')
        self.last_osqp_ms = float('nan')
        self.last_status = 0
        self.last_costs = [float('nan')] * 13   # see cost_terms()
        if self.publish_debug:
            self.stats_pub = self.create_publisher(Float32MultiArray, '/mpc/solve_stats', 10)
            # transient local = "latched": a recorder that subscribes later still gets it
            latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
            self.config_pub = self.create_publisher(String, '/mpc/config', latched)
            self.config_pub.publish(String(data=self.config_json(waypoints_path)))

    def init_avoidance(self):
        """Obstacle avoidance (obstacle_avoidance_node.py) feeds detours that replace a stretch
        of the raceline. Off by default (avoidance:=false keeps the plain raceline MPC)."""
        P = self.declare_parameter
        self.avoidance = bool(P('avoidance', False).value)
        self.raceline = self.waypoints              # never changed
        self.active_wp = self.waypoints             # raceline, or raceline with a detour
        self.plan_time = None                       # time.monotonic() of the last plan message
        self.plan_active = False
        self.plan_stop = False
        self.plan_timeout = float(P('plan_timeout', 0.5).value)       # s
        self.stale_speed_cap = float(P('stale_speed_cap', 1.5).value)  # m/s if plans stop coming
        # safety check of the predicted path against the obstacle boxes (not the walls)
        self.safety_margin = float(P('safety_margin', 0.0).value)     # m, car side to box
        self.safety_front = float(P('safety_front', 0.19).value)      # m, body ahead of the pose
        self.safety_rear = float(P('safety_rear', 0.39).value)        # m, body behind the pose
        self.safety_half_width = float(P('safety_half_width', 0.155).value)
        self.safety_brake = float(P('safety_brake', 3.0).value)       # m/s^2
        self.obstacle_pts = np.zeros((0, 2))
        if not self.avoidance:
            return
        self.create_subscription(_F64, '/obstacle_avoidance/plan',
                                 self.plan_callback, 1)
        self.create_subscription(_F64, '/obstacle_avoidance/obstacles', self.obstacles_callback, 1)
        self.safety_pub = self.create_publisher(_F64, '/mpc/safety', 10)
        self.active_pub = self.create_publisher(Marker, '/mpc/active_path', 1)
        self.get_logger().info('obstacle avoidance on: following /obstacle_avoidance/plan')

    def plan_callback(self, msg):
        try:
            plan = avoidance_msgs.decode_plan(msg.data)
        except ValueError as e:
            self.get_logger().warn(f'bad plan message: {e}', throttle_duration_sec=2.0)
            return
        self.plan_time = time.monotonic()
        if plan['pts'] is None:
            self.active_wp = self.raceline
            self.plan_active = False
            self.plan_stop = False
            return
        self.active_wp = avoidance_msgs.splice(self.raceline, plan['i_start'], plan['i_end'], plan['pts'])
        self.plan_active = True
        self.plan_stop = plan['stop']
        self.active_pub.publish(self.make_marker(plan['pts'][:, 0], plan['pts'][:, 1],
                                                 Marker.LINE_STRIP, (0.1, 0.5, 1.0), 0.05))

    def obstacles_callback(self, msg):
        try:
            _, self.obstacle_pts = avoidance_msgs.decode_obstacles(msg.data)
        except ValueError as e:
            self.get_logger().warn(f'bad obstacles message: {e}', throttle_duration_sec=2.0)

    def reference_waypoints(self):
        """Waypoints for this solve: the raceline, or the raceline with the current detour.
        If plans stop arriving during a detour, keep the detour but cap the speed."""
        wp = self.active_wp
        if self.avoidance and self.plan_active and self.plan_time is not None \
                and time.monotonic() - self.plan_time > self.plan_timeout:
            wp = wp.copy()
            wp[:, 3] = np.minimum(wp[:, 3], self.stale_speed_cap)
            self.get_logger().warn('no obstacle plan for %.1f s: keeping the last detour, slow'
                                   % (time.monotonic() - self.plan_time), throttle_duration_sec=1.0)
        return wp

    def safety_speed(self, ox, oy, oyaw, ov):
        """Highest speed that still stops before the predicted path would reach an obstacle
        box (inf if it never does). Steering is left as the MPC planned it."""
        pts = self.obstacle_pts
        if not self.avoidance or len(pts) == 0 or ox is None:
            return math.inf
        dist_along = 0.0
        reach = self.safety_half_width + self.safety_margin

        def footprint(k):
            c, s = math.cos(oyaw[k]), math.sin(oyaw[k])
            return [(ox[k] + a * c, oy[k] + a * s) for a in
                    (self.safety_front, 0.5 * (self.safety_front - self.safety_rear), -self.safety_rear)]
        # box points the car already overlaps now are a wrong estimate (it has not hit
        # anything): ignore them, or the car would stop forever beside the box
        near_now = np.zeros(len(pts), bool)
        for px, py in footprint(0):
            near_now |= np.hypot(pts[:, 0] - px, pts[:, 1] - py) < reach
        pts = pts[~near_now]
        if len(pts) == 0:
            return math.inf
        for k in range(1, len(ox)):
            dist_along += math.hypot(ox[k] - ox[k - 1], oy[k] - oy[k - 1])
            for px, py in footprint(k):
                d = np.min(np.hypot(pts[:, 0] - px, pts[:, 1] - py))
                if d < reach:
                    v_safe = math.sqrt(2.0 * self.safety_brake * max(dist_along - 0.10, 0.0))
                    self.safety_pub.publish(_F64(data=[time.monotonic(), v_safe, float(k), float(d)]))
                    self.get_logger().warn(f'predicted path reaches an obstacle in {dist_along:.2f} m: '
                                           f'speed <= {v_safe:.2f}', throttle_duration_sec=0.5)
                    return v_safe
        return math.inf

    def config_json(self, waypoints_path):
        """mpc_config + node parameters as JSON, so each recorded run knows its settings."""
        cfg = {}
        for name in self.config.__dataclass_fields__:
            value = getattr(self.config, name)
            if isinstance(value, np.ndarray):
                # the cost matrices are diagonal: store just the diagonal
                is_diag = np.count_nonzero(value - np.diag(np.diagonal(value))) == 0
                value = np.diagonal(value).tolist() if is_diag else value.tolist()
            elif isinstance(value, np.generic):
                value = value.item()
            cfg[name] = value
        params = {n: self.get_parameter(n).value
                  for n in self.list_parameters([], 1).names}
        return json.dumps({'mpc_config': cfg, 'parameters': params,
                           'waypoints_path': waypoints_path})

    def publish_stats(self, t_start):
        """[solve_ms, osqp_ms, callback_ms, status, *cost_terms()] (17 numbers);
        status 2=optimal, 1=inaccurate, 0=failed. tools/run_recorder.py stores them all."""
        if not self.publish_debug:
            return
        callback_ms = (time.perf_counter() - t_start) * 1e3
        self.stats_pub.publish(Float32MultiArray(data=[
            self.last_solve_ms, self.last_osqp_ms, callback_ms, float(self.last_status),
            *self.last_costs]))

    def cost_terms(self):
        """The solved QP's objective split into its parts (13 numbers):
          [total,
           Q  x, y, v, yaw   (stage cost, columns 0..TK-1 of the plan)
           Qf x, y, v, yaw   (terminal cost, column TK)
           R  accel, steer   (input size)
           Rd accel, steer]  (input change between plan steps)
        Each part is e_i * (M e)_i summed over the horizon, so the parts add up to
        e^T M e exactly, even if a weight matrix is not diagonal.
        """
        c = self.config
        err = self.xk.value - self.ref_traj_k.value           # (NXK, TK+1)
        u = self.uk.value                                      # (NU, TK)
        du = np.diff(u, axis=1)                                # (NU, TK-1)
        stage = (err[:, :-1] * (c.Qk @ err[:, :-1])).sum(axis=1)
        final = err[:, -1] * (c.Qfk @ err[:, -1])
        # slip model: fold the yaw_rate/beta parts (weight 0 for now) into "yaw", because the
        # recorder expects exactly 13 numbers
        stage = [*stage[:3], sum(stage[3:])]
        final = [*final[:3], sum(final[3:])]
        effort = (u * (c.Rk @ u)).sum(axis=1)
        change = (du * (c.Rdk @ du)).sum(axis=1)
        return [float(self.MPC_prob.value), *stage, *final, *effort, *change]


    def make_marker(self, xs, ys, mtype, rgb, size):
        m = Marker()
        m.header.frame_id = 'map'
        m.header.stamp = self.get_clock().now().to_msg()
        m.type = mtype
        m.action = Marker.ADD
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = size
        m.color.r, m.color.g, m.color.b = rgb
        m.color.a = 1.0
        m.points = [Point(x=float(x), y=float(y), z=0.05) for x, y in zip(xs, ys)]
        return m

    def predict_delay(self, x, y, yaw, v, now):
        """Move the measured state forward by delay_comp seconds (kinematic bicycle, speed
        held constant).

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
        '''
        get state
        build reference trajectory (starting from closest line connecting two waypoints at the speeds dictated by the waypoint csv file)
        linearize dynamics about expected path
        Solve QP
        apply u_0
        '''
        t_start = time.perf_counter()  # for publish_stats (debug only)
        
        # extract pose from ROS msg
        pos = odom_msg.pose.pose.position
        q = odom_msg.pose.pose.orientation

        x = pos.x
        y = pos.y
        v = odom_msg.twist.twist.linear.x
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        # particle-filter pose is the LiDAR, pose_offset metres ahead of the rear axle;
        # the kinematic model (and the sim's odom) is for the rear axle
        x -= self.pose_offset * math.cos(yaw)
        y -= self.pose_offset * math.sin(yaw)

        if self.model == 'slip':
            # twist is in the car's frame: linear.x forward, linear.y sideways (= the slip)
            vx, vy = odom_msg.twist.twist.linear.x, odom_msg.twist.twist.linear.y
            v = math.hypot(vx, vy)                         # slip model uses the full speed
            yaw_rate = odom_msg.twist.twist.angular.z
            beta = math.atan2(vy, vx) if v > 0.1 else 0.0  # undefined when (nearly) stopped

        # where the car will be when this command takes effect (no-op if delay_comp = 0)
        x, y, yaw = self.predict_delay(x, y, yaw, v, time.monotonic())

        vehicle_state =  State(x=x, y=y, v=v, yaw=yaw)# (x, y, v, yaw)

        # Calculate the next reference trajectory for the next T steps
        #       with current vehicle pose.
        #       ref_x, ref_y, ref_yaw, ref_v are columns of self.waypoints
        wp = self.reference_waypoints()
        ref_x, ref_y, ref_yaw, ref_v = wp[:, 0], wp[:, 1], wp[:, 2], wp[:, 3]
        ref_path = self.calc_ref_trajectory(vehicle_state, ref_x, ref_y, ref_yaw, ref_v)
        ref_path[3] = yaw + np.arctan2(np.sin(ref_path[3] - yaw), np.cos(ref_path[3] - yaw))

        x0 = [vehicle_state.x, vehicle_state.y, vehicle_state.v, vehicle_state.yaw]
        if self.model == 'slip':
            x0 += [yaw_rate, beta]
            # reference rows for yaw_rate and beta: zeros (their weights are 0 anyway)
            ref_path = np.vstack([ref_path, np.zeros((2, ref_path.shape[1]))])

        # TODO: solve the MPC control problem
        (
            self.oa,
            self.odelta,
            ox,
            oy,
            oyaw,
            ov,
            state_predict,
        ) = self.linear_mpc_control(ref_path, x0, self.oa, self.odelta)

        # Cost (quadratic): Q, Q_f, R, Rd
        # constraints: linearized dynamics, steering limits, acceleration / velocity limits
        # solver
        # warm start (from last solution)
        # initial state = current /odom pose
        if self.odelta is None or self.oa is None:
            self.get_logger().warn('MPC solve failed, skipping this drive command',
                                throttle_duration_sec=1.0)
            self.publish_stats(t_start)
            return

        # publish drive message.
        steer_output = self.odelta[0]

        # # compute speed output (matching gains to sim gains)
        # if self.oa[0] >= 0: # if speeding up or maintaining speed
        #     kp = 4.755 # a = kp * (v_cmd - v) 
        # else: # else braking
        #     kp = 19.02
        
        # t_end = time.perf_counter()
        # lag = (t_end - t_start) + self.CMD_PERIOD * 0.5
        # speed_after_delay = vehicle_state.v + lag * self.oa[0] # speed after time spent solving (TODO:? MAYBE ADD TIME TO SEND COMMAND ON REAL CAR?)
        # speed_output = speed_after_delay + self.oa[0] / kp # TODO: THIS CONVERSITON WILL LIKELY CHANGE ON REAL CAR (KP)

        # fixed 0.1 s (not DTK): keeps the car's speeding up / braking as tuned with DTK 0.1
        speed_output = vehicle_state.v + self.oa[0] * 0.1
        # obstacle safety check: never steering 0, only a lower speed
        # speed_output = min(speed_output, self.safety_speed(ox, oy, oyaw, ov))

        

        # write drive message
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle = float(steer_output)
        drive_msg.drive.speed = float(speed_output)
        self.drive_pub.publish(drive_msg)
        self.cmd_hist.append((time.monotonic(), float(speed_output), float(steer_output)))

        self.ref_pub.publish(self.make_marker(ref_path[0], ref_path[1], Marker.SPHERE_LIST, (0.1, 0.9, 0.2), 0.12))
        self.pred_pub.publish(self.make_marker(ox, oy, Marker.LINE_STRIP, (1.0, 0.2, 0.1), 0.06))
        self.publish_stats(t_start)


    def mpc_prob_init(self):
        """
        Create MPC quadratic optimization problem using cvxpy, solver: OSQP
        Will be solved every iteration for control.
        More MPC problem information here: https://osqp.org/docs/examples/mpc.html
        More QP example in CVXPY here: https://www.cvxpy.org/examples/basic/quadratic_program.html
        """
        # Initialize and create vectors for the optimization problem
        # Vehicle State Vector
        self.xk = cvxpy.Variable(
            (self.config.NXK, self.config.TK + 1)
        )
        # Control Input vector
        self.uk = cvxpy.Variable(
            (self.config.NU, self.config.TK)
        )
        objective =  0.0 # Objective value of the optimization problem
        constraints = []  # Create constraints array

        # Initialize reference vectors
        self.x0k = cvxpy.Parameter((self.config.NXK,))
        self.x0k.value = np.zeros((self.config.NXK,))

        # Initialize reference trajectory parameter
        self.ref_traj_k = cvxpy.Parameter((self.config.NXK, self.config.TK + 1))
        self.ref_traj_k.value = np.zeros((self.config.NXK, self.config.TK + 1))

        # Initializes block diagonal form of R = [R, R, ..., R] (NU*T, NU*T)
        R_block = block_diag(tuple([self.config.Rk] * self.config.TK))

        # Initializes block diagonal form of Rd = [Rd, ..., Rd] (NU*(T-1), NU*(T-1))
        Rd_block = block_diag(tuple([self.config.Rdk] * (self.config.TK - 1)))

        # Initializes block diagonal form of Q = [Q, Q, ..., Qf] (NX*T, NX*T)
        Q_block = [self.config.Qk] * (self.config.TK)
        Q_block.append(self.config.Qfk)
        Q_block = block_diag(tuple(Q_block))

        # Formulate and create the finite-horizon optimal control problem (objective function)
        # The FTOCP has the horizon of T timesteps

        # --------------------------------------------------------
        # fill in the objectives here, you should be using cvxpy.quad_form() somehwhere

        # Objective part 1: Influence of the control inputs: Inputs u multiplied by the penalty R
        objective += cvxpy.quad_form(cvxpy.vec(self.uk, order='F'), R_block)
        # Objective part 2: Deviation of the vehicle from the reference trajectory weighted by Q, including final Timestep T weighted by Qf
        objective += cvxpy.quad_form(cvxpy.vec(self.xk - self.ref_traj_k, order='F'), Q_block)
        # Objective part 3: Difference from one control input to the next control input weighted by Rd
        objective += cvxpy.quad_form(cvxpy.vec(self.uk[:, 1:] - self.uk[:, :-1], order='F'), Rd_block)
        # --------------------------------------------------------

        # Constraints 1: Calculate the future vehicle behavior/states based on the vehicle dynamics model matrices
        # Evaluate vehicle Dynamics for next T timesteps
        # init path to zeros (only the sizes matter here; real values are set every solve)
        path_predict = np.zeros((self.config.NXK, self.config.TK + 1))
        zeros_u = [0.0] * self.config.TK
        A_block, B_block, C_block = self.model_matrices(path_predict, zeros_u, zeros_u)

        # [AA] Sparse matrix to CVX parameter for proper stuffing
        # Reference: https://github.com/cvxpy/cvxpy/issues/1159#issuecomment-718925710
        m, n = A_block.shape
        self.Annz_k = cvxpy.Parameter(A_block.nnz)
        data = np.ones(self.Annz_k.size)
        rows = A_block.row * n + A_block.col
        cols = np.arange(self.Annz_k.size)
        Indexer = csc_matrix((data, (rows, cols)), shape=(m * n, self.Annz_k.size))

        # Setting sparse matrix data
        self.Annz_k.value = A_block.data

        # Now we use this sparse version instead of the old A_ block matrix
        self.Ak_ = cvxpy.reshape(Indexer @ self.Annz_k, (m, n), order="C")

        # Same as A
        m, n = B_block.shape
        self.Bnnz_k = cvxpy.Parameter(B_block.nnz)
        data = np.ones(self.Bnnz_k.size)
        rows = B_block.row * n + B_block.col
        cols = np.arange(self.Bnnz_k.size)
        Indexer = csc_matrix((data, (rows, cols)), shape=(m * n, self.Bnnz_k.size))
        self.Bk_ = cvxpy.reshape(Indexer @ self.Bnnz_k, (m, n), order="C")
        self.Bnnz_k.value = B_block.data

        # No need for sparse matrices for C as most values are parameters
        self.Ck_ = cvxpy.Parameter(C_block.shape)
        self.Ck_.value = C_block

        # -------------------------------------------------------S------
        # Constraint part 1:
        #       Add dynamics constraints to the optimization problem
        #       This constraint should be based on a few variables:
        #       self.xk, self.Ak_, self.Bk_, self.uk, and self.Ck_
        constraints += [
            cvxpy.vec(self.xk[:, 1:], order='F')
            == self.Ak_ @ cvxpy.vec(self.xk[:, :-1], order='F')
            + self.Bk_ @ cvxpy.vec(self.uk, order='F')
            + self.Ck_
        ] # x_kp1 = A*x_k + B*u_k + C
        
        # Constraint part 2:
        #       Add constraints on steering, change in steering angle
        #       cannot exceed steering angle speed limit. Should be based on:
        #       self.uk, self.config.MAX_DSTEER, self.config.DTK
        constraints += [
            cvxpy.abs(self.uk[1, 1:] - self.uk[1, :-1]) <= self.config.MAX_DSTEER * self.config.DTK
        ]
        # abs(steer_kp1 - steer_k) <= max_steering_diff


        # Constraint part 3:
        #       Add constraints on upper and lower bounds of states and inputs
        #       and initial state constraint, should be based on:
        #       self.xk, self.x0k, self.config.MAX_SPEED, self.config.MIN_SPEED,
        #       self.uk, self.config.MAX_ACCEL, self.config.MAX_STEER
        # steer_min <= steer_k <= steer_max
        # vel_min <= vel_k <= vel_max
        # max for accel
        # x0 = x_measured
        # initial state: the plan starts where the car is now
        constraints += [self.xk[:, 0] == self.x0k]

        # speed limits (row 2 of xk), from step 1 onward
        constraints += [self.xk[2, 1:] <= self.config.MAX_SPEED,
                        self.xk[2, 1:] >= self.config.MIN_SPEED]

        # acceleration limit (row 0 of uk)
        constraints += [cvxpy.abs(self.uk[0, :]) <= self.config.MAX_ACCEL]

        # steering angle limit (row 1 of uk)
        constraints += [cvxpy.abs(self.uk[1, :]) <= self.config.MAX_STEER]

        
        # -------------------------------------------------------------

        # Create the optimization problem in CVXPY and setup the workspace
        # Optimization goal: minimize the objective function
        self.MPC_prob = cvxpy.Problem(cvxpy.Minimize(objective), constraints)

    def calc_ref_trajectory(self, state, cx, cy, cyaw, sp):
        """
        Calculate interpolated reference trajectory for T steps: [x, y, v, yaw]

        Uses proper linear interpolation between waypoints based on velocity
        propagation, rather than snapping to discrete waypoint indices.

        :param state: Current vehicle state
        :param cx: Course X-Position
        :param cy: Course y-Position
        :param cyaw: Course Heading
        :param sp: speed profile
        :return: reference trajectory ref_traj (NXK x TK+1)
        """
        return calc_interpolated_ref_trajectory(
            state.x, state.y, state.v,
            cx, cy, sp, cyaw,
            self.config.DTK,
            self.config.TK,
            self.config.MAX_ACCEL
        )

    def predict_motion(self, x0, oa, od, xref):
        if self.model == 'slip':
            # nominal trajectory from the nonlinear slip model (Euler, N_SUB sub-steps per DTK)
            c = self.config
            return slip_model.rollout(x0, oa, od, c.DTK, c.N_SUB, self.slip_params,
                                      c.MIN_SPEED, c.MAX_SPEED, c.MAX_STEER)

        path_predict = xref * 0.0
        for i, _ in enumerate(x0):
            path_predict[i, 0] = x0[i]

        state = State(x=x0[0], y=x0[1], yaw=x0[3], v=x0[2])
        for (ai, di, i) in zip(oa, od, range(1, self.config.TK + 1)):
            state = self.update_state(state, ai, di)
            path_predict[0, i] = state.x
            path_predict[1, i] = state.y
            path_predict[2, i] = state.v
            path_predict[3, i] = state.yaw

        return path_predict

    def update_state(self, state, a, delta):

        # input check
        if delta >= self.config.MAX_STEER:
            delta = self.config.MAX_STEER
        elif delta <= -self.config.MAX_STEER:
            delta = -self.config.MAX_STEER

        state.x = state.x + state.v * math.cos(state.yaw) * self.config.DTK
        state.y = state.y + state.v * math.sin(state.yaw) * self.config.DTK
        state.yaw = (
            state.yaw + (state.v / self.config.WB) * math.tan(delta) * self.config.DTK
        )
        state.v = state.v + a * self.config.DTK

        if state.v > self.config.MAX_SPEED:
            state.v = self.config.MAX_SPEED
        elif state.v < self.config.MIN_SPEED:
            state.v = self.config.MIN_SPEED

        return state

    def get_model_matrix(self, v, phi, delta):
        """
        Calc linear and discrete time dynamic model-> Explicit discrete time-invariant
        Linear System: Xdot = Ax +Bu + C
        State vector: x=[x, y, v, yaw]
        :param v: speed
        :param phi: heading angle of the vehicle
        :param delta: steering angle: delta_bar
        :return: A, B, C
        """

        # State (or system) matrix A, 4x4
        A = np.zeros((self.config.NXK, self.config.NXK))
        A[0, 0] = 1.0
        A[1, 1] = 1.0
        A[2, 2] = 1.0
        A[3, 3] = 1.0
        A[0, 2] = self.config.DTK * math.cos(phi)
        A[0, 3] = -self.config.DTK * v * math.sin(phi)
        A[1, 2] = self.config.DTK * math.sin(phi)
        A[1, 3] = self.config.DTK * v * math.cos(phi)
        A[3, 2] = self.config.DTK * math.tan(delta) / self.config.WB

        # Input Matrix B; 4x2
        B = np.zeros((self.config.NXK, self.config.NU))
        B[2, 0] = self.config.DTK
        B[3, 1] = self.config.DTK * v / (self.config.WB * math.cos(delta) ** 2)

        C = np.zeros(self.config.NXK)
        C[0] = self.config.DTK * v * math.sin(phi) * phi
        C[1] = -self.config.DTK * v * math.cos(phi) * phi
        C[3] = -self.config.DTK * v * delta / (self.config.WB * math.cos(delta) ** 2)

        return A, B, C

    def model_matrices(self, path_predict, oa, od):
        """Stacked linearized dynamics along the nominal trajectory, for both models.

        kinematic: the original get_model_matrix (linearized at steering 0, Euler).
        slip: slip_model.linearize_discrete at the full nominal state and the nominal
              inputs (oa[t], od[t]) of each step (finite differences + matrix exponential).
        Both return dense numpy blocks, so A_block.nnz never changes between solves
        (needed by the fixed-size Annz_k / Bnnz_k parameters).
        """
        A_block, B_block, C_block = [], [], []
        for t in range(self.config.TK):
            if self.model == 'slip':
                A, B, C = slip_model.linearize_discrete(
                    path_predict[:, t], (oa[t], od[t]), self.config.DTK, self.slip_params)
            else:
                A, B, C = self.get_model_matrix(path_predict[2, t], path_predict[3, t], 0.0)
            A_block.append(A)
            B_block.append(B)
            C_block.extend(C)
        return block_diag(tuple(A_block)), block_diag(tuple(B_block)), np.array(C_block)

    def mpc_prob_solve(self, ref_traj, path_predict, x0, oa, od):
        self.x0k.value = x0

        A_block, B_block, C_block = self.model_matrices(path_predict, oa, od)

        self.Annz_k.value = A_block.data
        self.Bnnz_k.value = B_block.data
        self.Ck_.value = C_block

        self.ref_traj_k.value = ref_traj

        # Solve the optimization problem in CVXPY
        # Solver selections: cvxpy.OSQP; cvxpy.GUROBI
        t_solve = time.perf_counter()
        self.MPC_prob.solve(solver=cvxpy.OSQP, verbose=False, warm_start=False) # if error --> warm_start=False ???
        # wall time of solve() = cvxpy setup + OSQP; solver_stats.solve_time = OSQP alone
        self.last_solve_ms = (time.perf_counter() - t_solve) * 1e3
        osqp_s = getattr(self.MPC_prob.solver_stats, 'solve_time', None)
        self.last_osqp_ms = osqp_s * 1e3 if osqp_s is not None else float('nan')
        self.last_status = {cvxpy.OPTIMAL: 2, cvxpy.OPTIMAL_INACCURATE: 1}.get(
            self.MPC_prob.status, 0)

        if (
            self.MPC_prob.status == cvxpy.OPTIMAL
            or self.MPC_prob.status == cvxpy.OPTIMAL_INACCURATE
        ):
            ox = np.array(self.xk.value[0, :]).flatten()
            oy = np.array(self.xk.value[1, :]).flatten()
            ov = np.array(self.xk.value[2, :]).flatten()
            oyaw = np.array(self.xk.value[3, :]).flatten()
            oa = np.array(self.uk.value[0, :]).flatten()
            odelta = np.array(self.uk.value[1, :]).flatten()
            if getattr(self, 'publish_debug', False):
                self.last_costs = self.cost_terms()

        else:
            self.last_costs = [float('nan')] * 13
            print("Error: Cannot solve mpc..")
            oa, odelta, ox, oy, oyaw, ov = None, None, None, None, None, None

        return oa, odelta, ox, oy, oyaw, ov

    def linear_mpc_control(self, ref_path, x0, oa, od):
        """
        MPC contorl with updating operational point iteraitvely
        :param ref_path: reference trajectory in T steps
        :param x0: initial state vector
        :param oa: acceleration of T steps of last time
        :param od: delta of T steps of last time
        """

        if oa is None or od is None:
            oa = [0.0] * self.config.TK
            od = [0.0] * self.config.TK

        # Call the Motion Prediction function: Predict the vehicle motion for x-steps
        path_predict = self.predict_motion(x0, oa, od, ref_path)
        poa, pod = oa[:], od[:]

        # Run the MPC optimization: Create and solve the optimization problem
        mpc_a, mpc_delta, mpc_x, mpc_y, mpc_yaw, mpc_v = self.mpc_prob_solve(
            ref_path, path_predict, x0, poa, pod
        )

        return mpc_a, mpc_delta, mpc_x, mpc_y, mpc_yaw, mpc_v, path_predict

def main(args=None):
    rclpy.init(args=args)
    print("MPC Initialized")
    mpc_node = MPC()
    try:
        rclpy.spin(mpc_node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass  # Ctrl-C: stop quietly
    finally:
        mpc_node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
