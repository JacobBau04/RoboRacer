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
from infer_env import InferEnv, load_wall_distance_field
from rclpy.qos import qos_profile_sensor_data

class WallEnv(InferEnv):
    def __init__(self, map_yaml):
        sdf, origin, resolution = load_wall_distance_field(map_yaml)

        self.wall_sdf = jnp.asarray(sdf, dtype=jnp.float32)
        self.wall_origin = jnp.asarray(origin, dtype=jnp.float32)
        self.wall_resolution = float(resolution)

@dataclass(frozen=True)
class MPPIConfig:
    # frozen so jax.jit can take it as a static argument: use tuples, not arrays
    NX: int = 4  # length of state vector: x = [p_x, p_y, yaw, v]
    NU: int = 2  # length of input vector: u = [a, delta]
    T: int = 1024  # number of sampled control sequences
    H: int = 20  # horizon length
    DT: float = 0.05  # time step [s]
    INTEGRATOR: str = "euler"  # "euler" or "rk4" (bonus)

    # ---------------------------------------------------
    # TODO: you may need to tune the following parameters
    LAMBDA: float = 0.01  # temperature
    U_STD: tuple = (1.0, 0.25)  # std of the sampled inputs [a, delta]
    WB: float = 0.33  # Wheelbase [m]
    MIN_STEER: float = -0.4189  # minimum steering angle [rad]
    MAX_STEER: float = 0.4189  # maximum steering angle [rad]
    MIN_ACCEL: float = -3.0  # minimum acceleration [m/ss]
    MAX_ACCEL: float = 3.0  # maximum acceleration [m/ss]
    MIN_SPEED: float = 0.0  # minimum speed [m/s]
    MAX_SPEED: float = 3.0  # maximum speed [m/s]
    TRACK_DEVIATION_PENALTY = 0.5
    HEADING_PENALTY = 1.0

    WALL_AVOIDANCE_PENALTY = 500.0
    OPPONENT_AVOIDANCE_PENALTY = 150.0

    ACCELLERATION_PENALTY = 0.5
    STEERING_ANGLE_PENALTY = 0.25
    VELOCITY_PENALTY = 0.1

    CAR_RADIUS = 0.20
    WALL_BUBBLE = 0.30
    OPPONENT_BUBBLE = 0.40
    PROGRESS_PENALTY = 50.0

def dynamics(x, u, cfg):
    """
    Kinematic bicycle model f(x, u) of a single state (README Section II.a)
    Use jnp instead of np or math so JAX can trace it.
    :param x: state [p_x, p_y, yaw, v], shape (NX,)
    :param u: input [a, delta],def dynamics(x, u, cfg):
    """
    wheelbase = cfg.WB

    x_next = x[3] * jnp.cos(x[2])
    y_next = x[3] * jnp.sin(x[2])
    theta_next = (x[3]/wheelbase) * jnp.tan(u[1])
    v_next = u[0]
    
    return jnp.array([x_next,y_next,theta_next,v_next])

def step_euler(x, u, cfg):
    """
    Forward Euler step: x_{k+1} = x_k + f(x_k, u_k) * DT (README Section II.b.i)
    :return: next state, shape (NX,)
    """

    step = x + dynamics(x,u,cfg) * cfg.DT

    velocity = jnp.clip(step[3], cfg.MIN_SPEED, cfg.MAX_SPEED)

    step = step.at[3].set(velocity)

    return step


def step_rk4(x, u, cfg):
    """
    Bonus: Runge-Kutta 4 step (README Section II.b.ii)
    :return: next state, shape (NX,)
    """
    # TODO: fill in the RK4 step
    pass


def step(x, u, cfg):
    """
    Discretized dynamics F(x, u): the state after DT, holding u constant
    """
    if cfg.INTEGRATOR == "rk4":
        return step_rk4(x, u, cfg)
    return step_euler(x, u, cfg)


def simulate(x0, U, cfg):
    """
    Simulate one control sequence from x0
    :param x0: initial state, shape (NX,)
    :param U: control sequence u_0 ... u_{H-1}, shape (H, NU)
    :return: states x_0 ... x_H, shape (H+1, NX), starting with x0
    """
    def f(x, u):
        x_next = step(x, u, cfg)
        return x_next, x_next

    _, trajectories = jax.lax.scan(f, x0, U)  
    return jnp.vstack((x0, trajectories))


def rollout(x0, U, cfg):
    """
    Simulate all T sampled control sequences from x0, in parallel
    :param x0: initial state, shape (NX,)
    :param U: sampled control sequences, shape (T, H, NU)
    :return: states, shape (T, H+1, NX)
    """

    rollout_batched = jax.vmap(simulate,in_axes=(None,0,None))
    rollout_trajectories = rollout_batched(x0,U,cfg)


    return rollout_trajectories


def cost(X, U, ref, cfg,opponent_xy,wall_env,debug=False):
    """
    Cost of each step of one trajectory (README Section III.a): c[t] is the cost
    of applying u_t and reaching x_{t+1}
        c[t] = (x_{t+1} - ref_{t+1})^T Q (x_{t+1} - ref_{t+1})   (Qf for the last step)
             + u_t^T R u_t
             + (u_t - u_{t-1})^T Rd (u_t - u_{t-1})              (from t = 1)
    :param X: states x_0 ... x_H, shape (H+1, NX)
    :param U: inputs u_0 ... u_{H-1}, shape (H, NU)
    :param ref: reference states, shape (H+1, NX)
    :return: c, shape (H,)
    """

    
    def track_deviation_penalty(X,ref):
        """
        Penalty for raceline deviation
        """
        XY_coords = X[1:,:2]
        ref_coords = ref[1:,:2]
        deltas = jnp.linalg.norm(XY_coords-ref_coords,axis=1)
        return deltas **2

    def heading_error(X,ref):
        """
        Keep heading relatively close to the waypoint heading
        """
        heading_X = X[1:, 2]
        heading_ref = ref[1:, 2]
        heading_errors = jnp.arctan2(
            jnp.sin(heading_ref-heading_X),
            jnp.cos(heading_ref-heading_X)
            )
        return heading_errors **2
      
    def progress_around_track(X,ref):
        """
        Measure Arclength progress on the raceline
        """

        car_position_initial = X[0,:2]
        car_position_final = X[-1,:2]

        segment_starts = ref[:-1, :2]
        segment_ends = ref[1:, :2]

        segment_vectors = segment_ends - segment_starts

        normed_delta = jnp.maximum(jnp.linalg.norm(segment_vectors, axis=1), 1e-6)


        n_hat= segment_vectors/normed_delta[:,None]
        t_initial = jnp.sum((car_position_initial - segment_starts) * n_hat, axis=1)
        a_initial = jnp.clip(t_initial / normed_delta, 0.0, 1.0)
        projected_initial = segment_starts + a_initial[:, None] * segment_vectors

        t_final = jnp.sum((car_position_final - segment_starts) * n_hat,axis=1)
        a_final = jnp.clip(t_final / normed_delta, 0.0, 1.0)
        projected_final = segment_starts + a_final[:, None] * segment_vectors

        distances_initial = jnp.linalg.norm(projected_initial - car_position_initial,axis=1)
        distances_final = jnp.linalg.norm(projected_final - car_position_final,axis=1)


        closest_segment_initial = jnp.argmin(distances_initial)
        closest_segment_final = jnp.argmin(distances_final)


        # SELECT THE CORRECT PROJECTIONS

        projected_initial = projected_initial[closest_segment_initial]
        projected_final = projected_final[closest_segment_final]

        a_initial = a_initial[closest_segment_initial]
        a_final = a_final[closest_segment_final]


        segment_lengths = jnp.linalg.norm(segment_vectors, axis=1)

        cumulative_lengths = jnp.concatenate((
            jnp.zeros(1),
            jnp.cumsum(segment_lengths)
        ))

        s_initial = (
            cumulative_lengths[closest_segment_initial]
            + a_initial * segment_lengths[closest_segment_initial]
        )

        s_final = (
            cumulative_lengths[closest_segment_final]
            + a_final * segment_lengths[closest_segment_final]
        )

        arclength = s_final - s_initial

        return arclength
    
    """OLD OBSTACLE AVOIDANCE CODE BASED ON SCAN CALLBACK
    def obstacle__avoidance(X,wall_list):
            
            Penalty for avoiding obstacles. Do not use for race 3
            
        
            if len(obstacle_list) == 0:
                return jnp.zeros(cfg.H)
        
            XY_coords = X[1:,:2]
            deltas = XY_coords[:, None, :] - obstacle_list[None, :, :]
            distances = jnp.linalg.norm(deltas, axis=2)
            min_distance_from_obs = jnp.min(distances, axis=1)
            clearance = min_distance_from_obs - cfg.CAR_RADIUS        
        
            collision_cost = jnp.where(
            clearance <= 0.0,
            1000.0,
            0.0)
        
            return collision_cost
    """
    
    def wall_avoidance(X,wall_env,cfg):
            """
            Penalty for avoiding walls
            """
            wall_distances = wall_env.sample_wall_distance(X[1:, :2])

            clearance_violation = jnp.maximum(
                cfg.CAR_RADIUS + cfg.WALL_BUBBLE - wall_distances,
                0.0
                    )
            
            return clearance_violation ** 2

    def opponent_avoidance(X, opponent_xy, cfg):
        """
        Lazy Implementation. Does not use Gaussians - Needs Updates
        """

        ego_xy = X[1:, :2]

        distances = jnp.linalg.norm(
            ego_xy - opponent_xy,
            axis=1
        )

        clearance_violation = jnp.maximum(
            cfg.OPPONENT_BUBBLE - distances,
            0.0
        )

        return clearance_violation ** 2

    def accelleration_delta(U):
        """
        Penalty for jerky accelleration
        """
        accelleration= U[:,0]
        accelleration_second = accelleration[1:]
        accelleration_first = accelleration[:-1]
        deltas = accelleration_second - accelleration_first
        return jnp.concatenate((jnp.zeros(1), deltas**2))
        
    def steering_angle_delta(U):
        """
        Penalty for fast steering angle changes
        """
        steering_angle = U[:,-1]
        steering_angle_second = steering_angle[1:]
        steering_angle_first = steering_angle[:-1]
        deltas = steering_angle_second - steering_angle_first
        return jnp.concatenate((jnp.zeros(1), deltas**2))

    def velocity_error(X, ref):
        """
        Penalty velocity delta
        """
        velocity_X = X[1:, 3]
        velocity_ref = ref[1:, 3]
        return (velocity_X - velocity_ref) **2

    track_cost = track_deviation_penalty(X, ref) * cfg.TRACK_DEVIATION_PENALTY

    heading_cost = heading_error(X, ref) * cfg.HEADING_PENALTY

    opponent_cost = opponent_avoidance(X, opponent_xy,cfg) * cfg.OPPONENT_AVOIDANCE_PENALTY

    wall_cost = wall_avoidance(X,wall_env,cfg) * cfg.WALL_AVOIDANCE_PENALTY

    accel_cost = accelleration_delta(U) * cfg.ACCELLERATION_PENALTY

    steering_cost = steering_angle_delta(U) * cfg.STEERING_ANGLE_PENALTY

    velocity_cost = velocity_error(X, ref) * cfg.VELOCITY_PENALTY

    progress_cost = -progress_around_track(X, ref) * cfg.PROGRESS_PENALTY / cfg.H

    cost_array = (
        track_cost
        + heading_cost
        + opponent_cost
        + accel_cost
        + steering_cost
        + velocity_cost
        + wall_cost
        + progress_cost
    )

    if debug:
        print(
            f"TRACK: {float(jnp.sum(track_cost)):.2f} | "
            f"HEADING: {float(jnp.sum(heading_cost)):.2f} | "
            f"Opponent: {float(jnp.sum(opponent_cost)):.2f} | "
            f"Wall: {float(jnp.sum(wall_cost)):.2f} | "
            f"ACCEL: {float(jnp.sum(accel_cost)):.2f} | "
            f"STEERING: {float(jnp.sum(steering_cost)):.2f} | "
            f"VELOCITY: {float(jnp.sum(velocity_cost)):.2f} | "
            f"PROGRESS: {float(jnp.sum(progress_cost)):.2f} | "
            f"TOTAL: {float(jnp.sum(cost_array)):.2f}"
        )

    return cost_array


@partial(jax.jit, static_argnames=("cfg", "wall_env"))
def mppi_step(key, U, x0, ref, cfg, opponent_xy, wall_env):
    """
    One MPPI update (README Section IV)
    :param key: JAX random key, a new one at every call
    :param U: mean control sequence, already warm started, shape (H, NU)
    :param x0: current state, shape (NX,)
    :param ref: reference states, shape (H+1, NX)
    :param cfg: MPPIConfig
    :return: the new mean control sequence, shape (H, NU),
             the sampled trajectories (for visualization), shape (T, H+1, NX),
             and the weights of the samples at each step, shape (T, H)
    For the covariance update (bonus), you may change the inputs and outputs.
    """
    # --------------------------------------------------------
    # TODO 1. Sample: T control sequences around U with noise of std cfg.U_STD
    #         (jax.random.normal(key, shape)), clipped to the input bounds.
    #         Bonus: sample from a truncated distribution instead of clipping
    
    noise_samples= jax.random.normal(key,shape=(cfg.T,cfg.H,cfg.NU)) * jnp.array(cfg.U_STD)

    """ Set samples"""

    U_min = jnp.array([cfg.MIN_ACCEL,-cfg.MAX_STEER])
    U_max = jnp.array([cfg.MAX_ACCEL,cfg.MAX_STEER])

    sampled_controls = jnp.clip(U[None, :, :] + noise_samples,jnp.array(U_min),jnp.array(U_max))
    # TODO 2. Simulate: roll out all T sequences from x0

    all_trajectories = rollout(x0, sampled_controls, cfg)

    # TODO 3. Score: the cost of each step of each sample, c of shape (T, H)
    #         (vectorize cost() with jax.vmap), then the cost-to-go
    #         S[k, t] = c[k, t] + ... + c[k, H-1] (cumsum of the reversed steps)

    cost_vectorized= jax.vmap(cost,in_axes=(0,0,None,None,None,None))
    S_array = cost_vectorized(all_trajectories,sampled_controls,ref,cfg,opponent_xy,wall_env)
    cost_to_go = jnp.flip(jnp.cumsum(jnp.flip(S_array, axis=1), axis=1),axis=1)
    
    # TODO 4. Weight: for each step t, normalize S[:, t] to [0, 1] over the
    #         samples, (S - min) / (max - min + 1e-6), then take its softmax
    #         over the samples, w[:, t] = softmax(-S[:, t] / cfg.LAMBDA)

    normalized = (cost_to_go-jnp.min(cost_to_go,axis=0))/ (jnp.max(cost_to_go,axis=0)-jnp.min(cost_to_go,axis=0)+1e-6)
    weights = jax.nn.softmax(-normalized / cfg.LAMBDA, axis=0)
    

    # TODO 5. Update: U_new[t] = sum over the samples k of w[k, t] * u_k[t]
    #         Bonus: also update the covariance
    # --------------------------------------------------------
    update = jnp.sum(weights[:, :, None] * sampled_controls, axis=0)

    return update


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

        # TODO: create ROS subscribers and publishers
        #       use MPPI as a tracker (similar to pure pursuit)

        self.pose_sub_ = self.create_subscription(Odometry, '/ego_racecar/odom', self.odom_callback, 1)
        self.drive_pub_ = self.create_publisher(AckermannDriveStamped, '/drive', 1)
        self.path_pub_ = self.create_publisher(Path, '/mppi/selected_path', 10)
        self.opponent_sub_ = self.create_subscription(Path,'/opponent/predicted_path',self.opponent_callback,qos_profile_sensor_data)
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

        self.config = MPPIConfig()
        self.U = jnp.zeros((self.config.H, self.config.NU))  # mean control sequence
        self.key = jax.random.PRNGKey(0)
        self.opponent_xy = jnp.full((self.config.H, 2),1000.0)
        self.opponent_path_time = None
        self.opponent_timeout = 0.5

        mppi_step(
            self.key,
            self.U,
            jnp.zeros(self.config.NX),
            jnp.zeros((self.config.H + 1, self.config.NX)),
            self.config,
            self.opponent_xy,
            self.wall_env
        )

        
    def opponent_callback(self, msg):
        if len(msg.poses) == 0:
            self.opponent_path_time = None
            return

        points = np.array([
            [pose.pose.position.x, pose.pose.position.y]
            for pose in msg.poses
        ], dtype=np.float32).reshape(-1, 2)

        if len(points) < 2:
            self.opponent_path_time = None
            return

        if not np.isfinite(points).all():
            self.opponent_path_time = None
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

        self.opponent_xy = jnp.asarray(predicted)

        self.opponent_path_time = (
            self.get_clock().now().nanoseconds * 1e-9
        )  

    def publish_drive(self, speed, steering):
        msg = AckermannDriveStamped()
        msg.drive.speed = float(speed)
        msg.drive.steering_angle = float(steering)
        self.drive_pub_.publish(msg)

    def odom_callback(self, odom_msg):
        # TODO: extract the state x0 = [p_x, p_y, yaw, v] from the ROS msg

        self.car_x = odom_msg.pose.pose.position.x
        self.car_y = odom_msg.pose.pose.position.y
                
    

        quaternion = np.array([odom_msg.pose.pose.orientation.x, 
                                            odom_msg.pose.pose.orientation.y, 
                                            odom_msg.pose.pose.orientation.z, 
                                            odom_msg.pose.pose.orientation.w])
        self.yaw = np.arctan2(2 * (quaternion[3] * quaternion[2] + quaternion[0] * quaternion[1]),
                                    1 - 2 * (quaternion[1]**2 + quaternion[2]**2))
                
        vehicle_state = jnp.array([self.car_x, self.car_y, self.yaw, odom_msg.twist.twist.linear.x])

        if not np.isfinite(np.asarray(vehicle_state)).all():
            self.publish_drive(0.0, 0.0)
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

        opponent_active = (
            self.opponent_path_time is not None
            and 0.0 <= now - self.opponent_path_time <= self.opponent_timeout
        )

        opponent_xy = (
            self.opponent_xy
            if opponent_active
            else jnp.full((self.config.H, 2), 1000.0)
        )
        # TODO: run MPPI, keep the new mean control sequence in self.U

        self.U = mppi_step(subkey,self.U,vehicle_state,ref_path,self.config,opponent_xy,self.wall_env)
        selected_trajectory = simulate(
                vehicle_state,
                self.U,
                self.config
            )

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

        selected_cost = cost(
                selected_trajectory,
                self.U,
                ref_path,
                self.config,
                opponent_xy,
                self.wall_env,
                debug=True)
      
    

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
3. Implement Cedric's v1 opponent passing code with just bubble cost -DONE
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
