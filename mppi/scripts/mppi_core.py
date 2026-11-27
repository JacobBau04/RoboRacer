# MPPI background: Williams et al. (2017), doi:10.2514/1.G001921.
"""ROS-independent bicycle model and original MPPI objective/update.

Kept separate so offline tests exercise the same dynamics used by the ROS node.
The constrained LEFT/RIGHT/FOLLOW planner is in physics_overtaking.py.
"""
from dataclasses import dataclass
from functools import partial
import jax
import jax.numpy as jnp

@dataclass(frozen=True)
class MPPIConfig:
    # frozen so jax.jit can take it as a static argument: use tuples, not arrays
    NX: int = 4  # length of state vector: x = [p_x, p_y, yaw, v]
    NU: int = 2  # length of input vector: u = [a, delta]
    T: int = 1024 # number of sampled control sequences
    H: int = 20  # horizon length
    DT: float = 0.1  # time step [s]
    INTEGRATOR: str = "rk4"  # "euler" or "rk4" (bonus)

    # ---------------------------------------------------
    # TODO: you may need to tune the following parameters
    LAMBDA: float = 0.05  # temperature
    U_STD: tuple = (1.0, 0.25)  # std of the sampled inputs [a, delta] (car: steer was 0.5, wider than the
    # +-0.38 limit, so ~45% of samples sat at full lock; on the narrow levine_2nd hallways it lost the line
    # after 9 m in a closed-loop sim, with 0.15 it tracked 10.8 cm mean, 2026-11-24)
    WB: float = 0.33  # Wheelbase [m]
    MIN_STEER: float = -0.38  # minimum steering angle [rad] (car: was -0.4189; the MPC raced with 0.38)
    MAX_STEER: float = 0.38  # maximum steering angle [rad] (car: was 0.4189)
    MIN_ACCEL: float = -3.0  # minimum acceleration [m/ss]
    MAX_ACCEL: float = 3.0  # maximum acceleration [m/ss]
    MIN_SPEED: float = 0.0  # minimum speed [m/s]
    MAX_SPEED: float = 2.5  # maximum speed [m/s] (car first run: was 2.5; levine_race asks 2.1-2.3)



    # Tracking
    TRACK_DEVIATION_PENALTY = 0.25
    HEADING_PENALTY = 0.2
    VELOCITY_PENALTY = 1

    # Collision avoidance
    WALL_AVOIDANCE_PENALTY = 600.0

    OPPONENT_AVOIDANCE_PENALTY = 450.0

    # Control smoothness
    ACCELLERATION_PENALTY = 0.1
    STEERING_ANGLE_PENALTY = 0.1

    # Progress
    PROGRESS_PENALTY = 0

    # Clearance
    CAR_RADIUS = 0.16
    WALL_BUBBLE = 0.2
    SOFT_WALL_BUBBLE = 0.6
    SOFT_WALL_AVOIDANCE_PENALTY = 0.05
    OPPONENT_BUBBLE = 0.3

def dynamics(x, u, cfg):
    """
    Kinematic bicycle model f(x, u) of a single state (README Section II.a)
    Use jnp instead of np or math so JAX can trace it.
    :param x: state [p_x, p_y, yaw, v], shape (NX,)
    :param u: input [a, delta],def dynamics(x, u, cfg):
    """
    wheelbase = cfg.WB

    # RK4 evaluates intermediate states too. Clipping only the final speed
    # made braking at rest predict backwards motion within every time step.
    speed = jnp.clip(x[3], cfg.MIN_SPEED, cfg.MAX_SPEED)
    x_next = speed * jnp.cos(x[2])
    y_next = speed * jnp.sin(x[2])
    theta_next = (speed/wheelbase) * jnp.tan(u[1])
    outward = ((x[3] <= cfg.MIN_SPEED) & (u[0] < 0)) | ((x[3] >= cfg.MAX_SPEED) & (u[0] > 0))
    v_next = jnp.where(outward, 0.0, u[0])

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
    def step_rk4(x, u, cfg):
    """
    dt = cfg.DT

    k1 = dynamics(x, u, cfg)
    k2 = dynamics(x + 0.5 * dt * k1, u, cfg)
    k3 = dynamics(x + 0.5 * dt * k2, u, cfg)
    k4 = dynamics(x + dt * k3, u, cfg)

    x_next = x + (dt / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)

    # Enforce velocity limits
    velocity = jnp.clip(x_next[3], cfg.MIN_SPEED, cfg.MAX_SPEED)
    x_next = x_next.at[3].set(velocity)

    return x_next


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


# car: compiled once. Called outside jit, simulate's lax.scan was re-traced and re-compiled on every
# pose (~250 ms of a ~300 ms callback on the Jetson CPU, measured 2026-11-24). Same result.
simulate_jit = jax.jit(simulate, static_argnames=("cfg",))


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


def cost(X, U, ref, cfg, opponent_xy_0,opponent_xy_1,opponent_xy_2, wall_env,
         debug=False, return_components=False):
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


    def opponent_avoidance(X, opponent_xy_0,opponent_xy_1,opponent_xy_2,cfg):
        """
        Lazy Implementation. Does not use Gaussians - Needs Updates
        """

        ego_xy = X[1:, :2]

        distances_0 = jnp.linalg.norm(
            ego_xy - opponent_xy_0,
            axis=1
        )

        distances_1 = jnp.linalg.norm(
            ego_xy - opponent_xy_1,
            axis=1
        )

        distances_2 = jnp.linalg.norm(
                    ego_xy - opponent_xy_2,
                    axis=1
                )


        clearance_violation_0 = jnp.maximum(
            cfg.CAR_RADIUS + cfg.OPPONENT_BUBBLE - distances_0,
            0.0
        )
        clearance_violation_1 = jnp.maximum(
                    cfg.CAR_RADIUS + cfg.OPPONENT_BUBBLE - distances_1,
                    0.0)
        clearance_violation_2 = jnp.maximum(
                            cfg.CAR_RADIUS + cfg.OPPONENT_BUBBLE - distances_2,
                            0.0)


        return (clearance_violation_0 ** 2) + (clearance_violation_1 **2) + (clearance_violation_2 **2)

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

    opponent_cost = opponent_avoidance(X, opponent_xy_0,opponent_xy_1,opponent_xy_2,cfg) * cfg.OPPONENT_AVOIDANCE_PENALTY

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

    if return_components:
        return {
            "Track": jnp.sum(track_cost),
            "Heading": jnp.sum(heading_cost),
            "Opponent": jnp.sum(opponent_cost),
            "Wall": jnp.sum(wall_cost),
            "Acceleration": jnp.sum(accel_cost),
            "Steering": jnp.sum(steering_cost),
            "Velocity": jnp.sum(velocity_cost),
            "Progress": jnp.sum(progress_cost),
            "TOTAL": jnp.sum(cost_array)
        }

    return cost_array


@partial(jax.jit, static_argnames=("cfg", "wall_env"))
def mppi_step(key, U, x0, ref, cfg, opponent_xy_0,opponent_xy_1,opponent_xy_2, wall_env):
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

    cost_vectorized = jax.vmap(cost,in_axes=(0, 0, None, None, None, None, None, None))
    S_array = cost_vectorized(all_trajectories,sampled_controls,ref,cfg,opponent_xy_0,opponent_xy_1,opponent_xy_2,wall_env)
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

    return update, all_trajectories, cost_to_go[:, 0], sampled_controls
