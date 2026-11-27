# Related: Patrick & Bakolas (2024), arxiv.org/abs/2403.18066.
# Uses explicit behavior modes rather than their rollout clustering.
"""Physics-only LEFT / RIGHT / FOLLOW planner. No ROS imports or learned policy.

The state is [rear_axle_x, rear_axle_y, yaw, speed]. Opponent points are
vehicle centres in the same map frame, on a common future time grid.
Separate MPPI updates prevent averaging a left pass with a right pass.
"""
from dataclasses import dataclass, replace
from functools import partial
import math

import jax
import jax.numpy as jnp
import numpy as np

from mppi_core import MPPIConfig, cost, rollout, simulate, simulate_jit


@dataclass(frozen=True)
class PlannerConfig:
    # Configurable starting assumptions, exposed as overtaking.* ROS parameters.
    samples_per_mode: int = 256
    car_length: float = 0.51
    car_width: float = 0.28
    rear_to_center: float = 0.13
    clearance: float = 0.08
    uncertainty: float = 0.03
    uncertainty_growth: float = 0.03  # metres / second of prediction
    max_lateral_accel: float = 3.0
    max_steering_rate: float = 2.5    # rad/s
    standstill_gap: float = 0.9      # rear axle to opponent centre
    time_headway: float = 0.6
    pass_offset: float = 0.85       # minimum separation; increased for opponent drift
    change_distance: float = 2.5    # smooth lateral transition along the track
    trigger_distance: float = 5.0
    complete_gap: float = 0.9
    switch_penalty: float = 0.20    # metres of progress needed to justify switching
    stale_timeout: float = 0.5
    opponent_dt: float = 0.1
    opponent_braking: float = 3.0          # positive deceleration magnitude [m/s²]
    opponent_lateral_displacement: float = 0.65  # maximum scenario shift [m]
    opponent_lateral_speed: float = 0.5    # [m/s]
    opponent_lateral_accel: float = 1.0    # [m/s²], relative to track
    wall_substeps: int = 8

    def __post_init__(self):
        positive = ('samples_per_mode', 'car_length', 'car_width', 'max_lateral_accel',
                    'max_steering_rate', 'standstill_gap', 'time_headway', 'pass_offset',
                    'change_distance', 'trigger_distance', 'complete_gap', 'stale_timeout',
                    'opponent_dt', 'opponent_braking', 'opponent_lateral_displacement',
                    'opponent_lateral_speed', 'opponent_lateral_accel', 'wall_substeps')
        if any(not math.isfinite(getattr(self, k)) or getattr(self, k) <= 0 for k in positive):
            raise ValueError('Planner dimensions, limits, timing and sample counts must be positive')
        if min(self.clearance, self.uncertainty, self.uncertainty_growth, self.switch_penalty) < 0:
            raise ValueError('Margins and switching penalty must be nonnegative')

    @property
    def radius(self):
        return math.hypot(self.car_length / 2, self.car_width / 2)


@dataclass
class OpponentPrediction:
    points: np.ndarray             # includes position at the message stamp
    stamp: float
    dt: float = 0.1


def align_prediction(prediction, now, horizon, dt, start_delay=0.0):
    """Resample by source time; allow at most 0.35 s of constant-velocity extension.

    The predictor and controller have equal horizons, so a small bounded tail
    extension is needed for transport/actuation latency. Short paths fail closed.
    """
    points = np.asarray(prediction.points, dtype=float)
    if (points.ndim != 2 or points.shape[1] != 2 or len(points) < 2
            or not np.isfinite(points).all() or prediction.dt <= 0):
        raise ValueError('Expected finite opponent points [N, 2] with positive dt')
    age = now - prediction.stamp
    query = age + start_delay + np.arange(horizon + 1) * dt
    source = np.arange(len(points)) * prediction.dt
    aligned = np.column_stack([np.interp(query, source, points[:, axis]) for axis in (0, 1)])
    velocity = (points[-1] - points[-2]) / prediction.dt
    aligned += np.maximum(query - source[-1], 0)[:, None] * velocity
    covered = age >= -0.05 and query[0] >= -0.05 and query[-1] <= source[-1] + 0.35
    return aligned, bool(covered)


class Track:
    """Closed polyline with arc-length sampling and signed lateral projection."""
    def __init__(self, xy, speed):
        self.xy = np.asarray(xy, dtype=float)
        self.speed = np.asarray(speed, dtype=float)
        if self.xy.ndim != 2 or self.xy.shape[1] != 2 or len(self.xy) < 3:
            raise ValueError('Track needs at least three [x, y] points')
        if len(self.speed) != len(self.xy) or not np.isfinite(self.xy).all() or not np.isfinite(self.speed).all():
            raise ValueError('Track positions and speeds must be finite and have equal lengths')
        # Some shipped CSVs explicitly repeat the first point to close the lap.
        # This class adds the closing segment itself, so remove that one repeat.
        if len(self.xy) > 3 and np.linalg.norm(self.xy[-1] - self.xy[0]) < 1e-6:
            self.xy, self.speed = self.xy[:-1], self.speed[:-1]
        self.delta = np.roll(self.xy, -1, axis=0) - self.xy
        self.lengths = np.linalg.norm(self.delta, axis=1)
        if np.any(self.lengths < 1e-6):
            raise ValueError('Remove duplicate consecutive/closing waypoints')
        self.s = np.r_[0.0, np.cumsum(self.lengths)]
        self.length = self.s[-1]

    def project(self, point):
        fraction = np.clip(np.sum((point - self.xy) * self.delta, axis=1) / self.lengths**2, 0, 1)
        projection = self.xy + fraction[:, None] * self.delta
        i = int(np.argmin(np.sum((point - projection)**2, axis=1)))
        # Positive lateral distance is left of the track tangent.
        tangent = self.delta[i] / self.lengths[i]
        lateral = float(np.dot(point - projection[i], [-tangent[1], tangent[0]]))
        return float(self.s[i] + fraction[i] * self.lengths[i]), lateral

    def gap(self, s, origin):
        return (s - origin + self.length / 2) % self.length - self.length / 2

    def sample(self, s):
        s = np.asarray(s) % self.length
        i = np.minimum(np.searchsorted(self.s, s, side='right') - 1, len(self.xy) - 1)
        f = (s - self.s[i]) / self.lengths[i]
        xy = self.xy[i] + f[..., None] * self.delta[i]
        yaw = np.arctan2(self.delta[i, 1], self.delta[i, 0])
        speed = (1 - f) * self.speed[i] + f * self.speed[(i + 1) % len(self.xy)]
        return xy, yaw, speed


# FOLLOW uses the BRAKE scenario at index 1.
SCENARIO_NAMES = ('CONTINUE', 'BRAKE', 'LEFT', 'RIGHT', 'BRAKE_LEFT', 'BRAKE_RIGHT')


def predict_opponent_scenarios(track, prediction, now, horizon, dt, settings, start_delay=0.0):
    """Six alternative futures from the same timestamped observation, never averaged.

    CONTINUE preserves the supplied nominal path. Others use its initial
    forward speed and current track-relative offset. Braking stops at zero;
    lateral shifts have bounded displacement, speed and acceleration. These
    are finite hypotheses, not probabilities or a complete reachable set.
    """
    if (not all(math.isfinite(t) for t in (now, prediction.stamp, dt, start_delay))
            or dt <= 0 or start_delay < 0 or horizon < 1):
        raise ValueError('Scenario times must be finite and delays nonnegative')
    nominal, covered = align_prediction(prediction, now, horizon, dt, start_delay)
    origin_s, origin_d = track.project(np.asarray(prediction.points[0]))
    next_s, _ = track.project(np.asarray(prediction.points[1]))
    speed = max(0.0, track.gap(next_s, origin_s) / prediction.dt)
    # A maneuver could already have started during transport/actuation delay.
    times = np.maximum(0.0, now - prediction.stamp + start_delay + np.arange(horizon + 1)*dt)
    stop_time = speed / settings.opponent_braking
    braking_time = np.minimum(times, stop_time)
    brake_progress = speed*braking_time - 0.5*settings.opponent_braking*braking_time**2
    cruise_progress = speed * times

    def shifted(progress, deceleration, side):
        if speed < 1e-6 or side == 0:
            shift = np.zeros_like(progress)
        else:
            displacement = settings.opponent_lateral_displacement
            # Quintic smoothstep has max |f'|=1.875 and |f''|<5.774.
            # Limit sideways velocity by forward speed too (no sideways motion
            # from rest). Split acceleration budget between curvature of the
            # shift and longitudinal braking's effect on its time derivative.
            lateral_speed = min(settings.opponent_lateral_speed, 0.5*speed)
            duration = max(1.875*displacement/lateral_speed,
                           math.sqrt(2*5.774*displacement/settings.opponent_lateral_accel),
                           2*1.875*displacement*deceleration/(settings.opponent_lateral_accel*speed))
            q = np.clip(progress/(speed*duration), 0.0, 1.0)
            shift = side * displacement*q**3*(10 + q*(-15 + 6*q))
        xy, yaw, _ = track.sample(origin_s + progress)
        offset = origin_d + shift
        return xy + offset[:, None]*np.column_stack((-np.sin(yaw), np.cos(yaw)))

    paths = np.stack((nominal,
                      shifted(brake_progress, settings.opponent_braking, 0),
                      shifted(cruise_progress, 0, 1), shifted(cruise_progress, 0, -1),
                      shifted(brake_progress, settings.opponent_braking, 1),
                      shifted(brake_progress, settings.opponent_braking, -1)))
    return paths, covered


def opponent_scenario_batch(track, opponents, active, dt, settings):
    """Offline/already-aligned input: branch from time zero for each active car."""
    paths = np.repeat(opponents[:, None], len(SCENARIO_NAMES), axis=1)
    for i in np.flatnonzero(active):
        prediction = OpponentPrediction(opponents[i], 0.0, dt)
        paths[i], _ = predict_opponent_scenarios(track, prediction, 0.0, len(opponents[i]) - 1, dt, settings)
    return paths


def trajectory_valid(X, U, opponents, active, cfg, safety, wall_env, previous_steer=0.0):
    """Swept circular footprints + bounded dynamics, including the initial pose.

    Circles circumscribe the complete rectangles. Relative line-segment closest
    approach catches opponents crossing between time samples. Wall samples use
    an additional half-subinterval displacement margin. This checks the finite
    horizon under the bicycle/linear-interpolation model, not formal safety.
    """
    center = X[:, :2] + safety.rear_to_center * jnp.stack((jnp.cos(X[:, 2]), jnp.sin(X[:, 2])), axis=-1)
    fraction = jnp.linspace(0.0, 1.0, safety.wall_substeps + 1)
    sweep = center[:-1, None, :] + fraction[None, :, None] * jnp.diff(center, axis=0)[:, None, :]
    margin = jnp.linalg.norm(jnp.diff(center, axis=0), axis=-1) / (2 * safety.wall_substeps)
    wall_ok = jnp.all(wall_env.sample_wall_distance(sweep) >= safety.radius + safety.clearance + margin[:, None])
    relative = center[None, :, :] - opponents
    dr = jnp.diff(relative, axis=1)
    f = jnp.clip(-jnp.sum(relative[:, :-1] * dr, axis=-1) / jnp.maximum(jnp.sum(dr**2, axis=-1), 1e-9), 0, 1)
    closest = relative[:, :-1] + f[..., None] * dr
    uncertainty = safety.uncertainty + safety.uncertainty_growth * (jnp.arange(cfg.H) + 1) * cfg.DT
    radius = 2 * safety.radius + safety.clearance + uncertainty
    opponent_ok = jnp.all((jnp.linalg.norm(closest, axis=-1) >= radius[None, :]) | ~active[:, None])
    steer_delta = jnp.diff(jnp.concatenate((jnp.asarray([previous_steer]), U[:, 1])))
    lateral_accel = X[:-1, 3]**2 * jnp.tan(U[:, 1]) / cfg.WB
    bounded = (jnp.all(jnp.abs(steer_delta) <= safety.max_steering_rate * cfg.DT + 1e-5)
               & jnp.all(jnp.abs(lateral_accel) <= safety.max_lateral_accel + 1e-5)
               & jnp.all((U[:, 0] >= cfg.MIN_ACCEL) & (U[:, 0] <= cfg.MAX_ACCEL))
               & jnp.all((U[:, 1] >= cfg.MIN_STEER) & (U[:, 1] <= cfg.MAX_STEER))
               & jnp.all((X[:, 3] >= cfg.MIN_SPEED - 1e-5) & (X[:, 3] <= cfg.MAX_SPEED + 1e-5)))
    return wall_ok & opponent_ok & bounded & jnp.all(jnp.isfinite(X)) & jnp.all(jnp.isfinite(U))


def side_valid(X, reference, target_path, side):
    """When alongside the target, LEFT/RIGHT must stay on that track-relative side."""
    tangent = jnp.stack((jnp.cos(reference[:, 2]), jnp.sin(reference[:, 2])), axis=-1)
    normal = jnp.stack((-tangent[:, 1], tangent[:, 0]), axis=-1)
    relative = X[:, :2] - target_path
    along = jnp.sum(relative * tangent, axis=-1)
    lateral = jnp.sum(relative * normal, axis=-1)
    return (side == 0) | jnp.all((jnp.abs(along) > 1.0) | (side * lateral >= 0))


@partial(jax.jit, static_argnames=('cfg', 'safety', 'wall_env'))
def mode_update(key, warm, seed, state, reference, opponents, active, previous_steer, cfg, safety, wall_env,
                side=0.0, target_index=0, scenario_paths=None, scenario_active=None):
    """One mode only. Invalid samples have exactly zero weight, including suffixes."""
    noise = jax.random.normal(key, (cfg.T, cfg.H, cfg.NU)) * jnp.asarray(cfg.U_STD)
    means = jnp.where((jnp.arange(cfg.T) < cfg.T // 2)[:, None, None], seed, warm)
    controls = jnp.clip(means + noise, jnp.array([cfg.MIN_ACCEL, cfg.MIN_STEER]),
                        jnp.array([cfg.MAX_ACCEL, cfg.MAX_STEER]))
    # Smooth sampled controls before propagation, not after collision checks.
    def limit(previous, u):
        steer = jnp.clip(u[:, 1], previous - safety.max_steering_rate * cfg.DT,
                         previous + safety.max_steering_rate * cfg.DT)
        return steer, u.at[:, 1].set(steer)
    _, limited = jax.lax.scan(limit, jnp.full(cfg.T, previous_steer), controls.swapaxes(0, 1))
    controls = limited.swapaxes(0, 1).at[0].set(seed)
    paths = rollout(state, controls, cfg)
    # Reject against all scenarios; nominal paths only determine soft costs.
    checks = opponents if scenario_paths is None else scenario_paths
    check_active = active if scenario_active is None else scenario_active
    valid = jax.vmap(trajectory_valid, in_axes=(0, 0, None, None, None, None, None, None))(
        paths, controls, checks, check_active, cfg, safety, wall_env, previous_steer)
    valid &= jax.vmap(side_valid, in_axes=(0, None, None, None))(
        paths, reference, opponents[target_index], side)
    costs = jax.vmap(cost, in_axes=(0, 0, None, None, None, None, None, None))(
        paths, controls, reference, cfg, opponents[0, 1:], opponents[1, 1:], opponents[2, 1:], wall_env)
    totals = jnp.sum(costs, axis=1)
    valid = valid & jnp.isfinite(totals)
    count = jnp.sum(valid)
    # Exclude invalid samples from normalization, including all-invalid batches.
    low = jnp.min(jnp.where(valid, totals, jnp.inf))
    high = jnp.max(jnp.where(valid, totals, -jnp.inf))
    low = jnp.where(count > 0, low, 0.0)
    high = jnp.where(count > 0, high, 1.0)
    normalized = (totals - low) / jnp.maximum(high - low, 1e-6)
    weights = jnp.where(valid, jnp.exp(-normalized / cfg.LAMBDA), 0.0)
    weights /= jnp.maximum(jnp.sum(weights), 1e-12)
    update = jnp.sum(weights[:, None, None] * controls, axis=0)
    path = simulate(state, update, cfg)
    update_ok = trajectory_valid(path, update, checks, check_active, cfg, safety, wall_env, previous_steer)
    update_ok &= side_valid(path, reference, opponents[target_index], side)
    # Even same-mode averaging can be unsafe. Use a checked sample if needed.
    best = jnp.argmin(jnp.where(valid, totals, jnp.inf))
    use_average = (count > 0) & update_ok
    update = jnp.where(use_average, update, controls[best])
    path = jnp.where(use_average, path, paths[best])
    return update, path, count, count > 0


@dataclass
class Plan:
    behavior: str
    controls: np.ndarray
    trajectory: np.ndarray
    valid: bool
    reason: str
    valid_samples: dict


class OvertakingPlanner:
    def __init__(self, track, wall_env, vehicle=None, safety=None):
        self.track, self.wall_env = track, wall_env
        self.safety = safety or PlannerConfig()
        self.cfg = replace(vehicle or MPPIConfig(), T=self.safety.samples_per_mode)
        self.behavior = 'FOLLOW'
        self.target = None
        self.warm = {}
        self.key = jax.random.PRNGKey(0)

    def reference(self, state, mode, opponents, active, follow_opponents=None):
        cfg, p = self.cfg, self.safety
        s0, d0 = self.track.project(state[:2])
        lead = None
        for i in np.flatnonzero(active):
            so, do = self.track.project(opponents[i, 0])
            gap = self.track.gap(so, s0)
            if -p.complete_gap < gap < p.trigger_distance and abs(do - d0) < 2 * p.pass_offset:
                if lead is None or gap < lead[1]:
                    lead = (i, gap, do)
        # Keep the same target until our entire car is past it.
        if self.target is not None and active[self.target]:
            so, do = self.track.project(opponents[self.target, 0])
            gap = self.track.gap(so, s0)
            if gap > -p.complete_gap:
                lead = (self.target, gap, do)
        # A pass must leave room for a sideways-moving opponent as well as its
        # nominal centreline. Otherwise even a wide track only receives narrow,
        # doomed passing seeds. The collision gate remains authoritative.
        pass_offset = max(p.pass_offset, p.opponent_lateral_displacement + 2*p.radius
                          + p.clearance + p.uncertainty + p.uncertainty_growth*cfg.H*cfg.DT)
        lateral_target = 0.0 if mode == 'FOLLOW' or lead is None else lead[2] + (pass_offset if mode == 'LEFT' else -pass_offset)
        change_distance = max(p.change_distance, float(state[3]) * math.sqrt(
            6*abs(lateral_target - d0)/p.max_lateral_accel))
        s, v = s0, max(0.0, float(state[3]))
        refs = []
        for k in range(cfg.H + 1):
            xy, yaw, cruise = self.track.sample(s)
            target_v = min(float(cruise), cfg.MAX_SPEED)
            if mode == 'FOLLOW' and lead is not None and lead[1] > -p.complete_gap:
                i = lead[0]
                follow = opponents if follow_opponents is None else follow_opponents
                so, _ = self.track.project(follow[i, k])
                gap = self.track.gap(so, s)
                ko = min(k + 1, cfg.H)
                speed = np.linalg.norm(follow[i, ko] - follow[i, max(0, ko - 1)]) / cfg.DT
                desired = p.standstill_gap + p.time_headway * v
                target_v = min(target_v, max(0.0, speed + (gap - desired)))
            t = np.clip((s - s0) / change_distance, 0, 1)
            blend = t*t*t*(10 + t*(-15 + 6*t))  # quintic: zero slope/curvature at ends
            d = d0 + blend * (lateral_target - d0)
            refs.append([xy[0] - np.sin(yaw)*d, xy[1] + np.cos(yaw)*d, yaw, target_v])
            v += np.clip(target_v - v, cfg.MIN_ACCEL*cfg.DT, cfg.MAX_ACCEL*cfg.DT)
            s += v * cfg.DT
        ref = np.asarray(refs)
        direction = np.diff(ref[:, :2], axis=0)
        moving = np.linalg.norm(direction, axis=1) > 1e-5
        ref[:-1, 2] = np.where(moving, np.arctan2(direction[:, 1], direction[:, 0]), ref[:-1, 2])
        ref[-1, 2] = ref[-2, 2]
        return ref, lead

    def tracking_seed(self, state, reference, previous_steer):
        """Bicycle-model seed: look ahead along each behavior's reference, not raw steering bias."""
        cfg, p = self.cfg, self.safety
        x, y, yaw, v = map(float, state)
        controls = []
        steer = previous_steer
        for k in range(cfg.H):
            goal = reference[min(k + 4, cfg.H), :2]
            dx, dy = goal - np.array([x, y])
            lateral = -np.sin(yaw)*dx + np.cos(yaw)*dy
            request = np.arctan2(2 * cfg.WB * lateral, max(dx*dx + dy*dy, 0.16))
            steering_limit = min(cfg.MAX_STEER, np.arctan(p.max_lateral_accel * cfg.WB / max(v*v, 0.1)))
            request = np.clip(request, -steering_limit, steering_limit)
            steer = float(np.clip(request, steer - p.max_steering_rate*cfg.DT, steer + p.max_steering_rate*cfg.DT))
            acceleration = float(np.clip(3 * (reference[k, 3] - v), cfg.MIN_ACCEL, cfg.MAX_ACCEL))
            controls.append([acceleration, steer])
            # Seed generation only; rollout()/RK4 remains authoritative for validation.
            x += v*np.cos(yaw)*cfg.DT
            y += v*np.sin(yaw)*cfg.DT
            yaw += v*np.tan(steer)/cfg.WB*cfg.DT
            v = np.clip(v + acceleration*cfg.DT, cfg.MIN_SPEED, cfg.MAX_SPEED)
        return np.asarray(controls)

    def braking_plan(self, state, opponents, active, previous_steer, reason):
        cfg, p = self.cfg, self.safety
        # Keep initial steering continuous, then unwind while braking.
        steer = np.sign(previous_steer) * np.maximum(abs(previous_steer) - np.arange(1, cfg.H + 1)*p.max_steering_rate*cfg.DT, 0)
        controls = np.column_stack((np.full(cfg.H, cfg.MIN_ACCEL), steer))
        path = np.asarray(simulate_jit(jnp.asarray(state), jnp.asarray(controls), cfg))
        valid = bool(trajectory_valid(jnp.asarray(path), jnp.asarray(controls), jnp.asarray(opponents),
                                     jnp.asarray(active), cfg, p, self.wall_env, previous_steer))
        # Invalid braking plans trigger a stop request, not a certified escape.
        self.behavior = 'BRAKE'
        self.warm.clear()
        return Plan('BRAKE', controls, path, valid, reason, {})

    def plan(self, state, opponents=None, active=None, previous_steer=0.0, stale=False, scenarios=None):
        cfg, p = self.cfg, self.safety
        state = np.asarray(state, dtype=float)
        opponents = np.full((3, cfg.H + 1, 2), 1e3) if opponents is None else np.asarray(opponents, dtype=float)
        active = np.zeros(3, dtype=bool) if active is None else np.asarray(active, dtype=bool)
        if state.shape != (4,) or not np.isfinite(state).all() or not np.isfinite(previous_steer):
            raise ValueError('State and previous steering must be finite')
        if opponents.shape != (3, cfg.H + 1, 2) or active.shape != (3,) or not np.isfinite(opponents).all():
            raise ValueError('Expected finite opponents [3, H+1, 2] and active mask [3]')
        if scenarios is None:
            scenarios = opponent_scenario_batch(self.track, opponents, active, cfg.DT, p)
        scenarios = np.asarray(scenarios, dtype=float)
        if scenarios.shape != (3, len(SCENARIO_NAMES), cfg.H + 1, 2) or not np.isfinite(scenarios).all():
            raise ValueError('Expected finite scenarios [3, 6, H+1, 2]')
        # Keep opponent-major order consistent with the repeated active mask.
        checks = scenarios.reshape(-1, cfg.H + 1, 2)
        check_active = np.repeat(active, len(SCENARIO_NAMES))
        if stale:
            return self.braking_plan(state, checks, check_active, previous_steer, 'stale_or_short_prediction')
        _, lead = self.reference(state, 'FOLLOW', opponents, active)
        if lead is None:
            self.target = None
        modes = ['FOLLOW'] if lead is None else ['FOLLOW', 'LEFT', 'RIGHT']
        # Do not reverse passing side directly while alongside an opponent.
        if self.behavior in ('LEFT', 'RIGHT') and lead is not None:
            modes.remove('RIGHT' if self.behavior == 'LEFT' else 'LEFT')
        choices, counts = [], {}
        s0, _ = self.track.project(state[:2])
        for mode in modes:
            ref, _ = self.reference(state, mode, opponents, active, follow_opponents=scenarios[:, 1])
            seed = self.tracking_seed(state, ref, previous_steer)
            old = self.warm.get(mode, seed)
            warm = np.vstack((old[1:], old[-1])) if mode in self.warm else seed
            self.key, key = jax.random.split(self.key)
            u, path, count, ok = mode_update(key, jnp.asarray(warm), jnp.asarray(seed), jnp.asarray(state),
                                           jnp.asarray(ref), jnp.asarray(opponents), jnp.asarray(active),
                                           previous_steer, cfg, p, self.wall_env,
                                           1.0 if mode == 'LEFT' else -1.0 if mode == 'RIGHT' else 0.0,
                                           lead[0] if lead else 0, jnp.asarray(checks), jnp.asarray(check_active))
            counts[mode] = int(count)
            if not bool(ok):
                continue
            u, path = np.asarray(u), np.asarray(path)
            self.warm[mode] = u
            end_s, end_d = self.track.project(path[-1, :2])
            score = -self.track.gap(end_s, s0) + 0.05 * abs(end_d)
            score += p.switch_penalty if mode != self.behavior else 0.0
            choices.append((score, mode, u, path))
        if not choices:
            result = self.braking_plan(state, checks, check_active, previous_steer, 'no_valid_candidate')
            result.valid_samples = counts
            return result
        _, mode, u, path = min(choices, key=lambda item: item[0])
        # Authoritative final gate, also protects future changes to selection logic.
        if not bool(trajectory_valid(jnp.asarray(path), jnp.asarray(u), jnp.asarray(checks),
                                     jnp.asarray(check_active), cfg, p, self.wall_env, previous_steer)):
            return self.braking_plan(state, checks, check_active, previous_steer, 'final_validation_failed')
        self.behavior = mode
        self.target = lead[0] if mode in ('LEFT', 'RIGHT') and lead else None
        return Plan(mode, u, path, True, 'selected_valid_candidate', counts)
