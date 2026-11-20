"""FILTER 1C: shadow EKF1 with acceleration as an INPUT, computed the way the simulator computes it.
OFFLINE ONLY for now (no ROS node): replayed from bags by tools/compare_vdot_ekfs.py, which feeds
every message in STAMP order. Sim-only model: the real car's VESC needs its own fitted parameters
(VDOT_CAR_PROCEDURE.md).

Same as 1B (student_ekf + ShadowFilter, 5 states x, y, theta, v, omega; same sensors) except:
Dynamics (explicit Euler, sub-steps of at most 0.01 s = the sim step; the braking gain makes longer
Euler steps unstable):
    v' = v + a dt,  a = gym_speed_controller(v, v_cmd), copied from f1tenth_gym (pid_accl + accl_constraints):
        a_des = kp (v_cmd - v),  kp = 10 a_max / v_max = 4.755 speeding up (v > 0),
                                      10 a_max / |v_min| = 19.02 braking (v > 0); 2 a_max/.. when v <= 0
        a = clip(a_des, -a_max, upper(v)),  upper = a_max (v <= v_switch) else a_max v_switch / v
        a = 0 at v_min / v_max          (a_max 9.51, v_max 20, v_min -5, v_switch 7.319)
    Jacobian: dv'/dv = 1 + (da/dv) dt  (chain rule; da/dv = -kp in the linear zone, 0 when clipped)
    theta, x, y, omega lines: as 1B.
Sensors: as 1B: /prep/wheel_twist 50 Hz (v), /prep/imu 100 Hz (omega).
Inputs: steering delta (/prep/steering, ~45 Hz) and the speed command v_cmd (/drive, ~45 Hz, the MPC's
    command, time = receive time because the MPC leaves the stamp empty). Zero-order hold, no delay.
R: as 1B. Q: as 1B but q_v = 0.02 (best in the sweep; pass ShadowNoise with q_diag[3] = 0.02).
"""
# TODO (Jacob): think about implementing the commanded steering angle into the dynamics model or measurement for EKF_1. Would this be helpful?
# Now: 1C uses the speed command for acceleration; its yaw model is 1B's (steering command, bicycle).
# Same question as 1B: is the steering command worth it, given the 0.77-0.89x steering response?
import math
from dataclasses import dataclass

import numpy as np

from . import student_ekf as ekf
from .shadow_core import ShadowFilter


@dataclass
class GymSpeedParams:
    """f1tenth_gym 'f1tenth' vehicle parameters (envs/dynamic_models/__init__.py)."""
    a_max: float = 9.51
    v_max: float = 20.0
    v_min: float = -5.0
    v_switch: float = 7.319


def gym_accel(v, v_cmd, g: GymSpeedParams):
    """Acceleration the sim applies for speed command v_cmd at speed v, and its derivative d a / d v."""
    vel_diff = v_cmd - v
    if v > 0.0:
        kp = 10.0 * g.a_max / g.v_max if vel_diff > 0 else 10.0 * g.a_max / (-g.v_min)
    else:
        kp = 2.0 * g.a_max / g.v_max if vel_diff > 0 else 2.0 * g.a_max / (-g.v_min)
    a_des = kp * vel_diff

    if v > g.v_switch:
        upper, d_upper = g.a_max * g.v_switch / v, -g.a_max * g.v_switch / v ** 2
    else:
        upper, d_upper = g.a_max, 0.0

    if (v <= g.v_min and a_des <= 0) or (v >= g.v_max and a_des >= 0):
        return 0.0, 0.0
    if a_des <= -g.a_max:
        return -g.a_max, 0.0
    if a_des >= upper:
        return upper, d_upper
    return a_des, -kp


def predict_accel(mu, Sigma, v_cmd, delta, dt, Q, p, g: GymSpeedParams):
    """One prediction with a from the gym controller. Reuses Jacob's motion_model and jacobian_A."""
    mu = np.asarray(mu, dtype=float)
    a, dadv = gym_accel(float(mu[3]), v_cmd, g)
    u = np.array([a, delta])
    A = ekf.jacobian_A(mu, u, dt, p)
    A[3, 3] += dadv * dt                       # a depends on v: chain rule
    mu_new = ekf.motion_model(mu, u, dt, p)
    Sigma_new = A @ np.asarray(Sigma, dtype=float) @ A.T + np.asarray(Q, dtype=float) * dt
    return mu_new, 0.5 * (Sigma_new + Sigma_new.T)


class ShadowFilterAccel(ShadowFilter):
    """ShadowFilter with the speed command as an extra input. Predicts in sub-steps of at most
    sim_dt (the sim's own step), because the braking gain (19 1/s) makes long Euler steps unstable."""

    def __init__(self, *args, gym: GymSpeedParams = None, sim_dt=0.01, **kw):
        super().__init__(*args, **kw)
        self.g = gym or GymSpeedParams()
        self.sim_dt = sim_dt
        self.cmds = []                         # (t, v_cmd) in time order

    def add_command(self, t, v_cmd):
        if self.cmds and t < self.cmds[-1][0]:
            return
        self.cmds.append((float(t), float(v_cmd)))
        if len(self.cmds) > 400:
            del self.cmds[:200]

    def command_at(self, t):
        """Newest speed command with time <= t. Before the first command: hold the current estimate."""
        for s, v in reversed(self.cmds):
            if s <= t:
                return v
        return float(self.mu[3])

    def _predict_to(self, t):
        dt = t - self.t
        n = max(1, int(math.ceil(dt / self.sim_dt - 1e-9)))
        h = dt / n
        for i in range(n):
            ts = self.t + i * h
            self.mu, self.Sigma = predict_accel(self.mu, self.Sigma, self.command_at(ts),
                                                self.steering_at(ts), h, self.Q, self.p, self.g)
            self.c.predicts += 1
        self.t = t
        return dt
