"""Dynamic single-track ("slip") model for the MPC: the same equations as the simulator.

Source of the equations: f1tenth_gym/envs/dynamic_models/single_track.py (DynamicModel.ST,
which f1tenth_gym_ros uses). Parameter values: F1TENTH_VEHICLE_PARAMETERS in
f1tenth_gym/envs/dynamic_models/__init__.py.

State  x = [X, Y, v, psi, psi_dot, beta]
       X, Y     position of the centre of gravity (what the sim reports in /odom)
       v        speed (magnitude of the velocity, not just its forward part)
       psi      heading (where the car points)
       psi_dot  yaw rate
       beta     slip angle: the car travels along psi + beta, not psi
Input  u = [a, delta]   acceleration, steering angle

Simple first version (see the plan in the conversation / TUNING_LOG):
  * weight transfer: braking loads the front tires and unloads the rear (h > 0);
    set SlipParams.h = 0 to switch it off
  * delta is an input, the sim's ~26 ms steering lag is ignored
  * below V_KIN the kinematic model is used, like the sim does
  * rollout: Euler with small sub-steps; linearization: finite differences;
    discretization: matrix exponential (stable at any step size)
"""
import math
from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm

V_KIN = 0.5      # m/s, below this the sim (and this model) switch to the kinematic model
TAU_KIN = 0.02   # s, in kinematic mode psi_dot and beta relax to their kinematic values this fast


@dataclass
class SlipParams:
    mu: float = 1.0489       # friction coefficient (scales the tire stiffness in this model)
    C_Sf: float = 4.718      # front cornering stiffness [1/rad] (per unit tire load)
    C_Sr: float = 5.4562     # rear cornering stiffness [1/rad]
    lf: float = 0.1558       # centre of gravity -> front axle [m] (sim split scaled to WB 0.324)
    lr: float = 0.1682       # centre of gravity -> rear axle [m] (lf + lr = 0.324)
    m: float = 3.74          # mass [kg]
    I: float = 0.04712       # yaw inertia [kg m^2]
    g: float = 9.81          # gravity [m/s^2]
    h: float = 0.074         # centre of gravity height [m] (weight transfer; 0 = off)


def f_cont(x, u, p, kinematic=None):
    """Continuous-time dynamics x_dot = f(x, u).

    kinematic: None = pick the model from the speed (v < V_KIN). The linearization passes
    True/False explicitly so that its small finite-difference nudges never switch models.
    """
    _, _, v, psi, r, beta = x
    a, delta = u
    L = p.lf + p.lr
    if kinematic is None:
        kinematic = v < V_KIN

    if kinematic:
        # Kinematic bicycle at the centre of gravity (the sim's low-speed branch).
        beta_k = math.atan(math.tan(delta) * p.lr / L)
        r_k = v * math.cos(beta_k) * math.tan(delta) / L
        return np.array([
            v * math.cos(psi + beta_k),
            v * math.sin(psi + beta_k),
            a,
            r_k,                          # heading follows the kinematic yaw rate
            (r_k - r) / TAU_KIN,          # keep the psi_dot state close to it, so the
            (beta_k - beta) / TAU_KIN,    # switch to the dynamic model is smooth
        ])

    # Dynamic single-track model with linear tires. Tire load per unit mass, with weight
    # transfer: braking (a < 0) moves load to the front tires (more front grip) and off the
    # rear (less rear grip), so the car rotates into the corner more. The sim calls these
    # glr / glf. With h = 0 they are the static loads g*lr and g*lf.
    Ff = p.g * p.lr - a * p.h
    Fr = p.g * p.lf + a * p.h
    psi_ddot = (p.mu * p.m / (p.I * L)) * (
        p.lf * p.C_Sf * Ff * delta
        + (p.lr * p.C_Sr * Fr - p.lf * p.C_Sf * Ff) * beta
        - (p.lf ** 2 * p.C_Sf * Ff + p.lr ** 2 * p.C_Sr * Fr) * r / v
    )
    beta_dot = (p.mu / (v * L)) * (
        p.C_Sf * Ff * delta
        - (p.C_Sr * Fr + p.C_Sf * Ff) * beta
        + (p.C_Sr * Fr * p.lr - p.C_Sf * Ff * p.lf) * r / v
    ) - r
    return np.array([
        v * math.cos(psi + beta),         # travel direction is psi + beta: this is the slip
        v * math.sin(psi + beta),
        a,
        r,
        psi_ddot,
        beta_dot,
    ])


def rollout(x0, accels, steers, dt, n_sub, p, v_min, v_max, max_steer):
    """Nominal trajectory: integrate the nonlinear model with the given inputs.

    Euler with n_sub sub-steps per MPC step (dt / n_sub = 5 ms is stable down to V_KIN).
    Returns an array (6, len(accels) + 1); column 0 is x0.
    """
    h = dt / n_sub
    x = np.array(x0, dtype=float)
    out = np.zeros((6, len(accels) + 1))
    out[:, 0] = x
    for k, (a, delta) in enumerate(zip(accels, steers)):
        u = (a, min(max(delta, -max_steer), max_steer))
        for _ in range(n_sub):
            x = x + h * f_cont(x, u, p)
            x[2] = min(max(x[2], v_min), v_max)
        out[:, k + 1] = x
    return out


def linearize_discrete(x_bar, u_bar, dt, p, eps=1e-5):
    """Discrete model x_{k+1} ~= A_d x_k + B_d u_k + C_d around (x_bar, u_bar).

    1. Jacobians A_c = df/dx, B_c = df/du by central finite differences.
    2. Affine term c = f(x_bar, u_bar) - A_c x_bar - B_c u_bar, so that
       x_dot ~= A_c x + B_c u + c near the nominal point.
    3. Exact discretization over dt with u held constant (matrix exponential of the
       augmented matrix [[A_c, B_c, c], [0, 0, 0]] * dt).
    """
    x_bar = np.asarray(x_bar, dtype=float)
    u_bar = np.asarray(u_bar, dtype=float)
    n, m = len(x_bar), len(u_bar)
    kin = x_bar[2] < V_KIN                  # same model for every nudge below

    A_c = np.zeros((n, n))
    for j in range(n):
        d = np.zeros(n)
        d[j] = eps
        A_c[:, j] = (f_cont(x_bar + d, u_bar, p, kin) - f_cont(x_bar - d, u_bar, p, kin)) / (2 * eps)
    B_c = np.zeros((n, m))
    for j in range(m):
        d = np.zeros(m)
        d[j] = eps
        B_c[:, j] = (f_cont(x_bar, u_bar + d, p, kin) - f_cont(x_bar, u_bar - d, p, kin)) / (2 * eps)
    c = f_cont(x_bar, u_bar, p, kin) - A_c @ x_bar - B_c @ u_bar

    aug = np.zeros((n + m + 1, n + m + 1))
    aug[:n, :n] = A_c
    aug[:n, n:n + m] = B_c
    aug[:n, n + m] = c
    E = expm(aug * dt)
    return E[:n, :n], E[:n, n:n + m], E[:n, n + m]
