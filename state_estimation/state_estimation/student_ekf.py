"""Shadow EKF1: Jacob's own EKF (plan section 10). YOU write the function bodies.
Used by filter 1B (shadow_ekf1_node.py, a = 0) and filter 1C (shadow_accel.py, a from the sim's speed controller).

State  z = [x, y, theta, v, omega]   (odom frame, rear axle; m, m, rad, m/s, rad/s)
Input  u = [a, delta]                (longitudinal acceleration m/s^2 (0 in v1), steering angle rad)
Params p.L (wheelbase, m), p.tau (yaw-rate response time constant, s)   -> state_estimation.params.ModelParams

Motion model (explicit Euler over dt):
    x'     = x + v cos(theta) dt
    y'     = y + v sin(theta) dt
    theta' = wrap(theta + omega dt)
    v'     = v + a dt
    omega' = omega + (dt / tau) * (v tan(delta) / L - omega)
Covariance: Sigma' = A Sigma A^T + Q dt          (Q = process noise per second, 5x5)
Update (Joseph form):
    S = C Sigma C^T + R,  K = Sigma C^T S^-1,  mu' = mu + K (y - C mu),
    Sigma' = (I - K C) Sigma (I - K C)^T + K R K^T

Rules: return NEW arrays, never modify inputs. Angles wrapped to [-pi, pi).

How to check yourself (see ~/1_sim_ws/STUDENT_HANDOVER.md):
    pytest src/1_stage1_estimation/test/test_student_ekf.py -v
    python3 src/1_stage1_estimation/tools/offline_sim.py --level L1 --out results/offline
A working reference is in ~/1_sim_ws/reference_solutions/ (try first!).
"""
import math  # noqa: F401  (you will need it)

import numpy as np  # noqa: F401


def wrap_angle(a: float) -> float:
    """Wrap an angle (rad) to [-pi, pi). Example: wrap_angle(pi + 0.1) == -pi + 0.1."""
    # Shift so [-pi, pi) becomes [0, 2pi), take the remainder, then shift back (note: MINUS pi).
    return float((a + math.pi) % (2.0 * math.pi) - math.pi)


def motion_model(z, u, dt, p) -> np.ndarray:
    """f(z, u): state after dt seconds.

    z: (5,) [x, y, theta, v, omega];  u: (2,) [a, delta];  dt: float s;  p: ModelParams (L, tau)
    returns: (5,) new state, theta wrapped.
    """
    x, y, th, v, w = (float(e) for e in z)
    a, delta = (float(e) for e in u)
    w_steer = v * math.tan(delta) / p.L          # yaw rate the steering angle asks for
    return np.array([
        x + v * math.cos(th) * dt,
        y + v * math.sin(th) * dt,
        wrap_angle(th + w * dt),
        v + a * dt,
        w + (dt / p.tau) * (w_steer - w),        # first-order lag toward w_steer
    ])


def jacobian_A(z, u, dt, p) -> np.ndarray:
    """df/dz at (z, u): shape (5, 5)."""
    _, _, th, v, _ = (float(e) for e in z)
    delta = float(u[1])
    k = dt / p.tau
    A = np.eye(5)
    A[0, 2] = -v * math.sin(th) * dt             # dx'/dtheta
    A[0, 3] = math.cos(th) * dt                  # dx'/dv
    A[1, 2] = v * math.cos(th) * dt              # dy'/dtheta
    A[1, 3] = math.sin(th) * dt                  # dy'/dv
    A[2, 4] = dt                                 # dtheta'/domega
    A[4, 3] = k * math.tan(delta) / p.L          # domega'/dv
    A[4, 4] = 1.0 - k                            # domega'/domega
    return A


def jacobian_B(z, u, dt, p) -> np.ndarray:
    """df/du at (z, u): shape (5, 2), columns [a, delta]."""
    v = float(z[3])
    delta = float(u[1])
    B = np.zeros((5, 2))
    B[3, 0] = dt                                             # dv'/da
    B[4, 1] = (dt / p.tau) * v / (p.L * math.cos(delta) ** 2)  # domega'/ddelta
    return B


def predict(mu, Sigma, u, dt, Q, p) -> tuple:
    """One EKF prediction over dt.

    mu: (5,), Sigma: (5, 5), u: (2,), dt: float, Q: (5, 5) per second, p: ModelParams
    returns: (mu_new (5,), Sigma_new (5, 5)) -- Sigma_new symmetric.
    """
    mu = np.asarray(mu, dtype=float)
    Sigma = np.asarray(Sigma, dtype=float)
    A = jacobian_A(mu, u, dt, p)                 # linearise at the OLD mean
    mu_new = motion_model(mu, u, dt, p)          # mean through the nonlinear f
    Sigma_new = A @ Sigma @ A.T + np.asarray(Q, dtype=float) * dt
    return mu_new, 0.5 * (Sigma_new + Sigma_new.T)


def update(mu, Sigma, y, C, R, angle_idx=None) -> tuple:
    """One EKF measurement update (Joseph form).

    mu: (5,), Sigma: (5, 5), y: (m,), C: (m, 5), R: (m, m)
    angle_idx: optional list of measurement rows that are angles (their innovation is wrapped)
    returns: (mu_new (5,), Sigma_new (5, 5), innovation (m,), S (m, m))
    """
    mu = np.asarray(mu, dtype=float)
    Sigma = np.asarray(Sigma, dtype=float)
    y = np.atleast_1d(np.asarray(y, dtype=float))
    C = np.atleast_2d(np.asarray(C, dtype=float))
    R = np.atleast_2d(np.asarray(R, dtype=float))

    innov = y - C @ mu
    if angle_idx is not None:
        for i in angle_idx:
            innov[i] = wrap_angle(innov[i])

    S = C @ Sigma @ C.T + R
    K = np.linalg.solve(S, C @ Sigma).T          # = Sigma C^T S^-1 (S and Sigma symmetric)

    mu_new = mu + K @ innov
    mu_new[2] = wrap_angle(mu_new[2])

    I_KC = np.eye(mu.size) - K @ C
    Sigma_new = I_KC @ Sigma @ I_KC.T + K @ R @ K.T
    return mu_new, 0.5 * (Sigma_new + Sigma_new.T), innov, S
