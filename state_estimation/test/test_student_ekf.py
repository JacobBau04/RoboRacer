"""Jacob's checklist for student_ekf.py (plan section 10). Run:
    pytest src/1_state_estimation/test/test_student_ekf.py -v
"""
import math

import numpy as np
import pytest

from state_estimation import student_ekf as ekf
from state_estimation.params import ModelParams, ShadowNoise

P = ModelParams()
rng = np.random.default_rng(0)


def rand_state():
    return np.array([rng.uniform(-10, 10), rng.uniform(-10, 10), rng.uniform(-math.pi, math.pi),
                     rng.uniform(0.0, 12.0), rng.uniform(-3, 3)])


def rand_input():
    return np.array([rng.uniform(-5, 5), rng.uniform(-0.42, 0.42)])


def num_jac(f, x, eps=1e-6):
    x = np.asarray(x, dtype=float)
    f0 = f(x)
    J = np.zeros((len(f0), len(x)))
    for i in range(len(x)):
        d = np.zeros_like(x)
        d[i] = eps
        diff = f(x + d) - f(x - d)
        if len(f0) == 5:
            diff[2] = ekf.wrap_angle(diff[2])     # theta difference across the seam
        J[:, i] = diff / (2 * eps)
    return J


def test_jacobian_A_matches_finite_differences():
    for _ in range(100):
        z, u, dt = rand_state(), rand_input(), 0.02
        A = ekf.jacobian_A(z, u, dt, P)
        assert A.shape == (5, 5)
        assert np.allclose(A, num_jac(lambda zz: ekf.motion_model(zz, u, dt, P), z), atol=1e-6)


def test_jacobian_B_matches_finite_differences():
    for _ in range(100):
        z, u, dt = rand_state(), rand_input(), 0.02
        B = ekf.jacobian_B(z, u, dt, P)
        assert B.shape == (5, 2)
        assert np.allclose(B, num_jac(lambda uu: ekf.motion_model(z, uu, dt, P), u), atol=1e-6)


def test_circle_with_large_tau_and_straight_line():
    p = ModelParams(L=0.33, tau=1e9)               # tau -> infinity: omega stays constant
    v, w, dt = 3.0, 1.0, 1e-4
    z = np.array([0.0, 0.0, 0.0, v, w])
    T = 2 * math.pi / w
    for _ in range(int(round(T / dt))):
        z = ekf.motion_model(z, [0.0, 0.0], dt, p)
    assert math.hypot(z[0], z[1]) < 5e-3            # back at the start (explicit-Euler tolerance)
    z = np.array([0.0, 0.0, 0.3, 2.0, 0.0])
    for _ in range(100):
        z = ekf.motion_model(z, [0.0, 0.0], 0.01, p)
    assert z[1] / z[0] == pytest.approx(math.tan(0.3))


def test_omega_converges_to_steering_rate():
    v, delta, dt = 4.0, 0.2, 0.001
    z = np.array([0.0, 0.0, 0.0, v, 0.0])
    for _ in range(int(3 * P.tau / dt)):
        z = ekf.motion_model(z, [0.0, delta], dt, P)
    target = v * math.tan(delta) / P.L
    assert abs(z[4] - target) < 0.05 * abs(target)


def test_covariance_stays_symmetric_positive_definite():
    mu, Sig = rand_state(), ShadowNoise().Sigma0()
    Q = ShadowNoise().Q()
    for k in range(10000):
        mu, Sig = ekf.predict(mu, Sig, rand_input() * [0, 1], 0.01, Q, P)
        if k % 2 == 0:
            mu, Sig, _, _ = ekf.update(mu, Sig, [mu[3] + 0.05 * rng.standard_normal()],
                                       [[0, 0, 0, 1, 0]], [[0.0025]])
        else:
            mu, Sig, _, _ = ekf.update(mu, Sig, [mu[4] + 0.02 * rng.standard_normal()],
                                       [[0, 0, 0, 0, 1]], [[0.0004]])
        mu[3] = np.clip(mu[3], 0, 12)
    assert np.allclose(Sig, Sig.T)
    assert np.min(np.linalg.eigvalsh(Sig)) > 0


def test_update_never_increases_measured_variance():
    for _ in range(200):
        mu = rand_state()
        M = rng.standard_normal((5, 5))
        Sig = M @ M.T + 1e-3 * np.eye(5)
        for row in (3, 4):
            C = np.zeros((1, 5))
            C[0, row] = 1
            _, S2, _, _ = ekf.update(mu, Sig, [mu[row] + 0.1], C, [[0.01]])
            assert S2[row, row] <= Sig[row, row] + 1e-12


def test_stacked_update_equals_sequential():
    mu = rand_state()
    M = rng.standard_normal((5, 5))
    Sig = M @ M.T + 1e-3 * np.eye(5)
    y = np.array([mu[3] + 0.2, mu[4] - 0.1])
    C = np.array([[0, 0, 0, 1, 0], [0, 0, 0, 0, 1]], dtype=float)
    R = np.diag([0.0025, 0.0004])
    m1, S1, _, _ = ekf.update(mu, Sig, y, C, R)
    m2, S2, _, _ = ekf.update(mu, Sig, y[:1], C[:1], R[:1, :1])
    m2, S2, _, _ = ekf.update(m2, S2, y[1:], C[1:], R[1:, 1:])
    assert np.allclose(m1, m2) and np.allclose(S1, S2)


def test_wrap_angle_and_theta_stays_wrapped():
    assert ekf.wrap_angle(math.pi + 0.1) == pytest.approx(-math.pi + 0.1)
    assert ekf.wrap_angle(-math.pi) == pytest.approx(-math.pi)
    z = np.array([0.0, 0.0, 0.0, 5.0, 2.0])
    for _ in range(int(100 * 2 * math.pi / 2.0 / 0.01)):        # 100 laps of a circle
        z = ekf.motion_model(z, [0.0, 0.0], 0.01, ModelParams(tau=1e9))
        assert -math.pi <= z[2] < math.pi


def test_inputs_never_modified():
    mu, Sig = rand_state(), ShadowNoise().Sigma0()
    u, Q = rand_input(), ShadowNoise().Q()
    y, C, R = np.array([1.0]), np.array([[0, 0, 0, 1.0, 0]]), np.array([[0.01]])
    copies = [a.copy() for a in (mu, Sig, u, Q, y, C, R)]
    ekf.motion_model(mu, u, 0.02, P)
    ekf.jacobian_A(mu, u, 0.02, P)
    ekf.jacobian_B(mu, u, 0.02, P)
    ekf.predict(mu, Sig, u, 0.02, Q, P)
    ekf.update(mu, Sig, y, C, R)
    for a, b in zip((mu, Sig, u, Q, y, C, R), copies):
        assert np.array_equal(a, b)


def test_shapes():
    mu, Sig = rand_state(), ShadowNoise().Sigma0()
    m2, S2 = ekf.predict(mu, Sig, [0.0, 0.1], 0.02, ShadowNoise().Q(), P)
    assert m2.shape == (5,) and S2.shape == (5, 5)
    m3, S3, inn, S = ekf.update(m2, S2, [1.0], [[0, 0, 0, 1, 0]], [[0.01]])
    assert m3.shape == (5,) and S3.shape == (5, 5) and inn.shape == (1,) and S.shape == (1, 1)
