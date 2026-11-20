"""Filter logic of 1B (shadow_ekf1_node.py); 1C (shadow_accel.py) subclasses it.
Event-driven wrapper around student_ekf (no ROS): Δt guards, steering zero-order hold, counters.
Used by shadow_ekf1_node (live) and tools/offline_sim.py, so both run exactly the same logic.

Δt guards (plan section 7):
    dt < 0            drop the message, count it
    dt = 0            update only
    0 < dt <= 0.1     one predict, then update
    0.1 < dt <= 1     predict in sub-steps of at most 0.02 s, then update (counted)
    dt > 1            reset the filter from this measurement (counted)
"""
from collections import deque

import numpy as np

from . import student_ekf as ekf
from .params import Counters, ModelParams, ShadowNoise

C_SPEED = np.array([[0.0, 0.0, 0.0, 1.0, 0.0]])
C_GYRO = np.array([[0.0, 0.0, 0.0, 0.0, 1.0]])


class ShadowFilter:

    def __init__(self, p: ModelParams = None, noise: ShadowNoise = None, q_scale=1.0,
                 max_step=0.1, substep=0.02, reset_gap=1.0, stale_steering=0.2):
        self.p = p or ModelParams()
        self.noise = noise or ShadowNoise()
        self.Q = self.noise.Q(q_scale)
        self.max_step, self.substep, self.reset_gap = max_step, substep, reset_gap
        self.stale_steering = stale_steering
        self.steer = deque(maxlen=200)       # (t, delta) in stamp order
        self.mu = None
        self.Sigma = None
        self.t = None
        self.c = Counters()
        self.history = deque(maxlen=500)     # (t, mu, Sigma, u): unused until stage 2

    # ---------------------------------------------------------------- inputs
    def add_steering(self, t, delta):
        if self.steer and t < self.steer[-1][0]:
            return
        self.steer.append((float(t), float(delta)))

    def steering_at(self, t):
        """Command in effect at time t (newest with stamp <= t); 0 before the first command."""
        d, ts = 0.0, None
        for s, v in self.steer:
            if s <= t:
                d, ts = v, s
            else:
                break
        if ts is None or t - ts > self.stale_steering:
            self.c.stale_steering += 1
        return d

    # ---------------------------------------------------------------- filter
    def _init(self, t, sensor, y):
        self.mu = np.zeros(5)
        if sensor == 'speed':
            self.mu[3] = y
        else:
            self.mu[4] = y
        self.Sigma = self.noise.Sigma0()
        self.t = t

    def _predict_to(self, t):
        dt = t - self.t
        n = 1 if dt <= self.max_step else int(np.ceil(dt / self.substep))
        if n > 1:
            self.c.substeps += 1
        h = dt / n
        for i in range(n):
            ts = self.t + i * h
            u = np.array([0.0, self.steering_at(ts)])
            self.mu, self.Sigma = ekf.predict(self.mu, self.Sigma, u, h, self.Q, self.p)
            self.c.predicts += 1
        self.history.append((t, self.mu.copy(), self.Sigma.copy(), self.steering_at(self.t)))
        self.t = t
        return dt

    def measure(self, t, sensor, y, R):
        """Process one measurement. Returns a dict for diagnostics, or None if dropped.
        sensor: 'speed' or 'gyro'; y: float; R: variance."""
        if self.mu is None:
            self._init(t, sensor, y)
            dt = 0.0
        else:
            dt = t - self.t
            if dt < 0.0:
                self.c.dropped_old += 1
                return None
            if dt > self.reset_gap:
                self.c.resets += 1
                self._init(t, sensor, y)
                dt = 0.0
            elif dt > 0.0:
                self._predict_to(t)
        C = C_SPEED if sensor == 'speed' else C_GYRO
        self.mu, self.Sigma, innov, S = ekf.update(self.mu, self.Sigma, [y], C, [[R]])
        self.c.updates += 1
        nu, s = float(innov[0]), float(S[0, 0])
        return {'t': t, 'sensor': sensor, 'y': y, 'innovation': nu, 'S': s, 'nis': nu * nu / s, 'dt': dt}
