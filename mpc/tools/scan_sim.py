"""Synthetic LiDAR scans for offline tests (no ROS): map walls by marching through the
wall-distance field, box obstacles by exact ray-rectangle intersection.

Default beam pattern = the car's SICK TiM571: 811 beams over 270 deg (0.333 deg).
"""
import math

import numpy as np


def box_corners(cx, cy, yaw, length, width):
    c, s = math.cos(yaw), math.sin(yaw)
    hl, hw = 0.5 * length, 0.5 * width
    pts = [(hl, hw), (-hl, hw), (-hl, -hw), (hl, -hw)]
    return np.array([(cx + c * a - s * b, cy + s * a + c * b) for a, b in pts])


def _ray_segments(ox, oy, dx, dy, A, B):
    """Distance along rays (o + t d) to segment A-B, inf if missed."""
    ex, ey = B[0] - A[0], B[1] - A[1]
    den = dx * ey - dy * ex
    wx, wy = A[0] - ox, A[1] - oy
    with np.errstate(divide='ignore', invalid='ignore'):
        t = (wx * ey - wy * ex) / den
        u = (wx * dy - wy * dx) / den
    ok = (np.abs(den) > 1e-12) & (t > 0) & (u >= 0) & (u <= 1)
    return np.where(ok, t, np.inf)


class ScanSim:
    def __init__(self, occ, boxes=(), n_beams=811, fov=math.radians(270.0), range_max=25.0):
        """occ: track_model.OccupancyMap; boxes: list of (4, 2) corner arrays."""
        self.occ = occ
        self.boxes = [np.asarray(b, float) for b in boxes]
        self.n = n_beams
        self.angle_min = -0.5 * fov
        self.angle_inc = fov / (n_beams - 1)
        self.range_max = range_max

    def scan(self, x, y, yaw, noise=0.0, rng=None):
        ang = yaw + self.angle_min + np.arange(self.n) * self.angle_inc
        dx, dy = np.cos(ang), np.sin(ang)
        t = np.zeros(self.n)
        done = np.zeros(self.n, bool)
        for _ in range(200):                  # sphere tracing through the wall distance
            px, py = x + t * dx, y + t * dy
            dist = self.occ.wall_distance(px, py)
            hit = dist < 0.01
            done |= hit | (t > self.range_max)
            if done.all():
                break
            t = np.where(done, t, t + np.maximum(dist, 0.01))
        t = np.minimum(t, self.range_max)
        for B in self.boxes:
            for k in range(4):
                t = np.minimum(t, _ray_segments(x, y, dx, dy, B[k], B[(k + 1) % 4]))
        if noise > 0:
            rng = rng or np.random.default_rng()
            t = t + rng.normal(0.0, noise, self.n)
        return t
