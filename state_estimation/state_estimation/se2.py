"""SE(2) pose math. Poses are (x, y, theta) tuples/arrays; theta in radians.

compose(a, b) = a ⊕ b: pose b expressed in frame a, mapped to a's parent frame.
All angle outputs are wrapped to [-pi, pi).
"""
import math

import numpy as np


def wrap(a):
    """Wrap an angle (float or array) to [-pi, pi)."""
    return (np.asarray(a) + math.pi) % (2.0 * math.pi) - math.pi if isinstance(a, np.ndarray) \
        else (a + math.pi) % (2.0 * math.pi) - math.pi


def compose(a, b):
    """T_a ∘ T_b."""
    ax, ay, at = a
    bx, by, bt = b
    c, s = math.cos(at), math.sin(at)
    return (ax + c * bx - s * by, ay + s * bx + c * by, wrap(at + bt))


def inverse(a):
    """T_a^{-1}."""
    ax, ay, at = a
    c, s = math.cos(at), math.sin(at)
    return (-c * ax - s * ay, s * ax - c * ay, wrap(-at))


def between(a, b):
    """Relative pose a^{-1} ∘ b (b expressed in frame a)."""
    return compose(inverse(a), b)


def interpolate(p0, p1, f):
    """Linear in x, y; shortest-angle in theta. f=0 -> p0, f=1 -> p1 (f may be outside [0, 1])."""
    dth = wrap(p1[2] - p0[2])
    return (p0[0] + f * (p1[0] - p0[0]), p0[1] + f * (p1[1] - p0[1]), wrap(p0[2] + f * dth))


def shift_forward(pose, d):
    """Move a pose d metres forward along its own heading (d < 0 moves backward)."""
    x, y, t = pose
    return (x + d * math.cos(t), y + d * math.sin(t), t)


def laser_to_rear_axle(pose_laser, d):
    """Pose of the rear axle given the pose of a LiDAR mounted d metres ahead of it (no yaw offset)."""
    return shift_forward(pose_laser, -d)


def rear_axle_to_laser(pose_ra, d):
    return shift_forward(pose_ra, d)


def cog_velocity_to_rear_axle(vx_cog, vy_cog, yaw_rate, lr):
    """Body-frame velocity of the rear axle from the CoG velocity (rigid body, rear axle lr behind CoG).
    v_ra = v_cog + ω × r with r = (-lr, 0): vx unchanged, vy - lr*ω."""
    return vx_cog, vy_cog - lr * yaw_rate


def yaw_from_quat(q):
    """Yaw from a geometry_msgs Quaternion-like object (x, y, z, w attributes)."""
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def quat_from_yaw(theta):
    """(x, y, z, w) quaternion for a pure yaw rotation."""
    return (0.0, 0.0, math.sin(theta / 2.0), math.cos(theta / 2.0))


def integrate_twist(pose, v, w, dt):
    """Move a pose forward by dt seconds at constant body speed v and yaw rate w (exact arc)."""
    dth = w * dt
    if abs(dth) < 1e-9:
        step = (v * dt, 0.0, 0.0)
    else:
        r = v / w
        step = (r * math.sin(dth), r * (1.0 - math.cos(dth)), dth)
    return compose(pose, step)
