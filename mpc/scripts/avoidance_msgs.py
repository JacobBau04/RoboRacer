"""Messages between obstacle_avoidance_node.py and mpc_node.py, and the waypoint splice.

Both are std_msgs/Float64MultiArray (no custom message package needed).

Plan:      [1, plan_id, i_start, i_end, stop, n, x[n], y[n], yaw[n], v[n]]
           The detour replaces raceline points i_start..i_end (inclusive, wrapping past the
           lap end). Its points are spaced like the raceline (uniform arc length), because
           the MPC's reference generation assumes uniform spacing. n = 0: follow the raceline.
Obstacles: [1, n_obs, n_obs x (id, s_min, s_len, d_min, d_max), n_pts, x[n_pts], y[n_pts]]
           The points sample each confirmed box (outline and inside) in the map frame.
"""
import numpy as np

VERSION = 1.0
OBS_FIELDS = 5


def resample_plan(track, plan):
    """Plan (from detour_planner) -> (i_start, i_end, (n, 4) array x, y, yaw, v)."""
    xy, v = plan.xy, plan.v
    seg = np.diff(xy, axis=0)
    l = np.concatenate([[0.0], np.cumsum(np.hypot(seg[:, 0], seg[:, 1]))])
    n = max(2, int(round(l[-1] / track.ds)) + 1)
    lk = np.linspace(0.0, l[-1], n)
    x = np.interp(lk, l, xy[:, 0])
    y = np.interp(lk, l, xy[:, 1])
    vv = np.interp(lk, l, v)
    fwd = np.gradient(np.column_stack((x, y)), axis=0)
    yaw = np.arctan2(fwd[:, 1], fwd[:, 0])
    i_start = int(track.index_at(plan.s_car + plan.u[0] + 1e-6)) % track.n
    i_end = int(track.index_at(plan.s_car + plan.u[-1] + 1e-6)) % track.n
    return i_start, i_end, np.column_stack((x, y, yaw, vv))


def encode_plan(plan_id, i_start=-1, i_end=-1, pts=None, stop=False):
    if pts is None or len(pts) == 0:
        return [VERSION, float(plan_id), -1.0, -1.0, 0.0, 0.0]
    pts = np.asarray(pts, float)
    return [VERSION, float(plan_id), float(i_start), float(i_end), float(stop), float(len(pts))] \
        + pts[:, 0].tolist() + pts[:, 1].tolist() + pts[:, 2].tolist() + pts[:, 3].tolist()


def decode_plan(data):
    """-> dict(plan_id, i_start, i_end, stop, pts (n, 4) or None)."""
    data = np.asarray(data, float)
    if len(data) < 6 or data[0] != VERSION:
        raise ValueError('bad plan message')
    n = int(data[5])
    out = {'plan_id': int(data[1]), 'i_start': int(data[2]), 'i_end': int(data[3]),
           'stop': bool(data[4]), 'pts': None}
    if n > 0:
        if len(data) != 6 + 4 * n:
            raise ValueError('bad plan length')
        out['pts'] = data[6:].reshape(4, n).T.copy()
    return out


def splice(raceline, i_start, i_end, pts):
    """Raceline (N, 4) with points i_start..i_end replaced by pts.

    The result starts right after the detour, so its first segment is a regular raceline
    segment (the MPC takes the waypoint spacing from the first segment)."""
    N = len(raceline)
    count = (i_end - i_start) % N + 1
    if pts is None or count >= N - 2:
        return raceline
    keep = (i_end + 1 + np.arange(N - count)) % N
    return np.vstack([raceline[keep], pts])


def obstacle_samples(track, ob, step=0.08):
    """Points covering a box (s_min, s_len, d_min, d_max) in the map frame."""
    nu = max(2, int(np.ceil(ob.s_len / step)) + 1)
    nd = max(2, int(np.ceil((ob.d_max - ob.d_min) / step)) + 1)
    uu, dd = np.meshgrid(np.linspace(0.0, ob.s_len, nu), np.linspace(ob.d_min, ob.d_max, nd))
    return track.frenet_to_xy(ob.s_min + uu.ravel(), dd.ravel())


def encode_obstacles(track, obstacles):
    head = [VERSION, float(len(obstacles))]
    pts = [obstacle_samples(track, ob) for ob in obstacles]
    for ob in obstacles:
        head += [float(ob.id), ob.s_min, ob.s_len, ob.d_min, ob.d_max]
    allp = np.vstack(pts) if pts else np.zeros((0, 2))
    return head + [float(len(allp))] + allp[:, 0].tolist() + allp[:, 1].tolist()


def decode_obstacles(data):
    """-> (boxes (n, 5) array id, s_min, s_len, d_min, d_max ; points (m, 2))."""
    data = np.asarray(data, float)
    if len(data) < 3 or data[0] != VERSION:
        raise ValueError('bad obstacles message')
    n = int(data[1])
    boxes = data[2:2 + OBS_FIELDS * n].reshape(n, OBS_FIELDS)
    k = 2 + OBS_FIELDS * n
    m = int(data[k])
    pts = data[k + 1:k + 1 + 2 * m].reshape(2, m).T.copy()
    return boxes, pts
