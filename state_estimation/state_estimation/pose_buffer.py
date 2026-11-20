"""Stamped ring buffer of SE(2) poses with lookup and interpolation by stamp."""
from collections import deque

from . import se2


class PoseBuffer:
    """Keeps (stamp, pose, extra) entries in stamp order covering at least the last `horizon` seconds.

    lookup(t) returns (pose, status):
      'ok'     t inside the buffer, interpolated (linear x, y; shortest-angle theta)
      'older'  t before the oldest entry -> oldest pose
      'newer'  t after the newest entry  -> newest pose
      'empty'  no entries -> None
    """

    def __init__(self, horizon=1.0):
        self.horizon = float(horizon)
        self._buf = deque()

    def __len__(self):
        return len(self._buf)

    def add(self, stamp, pose, extra=None):
        """Append an entry. Out-of-order entries (stamp <= newest) are rejected (returns False)."""
        if self._buf and stamp <= self._buf[-1][0]:
            return False
        self._buf.append((float(stamp), tuple(pose), extra))
        # Drop the oldest entry only if the next one already covers newest - horizon, so any lookup
        # inside the horizon is always bracketed (interpolated, never clamped).
        cutoff = self._buf[-1][0] - self.horizon
        while len(self._buf) > 1 and self._buf[1][0] <= cutoff:
            self._buf.popleft()
        return True

    def newest(self):
        return self._buf[-1] if self._buf else None

    def oldest(self):
        return self._buf[0] if self._buf else None

    def lookup(self, t):
        if not self._buf:
            return None, 'empty'
        if t <= self._buf[0][0]:
            return self._buf[0][1], ('ok' if t == self._buf[0][0] else 'older')
        if t >= self._buf[-1][0]:
            return self._buf[-1][1], ('ok' if t == self._buf[-1][0] else 'newer')
        # binary search for the bracketing pair
        lo, hi = 0, len(self._buf) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if self._buf[mid][0] <= t:
                lo = mid
            else:
                hi = mid
        t0, p0, _ = self._buf[lo]
        t1, p1, _ = self._buf[hi]
        return se2.interpolate(p0, p1, (t - t0) / (t1 - t0)), 'ok'

    def lookup_extra(self, t):
        """Extra payload of the newest entry with stamp <= t (zero-order hold), or None."""
        best = None
        for s, _, e in self._buf:
            if s <= t:
                best = e
            else:
                break
        return best
