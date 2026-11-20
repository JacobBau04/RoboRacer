"""state_assembler logic (no ROS): latency compensation of the particle-filter pose with EKF1 odometry.

  On a PF pose (laser pose in map, stamped with its scan time t_s):
      base_map(t_s)  = laser_to_rear_axle(pf_pose)
      odom_base(t_s) = EKF1 pose interpolated at t_s from a short buffer
      T_map<-odom    = base_map(t_s) ∘ odom_base(t_s)^-1
  On every EKF1 pose odom_base(now):
      compensation on  (config C): pose_now = T_map<-odom ∘ odom_base(now)
      compensation off (config B): pose_now = base_map(t_s)  (the latest PF pose, unmoved)
  Config C can also predict from the EKF1 stamp to the publish time with EKF1's v and yaw rate
  (estimate(..., v, w, dt_forward)), because EKF1 publishes on a timer up to one period after its
  last measurement.
"""
import math

from . import se2
from .pose_buffer import PoseBuffer


class Assembler:

    def __init__(self, lidar_offset=0.275, compensation=True, buffer_seconds=1.0, correction_tau=0.0):
        """correction_tau > 0: blend T_map<-odom toward each new PF-derived value with this time
        constant instead of jumping to it (PF frame-to-frame noise then does not reach the output
        as steps; odometry still moves the pose without lag)."""
        self.d = float(lidar_offset)
        self.tau = float(correction_tau)
        self._t_corr = None
        self.compensation = bool(compensation)
        self.buf = PoseBuffer(buffer_seconds)
        self.T_map_odom = None
        self.pf_base = None
        self.pf_stamp = None
        self.counts = {'pf': 0, 'pf_older_than_buffer': 0, 'pf_newer_than_odom': 0, 'pf_no_odom': 0}

    def add_odom(self, t, pose_odom):
        return self.buf.add(t, pose_odom)

    def add_pf(self, t_s, laser_pose_map):
        """Returns the lookup status ('ok', 'older', 'newer', 'empty')."""
        self.counts['pf'] += 1
        base_map = se2.laser_to_rear_axle(laser_pose_map, self.d)
        self.pf_base, self.pf_stamp = base_map, t_s
        odom_ts, status = self.buf.lookup(t_s)
        if status == 'empty':
            self.counts['pf_no_odom'] += 1
            return status
        if status == 'older':
            self.counts['pf_older_than_buffer'] += 1
        elif status == 'newer':
            self.counts['pf_newer_than_odom'] += 1
        T_new = se2.compose(base_map, se2.inverse(odom_ts))
        if self.T_map_odom is None or self.tau <= 0.0 or self._t_corr is None or t_s <= self._t_corr:
            self.T_map_odom = T_new
        else:
            k = 1.0 - math.exp(-(t_s - self._t_corr) / self.tau)
            # blend in the frame of the current estimate: rotation about the car, not the odom origin
            self.T_map_odom = self._blend(self.T_map_odom, T_new, k, odom_ts)
        self._t_corr = t_s
        return status

    @staticmethod
    def _blend(T_old, T_new, k, odom_ts):
        """Move a fraction k of the way from T_old to T_new, measured as the correction of the
        car's map pose at odom_ts (so heading corrections rotate about the car)."""
        p_old = se2.compose(T_old, odom_ts)
        p_new = se2.compose(T_new, odom_ts)
        p = se2.interpolate(p_old, p_new, k)
        return se2.compose(p, se2.inverse(odom_ts))

    def estimate(self, pose_odom_now, v=0.0, w=0.0, dt_forward=0.0):
        """Rear-axle pose in map for the EKF1 pose `pose_odom_now`, or None before the first PF pose.
        With compensation, dt_forward > 0 also moves it forward at (v, w) to the publish time."""
        if self.T_map_odom is None:
            return None
        if self.compensation:
            p = se2.compose(self.T_map_odom, pose_odom_now)
            return se2.integrate_twist(p, v, w, dt_forward) if dt_forward > 0.0 else p
        return self.pf_base
