"""viz_tf: DISPLAY-ONLY map -> odom TF for the car (C_car runs), so Foxglove/RViz can draw the LiDAR scan
and the car's frames on the map.

On the car, vesc_to_odom owns odom -> base_link, so EKF1 and state_assembler don't broadcast TF
(config/*_car.yaml). That leaves `map` disconnected from base_link/laser. This node closes the tree:

    map -> odom = (map -> base_link from /state_estimate) * inverse(odom -> base_link from vesc_to_odom)

so map -> odom -> base_link -> laser puts the car where stage 1 says it is.

Also for the race stack (A_car view): with the TF-off copy of the race PF (particle_filter_notf), run
  ros2 run state_estimation viz_tf --ros-args -p state_topic:=/pf/pose/odom -p pose_to_base_x:=-0.27
(/pf/pose/odom is the LASER pose; pose_to_base_x moves it to base_link, 0.27 m behind).
Nothing reads this TF for estimation or control. Never run it next to the ORIGINAL race PF, which publishes its own
'/map' -> '/laser' TF; with the TF-off copy it is the only map TF. Run only one viz_tf at a time.

  /state_estimate (nav_msgs/Odometry, map, rear axle) + TF odom -> base_link  ->  TF map -> odom
"""
import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import Buffer, TransformBroadcaster, TransformException, TransformListener

from . import se2


class VizTf(Node):

    def __init__(self):
        super().__init__('viz_tf')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('state_topic', '/state_estimate')
        self.declare_parameter('pose_to_base_x', 0.0)   # base_link in the pose's frame (-0.27: laser pose -> rear axle)
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.map_frame, self.odom_frame, self.base_frame = g('map_frame'), g('odom_frame'), g('base_frame')
        self.to_base = (float(g('pose_to_base_x')), 0.0, 0.0)
        self.buf = Buffer(cache_time=Duration(seconds=2.0))
        # own thread (TUNING_LOG O1: a listener sharing the executor starved and fell seconds behind)
        self.listener = TransformListener(self.buf, self, spin_thread=True)
        self.br = TransformBroadcaster(self)
        self.create_subscription(Odometry, g('state_topic'), self.on_state, 10)
        self.warned = False

    def on_state(self, msg):
        stamp = Time.from_msg(msg.header.stamp)
        try:     # odom -> base_link at the estimate's time; else the newest one (display only)
            t = self.buf.lookup_transform(self.odom_frame, self.base_frame, stamp)
        except TransformException:
            try:
                t = self.buf.lookup_transform(self.odom_frame, self.base_frame, Time())
            except TransformException as e:
                if not self.warned:
                    self.get_logger().warn(f'no {self.odom_frame} -> {self.base_frame} TF yet: {e}')
                    self.warned = True
                return
        odom_base = (t.transform.translation.x, t.transform.translation.y,
                     se2.yaw_from_quat(t.transform.rotation))
        p = msg.pose.pose
        map_base = se2.compose((p.position.x, p.position.y, se2.yaw_from_quat(p.orientation)), self.to_base)
        T = se2.compose(map_base, se2.inverse(odom_base))
        out = TransformStamped()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = self.map_frame
        out.child_frame_id = self.odom_frame
        out.transform.translation.x, out.transform.translation.y = float(T[0]), float(T[1])
        r = out.transform.rotation
        r.x, r.y, r.z, r.w = se2.quat_from_yaw(T[2])
        self.br.sendTransform(out)


def main():
    rclpy.init()
    rclpy.spin(VizTf())
