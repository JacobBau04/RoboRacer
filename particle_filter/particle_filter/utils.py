import numpy as np

from std_msgs.msg import Header
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point, Pose, PoseStamped, PoseArray, Quaternion, PolygonStamped, Polygon, Point32, PoseWithCovarianceStamped, PointStamped
# stage1: tf_transformations replaced by the yaw-only math below (no extra dependency)
import math
# import tf2_ros
import time

class CircularArray(object):
    """ Simple implementation of a circular array.
        You can append to it any number of times but only "size" items will be kept
    """
    def __init__(self, size):
        self.arr = np.zeros(size)
        self.ind = 0
        self.num_els = 0

    def append(self, value):
        if self.num_els < self.arr.shape[0]:
            self.num_els += 1
        self.arr[self.ind] = value
        self.ind = (self.ind + 1) % self.arr.shape[0]

    def mean(self):
        return np.mean(self.arr[:self.num_els])

    def median(self):
        return np.median(self.arr[:self.num_els])

class Timer:
    """ Simple helper class to compute the rate at which something is called.
        
        "smoothing" determines the size of the underlying circular array, which averages
        out variations in call rate over time.

        use timer.tick() to record an event
        use timer.fps() to report the average event rate.
    """
    def __init__(self, smoothing):
        self.arr = CircularArray(smoothing)
        self.last_time = time.time()

    def tick(self):
        t = time.time()
        self.arr.append(1.0 / (t - self.last_time))
        self.last_time = t

    def fps(self):
        return self.arr.mean()

def angle_to_quaternion(angle):
    """Convert an angle in radians into a quaternion _message_."""
    q_out = Quaternion()
    q_out.x = 0.0
    q_out.y = 0.0
    q_out.z = math.sin(angle / 2.0)
    q_out.w = math.cos(angle / 2.0)
    return q_out

def quaternion_to_angle(q):
    """Convert a quaternion _message_ into an angle in radians.
    The angle represents the yaw.
    This is not just the z component of the quaternion."""
    x, y, z, w = q.x, q.y, q.z, q.w
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

def rotation_matrix(theta):
    ''' Creates a rotation matrix for the given angle in radians '''
    c, s = np.cos(theta), np.sin(theta)
    return np.matrix([[c, -s], [s, c]])

def particle_to_pose(particle):
    ''' Converts a particle in the form [x, y, theta] into a Pose object '''
    pose = Pose()
    pose.position.x = particle[0]
    pose.position.y = particle[1]
    pose.orientation = angle_to_quaternion(particle[2])
    return pose

def particles_to_poses(particles):
    ''' Converts a two dimensional array of particles into an array of Poses. 
        Particles can be a array like [[x0, y0, theta0], [x1, y1, theta1]...]
    '''
    return list(map(particle_to_pose, particles))

# DEPRECATED: should make the header inside the node now
# def make_header(frame_id, stamp=None):
#     ''' Creates a Header object for stamped ROS objects '''
#     if stamp == None:
#         stamp = rospy.Time.now()
#     header = Header()
#     header.stamp = stamp
#     header.frame_id = frame_id
#     return header

def map_to_world_slow(x,y,t,map_info):
    ''' Converts given (x,y,t) coordinates from the coordinate space of the map (pixels) into world coordinates (meters).
        Provide the MapMetaData object from a map message to specify the change in coordinates.
        *** Logical, but slow implementation, when you need a lot of coordinate conversions, use the map_to_world function
    ''' 
    scale = map_info.resolution
    angle = quaternion_to_angle(map_info.origin.orientation)
    rot = rotation_matrix(angle)
    trans = np.array([[map_info.origin.position.x],
                      [map_info.origin.position.y]])

    map_c = np.array([[x],
                      [y]])
    world = (rot*map_c) * scale + trans

    return world[0,0],world[1,0],t+angle

def map_to_world(poses, map_info):
    ''' Takes a two dimensional numpy array of poses:
            [[x0,y0,theta0],
             [x1,y1,theta1],
             [x2,y2,theta2],
                   ...     ]
        And converts them from map coordinate space (pixels) to world coordinate space (meters).
        - Conversion is done in place, so this function does not return anything.
        - Provide the MapMetaData object from a map message to specify the change in coordinates.
        - This implements the same computation as map_to_world_slow but vectorized and inlined
    '''

    scale = map_info.resolution
    angle = quaternion_to_angle(map_info.origin.orientation)

    # rotation
    c, s = np.cos(angle), np.sin(angle)
    # we need to store the x coordinates since they will be overwritten
    temp = np.copy(poses[:,0])
    poses[:,0] = c*poses[:,0] - s*poses[:,1]
    poses[:,1] = s*temp       + c*poses[:,1]

    # scale
    poses[:,:2] *= float(scale)

    # translate
    poses[:,0] += map_info.origin.position.x
    poses[:,1] += map_info.origin.position.y
    poses[:,2] += angle

def world_to_map(poses, map_info):
    ''' Takes a two dimensional numpy array of poses:
            [[x0,y0,theta0],
             [x1,y1,theta1],
             [x2,y2,theta2],
                   ...     ]
        And converts them from world coordinate space (meters) to world coordinate space (pixels).
        - Conversion is done in place, so this function does not return anything.
        - Provide the MapMetaData object from a map message to specify the change in coordinates.
        - This implements the same computation as world_to_map_slow but vectorized and inlined
        - You may have to transpose the returned x and y coordinates to directly index a pixel array
    '''
    scale = map_info.resolution
    angle = -quaternion_to_angle(map_info.origin.orientation)

    # translation
    poses[:,0] -= map_info.origin.position.x
    poses[:,1] -= map_info.origin.position.y

    # scale
    poses[:,:2] *= (1.0/float(scale))

    # rotation
    c, s = np.cos(angle), np.sin(angle)
    # we need to store the x coordinates since they will be overwritten
    temp = np.copy(poses[:,0])
    poses[:,0] = c*poses[:,0] - s*poses[:,1]
    poses[:,1] = s*temp       + c*poses[:,1]
    poses[:,2] += angle

def world_to_map_slow(x,y,t, map_info):
    ''' Converts given (x,y,t) coordinates from the coordinate space of the world (meters) into map coordinates (pixels).
        Provide the MapMetaData object from a map message to specify the change in coordinates.
        *** Logical, but slow implementation, when you need a lot of coordinate conversions, use the world_to_map function
    ''' 
    scale = map_info.resolution
    angle = quaternion_to_angle(map_info.origin.orientation)
    rot = rotation_matrix(-angle)
    trans = np.array([[map_info.origin.position.x],
                      [map_info.origin.position.y]])

    world = np.array([[x],
                      [y]])
    map_c = rot*((world - trans) / float(scale))
    return map_c[0,0],map_c[1,0],t-angle


# ---------------------------------------------------------------- stage1 additions
def wrap_angle(a):
    """Wrap to [-pi, pi)."""
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def se2_compose(a, b):
    c, s = math.cos(a[2]), math.sin(a[2])
    return np.array([a[0] + c * b[0] - s * b[1], a[1] + s * b[0] + c * b[1], wrap_angle(a[2] + b[2])])


def se2_inverse(a):
    c, s = math.cos(a[2]), math.sin(a[2])
    return np.array([-c * a[0] - s * a[1], s * a[0] - c * a[1], wrap_angle(-a[2])])


def se2_between(a, b):
    """b expressed in frame a."""
    return se2_compose(se2_inverse(a), b)


def se2_interpolate(p0, p1, f):
    return np.array([p0[0] + f * (p1[0] - p0[0]), p0[1] + f * (p1[1] - p0[1]),
                     wrap_angle(p0[2] + f * wrap_angle(p1[2] - p0[2]))])


def circular_weighted_mean_pose(particles, weights):
    """Weighted mean pose with the heading averaged on the circle (atan2 of mean sin/cos)."""
    x = float(np.dot(particles[:, 0], weights))
    y = float(np.dot(particles[:, 1], weights))
    th = math.atan2(float(np.dot(np.sin(particles[:, 2]), weights)),
                    float(np.dot(np.cos(particles[:, 2]), weights)))
    return np.array([x, y, th])
