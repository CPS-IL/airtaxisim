import pygame
import pandas as pd
import numpy as np
from std_msgs.msg import Float32MultiArray        # See https://gist.github.com/jarvisschultz/7a886ed2714fac9f5226
from std_msgs.msg import MultiArrayDimension      # See http://docs.ros.org/api/std_msgs/html/msg/MultiArrayLayout.html
import tf

class FPSTimer(object):

    def __init__(self):
        self.server_clock = pygame.time.Clock()
        self.server_fps = None
        self.frame = None
        self.simulation_time = None

    def on_world_tick(self, timestamp):
        self.server_clock.tick()            
        self.server_fps = self.server_clock.get_fps()
        self.frame = timestamp.frame
        self.simulation_time = timestamp.elapsed_seconds


def pack_multiarray_ros_msg(msg_mat, np_array, row_label=None, col_label=None):
    msg_mat.layout.dim[0].size = np_array.shape[0]
    msg_mat.layout.dim[1].size = np_array.shape[1]
    msg_mat.layout.dim[0].stride = np_array.shape[0]*np_array.shape[1]
    msg_mat.layout.dim[1].stride = np_array.shape[1]
    msg_mat.layout.data_offset = 0
    msg_mat.data = np_array.flatten().tolist()
    if row_label is not None:
        msg_mat.layout.dim[0].label = row_label
    if col_label is not None:
        msg_mat.layout.dim[1].label = col_label
    return msg_mat

def pack_df_from_multiarray_msg(multiarray_msg):
    height = multiarray_msg.layout.dim[0].size
    width = multiarray_msg.layout.dim[1].size
    np_state_matrix = np.array(multiarray_msg.data).reshape((height, width))            
    column_index = pd.Index(multiarray_msg.layout.dim[1].label.split(','))
    row_index = pd.Index(multiarray_msg.layout.dim[0].label.split(','))
    df = pd.DataFrame(np_state_matrix, index = row_index, columns=column_index) 
    return df

def pack_np_matrix_from_multiarray_msg(multiarray_msg):
    height = multiarray_msg.layout.dim[0].size
    width = multiarray_msg.layout.dim[1].size
    np_matrix = np.array(multiarray_msg.data).reshape((height, width))            
    return np_matrix

def init_matrix_array_ros_msg():

    msg_mat = Float32MultiArray()
    msg_mat.layout.dim.append(MultiArrayDimension())
    msg_mat.layout.dim.append(MultiArrayDimension())
    msg_mat.layout.dim[0].label = "height"
    msg_mat.layout.dim[1].label = "width"

    return msg_mat


# a bridge from cv2 (np.uint8 image) image to ROS image
from cv_bridge import CvBridge
MYBRIDGE = CvBridge()
def pack_image_ros_msg(cv2_image, header, frame_id):
    img_ros = MYBRIDGE.cv2_to_imgmsg(cv2_image)
    header.frame_id = frame_id
    img_ros.header = header
    img_ros.encoding = "rgb8"
    return img_ros


class ROSMsgMatrix():
    def __init__(self):
        self.mat = Float32MultiArray()
        self.mat.layout.dim.append(MultiArrayDimension())
        self.mat.layout.dim.append(MultiArrayDimension())
        self.mat.layout.dim[0].label = "height"
        self.mat.layout.dim[1].label = "width"

def add_carla_rotations(rot0, rot1):
    
    rot0.pitch += rot1.pitch
    rot0.yaw   += rot1.yaw
    rot0.roll  += rot1.roll

    return rot0

def rad2deg(rad):
    return rad*180/np.pi

def deg2rad(deg):
    return deg*np.pi/180

def carla_transform_to_ros_xyz_quaternion(transform):
    
    location = transform.location
    rotation = transform.rotation
    x, y, z = (location.x, location.y, location.z)
    pitch, yaw, roll = (rotation.pitch, rotation.yaw, rotation.roll)
    quaternion = tf.transformations.quaternion_from_euler(deg2rad(-yaw), deg2rad(-pitch), deg2rad(roll), 'rzyx')
    
    return (x, -y, z), quaternion
