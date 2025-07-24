import pandas as pd
import numpy as np
from std_msgs.msg import Float32MultiArray        # See https://gist.github.com/jarvisschultz/7a886ed2714fac9f5226
from std_msgs.msg import MultiArrayDimension      # See http://docs.ros.org/api/std_msgs/html/msg/MultiArrayLayout.html

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