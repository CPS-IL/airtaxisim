#!/usr/bin/env python3
import numpy as np
import rospy
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Vector3

### ROS Subscriber Callback ###
TRACKING_ARRAY_RECEIVED = None
def fnc_callback(msg):
    global TRACKING_ARRAY_RECEIVED
    TRACKING_ARRAY_RECEIVED = msg

P_gain = 0.05
I_gain = 0.000
D_gain = 0.001

TARGET_CLASS_INDEX = 2
FREQ_LOW_LEVEL = 10

# Initialize previous error and integral for PID
previous_error_x = 0
previous_error_y = 0
previous_error_z = 0

integral_x = 0
integral_y = 0
integral_z = 0

if __name__ == '__main__':

    # rosnode node initialization
    rospy.init_node('controller_node')

    # subscriber init.
    sub = rospy.Subscriber('/yolo_node/sort_mot_predictions', Float32MultiArray, fnc_callback)

    # publishers init.
    pub_tgt_box = rospy.Publisher('/controller_node/tgt_box_rcvd', Vector3, queue_size=10)
    pub_vel_cmd = rospy.Publisher('/controller_node/vel_cmd', Vector3, queue_size=10)

    # Running rate
    rate = rospy.Rate(FREQ_LOW_LEVEL)

    # msg init.
    tgt_box = Vector3()
    vel_cmd_tracking = Vector3()

    t_step = 0

    ##############################
    ### Instructions in a loop ###
    ##############################
    while not rospy.is_shutdown():
        t_step += 1

        if TRACKING_ARRAY_RECEIVED is not None:
            height = TRACKING_ARRAY_RECEIVED.layout.dim[0].size
            width = TRACKING_ARRAY_RECEIVED.layout.dim[1].size
            np_tracking = np.array(TRACKING_ARRAY_RECEIVED.data).reshape((height, width))

            if len(np_tracking) > 0:
                the_obj = np_tracking[-1]
                x1, y1, x2, y2 = the_obj[0:4]

                x_ctr = (x1 + x2) / 2
                y_ctr = (y1 + y2) / 2
                size = (x2 - x1) * (y2 - y1) / 1000

                CTR_X_POS = 224
                CTR_Y_POS = 224
                AREA_SIZE = 50

                ### Calculate error ###
                error_x = x_ctr - CTR_X_POS
                error_y = y_ctr - CTR_Y_POS
                error_z = size ** 0.5 - (AREA_SIZE) ** 0.7

                ### Integral Term ###
                integral_x += error_x
                integral_y += error_y
                integral_z += error_z

                ### Derivative Term ###
                derivative_x = error_x - previous_error_x
                derivative_y = error_y - previous_error_y
                derivative_z = error_z - previous_error_z

                ### Update previous errors ###
                previous_error_x = error_x
                previous_error_y = error_y
                previous_error_z = error_z

                ### PID control signals ###
                cmd_vx = P_gain * error_x + I_gain * integral_x + D_gain * derivative_x
                cmd_vy = P_gain * -error_y + I_gain * integral_y + D_gain * -derivative_y
                cmd_vz = .5 * error_z
                ### Clipping ###
                cmd_vx = np.clip(cmd_vx, -1, 1)
                cmd_vy = np.clip(cmd_vy, -1, 1)
                cmd_vz = np.clip(cmd_vz, -1, 1)

                vel_cmd_tracking.y = cmd_vx  # if target is at the right then generate positive cmd_vx
                vel_cmd_tracking.x = cmd_vy  # if target is at the above then generate positive cmd_vy
                vel_cmd_tracking.z = cmd_vz  # if target is small then generate positive cmd_vz

                # print('vx', cmd_vx, 'vy', cmd_vy, 'vz', cmd_vz)

            else:
                vel_cmd_tracking.x = 0
                vel_cmd_tracking.y = 0
                vel_cmd_tracking.z = 0

        ### Publish ###
        pub_vel_cmd.publish(vel_cmd_tracking)

        rate.sleep()
