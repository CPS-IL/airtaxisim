#!/usr/bin/env python3

import rospy

from std_msgs.msg import Float32, Bool, Header, String, Float32MultiArray, MultiArrayDimension
from geometry_msgs.msg import Twist, Pose, PoseStamped
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
from tf.transformations import quaternion_from_euler, euler_from_quaternion
import argparse

try:
    import pygame
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_q
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

from tools.display import DisplayManager
from tools.input import InputControl
from tools.constants import COLOR_BLACK, COLOR_WHITE
from tools.text import HelpTextManager, InfoTextManager
from tools.utils import pack_multiarray_ros_msg, ROSMsgMatrix, pack_df_from_multiarray_msg
import pandas as pd

from tools.a_star import AStarPlanner


### ROS Subscriber Callback ###
VEHICLE_STATE_RECEIVED = None
def fnc_callback(msg):
    global VEHICLE_STATE_RECEIVED
    VEHICLE_STATE_RECEIVED = msg



FREQ = 20

class InfoTextManager(object):

    def __init__(self, display_man, display_pos):
        pygame.font.init()
        self.font = pygame.font.Font(pygame.font.get_default_font(), 16)
        self.surface = None
        self.info_text = None
        self.display_man = display_man
        self.display_pos = display_pos
        self.display_man.add_sensor(self)
        self.offset = self.display_man.get_display_offset(self.display_pos)

        rospy.Subscriber('/carla_node/world_state', Float32MultiArray, self.callback_world_state)
        self.df_world_state = None
        rospy.Subscriber('/carla_node/vehicles_state', Float32MultiArray, self.callback_vehicle_state)
        self.df_world_state = None
        rospy.Subscriber('/jaxguam/pose', PoseStamped, self.callback_guam_pose)
        self.pose_guam_xyz = None
        self.pose_guam_euler_ypr = None

    def callback_world_state(self, msg):

        self.df_world_state = pack_df_from_multiarray_msg(msg)

    def callback_vehicle_state(self, msg):
        self.df_vehicle_state = pack_df_from_multiarray_msg(msg)

    def callback_guam_pose(self, msg):
        self.pose_guam_xyz = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        # msg.pose.orientation.x
        # msg.pose.orientation.y
        # msg.pose.orientation.z
        # msg.pose.orientation.w
        quaternion = (
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w)
        (yaw, pitch, roll) = euler_from_quaternion(quaternion) # unit rad
        self.pose_guam_euler_ypr = np.array([yaw*180/np.pi, pitch*180/np.pi, roll*180/np.pi])


    def tick(self):

        if self.df_world_state is not None and self.df_vehicle_state is not None and self.pose_guam_xyz is not None:

            self.info_text = [
                '',
                '   Sever:  % 2.1f FPS'  % (self.df_world_state.loc['world']['server_fps']),
                '',
                '   Client: % 2.1f FPS'  % (self.df_world_state.loc['world']['client_fps']),
                '',
                '   CARLA LinPosXYZ: % 2.1f;  % 2.1f;  % 2.1f; '  % (self.df_vehicle_state.loc['ego_vehicle']['x'], self.df_vehicle_state.loc['ego_vehicle']['y'], self.df_vehicle_state.loc['ego_vehicle']['z']),
                '   CARLA AngPosXYZ: % 2.1f;  % 2.1f;  % 2.1f; '  % (self.df_vehicle_state.loc['ego_vehicle']['yaw'], self.df_vehicle_state.loc['ego_vehicle']['pitch'], self.df_vehicle_state.loc['ego_vehicle']['roll']),
                '   GUAM LinPosXYZ: % 2.1f;  % 2.1f;  % 2.1f; '  % (self.pose_guam_xyz[0], self.pose_guam_xyz[1], self.pose_guam_xyz[2]),
                '   GUAM AngPosXYZ: % 2.1f;  % 2.1f;  % 2.1f; '  % (self.pose_guam_euler_ypr[0], self.pose_guam_euler_ypr[1], self.pose_guam_euler_ypr[2]),
                ]

            self.surface = pygame.Surface(self.display_man.get_display_size())
            self.surface.fill(COLOR_BLACK)

            h_offset, v_offset = (0, 0)
            for item in self.info_text:
                text_surface = self.font.render(item, True, COLOR_WHITE)
                v_offset += 18
                self.surface.blit(text_surface, (h_offset, v_offset))

        self.render()

    def render(self):
        if self.surface is not None:
            self.display_man.display.blit(self.surface, self.offset)
        else:
            self.surface = pygame.Surface(self.display_man.get_display_size())
            self.surface.fill(COLOR_BLACK)
            self.display_man.display.blit(self.surface, self.offset)



class ROSImageListenNRenderer:

    def __init__(self, display_man, sensor_type, sensor_options, display_pos, display_scale=1):
        self.surface = None
        self.display_man = display_man
        self.display_pos = display_pos
        self.subscriber = self.init_sensor(sensor_type, sensor_options)
        self.sensor_options = sensor_options
        self.display_man.add_sensor(self)
        self.data = None
        self.offset = self.display_man.get_display_offset(self.display_pos)
        self.scale = display_scale

    def init_sensor(self, sensor_type, sensor_options):
        if sensor_type == 'ROSSubimage':
            return rospy.Subscriber(sensor_options['ros_topic'], Image, self.save_ros_image)   # subscriber init.
        else:
            return None

    def save_ros_image(self, msg):
        IMAGE_RECEIVED = msg
        np_im = np.frombuffer(IMAGE_RECEIVED.data, dtype=np.uint8).reshape(IMAGE_RECEIVED.height, IMAGE_RECEIVED.width, -1)
        self.data = np.array(np_im)

    def render(self):

        if self.data is not None:
            width, height = self.display_man.get_display_size()
            self.surface = pygame.surfarray.make_surface(self.data.swapaxes(0, 1))
            self.surface = pygame.transform.smoothscale(self.surface, (self.scale*width,self.scale*height))
            self.display_man.display.blit(self.surface, self.offset)




def run_input_node(args):

    # msg init. the msg is to send out numpy array.

    matrix_msg = ROSMsgMatrix()

    # rosnode node initialization
    rospy.init_node('display_node')

    # Running rate
    rate=rospy.Rate(FREQ)

    # Getting the world and
    display_man = DisplayManager(grid_size=[3, 5], window_size=[args.width, args.height])

    ROSImageListenNRenderer(display_man, 'ROSSubimage', {'ros_topic':'/carla_node/cam_overview/image_raw'}, display_pos=[0, 4], display_scale=1)
    ROSImageListenNRenderer(display_man, 'ROSSubimage', {'ros_topic':'/carla_node/cam_front/image_raw'}, display_pos=[1, 3])
    ROSImageListenNRenderer(display_man, 'ROSSubimage', {'ros_topic':'/carla_node/cam_left/image_raw'}, display_pos=[1, 2])
    ROSImageListenNRenderer(display_man, 'ROSSubimage', {'ros_topic':'/carla_node/cam_right/image_raw'}, display_pos=[1, 4])
    ROSImageListenNRenderer(display_man, 'ROSSubimage', {'ros_topic':'/carla_node/cam_back/image_raw'}, display_pos=[2, 3])
    ROSImageListenNRenderer(display_man, 'ROSSubimage', {'ros_topic':'/carla_node/cam_up/image_raw'}, display_pos=[0, 3])
    ROSImageListenNRenderer(display_man, 'ROSSubimage', {'ros_topic':'/carla_node/cam_down/image_raw'}, display_pos=[0, 2])
    ROSImageListenNRenderer(display_man, 'ROSSubimage', {'ros_topic':'/yolo_node/sort_mot_frame'}, display_pos=[0, 0])
    ROSImageListenNRenderer(display_man, 'ROSSubimage', {'ros_topic':'/yolo_node/yolo_pred_frame'}, display_pos=[0, 1])

    map_listen_renderer = ROSImageListenNRenderer(display_man, 'ROSSubimage', {'ros_topic':'/path_planner/map_image'}, display_pos=[0, 0], display_scale=2)



    # subscriber init.

    # vehicle state
    pub_control_state   = rospy.Publisher('/display_node/control_cmd', Twist, queue_size=1)
    pub_target_position = rospy.Publisher('/display_node/target_pos', Twist, queue_size=1)

    infotext_manager = InfoTextManager(display_man, display_pos=[2, 0])

    input_control = InputControl()
    client_clock = pygame.time.Clock()


    while not rospy.is_shutdown():
        client_clock.tick_busy_loop(FREQ)
        display_man.render()
        input_control.tick(client_clock)

        #######################
        ### Subscribed Data ###
        #######################
        infotext_manager.tick()

        ###############
        ### Publish ###
        ###############
        twist_msg = Twist()

        twist_msg.linear.x = input_control._longitudinal_move_cmd
        twist_msg.linear.y = input_control._lateral_move_cmd
        twist_msg.linear.z = input_control._vertical_move_cmd

        twist_msg.angular.x = input_control._pitch_rate_cmd*.2
        twist_msg.angular.y = input_control._roll_rate_cmd*0.2
        twist_msg.angular.z = input_control._yaw_rate_cmd*0.2

        pub_control_state.publish(twist_msg)

        # print('pos_click', input_control.mouse_pos_click[0], input_control.mouse_pos_click[1])


        twist_msg_tgt = Twist()
        width, height = map_listen_renderer.display_man.get_display_size()
        twist_msg_tgt.linear.y = np.clip(input_control.mouse_pos[0] - map_listen_renderer.offset[0], 0, width)
        twist_msg_tgt.linear.x = np.clip(input_control.mouse_pos[1] - map_listen_renderer.offset[1], 0, height)
        twist_msg_tgt.angular.y = np.clip(input_control.mouse_pos_click[0] - map_listen_renderer.offset[0], 0, width)
        twist_msg_tgt.angular.x = np.clip(input_control.mouse_pos_click[1] - map_listen_renderer.offset[1], 0, height)
        pub_target_position.publish(twist_msg_tgt)


        # Sleep for Set Rate
        rate.sleep()


def main():
    argparser = argparse.ArgumentParser(
        description='ROS DISPLAY NODE')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1600x800',
        help='window resolution (default: 1600x800)')

    ### Traffic Setting ###
    args, unknown = argparser.parse_known_args()
    args.width, args.height = [int(x) for x in args.res.split('x')]

    try:
        run_input_node(args)
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':

    try:
        main()
    except rospy.ROSInterruptException:
        pass