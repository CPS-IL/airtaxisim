#!/usr/bin/env python3

import rospy
import argparse
import numpy as np
import tf
import cv2
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import torch
from cv_bridge import CvBridge
from tools.a_star import AStarPlanner

COLOR_GRAY  = (85, 87, 83)
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)

torch_gray = torch.ByteTensor(COLOR_GRAY)
torch_white = torch.ByteTensor(COLOR_WHITE)
torch_black = torch.ByteTensor(COLOR_BLACK)



FREQ = 10


class Triangle:

    def __init__(self):

        self.vertice_angles = np.array([0, 2.5*np.pi/3,  -2.5*np.pi/3])
        self.length = 3

    def get_vertice_given_vehice_pos(self, trans, rot):

        x = trans[0]
        y = trans[1]
        roll, pitch, yaw = euler_from_quaternion(rot)

        angles = self.vertice_angles + yaw
        vertice = []
        for angle in angles:
            vertice.append([x+self.length*np.cos(angle), y+self.length*np.sin(angle)])

        return vertice

    def get_pixel_indice(self, trans, rot, map_info):

        vertice = self.get_vertice_given_vehice_pos(trans, rot)

        map_origin_x = map_info.origin.position.x
        map_origin_y = map_info.origin.position.y
        resolution = map_info.resolution

        indice_vertice = []
        for vertex in vertice:
            x = vertex[0]
            y = vertex[1]
            ind_x = int((x - map_origin_x)/resolution)
            ind_y = int((y - map_origin_y)/resolution)
            indice_vertice.append([ind_x, ind_y])

        return indice_vertice

    def draw_triangle(self, img, trans, rot, map_info):
        indice_vertice = self.get_pixel_indice(trans, rot, map_info)
        pts = np.array(indice_vertice, np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(img, [pts], (255, 0, 0))
        return img


def xy2xy_indice(x, y, origin_x, origin_y, resolution):
    ind_x = int((x - origin_x)/resolution)
    ind_y = int((y - origin_y)/resolution)
    return ind_x, ind_y



class PathPlanner:

    def __init__(self):

        self.occupancy_grid = None
        self.sub = rospy.Subscriber('/projected_map', OccupancyGrid, self.callback_occupancy_grid)
        self.sub1 = rospy.Subscriber('/display_node/target_pos', Twist, self.callback_target_msg)
        self.triangle_marker = Triangle()

        self.pub_map_image = rospy.Publisher('/path_planner/map_image', Image, queue_size=1)

         # a bridge from cv2 image to ROS image
        self.mybridge = CvBridge()

        self.target_xy = None

        self.astar_planner = AStarPlanner(rs=2)

    def callback_target_msg(self, msg):

        self.target_xy = (msg.angular.x, msg.angular.y)

    def callback_occupancy_grid(self, msg):

        width = msg.info.width
        height = msg.info.height
        array_2d = np.reshape(np.array(msg.data), (height, width))
        self.occupancy_grid = {'info':msg.info, 'array_2d':array_2d}

    def plan(self, tf_vehicle):


        ### Map Information ###
        map_info = self.occupancy_grid['info']
        array = self.occupancy_grid['array_2d']
        torch_array = torch.IntTensor(array)
        unkown_area   = (torch_array == -1).byte().unsqueeze(-1).repeat(1, 1, 3)
        occupied_area = (torch_array == 100).byte().unsqueeze(-1).repeat(1, 1, 3)
        free_area     = (torch_array == 0).byte().unsqueeze(-1).repeat(1, 1, 3)

        width = map_info.width
        height = map_info.height
        map_origin_x = map_info.origin.position.x
        map_origin_y = map_info.origin.position.y

        ### Vehicle Position ###
        trans, rot = tf_vehicle
        x_vehicle = trans[0]
        y_vehicle = trans[1]

        ind_x, ind_y = xy2xy_indice(x_vehicle, y_vehicle, map_origin_x, map_origin_y, map_info.resolution)



        trans, rot = tf_vehicle
        gx = int(self.target_xy[0])
        gy = int(self.target_xy[1])
        sx = ind_x
        sy = ind_y

        obs_map = (torch_array == 100).numpy()

        print(sx, sy, gx, gy)



        rx, ry = self.astar_planner.planning(sx, sy, gx, gy, obs_map.T)

        return rx, ry

    def plot_map(self, tf_vehicle):

        ### Map Information ###
        map_info = self.occupancy_grid['info']
        array = self.occupancy_grid['array_2d']
        torch_array = torch.IntTensor(array)
        unkown_area   = (torch_array == -1).byte().unsqueeze(-1).repeat(1, 1, 3)
        occupied_area = (torch_array == 100).byte().unsqueeze(-1).repeat(1, 1, 3)
        free_area     = (torch_array == 0).byte().unsqueeze(-1).repeat(1, 1, 3)

        width = map_info.width
        height = map_info.height
        map_origin_x = map_info.origin.position.x
        map_origin_y = map_info.origin.position.y

        ### Vehicle Position ###
        trans, rot = tf_vehicle
        x_vehicle = trans[0]
        y_vehicle = trans[1]

        ind_x, ind_y = xy2xy_indice(x_vehicle, y_vehicle, map_origin_x, map_origin_y, map_info.resolution)
        roll, pitch, yaw = euler_from_quaternion(rot)

        ### Visualization ###
        colored_map = torch.zeros_like(unkown_area)
        colored_map += unkown_area*torch_gray.unsqueeze(0).unsqueeze(0)
        colored_map += occupied_area*torch_black.unsqueeze(0).unsqueeze(0)
        colored_map += free_area*torch_white.unsqueeze(0).unsqueeze(0)
        image = colored_map.numpy()
        self.triangle_marker.draw_triangle(image, trans, rot, map_info)
        rx, ry = self.plan(tf_vehicle)
        points = np.array([rx, ry]).T
        pts = points.reshape(-1,1,2)
        image = cv2.polylines(image, [pts], isClosed=False, color=(0,0,255), thickness = 1)


        if self.target_xy is not None:

            x = int(self.target_xy[0])
            y = int(self.target_xy[1])
            color = (0, 0, 255)
            markerType = cv2.MARKER_CROSS
            markerSize = 15
            thickness = 2
            cv2.drawMarker(image, (x, y), color, markerType, markerSize, thickness)

        cv2_images_uint8 = cv2.flip(image, 0)


        cv2.imshow('map', cv2_images_uint8)

        image_message = self.mybridge.cv2_to_imgmsg(cv2_images_uint8, encoding="passthrough")
        self.pub_map_image.publish(image_message)





        cv2.waitKey(10)

        ##################
        ### TO DO LIST ###
        ##################

        '''
        1. Need to scale the map for visulization
        2. Add the marker for the vehicle
        3. Then, add the A Star Path Planning
        '''







        return 0


    def step(self, tf_vehicle):

        if self.occupancy_grid is not None:

            self.plot_map(tf_vehicle)

            # self.plan(tf_vehicle)






        return 0






def run_pathplanner(args):

    # rosnode node initialization
    rospy.init_node('pathplanner_node')

    # Running rate
    rate=rospy.Rate(FREQ)

    # subscriber init.
    tf_listener = tf.TransformListener()

    planner = PathPlanner()


    i = 0


    while not rospy.is_shutdown():

        try:
            tf_vehicle = tf_listener.lookupTransform('map', 'vehicle', rospy.Time(0))

            planner.step(tf_vehicle)








        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            continue

        # break






        # Sleep for Set Rate
        rate.sleep()

        #break

        # i = i + 1

        # if i > 5:
        #     break


def main():
    argparser = argparse.ArgumentParser(
        description='ROS DISPLAY NODE')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1600x800',
        help='window resolution (default: 800x400)')

    ### Traffic Setting ###
    args, unknown = argparser.parse_known_args()
    args.width, args.height = [int(x) for x in args.res.split('x')]

    try:
        run_pathplanner(args)
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':

    try:
        main()
    except rospy.ROSInterruptException:
        pass