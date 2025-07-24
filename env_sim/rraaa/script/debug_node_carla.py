#!/usr/bin/env python3
import glob
import os
import sys
import math
import argparse
import random
import time
import carla
import numpy as np
import torch, torchvision

from PIL import Image
from pathlib import Path

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

try:
    import pygame
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_q
    from pygame.locals import K_p
    from pygame.locals import K_r
    from pygame.locals import K_w
    from pygame.locals import K_s
    from pygame.locals import K_a
    from pygame.locals import K_d
    from pygame.locals import K_o
    from pygame.locals import K_l
    from pygame.locals import K_e
    from pygame.locals import K_z
    from pygame.locals import K_c
    from pygame.locals import K_f

except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

from tools.settings import LOOP_FREQ
from tools.sensors import SensorManager
from tools.display import DisplayManager
from tools.map import MapImage, Util
from tools.constants import PIXELS_PER_METER, DEVICE
from tools.map import MapManager

import rospy
from std_msgs.msg import Header, Float32, Bool
from geometry_msgs.msg import PoseStamped, Twist, Vector3
from sensor_msgs.msg import Image, CameraInfo, PointCloud2 
from gazebo_msgs.msg import ModelStates 
from nav_msgs.msg import Odometry
from scipy.spatial.transform import Rotation
import tf
import sensor_msgs.point_cloud2 as pcl2  #https://answers.ros.org/question/207071/how-to-fill-up-a-pointcloud-message-with-data-in-python





### TF from map to map_ned ###
QUAT_from_XYZ_to_NED = Rotation.from_euler('ZYX', np.array([90, 0, 180]), degrees=True).as_quat()  #x, y, z, w format

### Simple Tools ###
def substract_location(transform1, transform2):
    transform1.location.x = transform1.location.x - transform2.location.x
    transform1.location.y = transform1.location.y - transform2.location.y
    transform1.location.z = transform1.location.z - transform2.location.z
    return transform1

def calra_eulerangs_to_quaternion(rotation):

    roll = -(rotation.roll+00)/180*np.pi
    pitch= -(rotation.pitch+00)/180*np.pi
    yaw  = (rotation.yaw+00)/180*np.pi

    quaternion = tf.transformations.quaternion_from_euler(yaw, pitch, roll, 'rzyx')
    return quaternion


def run_carla_node(args, client):

    pygame.init()
    pygame.font.init()

    # Display Manager organize all the sensors an its display in a window
    # If can easily configure the grid and the total window size
    display_manager = DisplayManager(grid_size=[3, 4], window_size=[args.width, args.height])

    vehicle = None
    vehicle_list = []

    # rosnode node initialization
    rospy.init_node('carla_node')

    # subscriber init

    # publishers init.
    pub_lidar_point_cloud  = rospy.Publisher('/carla_node/lidar_point_cloud', PointCloud2, queue_size=1)

    # tf broadcaster init.
    br0 = tf.TransformBroadcaster()

    # Running rate
    rate=rospy.Rate(LOOP_FREQ)

    try:

        # Getting the world and
        world = client.get_world()
        town_map = world.get_map()
        original_settings = world.get_settings()
        settings = world.get_settings()
        settings.no_rendering_mode = False
        world.apply_settings(settings)
        print(settings)

        # Initialize the static map image using MapImage.
        mapimage = MapImage(world, town_map, PIXELS_PER_METER, False, False, False)

        # Instanciating the vehicle to which we attached the sensors
        bp = world.get_blueprint_library().filter('charger_2020')[0]
        bp.set_attribute('role_name', 'hero')
        vehicle = world.spawn_actor(bp, random.choice(world.get_map().get_spawn_points()))
        print('Ego vehicle ID:', vehicle.id)
        vehicle_list.append(vehicle)

        # Disable physics for Overriding #
        vehicle.set_autopilot(False)
        vehicle.set_simulate_physics(False)
        vehicle.set_enable_gravity(False)
        world.tick() # <--- this tick was necessary to move the vehicle to random initial position.
        initial_transform = vehicle.get_transform()


        # Map Manager
        MapManager(world, mapimage, display_manager, vehicle, display_pos=[0,3])

        # Blank displays
        SensorManager(world, display_manager, 'None', None, None, None, display_pos=[1, 0])  # To add black area.
        #SensorManager(world, display_manager, 'None', None, None, None, display_pos=[1, 2])  # To add black area.
                      
        # Then, SensorManager can be used to spawn RGBCamera, LiDARs and SemanticLiDARs as needed
        # and assign each of them to a grid position, 
        SensorManager(world, display_manager, 'RGBCamera', carla.Transform(carla.Location(x=0, z=2.4), carla.Rotation(yaw=-90)), 
                      vehicle, {}, display_pos=[0, 0])
        SensorManager(world, display_manager, 'RGBCamera', carla.Transform(carla.Location(x=0, z=2.4), carla.Rotation(yaw=+00)), 
                      vehicle, {}, display_pos=[0, 1])
        SensorManager(world, display_manager, 'RGBCamera', carla.Transform(carla.Location(x=0, z=2.4), carla.Rotation(yaw=+90)), 
                      vehicle, {}, display_pos=[0, 2])
        SensorManager(world, display_manager, 'RGBCamera', carla.Transform(carla.Location(x=0, z=2.4), carla.Rotation(yaw=180)), 
                      vehicle, {}, display_pos=[1, 1])
        
        SensorManager(world, display_manager, 'SemanticSegmenation', carla.Transform(carla.Location(x=0, z=40), carla.Rotation(pitch=-90)), 
                      vehicle, {}, display_pos=[1, 3])

        lidar_sensor_manager = SensorManager(world, display_manager, 'LiDAR', carla.Transform(carla.Location(x=0, z=3.0), carla.Rotation(yaw=+00)), 
                      vehicle, {'channels' : '64', 'range' : '100',  'points_per_second': '250000', 'rotation_frequency': '20'}, display_pos=[1, 2])

        

        #Simulation loop
        call_exit = False
        clock = pygame.time.Clock()

        start_tick = pygame.time.get_ticks()

        while not rospy.is_shutdown():

            # Carla Tick
            clock.tick(LOOP_FREQ)
            world.wait_for_tick() # asynchronous mode

            # Render received data
            display_manager.render()

            # Get Joystick Input and Translate and Rotate the Vehicle #
            veh_transform = vehicle.get_transform()
        
            ### Overiding vehicle position with GAZEBO Pose ###

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    call_exit = True
                elif event.type == pygame.KEYDOWN:
                    if event.key == K_ESCAPE:
                        call_exit = True
                        break
                    elif event.key == K_w:
                        veh_transform.location.x += 0.5
                    elif event.key == K_s:
                        veh_transform.location.x -= 0.5
                    elif event.key == K_a:
                        veh_transform.location.y += 0.5
                    elif event.key == K_d:
                        veh_transform.location.y -= 0.5
                    elif event.key == K_o:
                        veh_transform.location.z += 0.5
                    elif event.key == K_l:
                        veh_transform.location.z -= 0.5
                    elif event.key == K_q:
                        veh_transform.rotation.yaw   += 1
                    elif event.key == K_e:
                        veh_transform.rotation.yaw   -= 1
                    elif event.key == K_z:
                        veh_transform.rotation.roll   += 1
                    elif event.key == K_c:
                        veh_transform.rotation.roll   -= 1
                    elif event.key == K_r:
                        veh_transform.rotation.pitch   += 1
                    elif event.key == K_f:
                        veh_transform.rotation.pitch   -= 1

            
            vehicle.set_transform(veh_transform)

            print(veh_transform)
            print(initial_transform)


            # Render received data
            display_manager.render()

            ### Publish Sensor Data ###
            header = Header()
            header.stamp = rospy.Time.now()

            veh_transform = vehicle.get_transform()
            quat_from_euler = calra_eulerangs_to_quaternion(veh_transform.rotation)
            br0.sendTransform((veh_transform.location.x-initial_transform.location.x, veh_transform.location.y-initial_transform.location.y, veh_transform.location.z-initial_transform.location.z), quat_from_euler, rospy.Time.now(), 'vehicle', 'map')

            sensor_transform = lidar_sensor_manager.sensor.get_transform()
            # sensor_transform.rotation.roll = 0
            # sensor_transform.rotation.pitch = 0
            # sensor_transform.rotation.yaw = 0
            quat_from_euler = calra_eulerangs_to_quaternion(sensor_transform.rotation)
            br0.sendTransform((sensor_transform.location.x-initial_transform.location.x , sensor_transform.location.y-initial_transform.location.y, sensor_transform.location.z--initial_transform.location.z), quat_from_euler, rospy.Time.now(), 'sensor', "map")  


            header.frame_id = 'sensor'
            # points = lidar_sensor_manager.data[:,:3]  # The 4th componet is intensity.
            # print('points', points.shape)
            # scaled_polygon_pcl = pcl2.create_cloud_xyz32(header,lidar_sensor_manager.data[:,:3])
            pub_lidar_point_cloud.publish(pcl2.create_cloud_xyz32(header,lidar_sensor_manager.data[:,:3]))



            if call_exit:
                break

            
            #pub_camera_frame_left.publish(temp_float)

            # Sleep for Set Rate
            passed_ticks = pygame.time.get_ticks() - start_tick
            fps = clock.get_fps()
            if (passed_ticks > 5*1000) and abs(fps - LOOP_FREQ) > 0.1:
                print(fps, 'error in loop frequency', int(passed_ticks/1000))
            rate.sleep()

    finally:
        display_manager.destroy()
        client.apply_batch([carla.command.DestroyActor(x) for x in vehicle_list])
        world.apply_settings(original_settings)

def main():
    argparser = argparse.ArgumentParser(
        description='CARLA Sensor tutorial')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '--sync',
        action='store_true',
        help='Synchronous mode execution')
    argparser.set_defaults(sync=True)
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1200x900',
        help='window resolution (default: 1280x720)')

    args = argparser.parse_args()
    args.width, args.height = [int(x) for x in args.res.split('x')]

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(5.0)

        run_carla_node(args, client)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':

    try:
        main()
    except rospy.ROSInterruptException:
        pass