#!/usr/bin/env python3

import pygame
import numpy as np
import time
import carla
from tools.constants import COLOR_BLACK
import rospy
import weakref
import collections
import math
from geometry_msgs.msg import PoseStamped, Twist
from sensor_msgs.msg import Image
from tf.transformations import euler_from_quaternion


# ==============================================================================
# -- CollisionSensor -----------------------------------------------------------
# ==============================================================================


def get_actor_display_name(actor, truncate=250):
    """Method to get actor display name"""
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name

class CollisionSensor(object):
    """ Class for collision sensors"""

    def __init__(self, parent_actor, panic=False):   #, hud):
        """Constructor method"""
        self.sensor = None
        self.history = []
        self.panic = panic
        self._parent = parent_actor
        # self.hud = hud
        world = self._parent.get_world()
        blueprint = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(blueprint, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to
        # self to avoid circular reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

    def get_collision_history(self):
        """Gets the history of collisions"""
        history = collections.defaultdict(int)
        for frame, intensity in self.history:
            history[frame] += intensity
        return history

    @staticmethod
    def _on_collision(weak_self, event):
        """On collision method"""
        self = weak_self()
        if not self:
            return
        actor_type = get_actor_display_name(event.other_actor)
        # self.hud.notification('Collision with %r' % actor_type)
        # print('Collision with %r' % actor_type)
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
        self.history.append((event.frame, intensity))
        if len(self.history) > 4000:
            self.history.pop(0)
        if self.panic:
            assert False, "COLLISION, COLLISION"



class CustomTimer:
    def __init__(self):
        try:
            self.timer = time.perf_counter
        except AttributeError:
            self.timer = time.time

    def time(self):
        return self.timer()

class SensorManager:
    def __init__(self, world, sensor_type, transform, attached, sensor_options):
        self.surface = None
        self.world = world
        self.sensor = self.init_sensor(sensor_type, transform, attached, sensor_options)
        self.sensor_options = sensor_options
        self.timer = CustomTimer()

        self.time_processing = 0.0
        self.tics_processing = 0

        self.data = None

    def init_sensor(self, sensor_type, transform, attached, sensor_options):
        if sensor_type == 'RGBCamera':
            camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')

            for key in sensor_options:
                camera_bp.set_attribute(key, sensor_options[key])

            camera = self.world.spawn_actor(camera_bp, transform, attach_to=attached)
            camera.listen(self.save_rgb_image)

            return camera

        elif sensor_type == 'LiDAR':
            lidar_bp = self.world.get_blueprint_library().find('sensor.lidar.ray_cast')
            lidar_bp.set_attribute('range', '100')
            lidar_bp.set_attribute('horizontal_fov', '150')
            lidar_bp.set_attribute('dropoff_general_rate', lidar_bp.get_attribute('dropoff_general_rate').recommended_values[0])
            lidar_bp.set_attribute('dropoff_intensity_limit', lidar_bp.get_attribute('dropoff_intensity_limit').recommended_values[0])
            lidar_bp.set_attribute('dropoff_zero_intensity', lidar_bp.get_attribute('dropoff_zero_intensity').recommended_values[0])

            for key in sensor_options:
                lidar_bp.set_attribute(key, sensor_options[key])

            lidar = self.world.spawn_actor(lidar_bp, transform, attach_to=attached)

            lidar.listen(self.save_lidar_points)

            return lidar

        '''

        if sensor_type == 'SemanticSegmenation':
            semantic_segmentation_bp = self.world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
            disp_size = self.display_man.get_display_size()
            semantic_segmentation_bp.set_attribute('image_size_x', str(disp_size[0]))
            semantic_segmentation_bp.set_attribute('image_size_y', str(disp_size[1]))

            for key in sensor_options:
                semantic_segmentation_bp.set_attribute(key, sensor_options[key])

            camera = self.world.spawn_actor(semantic_segmentation_bp, transform, attach_to=attached)
            camera.listen(self.save_semantic_segmentation_image)   ### This assigns a callback function!

            return camera



        elif sensor_type == 'SemanticLiDAR':
            lidar_bp = self.world.get_blueprint_library().find('sensor.lidar.ray_cast_semantic')
            lidar_bp.set_attribute('range', '100')

            for key in sensor_options:
                lidar_bp.set_attribute(key, sensor_options[key])

            lidar = self.world.spawn_actor(lidar_bp, transform, attach_to=attached)

            lidar.listen(self.save_semanticlidar_image)

            return lidar

        elif sensor_type == "Radar":
            radar_bp = self.world.get_blueprint_library().find('sensor.other.radar')
            for key in sensor_options:
                radar_bp.set_attribute(key, sensor_options[key])

            radar = self.world.spawn_actor(radar_bp, transform, attach_to=attached)
            radar.listen(self.save_radar_image)

            return radar
        '''

        if sensor_type == 'None':
            return None
        else:
            return None


    def get_sensor(self):
        return self.sensor


    def save_rgb_image(self, image):
        t_start = self.timer.time()

        image.convert(carla.ColorConverter.Raw)
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        self.data = array

        t_end = self.timer.time()
        self.time_processing += (t_end-t_start)
        self.tics_processing += 1

    def save_lidar_points(self, image):
        t_start = self.timer.time()

        points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
        points = np.reshape(points, (int(points.shape[0] / 4), 4))# This lidar data has shape of (n_points, 4) where X, Y, Z, I acount for the 4 dimensions.
        self.data = points #Here, "I" referes to intensity. See https://carla.readthedocs.io/en/latest/ref_sensors/#lidar-sensor

        t_end = self.timer.time()
        self.time_processing += (t_end-t_start)
        self.tics_processing += 1

    '''
    def save_semantic_segmentation_image(self, image):
        t_start = self.timer.time()

        image.convert(carla.ColorConverter.CityScapesPalette)
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]

        self.array = array
        self.image = array.swapaxes(0, 1)

        t_end = self.timer.time()
        self.time_processing += (t_end-t_start)
        self.tics_processing += 1
    '''

class GuamVelocitySensor:

    def __init__(self):
        self.sub = rospy.Subscriber('/guam/velocity', Twist, self.on_new_msg)
        self.linear = carla.Vector3D()
        self.angular = carla.Vector3D()
        self.ready = False

    def __str__(self):
        return "x:{}, y:{}, z:{}, roll:{}, pitch:{}, yaw:{}".format(
            self.linear.x, self.linear.y, self.linear.z, self.angular.x, self.angular.y, self.angular.z)

    def on_new_msg(self, msg):
        if not self.ready:
            self.ready = True
        self.linear.x  = msg.linear.x
        self.linear.y  = msg.linear.y
        self.linear.z  = msg.linear.z
        self.angular.x = msg.angular.x
        self.angular.y = msg.angular.y
        self.angular.z = msg.angular.z

    def get_vel(self):
        return self.linear, self.angular

    def is_ready(self):
        return self.ready