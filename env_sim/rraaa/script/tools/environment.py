import os
import carla
import numpy as np
import pygame
import random
import rospy
import time
import tf

from std_msgs.msg import Float32, Bool, Header, String, Float32MultiArray, MultiArrayDimension
from geometry_msgs.msg import PoseStamped, Twist, Vector3
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
import sensor_msgs.point_cloud2 as pcl2  #https://answers.ros.org/question/207071/how-to-fill-up-a-pointcloud-message-with-data-in-python

from tools.sensors import CollisionSensor, SensorManager
from tools.utils import FPSTimer, pack_multiarray_ros_msg, pack_df_from_multiarray_msg, pack_image_ros_msg, ROSMsgMatrix
from tools.utils import add_carla_rotations, rad2deg, deg2rad, carla_transform_to_ros_xyz_quaternion

from scipy.spatial.transform import Rotation
QUAT_from_XYZ_to_NED = Rotation.from_euler('ZYX', np.array([90, 0, 180]), degrees=True).as_quat()  #x, y, z, w format

from geometry_msgs.msg import Pose, PoseStamped
from tf.transformations import quaternion_from_euler, euler_from_quaternion

from utils.config import log, write_shared_tmp_file
from utils import constants

import rosbag

class ComplexObject():
    def __init__(self, world, records_dir, object_name):
        self.world = world
        self.records_dir = records_dir
        self.object_name = object_name

        # Subscribe to the ROS topic which publishes whether the simulation has started
        self.sim_started = False
        rospy.Subscriber("/sim_start/started", Bool, self.sim_start_callback)

        # Get the directory of the pre-recorded path
        self.object_dir = self.get_dir()
        
        # Pose topic name
        self.pose_topic = "/jaxguam/pose"

        # Load the rosbag
        relevant_topics = [self.pose_topic]
        self.bag = self.get_bag(relevant_topics)

        # Retrieve the starting time of the recorded path
        self.recording_start_time = self.get_start_time()

        # Spawn the adversarial object
        self.object = self.spawn_object()

    def spawn_object(self):
        # Retrieve the blueprint
        adv_obj = self.world.get_blueprint_library().filter("uli_cora")[0]

        # Retrieve the starting pose
        start_pose = self.get_closest_record(self.recording_start_time)[1]

        # Spawn the actor
        adv_obj = self.world.spawn_actor(
            adv_obj,
            carla.Transform(
                location=carla.Location(
                    x=start_pose.pose.position.x,
                    y=start_pose.pose.position.y,
                    z=start_pose.pose.position.z
                ),
                rotation=carla.Rotation(
                    roll=0,
                    pitch=0,
                    yaw=0
                )
            )
        )

        adv_obj.set_autopilot(False)
        adv_obj.set_enable_gravity(False)
        self.initial_transform = adv_obj.get_transform()

        return adv_obj

    def get_dir(self):
        object_dir = os.path.join(self.records_dir, self.object_name, 'tmp')
        assert os.path.exists(object_dir)
        return object_dir
    
    def get_bag(self, relevant_topics):
        # Rosbag path
        bag_path = os.path.join(self.object_dir, "recorded_topics.bag")
        assert os.path.exists(bag_path), f"Rosbag not found at: {bag_path}"

        # Dictionary to store messages by topic
        bag_data = {topic: [] for topic in relevant_topics}

        # Load the data
        with rosbag.Bag(bag_path) as bag:
            for topic, msg, t in bag.read_messages():
                if topic in relevant_topics:
                    bag_data[topic].append((t, msg))  # Save timestamp and message
        
        return bag_data
    
    def get_start_time(self):
        assert self.bag is not None, f"No rosbag loaded!"
        assert self.pose_topic is not None, f"No pose topic!"

        # Retrieve the pose data topic
        pose_bag = self.bag[self.pose_topic]
        
        return min([pose_info[0] for pose_info in pose_bag])
    
    def sim_start_callback(self, msg):
        if msg.data and not self.sim_started:
            self.sim_started = True
            self.sim_start_time = rospy.Time.now()

    def update(self):
        # Retrieve the current time
        curr_time = rospy.Time().now()

        # Find the time delta
        time_delta = curr_time - self.sim_start_time

        # Add this delta to the start time of the recording
        recording_time = self.recording_start_time + time_delta

        # Find the closest recording
        curr_pose = self.get_closest_record(recording_time)[1]

        # Retrieve the position
        X, Y, Z = (curr_pose.pose.position.x, curr_pose.pose.position.y, curr_pose.pose.position.z)

        # Transform the pose into CARLA-readable format
        quaternion = (
            curr_pose.pose.orientation.x,
            curr_pose.pose.orientation.y,
            curr_pose.pose.orientation.z,
            curr_pose.pose.orientation.w)
        (yaw, pitch, roll) = euler_from_quaternion(quaternion) # unit rad

        # Retrieve the transform of the object
        transform = self.object.get_transform()

        # Update the transform
        transform.location = carla.Location(X, Y, Z)
        rotation = carla.Rotation(rad2deg(-pitch), rad2deg(-yaw), rad2deg(roll))  # The constructor method follows a specific order of declaration: (pitch, yaw, roll), which corresponds to (Y-rotation,Z-rotation,X-rotation).
        transform.rotation = add_carla_rotations(rotation, self.initial_transform.rotation)
        self.object.set_transform(transform)
    
    def get_closest_record(self, ros_time):
        return min(self.bag[self.pose_topic], key=lambda x: abs((x[0] - ros_time).to_sec()))

class Environment():

    def __init__(self, args, client, config):

        self.args = args
        self.client = client
        self.config = config
        self.world = client.get_world()
        # self.world = client.load_world(config['map'])


        ### Setting the world ###
        self.original_settings = self.world.get_settings()
        settings = self.world.get_settings()
        self.traffic_manager = self.client.get_trafficmanager(args.tm_port)
        if args.asynch:
            self.traffic_manager.set_synchronous_mode(False)
            settings.synchronous_mode = False
        else:
            self.traffic_manager.set_synchronous_mode(True)
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = .1
        settings.actor_active_distance  = 500
        self.world.apply_settings(settings)

        ### Initiate states and blank messages ###
        self._autopilot_on = False
        self.collision_n_count = 0
        self.df_msg_input_display    = None
        self.df_msg_tracking_control = None

        ### ROS msg publisher init. ###
        self.pub_vehicles_state = rospy.Publisher('/carla_node/vehicles_state', Float32MultiArray, queue_size=1)
        self.pub_world_state    = rospy.Publisher('/carla_node/world_state', Float32MultiArray, queue_size=1)

        self.pub_camera_frame_left  = rospy.Publisher('/carla_node/cam_left/image_raw', Image, queue_size=1)
        self.pub_camera_frame_front = rospy.Publisher('/carla_node/cam_front/image_raw',Image, queue_size=1)
        self.pub_camera_frame_right = rospy.Publisher('/carla_node/cam_right/image_raw',Image, queue_size=1)
        self.pub_camera_frame_back  = rospy.Publisher('/carla_node/cam_back/image_raw', Image, queue_size=1)
        self.pub_camera_frame_up    = rospy.Publisher('/carla_node/cam_up/image_raw',   Image, queue_size=1)
        self.pub_camera_frame_down  = rospy.Publisher('/carla_node/cam_down/image_raw', Image, queue_size=1)

        self.pub_camera_frame_overview = rospy.Publisher('/carla_node/cam_overview/image_raw', Image, queue_size=1)
        # self.pub_lidar_point_cloud = rospy.Publisher('/carla_node/lidar_point_cloud', PointCloud2, queue_size=1)
        self.pub_initial_transform = rospy.Publisher('/carla_node/initial_transform',  Twist, queue_size=1)
        # tf broadcaster init.
        self.tf_broadcaster = tf.TransformBroadcaster()

        ### Multiple lidars are published. ###
        self.pub_lidar_point_cloud_down     = rospy.Publisher('/carla_node/lidar_point_cloud_down',     PointCloud2, queue_size=10)
        self.pub_lidar_point_cloud_up       = rospy.Publisher('/carla_node/lidar_point_cloud_up',       PointCloud2, queue_size=10)
        self.pub_lidar_point_cloud_left     = rospy.Publisher('/carla_node/lidar_point_cloud_left',     PointCloud2, queue_size=10)
        self.pub_lidar_point_cloud_right    = rospy.Publisher('/carla_node/lidar_point_cloud_right',    PointCloud2, queue_size=10)
        self.pub_lidar_point_cloud_back     = rospy.Publisher('/carla_node/lidar_point_cloud_back',     PointCloud2, queue_size=10)
        self.pub_lidar_point_cloud_forward  = rospy.Publisher('/carla_node/lidar_point_cloud_forward',  PointCloud2, queue_size=10)

        ### ROS msg Subscriber init. ###
        self.vehicle_type = self.config['ego_vehicle']['type']
        self.sub_jax_guam_pose = rospy.Subscriber(f'/{self.vehicle_type}/pose', PoseStamped, self.callback_jax_guam_pose)

        ### Timer for frames per second (FPS) ###
        self.fps_timer = FPSTimer()
        self.client_clock = pygame.time.Clock()
        self.world.on_tick(self.fps_timer.on_world_tick)

        self.msg_mat = ROSMsgMatrix()

        self.spectator = self.world.get_spectator()

        ### Start the environment ###
        self.start()
        if self.config['optional_features']['record_static_obstacles']['enabled']:
            self.save_obstacles()
        log.info("Finished setting up environment.")

    def save_obstacles(self):
        # TODO: need to extend the list of static obstacles
        bounding_boxes = self.world.get_level_bbs()
        # bounding_boxes = self.world.get_level_bbs(carla.libcarla.CityObjectLabel.Buildings)

        if self.config['optional_features']['record_static_obstacles']['frame'] == 'vehicle':
            ego_transform = self.ego_vehicle.get_transform()
            ego_location = ego_transform.location
            ego_rotation = ego_transform.rotation

        bbs = []
        for bbox in bounding_boxes:

            corners_world = np.array([corner for corner in bbox.get_world_vertices(carla.Transform())])

            if self.config['optional_features']['record_static_obstacles']['frame'] == 'vehicle':
                corners_ego = []
                for corner in corners_world:
                    relative_location = np.array([
                        corner.x - ego_location.x,
                        corner.y - ego_location.y,
                        corner.z - ego_location.z
                    ])

                    yaw = -np.radians(ego_rotation.yaw)
                    rotation_matrix = np.array([
                        [np.cos(yaw), -np.sin(yaw), 0],
                        [np.sin(yaw),  np.cos(yaw), 0],
                        [0,            0,           1]
                    ])

                    rotated_vector = np.dot(rotation_matrix, relative_location)
                    corners_ego.append(rotated_vector)
                corners = np.array(corners_ego)
            else:
                corners = np.array(
                    [[corner.x, corner.y, corner.z] for corner in bbox.get_world_vertices(carla.Transform())])

            bbs.append({
                "x_min": np.min(corners[:, 0]),
                "x_max": np.max(corners[:, 0]),
                "y_min": np.min(corners[:, 1]),
                "y_max": np.max(corners[:, 1]),
                "z_min": np.min(corners[:, 2]),
                "z_max": np.max(corners[:, 2])
            })

        write_shared_tmp_file(constants.obstacles_static_list, bbs)


    def tick(self):

        ### Publish ROS messages ####
        self.broadcast_tf()
        self.publish_camera_image()
        self.publish_lidar()
        self.publish_states()

        ### Tick the Carla ###
        if self.args.asynch:
            self.world.wait_for_tick()
        else:
            self.world.tick()
        
        ### Tick complex adversarial objects ###
        try:
            self.update_complex_objects()
        except Exception as e:
            print(f"Updating a complex object failed due to the following error: {e}")

        self.update_spectator()
        return 0

    def update_spectator(self):
        try:
            if self.config['spectator_follows_ego_vehicle'] == False:
                return
        except KeyError:
            log.info("spectator_follows_ego_vehicle not found in config, skipping spectator update.")
            return

        vehicle_transform = self.ego_vehicle.get_transform()
        relative_location = vehicle_transform.location + vehicle_transform.get_forward_vector() * (-20)
        relative_location.z += 10

        transform = carla.Transform(carla.Location(relative_location), carla.Rotation(pitch=-30, yaw=180))
        self.spectator.set_transform(transform)


    def spawn_ego_vehicle(self):
        ego_bp = self.world.get_blueprint_library().filter(self.config['ego_vehicle']['model'])[0]
        ego_bp.set_attribute('role_name','ego')
        spawn_point = random.choice(self.world.get_map().get_spawn_points())

        # If the vehicle is MiniHawk, ensure that there is no rotation. We need it to make sure the pose processing happens correctly.
        if self.vehicle_type == 'minihawk':
            spawn_point = carla.Transform(
                location=spawn_point.location,
                rotation=carla.Rotation(
                    0,
                    0,
                    0
                )
            )

        try:
            loc = carla.Location(self.config['ego_vehicle']['location']['x'],
                                 self.config['ego_vehicle']['location']['y'],
                                 self.config['ego_vehicle']['location']['z'])
            rot = carla.Rotation(0, 0, 0)
            spawn_point = carla.Transform(location=loc, rotation=rot)
        except KeyError:
            pass

        # spawn_point.location.z = spawn_point.location.z + 100
        self.ego_vehicle = self.world.spawn_actor(ego_bp, spawn_point)
        self.ego_vehicle.set_autopilot(False)
        self.ego_vehicle.set_simulate_physics(False)
        self.ego_vehicle.set_enable_gravity(False)
        self.control_variable = carla.VehicleControl()
        self.ego_vehicle.apply_control(self.control_variable)
        self.world.tick()
        self.initial_transform = self.ego_vehicle.get_transform()

        # Manually move the ego vehicle to 100m elevation. Match to node_guam.py:guam_reference_init
        # location = self.ego_vehicle.get_location()
        # new_location = location + carla.Location(x=0.0, y=0.0, z=100.0)
        # self.ego_vehicle.set_location(new_location)
        # self.world.tick()
        self.update_spectator()

    def spawn_adversarial_object(self, adv_config):
        # Get the initial coordinates
        pose_init   = adv_config['pose']
        x_location  = pose_init['location']['x']
        y_location  = pose_init['location']['y']
        z_location  = pose_init['location']['z']
        roll_angle  = pose_init['rotation']['roll']
        pitch_angle = pose_init['rotation']['pitch']
        yaw_angle   = pose_init['rotation']['yaw']

        # Get the initial velocity
        vel_init    = adv_config['velocity']
        x_vel       = vel_init['x']
        y_vel       = vel_init['y']
        z_vel       = vel_init['z']

        # Add ego vehicle's coordinates if necessary
        if adv_config['pose']['type'] == 'relative':
            x_location  += self.config['ego_vehicle']['location']['x']
            y_location  += self.config['ego_vehicle']['location']['y']
            z_location  += self.config['ego_vehicle']['location']['z']
            # roll_angle  += self.ego_vehicle.get_transform().rotation.roll
            # pitch_angle += self.ego_vehicle.get_transform().rotation.pitch
            # yaw_angle   += self.ego_vehicle.get_transform().rotation.yaw
        elif adv_config['pose']['type'] == 'absolute':
            pass # Nothing to do
        else:
            raise ValueError(f"Unknown pose type: {adv_config['pose']['type']}.")

        # Spawn the adversarial object
        adv_obj = self.world.get_blueprint_library().filter(adv_config['model'])[0]
        adv_obj = self.world.spawn_actor(
            adv_obj,
            carla.Transform(
                location=carla.Location(
                    x=x_location,
                    y=y_location,
                    z=z_location
                ),
                rotation=carla.Rotation(
                    roll=roll_angle,
                    pitch=pitch_angle,
                    yaw=yaw_angle
                )
            )
        )

        # Set the target velocity
        adv_obj.set_target_velocity(
            velocity=carla.Vector3D(
                x_vel,
                y_vel,
                z_vel
            )
        )

        # Set some attributes. Disabling the physics simulation won't allow to move the vehicle.
        adv_obj.set_autopilot(False)
        adv_obj.set_enable_gravity(False)
        self.adv_obj = adv_obj

    def spawn_adversarial_objects(self):
        # TODO: I need to record the list of spawned vehicles into one of the above lists
        # TODO: add a config parameter to choose the vehicle (e.g. Dodge, Nissan, etc)
        # Check if adversarial objects are enabled
        if not self.config['adv_objects']['enabled']:
            self.adv_obj = None
            return

        # Spawn the adversarial objects
        for adv_obj in self.config['adv_objects']['objects']:
            self.spawn_adversarial_object(adv_obj)
        
        # Spawn complex adversarial objects
        self.complex_objects = [
            ComplexObject(
                self.world, 
                self.config['adv_objects']['complex_objects']['records_dir'],
                obj_name
            )
            for obj_name in self.config['adv_objects']['complex_objects']['object_names']
        ]
        print(f"Spawned {len(self.complex_objects)} complex objects")

    def update_complex_objects(self):
        for complex_object in self.complex_objects:
            complex_object.update()

    def start(self):

        self.vehicles_list = []
        self.walkers_list = []
        self.all_id = []

        ### spawn vehicles ###
        self.spawn_ego_vehicle()  # ego vehicle
        self.generate_traffic()

        ### spawn the adversarial object ###
        self.spawn_adversarial_objects()

        ### sensor initialization ###
        self.camera_front= SensorManager(self.world, 'RGBCamera', carla.Transform(carla.Location(x=2,  z=1.5), carla.Rotation(yaw=+00, pitch=-10)),
                                        self.ego_vehicle, {'fov':'90.0', 'image_size_x': '400', 'image_size_y': '400'})
        self.camera_left = SensorManager(self.world, 'RGBCamera', carla.Transform(carla.Location(x=0, z=2.4), carla.Rotation(yaw=-90)),
                                        self.ego_vehicle, {'fov':'90.0', 'image_size_x': '400', 'image_size_y': '400'})
        self.camera_right= SensorManager(self.world, 'RGBCamera', carla.Transform(carla.Location(x=0, z=2.4), carla.Rotation(yaw=+90)),
                                        self.ego_vehicle, {'fov':'90.0', 'image_size_x': '400', 'image_size_y': '400'})
        self.camera_back = SensorManager(self.world, 'RGBCamera', carla.Transform(carla.Location(x=-2,  z=1.5), carla.Rotation(yaw=+180, pitch=-10)),
                                        self.ego_vehicle, {'fov':'90.0', 'image_size_x': '400', 'image_size_y': '400'})
        self.camera_down = SensorManager(self.world, 'RGBCamera', carla.Transform(carla.Location(x=0,  z=-1.5), carla.Rotation(pitch=-90)),
                                        self.ego_vehicle, {'fov':'90.0', 'image_size_x': '448', 'image_size_y': '448'})
        self.camera_up   = SensorManager(self.world, 'RGBCamera', carla.Transform(carla.Location(x=0,  z=2.4), carla.Rotation(pitch= 90)),
                                        self.ego_vehicle, {'fov':'90.0', 'image_size_x': '400', 'image_size_y': '400'})
        self.camera_overview  = SensorManager(self.world, 'RGBCamera', carla.Transform(carla.Location(x=-1, z=7.0), carla.Rotation(pitch=-60)),
                                        self.ego_vehicle, {'fov':'60.0', 'image_size_x': '600', 'image_size_y': '600'})

        # Downward: pitch=0, yaw=0, roll=0
        # Upward: pitch=180, yaw=0, roll=0
        # Left: pitch=0, yaw=0, roll=90
        # Right: pitch=0, yaw=0, roll=-90
        # Backward: pitch=-90, yaw=0, roll=0
        # Forward:  pitch=90, yaw=0, roll=0
        # ND: pitch=0, yaw=-90, roll=0
        # ND: pitch=0, yaw=90, roll=0

        # Create multiple lidar sensors
        self.transform_lidar_from_vehicle_down      = carla.Transform(carla.Location(x=0, y=0, z=0), carla.Rotation(pitch=0, yaw=0, roll=0))
        self.transform_lidar_from_vehicle_up        = carla.Transform(carla.Location(x=0, y=0, z=0), carla.Rotation(pitch=180, yaw=0, roll=0))
        self.transform_lidar_from_vehicle_left      = carla.Transform(carla.Location(x=0, y=0, z=0), carla.Rotation(pitch=0, yaw=0, roll=-90))
        self.transform_lidar_from_vehicle_right     = carla.Transform(carla.Location(x=0, y=0, z=0), carla.Rotation(pitch=0, yaw=0, roll=90))
        self.transform_lidar_from_vehicle_back      = carla.Transform(carla.Location(x=0, y=0, z=0), carla.Rotation(pitch=90, yaw=0, roll=0))
        self.transform_lidar_from_vehicle_forward   = carla.Transform(carla.Location(x=0, y=0, z=0), carla.Rotation(pitch=-90, yaw=0, roll=0))
        self.lidar_sensor_down      = SensorManager(self.world, 'LiDAR', self.transform_lidar_from_vehicle_down, self.ego_vehicle, { 'channels' : '64', 'range' : '100', 'points_per_second': '230400', 'upper_fov': '-27', 'lower_fov': '-90', 'rotation_frequency': '10', 'horizontal_fov': '360', 'sensor_tick':'0.1', })
        self.lidar_sensor_up        = SensorManager(self.world, 'LiDAR', self.transform_lidar_from_vehicle_up, self.ego_vehicle, { 'channels' : '64', 'range' : '100', 'points_per_second': '230400', 'upper_fov': '-27', 'lower_fov': '-90', 'rotation_frequency': '10', 'horizontal_fov': '360', 'sensor_tick':'0.1', })
        self.lidar_sensor_left      = SensorManager(self.world, 'LiDAR', self.transform_lidar_from_vehicle_left, self.ego_vehicle, { 'channels' : '64', 'range' : '100', 'points_per_second': '230400', 'upper_fov': '-27', 'lower_fov': '-90', 'rotation_frequency': '10', 'horizontal_fov': '360', 'sensor_tick':'0.1', })
        self.lidar_sensor_right     = SensorManager(self.world, 'LiDAR', self.transform_lidar_from_vehicle_right, self.ego_vehicle, { 'channels' : '64', 'range' : '100', 'points_per_second': '230400', 'upper_fov': '-27', 'lower_fov': '-90', 'rotation_frequency': '10', 'horizontal_fov': '360', 'sensor_tick':'0.1', })
        self.lidar_sensor_back      = SensorManager(self.world, 'LiDAR', self.transform_lidar_from_vehicle_back, self.ego_vehicle, { 'channels' : '64', 'range' : '100', 'points_per_second': '230400', 'upper_fov': '-27', 'lower_fov': '-90', 'rotation_frequency': '10', 'horizontal_fov': '360', 'sensor_tick':'0.1', })
        self.lidar_sensor_forward   = SensorManager(self.world, 'LiDAR', self.transform_lidar_from_vehicle_forward, self.ego_vehicle, { 'channels' : '64', 'range' : '100', 'points_per_second': '230400', 'upper_fov': '-27', 'lower_fov': '-90', 'rotation_frequency': '10', 'horizontal_fov': '360', 'sensor_tick':'0.1', })

        self.transform_lidar_from_vehicle = carla.Transform(carla.Location(x=0, y=0, z=0), carla.Rotation(pitch=0, yaw=0, roll=0))
        self.lidar_sensor   = SensorManager(self.world, 'LiDAR', self.transform_lidar_from_vehicle, self.ego_vehicle, {
                                            'channels' : '64',
                                            'range' : '100',
                                            'points_per_second': '230400',
                                            'upper_fov': '-27',
                                            'lower_fov': '-90',
                                            'rotation_frequency': '10',
                                            'horizontal_fov': '360',
                                            'sensor_tick':'0.1',
                                            })
        self.collision_sensor_ego = CollisionSensor(self.ego_vehicle, panic=True)
        return 0


    def reset(self):
        print('RESET CALLED!')
        spawn_point = random.choice(self.world.get_map().get_spawn_points())
        self.ego_vehicle.set_transform(spawn_point)
        return 0


    def callback_jax_guam_pose(self, msg):

        ### Unpack PoseStamped msg ###
        X, Y, Z = (msg.pose.position.x, msg.pose.position.y, msg.pose.position.z) # meters

        quaternion = (
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w)
        (yaw, pitch, roll) = euler_from_quaternion(quaternion) # unit rad

        ### Overrisde the pose, i.e., transform ###
        transform = self.ego_vehicle.get_transform()
        # NOTE: the 2 lines below were causing incorrect behaviour in CARLA, because the vehicle was offset WRT CARLA coordinates
        # matlab_location = carla.Location(X, -Y, Z) # CARLA uses the Unreal Engine coordinates system. This is a Z-up left-handed system.
        # transform.location = matlab_location + self.initial_transform.location
        matlab_location = carla.Location(X, Y, Z) # CARLA uses the Unreal Engine coordinates system. This is a Z-up left-handed system.
        transform.location = matlab_location
        matlab_rotation = carla.Rotation(rad2deg(-pitch), rad2deg(-yaw), rad2deg(roll))  # The constructor method follows a specific order of declaration: (pitch, yaw, roll), which corresponds to (Y-rotation,Z-rotation,X-rotation).
        transform.rotation = add_carla_rotations(matlab_rotation, self.initial_transform.rotation)
        self.ego_vehicle.set_transform(transform)
        self.update_spectator()


    def broadcast_tf(self):

        ### Broadcast TF-vehicle from map ###
        veh_transform = self.ego_vehicle.get_transform()
        xyz, quaternion = carla_transform_to_ros_xyz_quaternion(veh_transform)
        self.tf_broadcaster.sendTransform(xyz, quaternion, rospy.Time.now(), 'vehicle', 'map')

        ### Broadcast TF-sensor from vehicle ###
        xyz, quaternion = carla_transform_to_ros_xyz_quaternion(self.transform_lidar_from_vehicle)
        self.tf_broadcaster.sendTransform(xyz, quaternion, rospy.Time.now(), 'sensor', 'vehicle')


    def publish_lidar(self):
        header = Header()
        header.frame_id = 'sensor'
        # if self.lidar_sensor is not None and self.lidar_sensor.data is not None:
        #     points = np.array(self.lidar_sensor.data[:,:3])
        #     points[:, 1] = -points[:, 1]
            # self.pub_lidar_point_cloud.publish(pcl2.create_cloud_xyz32(header,points))
        
        # Publish multiple lidars data
        # Down
        if self.lidar_sensor_down is not None and self.lidar_sensor_down.data is not None:
            points = np.array(self.lidar_sensor_down.data[:, :3])
            points[:, 1] = -points[:, 1]
            self.pub_lidar_point_cloud_down.publish(pcl2.create_cloud_xyz32(header, points))

        # Up
        if self.lidar_sensor_up is not None and self.lidar_sensor_up.data is not None:
            points = np.array(self.lidar_sensor_up.data[:, :3])
            points[:, 1] = -points[:, 1]
            self.pub_lidar_point_cloud_up.publish(pcl2.create_cloud_xyz32(header, points))

        # Left
        if self.lidar_sensor_left is not None and self.lidar_sensor_left.data is not None:
            points = np.array(self.lidar_sensor_left.data[:, :3])
            points[:, 1] = -points[:, 1]
            self.pub_lidar_point_cloud_left.publish(pcl2.create_cloud_xyz32(header, points))

        # Right
        if self.lidar_sensor_right is not None and self.lidar_sensor_right.data is not None:
            points = np.array(self.lidar_sensor_right.data[:, :3])
            points[:, 1] = -points[:, 1]
            self.pub_lidar_point_cloud_right.publish(pcl2.create_cloud_xyz32(header, points))

        # Back
        if self.lidar_sensor_back is not None and self.lidar_sensor_back.data is not None:
            points = np.array(self.lidar_sensor_back.data[:, :3])
            points[:, 1] = -points[:, 1]
            self.pub_lidar_point_cloud_back.publish(pcl2.create_cloud_xyz32(header, points))

        # Forward
        if self.lidar_sensor_forward is not None and self.lidar_sensor_forward.data is not None:
            points = np.array(self.lidar_sensor_forward.data[:, :3])
            points[:, 1] = -points[:, 1]
            self.pub_lidar_point_cloud_forward.publish(pcl2.create_cloud_xyz32(header, points))

    def publish_camera_image(self):
        header = Header()
        header.stamp = rospy.Time.now()
        if self.camera_left and self.camera_left.data is not None:
            self.pub_camera_frame_left.publish(pack_image_ros_msg(self.camera_left.data, header, 'left_camera'))
        if self.camera_right and self.camera_right.data is not None:
            self.pub_camera_frame_right.publish(pack_image_ros_msg(self.camera_right.data, header, 'right_camera'))
        if self.camera_overview and self.camera_overview.data is not None:
            self.pub_camera_frame_overview.publish(pack_image_ros_msg(self.camera_overview.data, header, 'overview_camera'))
        if self.camera_front and self.camera_front.data is not None:
            self.pub_camera_frame_front.publish(pack_image_ros_msg(self.camera_front.data, header, 'front_camera'))
        if self.camera_back and self.camera_back.data is not None:
            self.pub_camera_frame_back.publish(pack_image_ros_msg(self.camera_back.data, header, 'back_camera'))
        if self.camera_up and self.camera_up.data is not None:
            self.pub_camera_frame_up.publish(pack_image_ros_msg(self.camera_up.data, header,  'up_camera'))
        if self.camera_down and self.camera_down.data is not None:
            self.pub_camera_frame_down.publish(pack_image_ros_msg(self.camera_down.data, header,  'down_camera'))

    def publish_states(self):

        # World State
        world_state = np.array([[self.fps_timer.server_fps, self.client_clock.get_fps()]])
        label_row = 'world'
        label_col = 'server_fps,client_fps'
        self.pub_world_state.publish(pack_multiarray_ros_msg(self.msg_mat.mat, world_state, label_row, label_col))

        # Vehicle State
        state_key, values_1 = self.get_vehicle_state(self.ego_vehicle)
        _,         values_2 = self.get_vehicle_state(self.ego_vehicle)
        veh_list_str = ['ego_vehicle','forward_vehicle']
        state_val_np = np.array([values_1, values_2])
        multiarray_ros_msg = pack_multiarray_ros_msg(self.msg_mat.mat, state_val_np, ','.join(veh_list_str), ','.join(state_key))
        self.pub_vehicles_state.publish(multiarray_ros_msg)

        return 0


    @staticmethod
    def get_vehicle_state(vehicle):
        transform = vehicle.get_transform()
        values = [transform.location.x, transform.location.y, transform.location.z]
        keys   = ['x', 'y', 'z']
        values += [transform.rotation.roll, transform.rotation.pitch, transform.rotation.yaw]
        keys   += ['roll', 'pitch', 'yaw']
        return keys, values

    @staticmethod
    def get_actor_blueprints(world, filter, generation):
        bps = world.get_blueprint_library().filter(filter)

        # Remove the blueprints of aerial vehicles (they are tagged with `uli`)
        bps = [bp for bp in bps if "uli" not in bp.tags]

        if generation.lower() == "all":
            return bps
        # If the filter returns only one bp, we assume that this one needed
        # and therefore, we ignore the generation
        if len(bps) == 1:
            return bps

        try:
            int_generation = int(generation)
            # Check if generation is in available generations
            if int_generation in [1, 2]:
                bps = [x for x in bps if int(x.get_attribute('generation')) == int_generation]
                return bps
            else:
                print("   Warning! Actor Generation is not valid. No actor will be spawned.")
                return []
        except:
            print("   Warning! Actor Generation is not valid. No actor will be spawned.")
            return []

    def generate_traffic(self):
        synchronous_master = False
        blueprints = self.get_actor_blueprints(self.world, self.args.filterv, self.args.generationv)
        blueprintsWalkers = self.get_actor_blueprints(self.world, self.args.filterw, self.args.generationw)

        if self.args.safe:
            blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 4]
            blueprints = [x for x in blueprints if not x.id.endswith('microlino')]
            blueprints = [x for x in blueprints if not x.id.endswith('carlacola')]
            blueprints = [x for x in blueprints if not x.id.endswith('cybertruck')]
            blueprints = [x for x in blueprints if not x.id.endswith('t2')]
            blueprints = [x for x in blueprints if not x.id.endswith('sprinter')]
            blueprints = [x for x in blueprints if not x.id.endswith('firetruck')]
            blueprints = [x for x in blueprints if not x.id.endswith('ambulance')]

        blueprints = sorted(blueprints, key=lambda bp: bp.id)

        spawn_points = self.world.get_map().get_spawn_points()
        number_of_spawn_points = len(spawn_points)

        if self.args.number_of_vehicles < number_of_spawn_points:
            random.shuffle(spawn_points)
        elif self.args.number_of_vehicles > number_of_spawn_points:
            msg = 'requested %d vehicles, but could only find %d spawn points'
            log.warning(msg, self.args.number_of_vehicles, number_of_spawn_points)
            self.args.number_of_vehicles = number_of_spawn_points

        # @todo cannot import these directly.
        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        FutureActor = carla.command.FutureActor

        # --------------
        # Spawn vehicles
        # --------------
        batch = []
        hero = self.args.hero
        for n, transform in enumerate(spawn_points):
            if n >= self.args.number_of_vehicles:
                break
            blueprint = random.choice(blueprints)
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            if blueprint.has_attribute('driver_id'):
                driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                blueprint.set_attribute('driver_id', driver_id)
            if hero:
                blueprint.set_attribute('role_name', 'hero')
                hero = False
            else:
                blueprint.set_attribute('role_name', 'autopilot')

            # spawn the cars and set their autopilot and light state all together
            batch.append(SpawnActor(blueprint, transform).then(SetAutopilot(FutureActor, True, self.traffic_manager.get_port())))

        for response in self.client.apply_batch_sync(batch, synchronous_master):
            if response.error:
                log.error(response.error)
            else:
                self.vehicles_list.append(response.actor_id)

        # Set automatic vehicle lights update if specified
        if self.args.car_lights_on:
            all_vehicle_actors = self.world.get_actors(self.vehicles_list)
            for actor in all_vehicle_actors:
                self.traffic_manager.update_vehicle_lights(actor, True)

        # -------------
        # Spawn Walkers
        # -------------
        # some settings
        percentagePedestriansRunning = 0.0      # how many pedestrians will run
        percentagePedestriansCrossing = 0.0     # how many pedestrians will walk through the road

        if self.args.seedw:
            self.world.set_pedestrians_seed(self.args.seedw)
            random.seed(self.args.seedw)
        # 1. take all the random locations to spawn
        spawn_points = []
        for i in range(self.args.number_of_walkers):
            spawn_point = carla.Transform()
            loc = self.world.get_random_location_from_navigation()
            if (loc != None):
                spawn_point.location = loc
                spawn_points.append(spawn_point)
        # 2. we spawn the walker object
        batch = []
        walker_speed = []
        for spawn_point in spawn_points:
            walker_bp = random.choice(blueprintsWalkers)
            # set as not invincible
            if walker_bp.has_attribute('is_invincible'):
                walker_bp.set_attribute('is_invincible', 'false')
            # set the max speed
            if walker_bp.has_attribute('speed'):
                if (random.random() > percentagePedestriansRunning):
                    # walking
                    walker_speed.append(walker_bp.get_attribute('speed').recommended_values[1])
                else:
                    # running
                    walker_speed.append(walker_bp.get_attribute('speed').recommended_values[2])
            else:
                print("Walker has no speed")
                walker_speed.append(0.0)
            batch.append(SpawnActor(walker_bp, spawn_point))
        results = self.client.apply_batch_sync(batch, synchronous_master)
        walker_speed2 = []
        for i in range(len(results)):
            if results[i].error:
                log.error(results[i].error)
            else:
                self.walkers_list.append({"id": results[i].actor_id})
                walker_speed2.append(walker_speed[i])
        walker_speed = walker_speed2
        # 3. we spawn the walker controller
        batch = []
        walker_controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')
        for i in range(len(self.walkers_list)):
            batch.append(SpawnActor(walker_controller_bp, carla.Transform(), self.walkers_list[i]["id"]))
        results = self.client.apply_batch_sync(batch, synchronous_master)
        for i in range(len(results)):
            if results[i].error:
                log.error(results[i].error)
            else:
                self.walkers_list[i]["con"] = results[i].actor_id
        # 4. we put together the walkers and controllers id to get the objects from their id
        for i in range(len(self.walkers_list)):
            self.all_id.append(self.walkers_list[i]["con"])
            self.all_id.append(self.walkers_list[i]["id"])
        self.all_actors = self.world.get_actors(self.all_id)


        # 5. initialize each controller and set target to walk to (list is [controler, actor, controller, actor ...])
        # set how many pedestrians can cross the road
        self.world.set_pedestrians_cross_factor(percentagePedestriansCrossing)
        for i in range(0, len(self.all_id), 2):
            # start walker
            self.all_actors[i].start()
            # set walk to random point
            self.all_actors[i].go_to_location(self.world.get_random_location_from_navigation())
            # max speed
            self.all_actors[i].set_max_speed(float(walker_speed[int(i/2)]))

        print('spawned %d vehicles and %d walkers, press Ctrl+C to exit.' % (len(self.vehicles_list), len(self.walkers_list)))


    def destroy(self):
        log.trace('\nDestroying all CARLA actors')
        try:
            if self.camera_front:
                self.camera_front.sensor.destroy()
            if self.camera_left:
                self.camera_left.sensor.destroy()
            if self.camera_right:
                self.camera_right.sensor.destroy()
            if self.camera_back:
                self.camera_back.sensor.destroy()
            if self.camera_up:
                self.camera_up.sensor.destroy()
            if self.camera_down:
                self.camera_down.sensor.destroy()
            if self.camera_overview:
                self.camera_overview.sensor.destroy()
            if self.lidar_sensor:
                self.lidar_sensor.sensor.destroy()
            if self.collision_sensor_ego:
                self.collision_sensor_ego.sensor.destroy()
        except Exception as e:
            log.error(f"Error destroying sensors: {e}")

        try:
            if self.ego_vehicle:
                self.ego_vehicle.destroy()
                log.trace('Ego vehicle destroyed')
        except Exception as e:
            log.error(f"Error destroying ego vehicle: {e}")

        try:
            if self.adv_obj:
                self.adv_obj.destroy()
                log.trace('Adv. object destroyed')
        except Exception as e:
            log.error(f"Error destroying adv. object: {e}")

        if self.vehicles_list:
            self.client.apply_batch([carla.command.DestroyActor(vehicle) for vehicle in self.vehicles_list])
            log.trace(f"Destroyed {len(self.vehicles_list)} vehicles")
            self.vehicles_list = []

        # Stop and destroy walkers
        if self.walkers_list:
            for i in range(0, len(self.all_id), 2):
                try:
                    self.all_actors[i].stop()
                except Exception as e:
                    log.error(f"Error stopping walker controller: {e}")
            self.client.apply_batch([carla.command.DestroyActor(walker_id) for walker_id in self.all_id])
            log.trace(f"Destroyed {len(self.walkers_list)} walkers")
            self.walkers_list = []
            self.all_id = []

        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                self.client.reload_world()
                self.client.get_world().apply_settings(self.original_settings)
                log.info("Carla world reset successfully.")
                break
            except RuntimeError as e:
                log.error(f"Attempt {attempt + 1} to reset Carla world failed: {e}")
                time.sleep(5)