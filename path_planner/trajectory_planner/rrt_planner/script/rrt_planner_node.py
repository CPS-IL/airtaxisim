#!/usr/bin/env python3

import os
import sys
import math
import time
import rospy
import random
import numpy as np
from typing import List
from scipy.spatial import KDTree
from matplotlib import pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.collections import LineCollection

sys.path.append(os.path.abspath('/catkin_ws/src/scripts/utils'))
from utils import constants
from utils.config import load_yaml_file, log

import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Twist, PoseStamped

class Node:
    def __init__(self, x: float, y: float, z:float, children: List[int] = [], parent: int = None):
        self.x = x
        self.y = y
        self.z = z
        self.children = children
        self.parent = parent
    
    def add_child(self, child_idx: int):
        self.children.append(child_idx)
    
    def set_parent(self, parent_idx: int):
        assert self.parent is None, f"Parent is already set!"
        self.parent = parent_idx
    
    def __str__(self):
        return f"x: {self.x}, y: {self.y}, z: {self.z}, children indices: {self.children}"

class Obstacle:
    def __init__(self, obs_boundaries: tuple):
        self.obs_boundaries = obs_boundaries # (x_min, x_max, y_min, y_max, z_min, z_max)
        (
            self.x_min, self.x_max, 
            self.y_min, self.y_max, 
            self.z_min, self.z_max
        ) = self.obs_boundaries
        self.width = self.x_max - self.x_min
        self.height = self.y_max - self.y_min
        self.depth = self.z_max - self.z_min
    
    def is_inside(self, point: List) -> bool:
        x, y, z = point
        return (
            self.x_min <= x <= self.x_max and
            self.y_min <= y <= self.y_max and
            self.z_min <= z <= self.z_max
        )

class RRTPlanner:
    def __init__(self, 
                 start: List, 
                 goal: List,
                 map_boundaries: tuple,
                 obstacles: List[Obstacle] = [],
                 goal_thresh: float = 1.0,
                 step_size: float = 3.0,
                 max_iter: int = 100,
        ):
        self.start = start
        self.goal = goal
        self.map_boundaries = map_boundaries # (x_min, x_max, y_min, y_max, z_min, z_max)
        (
            self.x_min, self.x_max, 
            self.y_min, self.y_max, 
            self.z_min, self.z_max
        ) = self.map_boundaries
        self.obstacles = obstacles
        self.goal_thresh = goal_thresh
        self.step_size = step_size
        self.max_iter = max_iter

        # Tree
        self.tree = self.init_tree(start)

        # Current pose of the vehicle
        self.pose = None

        # Occupancy tree
        self.kdtree = None
        self.occupied_points = []

        # Path from the start to the goal
        self.path = None
    
    def init_tree(self, start: tuple) -> List[Node]:
        tree = []
        tree.append(Node(start[0], start[1], start[2], children=[], parent=None))
        return tree
    
    def set_start(self, new_start: tuple):
        # Update the start point
        self.start = new_start

        # Update the tree
        self.tree = self.init_tree(new_start)

    def set_pose(self, new_pose: tuple):
        self.pose = new_pose

    def plan(self) -> np.array:
        """
        Function that plans a route from the start position to the goal position.
        """
        iter = 0
        found = False
        while iter < self.max_iter and not found:
            # Draw a random sample
            sample = self.draw_sample()

            # Find the nearest node in the tree
            node_idx = self.get_nearest_node(sample)

            # Step toward the sample from the nearest node
            new_node = self.step(sample, node_idx)

            # Check if the step is valid
            if not self.is_valid(new_node):
                continue

            # Add the new edge and the new node to the tree
            self.add_node(new_node, node_idx)

            # Check if the new node is close enough to the goal
            is_goal = self.is_goal(new_node)
            if is_goal:
                # Before extracting the path, add goal as a node
                self.add_node(Node(self.goal[0], self.goal[1], self.goal[2]), len(self.tree) - 1)

                # Extract the path
                self.path = self.extract_path()
                found = True

            # Increment the iterations
            iter += 1
        
        return self.get_path_as_list()
    
    def get_path_as_list(self) -> List[List[float]]:
        return [[node.x, node.y, node.z] for node in self.path]
    
    def draw_sample(self) -> tuple:
        # Generate random coordinates
        return (
            random.uniform(self.x_min, self.x_max),
            random.uniform(self.y_min, self.y_max),
            random.uniform(self.z_min, self.z_max)
        )
    
    def get_nearest_node(self, sample: tuple):
        # Convert the sample into a numpy array
        sample = np.array(sample)

        # Convert the node coordinates into a numpy array
        nodes = np.array([
            [node_.x, node_.y, node_.z] for node_ in self.tree
        ])
        
        # Calculate the distances
        distances = np.sum((nodes - sample) ** 2, axis=1)

        # Find the index of the nearest node
        return np.argmin(distances)
    
    def step(self, sample, node_idx: int):
        # Find the direction vector
        dir_vec = sample - np.array([
            self.tree[node_idx].x,
            self.tree[node_idx].y,
            self.tree[node_idx].z
        ])
        magnitude = np.linalg.norm(dir_vec)
        
        # Normalize the direction vector
        if magnitude == 0:
            print(f"WARNING: discarding a zero vector.")
            return None
        else:
            dir_vec = dir_vec / magnitude

        # Step
        new_node = np.array([self.tree[node_idx].x, self.tree[node_idx].y, self.tree[node_idx].z])
        if magnitude > self.step_size:
            new_node = new_node + dir_vec * self.step_size
        else:
            new_node = new_node + dir_vec * magnitude
        
        # Return the new node
        new_node = Node(new_node[0], new_node[1], new_node[2], children=[])

        return new_node
    
    def is_valid(self, node: Node, inflation_radius: float = 3.0):
        """
        # Check for collisions with every single obstacle
        for obstacle in self.obstacles:
            is_inside = obstacle.is_inside([node.x, node.y, node.z])
            if is_inside:
                return False
        return True
        """
        # Assume the point is free if no occupancy data is available
        if self.kdtree is None:
            return True
        
        # Query for any point with inflation radius
        indices = self.kdtree.query_ball_point([node.x, node.y, node.z], inflation_radius)

        return len(indices) == 0
    
    def add_node(self, new_node: Node, closest_node_idx: int):
        # Set the parent of the new node
        new_node.set_parent(closest_node_idx)

        # Append the new node
        self.tree.append(new_node)

        # Add the new edge
        self.tree[closest_node_idx].add_child(len(self.tree) - 1)
    
    def is_goal(self, node):
        # Form the numpy arrays
        node = np.array([node.x, node.y, node.z])
        goal = np.array([self.goal[0], self.goal[1], self.goal[2]])

        # Compute the distance 
        dist = np.linalg.norm(node - goal)

        # Compare the distance with the threshold
        if dist <= self.goal_thresh:
            return True
        else:
            return False
    
    def extract_path(self):
        # Store the nodes of the path
        path = []

        # Move from the leaf node to the root node
        curr_idx = len(self.tree) - 1
        while curr_idx != 0:
            path.append(self.tree[curr_idx])
            curr_idx = self.tree[curr_idx].parent
        
        # Add the root node
        path.append(self.tree[0])

        # Reverse the path
        path.reverse()

        # Extract the path as a list of coordinates
        return path

    def plot_tree(self):
        """
        NOTE: plotting works only in 2D.
        """
        # Initialize the figure
        fig, ax = plt.subplots()

        # Plot the obstacles
        for obstacle in self.obstacles:
            rect = Rectangle(
                (obstacle.x_min, obstacle.y_min), obstacle.width, obstacle.height,
                edgecolor='none',
                facecolor='black'
            )
            ax.add_patch(rect)

        # Plot the nodes
        node_xs = [node.x for node in self.tree]
        node_ys = [node.y for node in self.tree]
        ax.scatter(node_xs, node_ys, s=30, facecolors='none', edgecolors='black')

        # Plot the edges
        edges = []
        for node in self.tree:
            for child_idx in node.children:
                edges.append(
                    [
                        (node.x, node.y), 
                        (self.tree[child_idx].x, self.tree[child_idx].y)
                    ]
                )
        
        # Collect the edges to draw
        lc = LineCollection(edges, colors='blue')
        ax.add_collection(lc)

        # Plot the path
        if self.path:
            path_xs = [coordinates[0] for coordinates in self.path]
            path_ys = [coordinates[1] for coordinates in self.path]
            ax.plot(path_xs, path_ys, linewidth=3, c='green')

        # Plot the start and goal nodes
        start_circle = Circle((self.start[0], self.start[1]), self.goal_thresh, edgecolor='green', facecolor='none', linewidth=2)
        goal_circle = Circle((self.goal[0], self.goal[1]), self.goal_thresh, edgecolor='red', facecolor='none', linewidth=2)
        ax.add_patch(start_circle)
        ax.add_patch(goal_circle)

        # Set the graph limits
        ax.set_xlim([self.x_min, self.x_max])
        ax.set_ylim([self.y_min, self.y_max])

        plt.show()

    def is_connected(self):
        """
        Checks that the tree is a single graph.
        """
        if len(self.tree) == 1:
            return True

        is_child = [False for _ in range(len(self.tree))]
        for node in self.tree:
            for child_idx in node.children:
                is_child[child_idx] = True
        return all(is_child[1:])
    
    def update_kdtree(self, occupied_points: List):
        # Update the octomap
        self.occupied_points = occupied_points
        if self.occupied_points:
            self.kdtree = KDTree(self.occupied_points)
        
        # Check that the current path is still free
        if not all([self.is_valid(node) for node in self.path]):
            print("REPLANNING")
            # If the current position of the vehicle is known, plan the new route starting from it
            if self.pose:
                print(f"Planning from the current position: {self.pose}")
                self.set_start(self.pose)

            return self.plan()
        else:
            # print("CURRENT PATH IS VALID")
            return self.get_path_as_list()

class RRTPlannerNode:
    def __init__(self, config):
        # Record the config
        self.config = config

        # Initialize the RRT planner node
        rospy.init_node('rrt_planner')

        self.last_speed_check_time = None # last speed check time in s
        self.last_speed_check_coordinate = None # last speed check position as a 3D numpy array (x, y, z)

        # Publish to the target waypoint topic
        self.target_waypoint_pub = rospy.Publisher('/target/waypoint', Float32MultiArray, queue_size=1)

        # Extract the start and target points
        log.info("Extrating the start and goal points...")
        self.start, self.goal = self.get_start_end_points(config)

        # Initialize the planner instance
        log.info("Initializing the internal RRT planner...")
        self.rrt_planner = RRTPlanner(self.start, self.goal, map_boundaries=(-100, 100, -100, 100, -5, 100), goal_thresh=30.0, step_size=10.0, max_iter=1000)

        # Generate the starting plan
        log.info("Planning the start path...")
        st = time.time()
        self.path = self.rrt_planner.plan()
        et = time.time()
        if self.path:
            log.info(f"Path has been found!")
            print(self.path)
            self.waypoint_counter = 0
        else:
            log.info("No path found!")
            self.waypoint_counter = None
        log.info(f"Planning took {round(et - st, 2)} seconds")

        # Subscribe to the octomap to read the occupancy map
        self.octomap_sub = rospy.Subscriber('/octomap_point_cloud_centers', PointCloud2, self.octomap_callback)

        # Subscribe to the goal topic
        self.global_target_sub = rospy.Subscriber(config['ego_vehicle']['reference_topic'], Twist, self.global_target_callback)

        # Subscribe to the vehicle pose topic
        self.pose_sub = rospy.Subscriber(f"/{config['ego_vehicle']['type']}/pose", PoseStamped, self.pose_callback)

    def get_start_end_points(self, config):
        # Start point
        start_point = np.array([
            float(config['ego_vehicle']['location']['x']),
            float(config['ego_vehicle']['location']['y']),
            float(config['ego_vehicle']['location']['z'])
        ])

        # End point
        end_point = np.array([
            float(config['target']['x']),
            float(config['target']['y']),
            float(config['target']['z'])
        ])
        if config['target']['type'] == 'relative':
            end_point += start_point
        elif config['target']['type'] == 'absolute':
            pass
        else:
            raise ValueError(f"Unknown target type {config['target']['type']}.")

        return (start_point, end_point)
    
    def global_target_callback(self, global_target):
        pass

    def octomap_callback(self, map_msg):
        # Update the octomap
        new_path = self.rrt_planner.update_kdtree([tuple(p) for p in pc2.read_points(map_msg, field_names=("x", "y", "z"), skip_nans=True)])

        # Check if the new path is actually new
        if new_path != self.path:
            # print("NEW PATH")
            # print(self.path)
            # print(new_path)
            self.path = new_path
            self.waypoint_counter = 0

    def run(self):
        log.info("Running RRT planner node...")
        r = rospy.Rate(10)
        start_time = time.time()

        while not rospy.is_shutdown():
            # Publish the current waypoint
            message = Float32MultiArray()
            message.data = [
                self.path[self.waypoint_counter][0],
                self.path[self.waypoint_counter][1],
                self.path[self.waypoint_counter][2],
                0,
                0,
                0
            ]

            # Run some pre-checks
            if time.time() - start_time < 5:
                message.data = [
                    self.config['ego_vehicle']['location']['x'],
                    self.config['ego_vehicle']['location']['y'],
                    self.config['ego_vehicle']['location']['z'],
                    0,
                    0,
                    0
                ]

            # Publish the data
            self.target_waypoint_pub.publish(message)

            r.sleep()

    def pose_callback(self, data):
        # Update the current pose in the planner
        self.rrt_planner.set_pose(
            (data.pose.position.x, data.pose.position.y, data.pose.position.z)
        )

        # Only do something if the waypoint counter is not the last counter
        if self.waypoint_counter < len(self.path) - 1:
            # Current position
            x_curr = data.pose.position.x
            y_curr = data.pose.position.y
            z_curr = data.pose.position.z

            # Current waypoint
            x_waypoint = self.path[self.waypoint_counter][0]
            y_waypoint = self.path[self.waypoint_counter][1]
            z_waypoint = self.path[self.waypoint_counter][2]

            # Compute the distance to the waypoint
            waypoint_dist = math.sqrt(
                (x_curr - x_waypoint) ** 2 +
                (y_curr - y_waypoint) ** 2 +
                (z_curr - z_waypoint) ** 2
            )

            # If the distance is below a threshold, move to the next waypoint
            if waypoint_dist < self.config['ego_vehicle']['planner']['distance_threshold']:
                self.waypoint_counter += 1

            # If too much time has passed, move to the next waypoint
            if self.last_speed_check_coordinate is not None and self.last_speed_check_time is not None and self.waypoint_counter > 0:
                if time.time() - self.last_speed_check_time >= self.config['ego_vehicle']['planner']['speed_check_time']:
                    last_coordinate_dist = math.sqrt(
                        (x_curr - self.last_speed_check_coordinate[0]) ** 2 +
                        (y_curr - self.last_speed_check_coordinate[1]) ** 2 +
                        (z_curr - self.last_speed_check_coordinate[2]) ** 2
                    )
                    if last_coordinate_dist / (time.time() - self.last_speed_check_time) < self.config['ego_vehicle']['planner']['speed_threshold']:
                        self.waypoint_counter += 1
                    self.last_speed_check_time = time.time()
                    self.last_speed_check_coordinate = np.array([x_curr, y_curr, z_curr])
            else:
                self.last_speed_check_time = time.time()
                self.last_speed_check_coordinate = np.array([x_curr, y_curr, z_curr])

def main():
    # Load the config file
    config = load_yaml_file(constants.merged_config_path, __file__)

    # Initialize the RRT planner node
    rrt_planner_node = RRTPlannerNode(config)

    # Run the RRT planner node
    rrt_planner_node.run()

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass