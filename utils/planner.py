import math
import json
import time
import rospy
import heapq
import numpy as np
import logging
# import pyvista as pv
from tqdm import tqdm
from typing import List, Dict, Optional, Set, Tuple
from matplotlib import pyplot as plt
from std_msgs.msg import Float32MultiArray, Float32, String
from geometry_msgs.msg import Twist, PoseStamped
from .mln_safety_verifier import (
    MLNSafetyVerifier, 
    PlanningAction, 
    Action,
    SafetyVerificationResult,
    SafetyVerificationError,
    EnvironmentalPredicate,
    Rule
)
from rraaa_sim.msg import WindStatus, ObstacleArray


class BasePathPlanner:
    """
    Abstract class for the path planner.
    """
    def __init__(self):
        self.hahaha = "hahaha"
        pass

    def get_waypoints(
            self, 
            start_point: np.array, 
            end_point: np.array
        ) -> List[np.array]:
        """
        Generate a series of waypoints that define a path from the start point to the end point in a 3D space.

        This method should be implemented in subclasses to compute and return the waypoints that guide the path from 
        the `start_point` to the `end_point`, considering obstacles, path constraints, or other factors relevant to 
        path planning.

        Args:
            - start_point (np.array): A 3D numpy array representing the starting point of the path.
            - end_point (np.array): A 3D numpy array representing the target end point of the path.

        Returns:
            - waypoints (List[np.array]): A list of 3D numpy arrays, each representing a waypoint along the path, 
              with each waypoint containing 3D coordinates (x, y, z) in the form of an array of shape (3,).
        """
        raise NotImplementedError
    
    def run(self):
        """
        Executes the path planning process and publishes the generated waypoints.

        This method should be implemented in subclasses to initiate the process of generating waypoints 
        from a given start point to an end point. The generated waypoints are then published or returned 
        to be used by other components, such as a navigation system, for path execution.

        The `run` method can include any additional logic required for path planning, such as handling 
        dynamic updates, communication with other systems, or managing resources during the planning process.

        Note:
            The specific implementation of this method depends on the path planning algorithm and 
            how the waypoints are intended to be used (e.g., sending to a robot's movement controller).

        Returns:
            - None: This method may handle the publishing or updating of waypoints through external 
              mechanisms, so no return value is necessary.
        """
        raise NotImplementedError
    





########## PRE-MADE PLANNERS ##########

def astar_3d(grid, start, goal):
    """
    Perform A* algorithm on a 3D grid to find the shortest path.

    Args:
        grid (list of list of list): The 3D grid representing the environment.
        start (tuple): The starting point as (depth, row, col).
        goal (tuple): The goal point as (depth, row, col).
    
    Returns:
        list: The path from start to goal as a list of (depth, row, col) tuples.
    """
    class Node:
        def __init__(self, x, y, z, g, h):
            self.x = x
            self.y = y
            self.z = z
            self.g = g  # Cost from the start node
            self.h = h  # Heuristic to the goal node
            self.f = g + h  # f = g + h
            self.parent = None  # To track the path

        def __lt__(self, other):
            return self.f < other.f  # For priority queue sorting by f-value
        
    # Directions (6 possible moves: up, down, left, right, forward, backward)
    DIRECTIONS = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]

    depth, rows, cols = len(grid), len(grid[0]), len(grid[0][0])
    
    # Heuristic: Manhattan distance in 3D
    def heuristic(x, y, z):
        # L1
        return abs(x - goal[0]) + abs(y - goal[1]) + abs(z - goal[2])

        # L2
        # return math.sqrt((x - goal[0]) ** 2 + (y - goal[1]) ** 2 + (z - goal[2]) ** 2)

    # Check if the cell is valid (within bounds and not blocked)
    def is_valid(x, y, z):
        return 0 <= x < depth and 0 <= y < rows and 0 <= z < cols and grid[x][y][z] == 0

    # Priority Queue (Min-Heap) to store the nodes, starting with the start node
    open_list = []
    start_node = Node(start[0], start[1], start[2], 0, heuristic(start[0], start[1], start[2]))
    heapq.heappush(open_list, start_node)

    # Set to track visited nodes
    visited = set()

    while open_list:
        # Get the node with the lowest f-value
        current = heapq.heappop(open_list)
        
        # If we have reached the goal, reconstruct the path
        if (current.x, current.y, current.z) == goal:
            path = []
            while current:
                path.append((current.x, current.y, current.z))
                current = current.parent
            return path[::-1]  # Reverse path to get from start to goal
        
        visited.add((current.x, current.y, current.z))

        # Explore neighbors
        for dx, dy, dz in DIRECTIONS:
            nx, ny, nz = current.x + dx, current.y + dy, current.z + dz

            if (nx, ny, nz) not in visited and is_valid(nx, ny, nz):
                g_cost = current.g + 1  # Assuming each move has a cost of 1
                h_cost = heuristic(nx, ny, nz)
                neighbor = Node(nx, ny, nz, g_cost, h_cost)
                neighbor.parent = current
                heapq.heappush(open_list, neighbor)

    return None  # If no path is found

def plot_grid_with_path_3d(grid, path, start, goal):
    """
    Plot the 3D grid and the path using matplotlib.

    Args:
        grid (list of list of list): The 3D grid representing the environment.
        path (list): The list of path nodes as (depth, row, col).
        start (tuple): The starting point as (depth, row, col).
        goal (tuple): The goal point as (depth, row, col).
    """
    depth, rows, cols = len(grid), len(grid[0]), len(grid[0][0])

    # Set up the 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the grid: obstacles are shown as black cubes
    for x in range(depth):
        for y in range(rows):
            for z in range(cols):
                if grid[x][y][z] == 1:
                    ax.scatter(z, y, x, color='black', marker='o')

    # Mark the path (if any)
    if path:
        path_x, path_y, path_z = zip(*path)
        ax.plot(path_z, path_y, path_x, color='blue', marker='o', markersize=5, label='Path')

    # Mark the start and goal points
    ax.scatter(start[2], start[1], start[0], color='green', s=100, label='Start')  # Start point (green)
    ax.scatter(goal[2], goal[1], goal[0], color='red', s=100, label='Goal')  # Goal point (red)

    # Set labels and the legend
    ax.set_xlabel('Z')
    ax.set_ylabel('Y')
    ax.set_zlabel('X')
    ax.legend()

    # Display the plot
    plt.show()

def find_path_3d(start, goal, obstacles=None, plot_flag=False):
    """
    A function that combines A* algorithm and 3D path visualization.
    
    Args:
        start (tuple): The starting point as (depth, row, col).
        goal (tuple): The goal point as (depth, row, col).
        obstacles (list): A list of obstacle coordinates as (depth, row, col).
        plot_flag (bool): If True, the grid and path will be plotted.
    
    Returns:
        list: The path from start to goal as a list of (depth, row, col) tuples.
    """
    # Save the original start and goal
    start_orig = start
    goal_orig = goal

    # Convert float coordinates to integers (round or floor)
    start = tuple(map(lambda x: int(round(x)), start))  # Rounding float coordinates
    goal = tuple(map(lambda x: int(round(x)), goal))    # Rounding float coordinates

    # Dynamically adjust grid size to cover both start and goal
    grid_boundaries = get_grid_boundaries(start, goal)
    x_max = grid_boundaries['x_max']
    x_min = grid_boundaries['x_min']
    y_max = grid_boundaries['y_max']
    y_min = grid_boundaries['y_min']
    z_max = grid_boundaries['z_max']
    z_min = grid_boundaries['z_min']

    # Create a grid that includes the bounding box of start and goal
    grid = np.zeros((x_max - x_min + 1, y_max - y_min + 1, z_max - z_min + 1))

    # Place obstacles in the grid (if any)
    if obstacles:
        for obs in obstacles:
            obs_x, obs_y, obs_z = obs
            # Check if the obstacle is within the grid bounds
            if x_min <= obs_x <= x_max and y_min <= obs_y <= y_max and z_min <= obs_z <= z_max:
                # Adjust obstacle coordinates based on the bounding box
                grid[obs_x - x_min, obs_y - y_min, obs_z - z_min] = 1

    # Check if the start or goal are blocked by obstacles
    if grid[start[0] - x_min, start[1] - y_min, start[2] - z_min] == 1:
        print("Start position is blocked by an obstacle!")
        return None
    
    if grid[goal[0] - x_min, goal[1] - y_min, goal[2] - z_min] == 1:
        print("Goal position is blocked by an obstacle!")
        return None

    # Adjust start and goal coordinates to the new grid
    start_adjusted = (start[0] - x_min, start[1] - y_min, start[2] - z_min)
    goal_adjusted = (goal[0] - x_min, goal[1] - y_min, goal[2] - z_min)

    # Find the path using A* algorithm
    path = astar_3d(grid.tolist(), start_adjusted, goal_adjusted)
    
    # Convert the path back to original coordinates
    if path:
        print("Path has been found!")
        path_original = [(x + x_min, y + y_min, z + z_min) for x, y, z in path]
        
        # Prepend the original start point and append the original goal point
        path_original.insert(0, start_orig)  # Prepend the original start point (float)
        path_original.append(goal_orig)      # Append the original goal point (float)
        
        # Plot the 3D grid and the path if requested
        if plot_flag:
            plot_grid_with_path_3d(grid, path, start_adjusted, goal_adjusted)
    else:
        print("No path found!")
        path_original = None

    return path_original

def get_grid_boundaries(start, goal, offset=10):
    x_min = math.floor(min(start[0], goal[0]))
    x_max = math.ceil(max(start[0], goal[0]))
    y_min = math.floor(min(start[1], goal[1]))
    y_max = math.ceil(max(start[1], goal[1]))
    z_min = math.floor(min(start[2], goal[2]))
    z_max = math.ceil(max(start[2], goal[2]))

    boundaries = {
        "x_min": x_min - offset,
        "x_max": x_max + offset,
        "y_min": y_min - offset,
        "y_max": y_max + offset,
        "z_min": z_min - offset,
        "z_max": z_max + offset
    }

    return boundaries

def intersect_bounding_boxes_with_grid(bounding_boxes, grid_boundaries):
    """
    Intersects a list of bounding boxes with the grid boundaries and preserves only the intersection.
    
    Args:
        bounding_boxes (list of dict): List of bounding box dictionaries with 'min' and 'max' coordinates.
        grid_boundaries (dict): Dictionary representing the grid's min/max bounds.
        
    Returns:
        list of dict: List of dictionaries representing the intersected bounding boxes.
    """
    intersected_boxes = []
    
    # Grid boundaries
    grid_x_min, grid_x_max = grid_boundaries['x_min'], grid_boundaries['x_max']
    grid_y_min, grid_y_max = grid_boundaries['y_min'], grid_boundaries['y_max']
    grid_z_min, grid_z_max = grid_boundaries['z_min'], grid_boundaries['z_max']
    
    for box in bounding_boxes:
        # Calculate intersection in each dimension
        x_min = max(box['x_min'], grid_x_min)
        x_max = min(box['x_max'], grid_x_max)
        
        y_min = max(box['y_min'], grid_y_min)
        y_max = min(box['y_max'], grid_y_max)
        
        z_min = max(box['z_min'], grid_z_min)
        z_max = min(box['z_max'], grid_z_max)
        
        # Check if the intersection is valid (min <= max in each dimension)
        if x_min < x_max and y_min < y_max and z_min < z_max:
            # Add the intersected box to the list
            intersected_boxes.append({
                'x_min': x_min, 'x_max': x_max,
                'y_min': y_min, 'y_max': y_max,
                'z_min': z_min, 'z_max': z_max
            })
    
    return intersected_boxes

def convert_obstacles(obstacle_dicts, offset=0):
    """
    Convert a list of obstacles in dictionary format to a set of grid coordinates.
    
    Args:
        obstacle_dicts (list): List of obstacle dictionaries, each with keys 'x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max'.
        
    Returns:
        set: A set of tuples representing the 3D grid coordinates that are blocked by obstacles.
    """
    obstacles_set = set()  # Set to store blocked grid coordinates
    
    for obstacle in tqdm(obstacle_dicts):

        # Extract the bounding box of the obstacle and convert to integer grid coordinates
        x_min = math.floor(obstacle["x_min"])   - offset
        x_max = math.ceil(obstacle["x_max"])    + offset
        y_min = math.floor(obstacle["y_min"])   - offset
        y_max = math.ceil(obstacle["y_max"])    + offset
        z_min = math.floor(obstacle["z_min"])   - offset
        z_max = math.ceil(obstacle["z_max"])    + offset
        
        # Add all coordinates in the bounding box to the set of obstacles
        for x in range(x_min, x_max + 1):
            for y in range(y_min, y_max + 1):
                for z in range(z_min, z_max + 1):
                    obstacles_set.add((x, y, z))  # Add the coordinate as a blocked cell

    return obstacles_set

def remove_large_obstacles(obstacles, threshold=100_000):
    filtered_obstacles = []
    for obstacle in obstacles:
        area = (obstacle['x_max'] - obstacle['x_min']) * (obstacle['y_max'] - obstacle['y_min'])
        if area <= threshold:
            filtered_obstacles.append(obstacle)
    return filtered_obstacles

def remove_ego_vehicle(obstacles, start, offset=0):
    """
    Remove obstacles that contain the ego vehicle's start position within their bounding box.

    Args:
        obstacles (list): List of obstacle dictionaries, each with keys 'x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max'.
        start (tuple or list): The (x, y, z) coordinates of the ego vehicle.

    Returns:
        list: The filtered list of obstacles that do not contain the ego vehicle's position.
    """
    filtered_obstacles = []
    
    for obstacle in obstacles:
        # Extract the bounding box coordinates
        x_min = math.floor(obstacle["x_min"])   - offset
        x_max = math.ceil(obstacle["x_max"])    + offset
        y_min = math.floor(obstacle["y_min"])   - offset
        y_max = math.ceil(obstacle["y_max"])    + offset
        z_min = math.floor(obstacle["z_min"])   - offset
        z_max = math.ceil(obstacle["z_max"])    + offset
        
        # Check if the start (ego vehicle) is inside the bounding box of the obstacle
        if not (x_min <= start[0] <= x_max and y_min <= start[1] <= y_max and z_min <= start[2] <= z_max):
            filtered_obstacles.append(obstacle)

    return filtered_obstacles

def load_obstacles(vehicle_type):
    # Load the file of obstacles
    obstacles_path = f"/catkin_ws/src/planner/script/utils/tmp/static_obstacles.json"
    with open(obstacles_path, 'r') as f:
        obstacles = json.load(f)

    return obstacles

def compute_velocities(points, velocity_magnitude=5.0):
    velocities = []

    # If there are no points or only one point, return an empty list or a list with one zero vector
    if len(points) <= 1:
        return [np.zeros(3)]

    for i in range(len(points)):
        if i == 0 or i == len(points) - 1:
            # For the first and last points, velocity is zero
            velocities.append(np.zeros(3))
        else:
            # Compute the velocity vector from points[i] to points[i+1]
            direction = points[i + 1] - points[i]  # Correct direction from points[i] to points[i+1]
            direction_norm = np.linalg.norm(direction)  # Calculate the norm of the direction vector

            # Normalize and scale the direction vector to have the desired velocity magnitude
            if direction_norm != 0:
                velocity_vector = (direction / direction_norm) * velocity_magnitude
            else:
                velocity_vector = np.zeros(3)

            velocities.append(velocity_vector)

    return velocities

def plot_path_pv(path, obstacles_as_points):
    if path:
        path = np.array(path)
        path_rgba = np.array([[1, 0, 0]] * path.shape[0]).astype(float)
    else:
        path = np.empty(shape=(0, 3))
        path_rgba = np.empty(shape=(0, 3))

    # Visualize the obstacles
    points = np.array(list(obstacles_as_points))
    print("Plotting")
    rgba = points - points.min(axis=0)
    rgba = rgba / rgba.max(axis=0)
    pv.plot(np.concatenate((points, path), axis=0), scalars=np.concatenate((rgba, path_rgba), axis=0), render_points_as_spheres=True, point_size=10, cpos='xy', rgba=True)

class PathPlanner(BasePathPlanner):
    """
    Path planner class. It utilizes the starting point and the target point to calculate the path.
    """
    def __init__(self, config) -> None:
        super().__init__()
        
        self.config = config
        
        # Start the ROS node
        rospy.init_node("planner")

        # Waypoint counter
        self.waypoint_counter = 0
        self.last_speed_check_time = None # last speed check time in s
        self.last_speed_check_coordinate = None # last speed check position as a 3D numpy array (x, y, z)

        # Publish to the target waypoint topic
        self.target_waypoint_pub = rospy.Publisher('/target/waypoint', Float32MultiArray, queue_size=1)

        # Calculate the path to the target
        self.start_point, self.end_point = self.get_start_end_points(config)
        self.waypoints, self.velocities = self.get_waypoints(self.start_point, self.end_point)

        # Subscribe to the global target topic
        self.global_target_sub = rospy.Subscriber(config['ego_vehicle']['reference_topic'], Twist, self.global_target_callback)
        self.pose_sub = rospy.Subscriber(f"/{config['ego_vehicle']['type']}/pose", PoseStamped, self.pose_callback)

    def get_waypoints(self, start_point: np.array, end_point: np.array) -> List[np.array]:
        """
        Identify the waypoints towards the global target.

        Args:
            - start_point (np.array): starting point
            - end_point (np.array): end point

        Return:
            - waypoints (List[np.array]): a list of numpy arrays of shape (3,), where each entry contains 3D coordinates
        """
        if self.config['ego_vehicle']['planner']['type'] == 'simple':
            waypoints = [end_point]
            velocities = [0]
        elif self.config['ego_vehicle']['planner']['type'] == 'a_star':
            # Load the obstacles
            time.sleep(10)
            print("Loading the obstacles...")
            obstacles = load_obstacles(self.config['ego_vehicle']['type'])

            # Remove obstacles with areas greater than 100k
            obstacles = remove_large_obstacles(obstacles)

            # Remove the ego vehicle
            offset = 3
            obstacles = remove_ego_vehicle(obstacles, start_point.tolist(), offset=offset)

            # Convert the obstacles into the required format
            obstacles = convert_obstacles(obstacles, offset=offset)
            print(f"Number of obstacle points: {len(obstacles)}")
            
            # Find the path
            print("Finding the path...")
            print(f"Start: {start_point.tolist()} | Goal: {end_point.tolist()}")
            waypoints = find_path_3d(tuple(start_point.tolist()), tuple(end_point.tolist()), obstacles=obstacles, plot_flag=False)

            # Plot the path
            plot_path = False
            if plot_path:
                plot_path_pv(waypoints, obstacles)

            velocities = compute_velocities([np.array(x) for x in waypoints], velocity_magnitude=0.5)
        else:
            raise ValueError(f"Unknown planner {self.config['ego_vehicle']['planner']['type']}")

        return waypoints, velocities

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

    def run(self):
        r = rospy.Rate(10)
        start_time = time.time()

        while not rospy.is_shutdown():
            # Publish the current waypoint
            message = Float32MultiArray()
            message.data = [
                self.waypoints[self.waypoint_counter][0],
                self.waypoints[self.waypoint_counter][1],
                self.waypoints[self.waypoint_counter][2],
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
        # Only do something if the waypoint counter is not the last counter
        if self.waypoint_counter < len(self.waypoints) - 1:
            # Current position
            x_curr = data.pose.position.x
            y_curr = data.pose.position.y
            z_curr = data.pose.position.z

            # Current waypoint
            x_waypoint = self.waypoints[self.waypoint_counter][0]
            y_waypoint = self.waypoints[self.waypoint_counter][1]
            z_waypoint = self.waypoints[self.waypoint_counter][2]

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

    def global_target_callback(self, data):
        pass

class SafePathPlanner(PathPlanner):
    """
    Enhanced path planner with safety verification capabilities.
    This class extends the base PathPlanner with MLN-based safety checks and real-time verification.
    """
    def __init__(self, config) -> None:
        try:
            # Initialize base planner
            super().__init__(config)
            
            # Initialize logger
            self.logger = logging.getLogger("SafePathPlanner")
            self.logger.setLevel(logging.INFO)
            if not self.logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)
                self.logger.addHandler(ROSLogHandler())  # Add ROS logging
            
            # Validate safety config
            safety_config = config.get('safety_config', {})
            if not isinstance(safety_config, dict):
                raise ValueError("safety_config must be a dictionary")
            
            # Initialize safety verifier with error handling
            try:
                self.safety_verifier = MLNSafetyVerifier(safety_config)
            except Exception as e:
                self.logger.error(f"Failed to initialize MLN Safety Verifier: {str(e)}")
                raise
            
            # Initialize safety-specific state with defaults
            self.current_position = None
            self.current_velocity = np.zeros(3)
            self.current_attitude_rates = np.zeros(3)
            self.current_battery_level = 100.0
            self.gps_signal_strength = 1.0
            self.wind_data = {'speed': 0.0, 'gusts': False, 'direction': 0.0}
            self.obstacles = []
            self.last_verification_result = None
            
            # Set up additional safety publishers with queue_size and latch
            self.safety_status_pub = rospy.Publisher(
                '/safety/status',
                String,
                queue_size=10,
                latch=True  # Keep last message
            )
            
            # Set up additional safety subscribers with error handling
            self._setup_safety_subscribers()
            
        except Exception as e:
            rospy.logerr(f"Failed to initialize SafePathPlanner: {str(e)}")
            raise

    def _setup_safety_subscribers(self):
        """Set up all necessary safety-related ROS subscribers with error handling."""
        try:
            # Core flight data
            self.velocity_sub = rospy.Subscriber(
                f"/{self.config['ego_vehicle']['type']}/velocity",
                Twist,
                self.velocity_callback,
                queue_size=10
            )
            self.attitude_rates_sub = rospy.Subscriber(
                f"/{self.config['ego_vehicle']['type']}/attitude_rates",
                Twist,
                self.attitude_rates_callback,
                queue_size=10
            )
            
            # Additional state data
            self.battery_sub = rospy.Subscriber(
                f"/{self.config['ego_vehicle']['type']}/battery",
                Float32,
                self.battery_callback,
                queue_size=10
            )
            self.gps_sub = rospy.Subscriber(
                f"/{self.config['ego_vehicle']['type']}/gps_status",
                Float32,
                self.gps_callback,
                queue_size=10
            )
            self.wind_sub = rospy.Subscriber(
                f"/{self.config['ego_vehicle']['type']}/wind",
                WindStatus,
                self.wind_callback,
                queue_size=10
            )
            self.obstacles_sub = rospy.Subscriber(
                f"/{self.config['ego_vehicle']['type']}/obstacles",
                ObstacleArray,
                self.obstacles_callback,
                queue_size=10
            )
        except Exception as e:
            rospy.logerr(f"Failed to set up safety subscribers: {str(e)}")
            raise

    def get_current_state_data(self) -> Dict:
        """Gather all current state data for safety verification."""
        if self.current_position is None:
            return {}

        try:
            # Calculate additional metrics
            path_deviation = 0.0
            if self.waypoint_counter < len(self.waypoints):
                current_waypoint = np.array(self.waypoints[self.waypoint_counter])
                path_deviation = np.linalg.norm(self.current_position - current_waypoint)

            # Get nearest obstacle distance
            nearest_obstacle_distance = float('inf')
            if self.obstacles:
                for obstacle in self.obstacles:
                    dist = np.linalg.norm(self.current_position - np.array([
                        (obstacle['x_min'] + obstacle['x_max']) / 2,
                        (obstacle['y_min'] + obstacle['y_max']) / 2,
                        (obstacle['z_min'] + obstacle['z_max']) / 2
                    ]))
                    nearest_obstacle_distance = min(nearest_obstacle_distance, dist)

            # Compile state data
            state_data = {
                # Position and movement
                'altitude': self.current_position[2],
                'velocity': np.linalg.norm(self.current_velocity),
                'vertical_speed': self.current_velocity[2],
                'horizontal_speed': np.linalg.norm(self.current_velocity[:2]),
                
                # Attitude
                'yaw_rate': abs(self.current_attitude_rates[2]),
                'roll_rate': abs(self.current_attitude_rates[0]),
                'pitch_rate': abs(self.current_attitude_rates[1]),
                
                # Flight phase
                'is_landing': self.waypoint_counter >= len(self.waypoints) - 3,
                'is_final_approach': self.waypoint_counter == len(self.waypoints) - 1,
                'is_takeoff': self.waypoint_counter < 3,
                'is_cruising': 3 <= self.waypoint_counter < len(self.waypoints) - 3,
                
                # Path information
                'path_deviation': path_deviation,
                'waypoint_distance': path_deviation,
                
                # Obstacles
                'nearest_obstacle_distance': nearest_obstacle_distance,
                'obstacle_density': len(self.obstacles) / 100.0,  # Normalized density
                
                # System status
                'battery_level': self.current_battery_level,
                'gps_strength': self.gps_signal_strength,
                
                # Environmental
                'wind_speed': self.wind_data['speed'],
                'wind_gust_detected': self.wind_data['gusts'],
            }

            return state_data
        except Exception as e:
            rospy.logerr(f"Error getting state data: {str(e)}")
            return {}

    def run(self):
        """Main execution loop with safety verification."""
        try:
            r = rospy.Rate(10)
            start_time = time.time()

            while not rospy.is_shutdown():
                try:
                    if self.current_position is None:
                        r.sleep()
                        continue

                    # Get current state data
                    state_data = self.get_current_state_data()
                    
                    # Get next waypoint and determine planning action
                    if self.waypoint_counter < len(self.waypoints):
                        next_waypoint = np.array(self.waypoints[self.waypoint_counter])
                        planning_action = self._determine_planning_action(
                            self.current_position, 
                            next_waypoint
                        )
                        
                        # Perform safety verification
                        try:
                            verification_result = self.safety_verifier.verify_safety(
                                planning_action=planning_action,
                                state_data=state_data,
                                current_waypoint=self.current_position,
                                next_waypoint=next_waypoint
                            )
                            
                            self.last_verification_result = verification_result
                            
                            # Log verification results
                            self.safety_verifier.log_verification_result(verification_result)
                            
                            # Publish safety status
                            self._publish_safety_status(verification_result)
                            
                            if verification_result.is_safe:
                                # Convert verified actions to waypoint adjustment
                                modified_waypoint = self.safety_verifier.convert_actions_to_waypoint_adjustment(
                                    verification_result.verified_actions,
                                    self.current_position,
                                    next_waypoint
                                )
                            else:
                                # If unsafe, hover in place or execute emergency action
                                modified_waypoint = self._handle_unsafe_condition(
                                    verification_result,
                                    self.current_position
                                )
                        except Exception as e:
                            rospy.logerr(f"Safety verification failed: {str(e)}")
                            # Default to current position if safety verification fails
                            modified_waypoint = self.current_position
                        
                        # Prepare message
                        message = Float32MultiArray()
                        message.data = modified_waypoint.tolist() + [0, 0, 0]  # Add orientation
                    else:
                        # Stay at current position if no more waypoints
                        message = Float32MultiArray()
                        message.data = self.current_position.tolist() + [0, 0, 0]

                    # Run some pre-checks
                    if time.time() - start_time < 5:
                        message.data = [
                            self.config['ego_vehicle']['location']['x'],
                            self.config['ego_vehicle']['location']['y'],
                            self.config['ego_vehicle']['location']['z'],
                            0, 0, 0
                        ]

                    # Publish the data with error handling
                    try:
                        self.target_waypoint_pub.publish(message)
                    except Exception as e:
                        rospy.logerr(f"Failed to publish waypoint: {str(e)}")

                    r.sleep()
                except Exception as e:
                    rospy.logerr(f"Error in main loop: {str(e)}")
                    r.sleep()

        except rospy.ROSInterruptException:
            rospy.loginfo("ROS node interrupted")
        except Exception as e:
            rospy.logerr(f"Fatal error in run loop: {str(e)}")

    def _determine_planning_action(self, 
                                 current_pos: np.ndarray, 
                                 next_waypoint: np.ndarray) -> PlanningAction:
        """Determine the primary planning action based on waypoint positions."""
        direction = next_waypoint - current_pos
        
        # Determine primary movement direction
        if abs(direction[2]) > max(abs(direction[0]), abs(direction[1])):
            return PlanningAction.UP if direction[2] > 0 else PlanningAction.DOWN
        elif abs(direction[0]) > abs(direction[1]):
            return PlanningAction.FORWARD if direction[0] > 0 else PlanningAction.BACKWARD
        else:
            return PlanningAction.RIGHT if direction[1] > 0 else PlanningAction.LEFT

    def _handle_unsafe_condition(self, 
                               verification_result: SafetyVerificationResult,
                               current_position: np.ndarray) -> np.ndarray:
        """Handle unsafe conditions by determining appropriate emergency action."""
        if EnvironmentalPredicate.OBSTACLES_IN_1M_RADIUS in verification_result.active_predicates:
            # Emergency stop - stay exactly where we are
            return current_position
            
        if EnvironmentalPredicate.BATTERY_CRITICAL in verification_result.active_predicates:
            # Emergency landing - move down while maintaining horizontal position
            return np.array([
                current_position[0],
                current_position[1],
                max(0, current_position[2] - 0.5)  # Descend slowly
            ])
            
        # Default safety behavior - hover with slight adjustments for stability
        return current_position + np.random.normal(0, 0.1, 3)

    def _publish_safety_status(self, verification_result: SafetyVerificationResult):
        """Publish safety status information with error handling."""
        try:
            status_msg = String()
            status = {
                'is_safe': verification_result.is_safe,
                'warnings': verification_result.safety_warnings,
                'score': float(verification_result.score),  # Ensure JSON serializable
                'active_predicates': [p.name for p in verification_result.active_predicates],
                'verified_actions': [a.name for a in verification_result.verified_actions],
                'timestamp': rospy.Time.now().to_sec()
            }
            status_msg.data = json.dumps(status)
            self.safety_status_pub.publish(status_msg)
        except Exception as e:
            rospy.logerr(f"Failed to publish safety status: {str(e)}")

    # Callback methods for subscribers
    def velocity_callback(self, data):
        """Store velocity data with error handling."""
        try:
            self.current_velocity = np.array([
                data.linear.x,
                data.linear.y,
                data.linear.z
            ])
        except Exception as e:
            rospy.logerr(f"Error in velocity callback: {str(e)}")

    def attitude_rates_callback(self, data):
        """Store attitude rates data with error handling."""
        try:
            self.current_attitude_rates = np.array([
                data.angular.x,  # Roll rate
                data.angular.y,  # Pitch rate
                data.angular.z   # Yaw rate
            ])
        except Exception as e:
            rospy.logerr(f"Error in attitude rates callback: {str(e)}")

    def battery_callback(self, data):
        """Store battery level data with error handling."""
        try:
            self.current_battery_level = data.data
        except Exception as e:
            rospy.logerr(f"Error in battery callback: {str(e)}")

    def gps_callback(self, data):
        """Store GPS signal strength data with error handling."""
        try:
            self.gps_signal_strength = data.data
        except Exception as e:
            rospy.logerr(f"Error in GPS callback: {str(e)}")

    def wind_callback(self, data):
        """Store wind data with error handling."""
        try:
            self.wind_data = {
                'speed': data.speed,
                'gusts': data.gusts,
                'direction': data.direction
            }
        except Exception as e:
            rospy.logerr(f"Error in wind callback: {str(e)}")

    def obstacles_callback(self, data):
        """Store obstacle data with error handling and validation."""
        try:
            self.obstacles = []
            for obstacle in data.obstacles:
                # Validate obstacle data
                if not all(hasattr(obstacle, attr) for attr in 
                         ['x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max', 'is_moving']):
                    rospy.logwarn("Received invalid obstacle data")
                    continue
                    
                # Validate coordinates
                if not (obstacle.x_min <= obstacle.x_max and 
                       obstacle.y_min <= obstacle.y_max and 
                       obstacle.z_min <= obstacle.z_max):
                    rospy.logwarn("Received invalid obstacle coordinates")
                    continue
                
                self.obstacles.append({
                    'x_min': obstacle.x_min,
                    'x_max': obstacle.x_max,
                    'y_min': obstacle.y_min,
                    'y_max': obstacle.y_max,
                    'z_min': obstacle.z_min,
                    'z_max': obstacle.z_max,
                    'is_moving': obstacle.is_moving,
                    'velocity': np.array(obstacle.velocity) if len(obstacle.velocity) == 3 else np.zeros(3)
                })
        except Exception as e:
            rospy.logerr(f"Error in obstacles callback: {str(e)}")

    def pose_callback(self, data):
        """Override base pose callback to store current position."""
        try:
            # Store current position
            self.current_position = np.array([
                data.pose.position.x,
                data.pose.position.y,
                data.pose.position.z
            ])
            
            # Call parent's pose callback
            super().pose_callback(data)
            
        except Exception as e:
            rospy.logerr(f"Error in pose callback: {str(e)}")

    def _initialize_rules(self) -> List[Rule]:
        """Initialize all safety rules with their weights."""
        rules = []
        
        # CRITICAL EMERGENCY RULES (Priority 9.5-10.0)
        rules.extend([
            # Existing emergency rules
            Rule(
                conditions=[EnvironmentalPredicate.OBSTACLES_IN_1M_RADIUS],
                output_actions=[Action.EMERGENCY_STOP],
                weight=10.0
            ),
            Rule(
                conditions=[EnvironmentalPredicate.BATTERY_CRITICAL],
                output_actions=[Action.EMERGENCY_LANDING_REQUIRED],
                weight=9.8
            ),
            
            # New emergency rules
            Rule(
                conditions=[
                    EnvironmentalPredicate.OBSTACLES_IN_2M_RADIUS,
                    EnvironmentalPredicate.VELOCITY_ABOVE_5
                ],
                output_actions=[Action.EMERGENCY_STOP],
                weight=9.7
            ),
            Rule(
                conditions=[
                    EnvironmentalPredicate.GPS_SIGNAL_WEAK,
                    EnvironmentalPredicate.VISIBILITY_LOW
                ],
                output_actions=[Action.HOVER_IN_PLACE, Action.STABILIZE_POSITION],
                weight=9.6
            ),
            Rule(
                conditions=[
                    EnvironmentalPredicate.WIND_STRONG,
                    EnvironmentalPredicate.ALTITUDE_ABOVE_100
                ],
                output_actions=[Action.SLOW_DESCENT, Action.STABILIZE_POSITION],
                weight=9.5
            )
        ])

        # LANDING SAFETY RULES (Priority 8.5-9.4)
        rules.extend([
            # Existing landing rules
            Rule(
                conditions=[
                    EnvironmentalPredicate.LANDING,
                    EnvironmentalPredicate.ALTITUDE_BELOW_10,
                    EnvironmentalPredicate.VELOCITY_ABOVE_2
                ],
                output_actions=[Action.DECELERATE],
                weight=9.4
            ),
            
            # New landing rules
            Rule(
                conditions=[
                    EnvironmentalPredicate.FINAL_APPROACH,
                    EnvironmentalPredicate.WIND_GUST_DETECTED
                ],
                output_actions=[Action.HOVER_IN_PLACE, Action.STABILIZE_POSITION],
                weight=9.3
            ),
            Rule(
                conditions=[
                    EnvironmentalPredicate.LANDING,
                    EnvironmentalPredicate.OBSTACLE_DENSITY_HIGH
                ],
                output_actions=[Action.HOVER_IN_PLACE],
                weight=9.2
            ),
            Rule(
                conditions=[
                    EnvironmentalPredicate.TOUCHDOWN_PHASE,
                    EnvironmentalPredicate.LANDING_SURFACE_FLAT,
                    EnvironmentalPredicate.LANDING_ZONE_CLEAR
                ],
                output_actions=[Action.SLOW_DESCENT],
                weight=9.0
            ),
            Rule(
                conditions=[
                    EnvironmentalPredicate.LANDING,
                    EnvironmentalPredicate.BATTERY_LOW
                ],
                output_actions=[Action.RAPID_DESCENT],
                weight=8.8
            )
        ])

        # STABILITY AND CONTROL RULES (Priority 7.5-8.4)
        rules.extend([
            Rule(
                conditions=[
                    EnvironmentalPredicate.YAW_RATE_HIGH,
                    EnvironmentalPredicate.ROLL_RATE_HIGH
                ],
                output_actions=[Action.STABILIZE_POSITION, Action.HOVER_IN_PLACE],
                weight=8.4
            ),
            Rule(
                conditions=[
                    EnvironmentalPredicate.ALTITUDE_CHANGE_RATE_HIGH,
                    EnvironmentalPredicate.PASSENGER_COMFORT_PRIORITY
                ],
                output_actions=[Action.STABILIZE_POSITION],
                weight=8.2
            ),
            Rule(
                conditions=[
                    EnvironmentalPredicate.TURBULENCE_HIGH,
                    EnvironmentalPredicate.VELOCITY_ABOVE_5
                ],
                output_actions=[Action.DECELERATE, Action.STABILIZE_POSITION],
                weight=8.0
            ),
            Rule(
                conditions=[
                    EnvironmentalPredicate.WIND_STRONG,
                    EnvironmentalPredicate.HORIZONTAL_SPEED_HIGH
                ],
                output_actions=[Action.DECELERATE],
                weight=7.8
            )
        ])

        # OBSTACLE AVOIDANCE RULES (Priority 6.5-7.4)
        rules.extend([
            Rule(
                conditions=[
                    EnvironmentalPredicate.OBSTACLE_AHEAD_2M,
                    EnvironmentalPredicate.VELOCITY_ABOVE_2
                ],
                output_actions=[Action.DECELERATE, Action.HOVER_IN_PLACE],
                weight=7.4
            ),
            Rule(
                conditions=[
                    EnvironmentalPredicate.OBSTACLE_MOVING,
                    EnvironmentalPredicate.OBSTACLES_IN_5M_RADIUS
                ],
                output_actions=[Action.HOVER_IN_PLACE],
                weight=7.2
            ),
            Rule(
                conditions=[
                    EnvironmentalPredicate.OBSTACLE_DENSITY_HIGH,
                    EnvironmentalPredicate.CRUISE_PHASE
                ],
                output_actions=[Action.DECELERATE],
                weight=7.0
            )
        ])

        # PATH FOLLOWING RULES (Priority 5.5-6.4)
        rules.extend([
            Rule(
                conditions=[
                    EnvironmentalPredicate.DEVIATION_FROM_PATH_HIGH,
                    EnvironmentalPredicate.VELOCITY_ABOVE_5
                ],
                output_actions=[Action.DECELERATE],
                weight=6.4
            ),
            Rule(
                conditions=[
                    EnvironmentalPredicate.WAYPOINT_NEAR,
                    EnvironmentalPredicate.VELOCITY_ABOVE_2
                ],
                output_actions=[Action.DECELERATE],
                weight=6.2
            ),
            Rule(
                conditions=[
                    EnvironmentalPredicate.WAYPOINT_FAR,
                    EnvironmentalPredicate.VELOCITY_BELOW_2,
                    EnvironmentalPredicate.BATTERY_SUFFICIENT
                ],
                output_actions=[Action.ACCELERATE],
                weight=6.0
            )
        ])

        # COMFORT AND EFFICIENCY RULES (Priority 4.5-5.4)
        rules.extend([
            Rule(
                conditions=[
                    EnvironmentalPredicate.PASSENGER_COMFORT_PRIORITY,
                    EnvironmentalPredicate.ACCELERATION_HIGH
                ],
                output_actions=[Action.DECELERATE],
                weight=5.4
            ),
            Rule(
                conditions=[
                    EnvironmentalPredicate.ENERGY_EFFICIENT_MODE,
                    EnvironmentalPredicate.CRUISE_PHASE,
                    EnvironmentalPredicate.ALTITUDE_STABLE
                ],
                output_actions=[Action.MAINTAIN],
                weight=5.2
            ),
            Rule(
                conditions=[
                    EnvironmentalPredicate.BATTERY_LOW,
                    EnvironmentalPredicate.WAYPOINT_FAR
                ],
                output_actions=[Action.DECELERATE],
                weight=5.0
            )
        ])

        # ENVIRONMENTAL ADAPTATION RULES (Priority 4.0-4.4)
        rules.extend([
            Rule(
                conditions=[
                    EnvironmentalPredicate.VISIBILITY_LOW,
                    EnvironmentalPredicate.CRUISE_PHASE
                ],
                output_actions=[Action.DECELERATE],
                weight=4.4
            ),
            Rule(
                conditions=[
                    EnvironmentalPredicate.WIND_GUST_DETECTED,
                    EnvironmentalPredicate.ALTITUDE_STABLE
                ],
                output_actions=[Action.STABILIZE_POSITION],
                weight=4.2
            ),
            Rule(
                conditions=[
                    EnvironmentalPredicate.GPS_SIGNAL_WEAK,
                    EnvironmentalPredicate.VELOCITY_ABOVE_2
                ],
                output_actions=[Action.DECELERATE],
                weight=4.0
            )
        ])

        return rules