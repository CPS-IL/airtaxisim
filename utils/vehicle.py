import rospy
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Twist

from loguru import logger

class Vehicle_Node:
    def __init__(self, config) -> None:

        self.config = config

        # Record the vehicle type
        self.vehicle_type = config['ego_vehicle']['type']

        # Initial position and velocity
        self.initial_position = [float(config['ego_vehicle']['location']['x']),
                                 float(config['ego_vehicle']['location']['y']),
                                 float(config['ego_vehicle']['location']['z'])]
        self.initial_velocity = [float(config['ego_vehicle']['velocity']['x']), 
                                 float(config['ego_vehicle']['velocity']['y']),
                                 float(config['ego_vehicle']['velocity']['z'])]

        # Construct the node
        logger.info(f"Constructing a {self.vehicle_type} node...")
        rospy.init_node(self.vehicle_type)
        self.vehicle_pose_pub = rospy.Publisher(f'/{self.vehicle_type}/pose', PoseStamped, queue_size=1)
        self.vehicle_pose_msg = PoseStamped()
        self.vehicle_vel_pub = rospy.Publisher(f'/{self.vehicle_type}/velocity', Twist, queue_size=1)
        self.vehicle_vel_msg = Twist()

    def main(self):
        raise NotImplementedError