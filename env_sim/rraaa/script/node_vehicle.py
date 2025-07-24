#!/usr/bin/env python3

import functools as ft
import ipdb
import rospy

from loguru import logger as log

from geometry_msgs.msg import PoseStamped

from utils.vehicle import Vehicle_Node
from utils.config import load_yaml_file
from utils import constants


class MiniHawk_Node(Vehicle_Node):
    def __init__(self, config) -> None:
        super().__init__(config)

        # Topic to which pose data is published
        self.pose_topic = "/minihawk_SIM/mavros/local_position/pose"

    def main(self):
        def pose_processor(self, pose_data):
            """
            Receive pose data from the Gazebo simulation, and publish it to the CARLA pose topic.
            """
            # Set the location
            # NOTE: I introduced a "-" sign at the Y-coordinate on Oct-27-2024, because the Y-axis was flipped.
            self.vehicle_pose_msg.pose.position.x = pose_data.pose.position.x
            self.vehicle_pose_msg.pose.position.y = -pose_data.pose.position.y
            self.vehicle_pose_msg.pose.position.z = pose_data.pose.position.z

            # Set the orientation
            # NOTE: Gazebo and CARLA have different frames of reference: right-handed and left-handed respectively. The code below accounts for it correctly.
            x = pose_data.pose.orientation.x
            y = pose_data.pose.orientation.y
            z = pose_data.pose.orientation.z
            w = pose_data.pose.orientation.w
            self.vehicle_pose_msg.pose.orientation.x = z
            self.vehicle_pose_msg.pose.orientation.y = y
            self.vehicle_pose_msg.pose.orientation.z = x
            self.vehicle_pose_msg.pose.orientation.w = w
            self.vehicle_pose_pub.publish(self.vehicle_pose_msg)

        def simulate():
            rospy.spin()

        # Subscribe to the ROS topic which publishes the pose
        pose_processor = ft.partial(pose_processor, self)
        rospy.Subscriber(
            self.pose_topic,
            PoseStamped,
            pose_processor
        )

        # Run the simulation
        simulate()

if __name__ == "__main__":
    config = load_yaml_file(constants.merged_config_path, __file__)

    vehicle_type = config['ego_vehicle']['type']
    if vehicle_type == 'minihawk':
        if config['ego_vehicle']['debug']:
            with ipdb.launch_ipdb_on_exception():
                minihawk_node = MiniHawk_Node(config)
                minihawk_node.main()
        else:
            minihawk_node = MiniHawk_Node(config)
            minihawk_node.main()
    else:
        log.info("This node only supports MiniHawk, skipping.")
        while not rospy.is_shutdown():
            rospy.sleep(1)  # Sleep to avoid busy-waiting, adjust as needed
