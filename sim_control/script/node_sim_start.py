#!/usr/bin/env python3

import math
import rospy
from rospy.msg import AnyMsg
from std_msgs.msg import Bool
from geometry_msgs.msg import PoseStamped

from utils.config import load_yaml_file, write_shared_tmp_file, log
from utils import constants

class SimStart:
    def __init__(self):
        self.config = load_yaml_file(constants.merged_config_path, __file__)

        rospy.init_node('sim_start')
        self.rate = rospy.Rate(constants.frequency_low)

        self.sim_start_pub = rospy.Publisher('/sim_start/started', Bool, queue_size=1)
        self.sim_start_pub.publish(False)

        rospy.Subscriber(f"/{self.config['ego_vehicle']['type']}/pose", PoseStamped, self.callback)

    def callback(self, msg: PoseStamped):
        # if msg:
            # log.trace("Simulator has started.")
            # self.sim_start_pub.publish(True)

        # Compute the distance
        x_curr = msg.pose.position.x
        y_curr = msg.pose.position.y
        z_curr = msg.pose.position.z
        x_init = self.config['ego_vehicle']['location']['x']
        y_init = self.config['ego_vehicle']['location']['y']
        z_init = self.config['ego_vehicle']['location']['z']
        dist = math.sqrt(
            (x_curr - x_init) ** 2 +
            (y_curr - y_init) ** 2 +
            (z_curr - z_init) ** 2
        )
        
        if dist < self.config['ego_vehicle']['planner']['distance_threshold']:
            log.trace("Simulator has started.")
            self.sim_start_pub.publish(True)
        else:
            log.trace("Simulator not started.")

    def run(self):
        while not rospy.is_shutdown():
            self.rate.sleep()


if __name__ == "__main__":
    simstart = SimStart()
    simstart.run()
