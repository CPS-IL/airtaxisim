#!/usr/bin/env python3

import rospy
import sensor_msgs.point_cloud2 as pcl2
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header
import numpy as np
import threading

class CarlaMultiLidarMerger:
    def __init__(self):
        rospy.init_node('carla_merged_lidar_publisher')

        self.lock = threading.Lock()
        self.latest_clouds = {}

        # List of topic names to subscribe to
        self.lidar_topics = [
            '/carla_node/lidar_point_cloud_down',
            '/carla_node/lidar_point_cloud_up',
            '/carla_node/lidar_point_cloud_left',
            '/carla_node/lidar_point_cloud_right',
            '/carla_node/lidar_point_cloud_back',
            '/carla_node/lidar_point_cloud_forward',
        ]

        # Subscribe to each topic
        for topic in self.lidar_topics:
            rospy.Subscriber(topic, PointCloud2, self.lidar_callback, callback_args=topic, queue_size=10)

        # Publisher for the merged cloud (what Octomap will listen to)
        self.pub_merged = rospy.Publisher('/carla_node/lidar_point_cloud', PointCloud2, queue_size=1)

        # Publish merged cloud periodically
        rospy.Timer(rospy.Duration(0.1), self.publish_merged_cloud)  # 10 Hz

    def lidar_callback(self, msg, topic):
        with self.lock:
            self.latest_clouds[topic] = msg

    def publish_merged_cloud(self, event):
        with self.lock:
            if not self.latest_clouds:
                return

            all_points = []

            for msg in self.latest_clouds.values():
                points = list(pcl2.read_points(msg, field_names=["x", "y", "z"], skip_nans=True))
                all_points.extend(points)

            if all_points:
                header = Header()
                header.stamp = rospy.Time.now()
                header.frame_id = "sensor"  # Adjust if needed for your TF tree

                merged_cloud = pcl2.create_cloud_xyz32(header, all_points)
                self.pub_merged.publish(merged_cloud)

if __name__ == '__main__':
    try:
        CarlaMultiLidarMerger()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
