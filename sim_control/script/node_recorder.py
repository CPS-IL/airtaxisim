#!/usr/bin/env python3

import atexit
import os
import rosbag
import rospy
import signal
from threading import Lock

from std_msgs.msg import Bool, String, Float32MultiArray
from geometry_msgs.msg import PoseStamped, Twist
from sensor_msgs.msg import Image

from utils.config import load_yaml_file, log
from utils import constants

class TopicRecorder:
    def __init__(self):
        self.config = load_yaml_file(constants.merged_config_path, __file__)

        rospy.init_node('sim_recorder')
        self.rate = rospy.Rate(constants.frequency_low)

        # Set up the topics to be recorded
        self.topics_to_record = [
            (f"/{self.config['ego_vehicle']['type']}/pose", PoseStamped),
            (self.config['ego_vehicle']['reference_topic'], Twist),
            (f"/{self.config['ego_vehicle']['type']}/velocity", Twist),
            ("/yolo_node/yolo_pred_frame", Image),
            ("/yolo_node/yolo_predictions", Float32MultiArray)
        ]

        self.recording = False
        self.recording_done = False
        self.bag = None

        try:
            self.recording_requested = self.config['record_rosbag']
        except KeyError:
            self.recording_requested = False

        if not self.recording_requested:
            return

        self.lock = Lock()

        # Subscribe to the trigger topic
        rospy.Subscriber('/sim_start/started', Bool, self.sim_start_callback)

        # Subscribe to the topics for recording
        self.subscribers = []
        for topic, msg_type in self.topics_to_record:
            self.subscribers.append(rospy.Subscriber(topic, msg_type, self.record_callback, callback_args=topic))

        atexit.register(self.terminate)
        signal.signal(signal.SIGINT, self.shutdown_handler)
        signal.signal(signal.SIGTERM, self.shutdown_handler)

    def sim_start_callback(self, msg):
        if msg.data and not self.recording and not self.recording_done:
            log.info("Recording started")
            self.recording = True
            rosbag_path = os.path.join(os.path.dirname(constants.merged_config_path), "recorded_topics.bag")
            log.info(f"Starting rosbag at {rosbag_path}")
            self.bag = rosbag.Bag(rosbag_path, 'w')

    def record_callback(self, msg, topic):
        if self.recording and self.bag:
            try:
                with self.lock:
                    self.bag.write(topic, msg, rospy.get_rostime())
                    self.bag.flush()
            except Exception as e:
                rospy.logerr(f"Failed to write to bag file: {e}")

    def shutdown_handler(self, signal_number, frame):
        self.terminate(signal_number)

    def terminate(self, signal_number=None):
        self.recording = False
        self.recording_done = True
        if self.bag:
            self.bag.flush()
            self.bag.close()
            self.bag = None
        if signal_number == None:
            log.success(f"Recording stopped")
        else:
            log.success(f"Recording stopped with signal {signal_number}")

    def run(self):
        try:
            while not rospy.is_shutdown():
                self.rate.sleep()
        except rospy.ROSInterruptException:
            self.terminate()

if __name__ == "__main__":
    recorder = TopicRecorder()
    if recorder.recording_requested:
        recorder.run()
        recorder.terminate()
