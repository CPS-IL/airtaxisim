#!/usr/bin/env python3
import rospy
import torch
import numpy as np


from std_msgs.msg import Float32, Bool
from std_msgs.msg import Float64MultiArray        # See https://gist.github.com/jarvisschultz/7a886ed2714fac9f5226
from std_msgs.msg import MultiArrayDimension      # See http://docs.ros.org/api/std_msgs/html/msg/MultiArrayLayout.html


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import rospy
from std_msgs.msg import String

class ROSNode:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('node_torch')

        # Create a publisher for a topic
        self.publisher = rospy.Publisher('example_topic', String, queue_size=10)

        # Create a subscriber for a topic
        rospy.Subscriber('/octomap_conv_array', Float64MultiArray, self.callback)

    def callback(self, msg):
        # This method is called when a new message is received on the subscribed topic
        # rospy.loginfo("Received message: %s", data.data)
        rospy.loginfo("Received")

        height = msg.layout.dim[0].size
        width = msg.layout.dim[1].size
        np_array = np.array(msg.data).reshape((height, width))

        torch_array = torch.FloatTensor(np_array).to(DEVICE)


        print("msg:", torch_array.size())

        occupancy = torch_array[:, -1]
        ind_occupied = occupancy > 0

        print("occupancy:", occupancy.size())

        torch_array_fil = torch_array[ind_occupied, :]

        print("fil_msg:", torch_array_fil.size())






        # index = torch_array[:][4] > 0

        # print(index.size())




        # torch_array = torch_array[torch_array[:][4] > 0][:]

        


        

        # print(torch_array[:5])



        # column_index = pd.Index(multiarray_msg.layout.dim[1].label.split(','))
        # row_index = pd.Index(multiarray_msg.layout.dim[0].label.split(','))
        # df = pd.DataFrame(np_state_matrix, index = row_index, columns=column_index) 



    def publish_message(self, message):
        # Publish a message to the topic
        self.publisher.publish(message)

if __name__ == '__main__':
    try:
        node = ROSNode()
        rate = rospy.Rate(10)  # 10 Hz (adjust the rate as needed)
        while not rospy.is_shutdown():
            message = "Hello, ROS!"
            node.publish_message(message)
            rate.sleep()
    except rospy.ROSInterruptException:
        pass
