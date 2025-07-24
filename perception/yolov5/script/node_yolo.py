#!/usr/bin/env python3
import rospy
import torch
import numpy as np
import PIL
import os
import argparse

from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32, Bool
from std_msgs.msg import Float32MultiArray        # See https://gist.github.com/jarvisschultz/7a886ed2714fac9f5226
from std_msgs.msg import MultiArrayDimension      # See http://docs.ros.org/api/std_msgs/html/msg/MultiArrayLayout.html
from cv_bridge import CvBridge

from tools.msgpacking import init_matrix_array_ros_msg, pack_multiarray_ros_msg
from tools.sort import Sort

import torch
import numpy as np
import cv2
import matplotlib
import random

import sys, os


def get_latest_model(directory):
    files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.pt')]
    if not files:
        raise FileNotFoundError("No model files found in the directory.")
    latest_file = max(files, key=os.path.getmtime)
    return latest_file


def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def color_list():
    # Return first 10 plt colors as (r,g,b) https://stackoverflow.com/questions/51350872/python-from-color-name-to-rgb
    def hex2rgb(h):
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))
    return [hex2rgb(h) for h in matplotlib.colors.TABLEAU_COLORS.values()]  # or BASE_ (8), CSS4_ (148), XKCD_ (949)

from ultralytics import YOLO

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(os.getcwd())

yolo_directory = os.path.expanduser('~/../catkin_ws/src/yolov5/models')

yolo_path = get_latest_model(yolo_directory)
print(f"Latest model loaded: {os.path.basename(yolo_path)}")



#changes made for bayesian OPT Looping
#yolo_path = os.path.expanduser('~/../catkin_ws/src/yolov5/models/best.pt')

YOLO_MODEL = YOLO(yolo_path) # Yolo v8
# YOLO_MODEL = YOLO("yolo_param/tasnim/best1.pt") # <--- This is Yolo v5
YOLO_MODEL.to(DEVICE)

FREQ_NODE = 10

### ROS Subscriber Callback ###
IMAGE_RECEIVED = None
def fnc_img_callback(msg):
    global IMAGE_RECEIVED
    IMAGE_RECEIVED = msg



def run_yolo_node(args):

    mot_tracker = Sort(max_age=args.max_age, min_hits=args.min_hits, iou_threshold=args.iou_threshold) #create instance of the SORT tracker
    colors = color_list()

    # rosnode node initialization
    rospy.init_node('perception_node')   # rosnode node initialization
    print("Perception_node is initialized at", os.getcwd())

    # subscriber init.
    sub_image = rospy.Subscriber('/carla_node/cam_down/image_raw', Image, fnc_img_callback)   # subscriber init.

    # publishers init.
    pub_yolo_prediction = rospy.Publisher('/yolo_node/yolo_predictions', Float32MultiArray, queue_size=10)   # publisher1 initialization.
    pub_yolo_boundingbox_video = rospy.Publisher('/yolo_node/yolo_pred_frame', Image, queue_size=10)   # publisher2 initialization.
    pub_sort_prediction = rospy.Publisher('/yolo_node/sort_mot_predictions', Float32MultiArray, queue_size=10)   # publisher1 initialization.
    pub_sort_boundingbox_video = rospy.Publisher('/yolo_node/sort_mot_frame', Image, queue_size=10)    # publisher3 initialization.

    rate=rospy.Rate(FREQ_NODE)   # Running rate at 20 Hz

    # a bridge from cv2 image to ROS image
    mybridge = CvBridge()

    # msg init. the msg is to send out numpy array.
    msg_mat = init_matrix_array_ros_msg()
    t_step = 0

    ##############################
    ### Instructions in a loop ###
    ##############################
    while not rospy.is_shutdown():

        t_step += 1

        if IMAGE_RECEIVED is not None:
            np_im = np.frombuffer(IMAGE_RECEIVED.data, dtype=np.uint8).reshape(IMAGE_RECEIVED.height, IMAGE_RECEIVED.width, -1)
            np_im = np.array(np_im)

            with torch.no_grad():
                x_image = torch.FloatTensor(np_im).to(DEVICE).permute(2, 0, 1).unsqueeze(0)/255
                results = YOLO_MODEL.predict(source=x_image, save=False, save_txt=False, verbose=False)  #
                cv2_img = results[0].plot()
                prediction_df = results[0].to_df()

            # Filter Helipads and Take Box coordinates.
            try:
                df_boxes = prediction_df.where(prediction_df['class']==0) # label 0 is helipad
                tgt_boxes = []
                for index, row in df_boxes.iterrows():
                    box = row['box']
                    tgt_boxes.append([box['x1'], box['y1'], box['x2'], box['y2']])

            except:
                tgt_boxes = []


            # Use SOT to track objects.
            if len(tgt_boxes) > 0:
                trackers = mot_tracker.update(np.array(tgt_boxes))  # Bayesian Update
            else:
                trackers = mot_tracker.update(np.empty((0, 5))) # Prediction

            # Draw the tracked detections.
            if len(trackers) > 0:
                for d in trackers:
                    d = d.astype(np.int32)
                    #print('d',d)
                    plot_one_box((d[0],d[1],d[2],d[3]), np_im, color=colors[1], label=str(d[4]%100))

            # Publish the image with the tracked detections.
            image_message = mybridge.cv2_to_imgmsg(np_im, encoding="passthrough")
            image_message.encoding = "rgb8"
            pub_sort_boundingbox_video.publish(image_message)

            try:
                pub_sort_prediction.publish(pack_multiarray_ros_msg(msg_mat, trackers))
            except:
                pub_sort_prediction.publish(pack_multiarray_ros_msg(msg_mat, np.array([[1]])))


            ### Publish the prediction results in results.xyxy[0]) ###
            #                   x1           y1           x2           y2   confidence        class
            # tensor([[7.50637e+02, 4.37279e+01, 1.15887e+03, 7.08682e+02, 8.18137e-01, 0.00000e+00],
            #         [9.33597e+01, 2.07387e+02, 1.04737e+03, 7.10224e+02, 5.78011e-01, 0.00000e+00],
            #         [4.24503e+02, 4.29092e+02, 5.16300e+02, 7.16425e+02, 5.68713e-01, 2.70000e+01]])
            try:
                prediction_np = []
                for index, row in df_boxes.iterrows():
                    box = row['box']
                    confidence = row['confidence']
                    class_label = row['class']
                    prediction_np.append([box['x1'], box['y1'], box['x2'], box['y2'], confidence, class_label])
                prediction_np = np.array(prediction_np)
                pub_yolo_prediction.publish(pack_multiarray_ros_msg(msg_mat, prediction_np))
            except:
                pub_yolo_prediction.publish(pack_multiarray_ros_msg(msg_mat, np.array([[1]])))

            ### Publish the bounding box image ###
            image_message = mybridge.cv2_to_imgmsg(cv2_img, encoding="passthrough")
            image_message.encoding = "rgb8"
            pub_yolo_boundingbox_video.publish(image_message)

        rate.sleep()


def main():

    parser = argparse.ArgumentParser(description='ROS YOLO NODE')
    parser.add_argument('--display', dest='display', help='Display online tracker output (slow) [False]',action='store_true')
    parser.add_argument("--seq_path", help="Path to detections.", type=str, default='data')
    parser.add_argument("--phase", help="Subdirectory in seq_path.", type=str, default='train')
    parser.add_argument("--max_age",
                        help="Maximum number of frames to keep alive a track without associated detections.",
                        type=int, default=10)
    parser.add_argument("--min_hits",
                        help="Minimum number of associated detections before track is initialised.",
                        type=int, default=3)
    parser.add_argument("--iou_threshold", help="Minimum IOU for match.", type=float, default=0.3)

    args, unknown = parser.parse_known_args()

    try:
        run_yolo_node(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')



if __name__ == '__main__':

    try:
        main()
    except rospy.ROSInterruptException:
        pass