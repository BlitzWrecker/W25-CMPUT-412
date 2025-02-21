#!/usr/bin/env python3

# potentially useful for question - 1.1 - 1.4 and 2.1

# import required libraries
import os
import rospy
import numpy as np
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage, Image

import cv2
from cv_bridge import CvBridge

class LaneDetectionNode(DTROS):
    def __init__(self, node_name):
        super(LaneDetectionNode, self).__init__(node_name=node_name, node_type=NodeType.PERCEPTION)
        # add your code here

        # camera calibration parameters (intrinsic matrix and distortion coefficients)
        self.camera_matrix = np.array([[729.3017308196419, 0.0, 296.9297699654982],
                                       [0.0, 714.8576567892494, 194.88265037301576],
                                       [0.0, 0.0, 1.0]])

        # Distortion coefficients (example values, replace with actual ones)
        self.dist_coeffs = np.array(
            [[-1.526832375685591], [2.217300696985744], [-0.00035517449407590306], [-0.013740460640726298], [0.0]])

        # color detection parameters in HSV format
        self.yellow_lower = np.array([20, 100, 100])
        self.yellow_upper = np.array([30, 255, 255])
        self.white_lower = np.array([0, 0, 200])
        self.white_upper = np.array([255, 30, 255])

        # initialize bridge and subscribe to camera feed
        self._vehicle_name = os.environ['VEHICLE_NAME']
        self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"

        self._bridge = CvBridge()
        self.pub = rospy.Publisher(f"/{self._vehicle_name}/processed_image", Image, queue_size=1)
        self.sub = rospy.Subscriber(self._camera_topic, CompressedImage, self.callback)

        # lane detection publishers

        # LED

        # ROI vertices
        self.roi = None

        # define other variables as needed
        self.rate = rospy.Rate(5)

    def undistort_image(self, image):
        h, w = image.shape[:2]
        new_camera_matrix, self.roi = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix, self.dist_coeffs, (w, h), 1, (w, h)
        )
        undistorted = cv2.undistort(image, self.camera_matrix, self.dist_coeffs, None, new_camera_matrix)
        return undistorted

    def preprocess_image(self, image):
        # add your code here
        if self.roi is None:
            return

        x, y, w, h = self.roi
        resized = image[y:y+h, x:x+w]
        return cv2.GaussianBlur(resized, (5, 5), 0)

    def detect_lane_color(self, **kwargs):
        # add your code here
        pass

    def detect_lane(self, **kwargs):
        # add your code here
        # potentially useful in question 2.1
        pass

    def callback(self, msg):
        # add your code here

        # convert compressed image to CV2
        image = self._bridge.compressed_imgmsg_to_cv2(msg)

        # undistort image
        undistorted_image = self.undistort_image(image)
        self.pub.publish(self._bridge.cv2_to_imgmsg(undistorted_image, encoding="bgr8"))

        # preprocess image
        preprocessed_image = self.preprocess_image(undistorted_image)

        # detect lanes - 2.1

        # publish lane detection results

        # detect lanes and colors - 1.3

        # publish undistorted image

        # control LEDs based on detected colors

        # anything else you want to add here
        self.rate.sleep()
        pass

    # add other functions as needed

if __name__ == '__main__':
    node = LaneDetectionNode(node_name='lane_detection_node')
    rospy.spin()