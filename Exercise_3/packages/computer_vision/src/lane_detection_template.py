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
        
        # color detection parameters in HSV format
        self.yellow_lower = np.array([20, 100, 100])
        self.yellow_upper = np.array([30, 255, 255])
        self.white_lower = np.array([0, 0, 200])
        self.white_upper = np.array([255, 30, 255])

        # initialize bridge and subscribe to camera feed

        # lane detection publishers

        # LED
        
        # ROI vertices
        
        # define other variables as needed
        self.rate = rospy.Rate(5)

    def undistort_image(self, image):
        h, w = image.shape[:2]
        new_camera_matrix, self.roi = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix, self.dist_coeffs, (w, h), 1, (w, h)
        )
        undistorted = cv2.undistort(image, self.camera_matrix, self.dist_coeffs, None, new_camera_matrix)
        return undistorted

    def preprocess_image(self, **kwargs):
        # add your code here
        pass
    
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
        
        # undistort image

        # preprocess image

        # detect lanes - 2.1 
        
        # publish lane detection results
        
        # detect lanes and colors - 1.3
        
        # publish undistorted image
        
        # control LEDs based on detected colors

        # anything else you want to add here
        
        pass

    # add other functions as needed

if __name__ == '__main__':
    node = LaneDetectionNode(node_name='lane_detection_node')
    rospy.spin()