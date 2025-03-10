#!/usr/bin/env python3

# potentially useful for part 2 of exercise 4

# import required libraries
import rospy
import os
import cv2
import numpy as np
from duckietown.dtros import DTROS, NodeType
from ex4.srv import NavigateCMD
from sensor_msgs.msg import CompressedImage

class CrossWalkNode(DTROS):

    def __init__(self, node_name):
        super(CrossWalkNode, self).__init__(node_name=node_name, node_type=NodeType.LOCALIZATION)

        # add your code here

        # call navigation control node
        self.naviagte_service = None

        try:
            rospy.wait_for_service('navigate_service', timeout=1)
            self.lane_hehavior_service = rospy.ServiceProxy('navigate_service', NavigateCMD)
        except rospy.ROSException:
            self.naviagte_service = None

        # subscribe to camera feed
        self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"
        self.sub = rospy.Subscriber(self._camera_topic, CompressedImage, self.image_callback)

        # define other variables as needed
        
        # Camera calibration parameters extracted from the file manager on the dashboard
        # Hard coding is a bad practice; We will have to hard code these parameters again if we switch to another Duckiebot
        # We found a ROS topic that gives us the intrinsic parameters, but not the extrinsict parameters (i.e. the homography matrix)
        self.camera_matrix = np.array([[729.3017308196419, 0.0, 296.9297699654982],
                                       [0.0, 714.8576567892494, 194.88265037301576],
                                       [0.0, 0.0, 1.0]])
        self.dist_coeffs = np.array(
            [[-1.526832375685591], [2.217300696985744], [-0.00035517449407590306], [-0.013740460640726298], [0.0]])

        self.homography = np.array([
            -4.3606292146280124e-05,
            0.0003805216196272236,
            0.2859625589246484,
            -0.00140575582723828,
            6.134315694680119e-05,
            0.396570514773939,
            -0.0001717830439245288,
            0.010136558604291714,
            -1.0992556526691932,
        ]).reshape(3, 3)

        # Precompute undistortion maps
        h, w = 480, 640  # Adjust to your image size
        self.new_camera_matrix, self.roi = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix, self.dist_coeffs, (w, h), 1, (w, h))
        self.map1, self.map2 = cv2.initUndistortRectifyMap(
            self.camera_matrix, self.dist_coeffs, None, self.new_camera_matrix, (w, h), cv2.CV_16SC2)
    
        # Color detection parameters in HSV format
        self.lower_blue = np.array([100, 150, 50])
        self.upper_blue = np.array([140, 255, 255])
    
        # Initialize bridge and subscribe to camera feed
        self._vehicle_name = os.environ['VEHICLE_NAME']

        # Color detection parameters in HSV format
        self.lower_blue = np.array([100, 150, 50])
        self.upper_blue = np.array([140, 255, 255])

        # Remember the last detected color. We only have to execute a different navigation control when there is a color
        # change
        self.last_color = None

    def detect_line(self, **kwargs):
        pass

    def detect_ducks(self, **kwargs):
        pass

    def image_callback(self, **kwargs):
        pass

if __name__ == '__main__':
    # create the node
    node = CrossWalkNode(node_name='april_tag_detector')
    rospy.spin()
