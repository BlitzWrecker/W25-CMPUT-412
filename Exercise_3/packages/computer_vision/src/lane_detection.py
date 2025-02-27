#!/usr/bin/env python3


# import required libraries
import os
import rospy
import numpy as np
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import String  # For publishing the lane color
import cv2
from cv_bridge import CvBridge
from computer_vision.srv import LaneBehaviorCMD


class LaneDetectionNode(DTROS):
    def __init__(self, node_name):
        super(LaneDetectionNode, self).__init__(node_name=node_name, node_type=NodeType.PERCEPTION)
       
        # Camera calibration parameters
        self.camera_matrix = np.array([[729.3017308196419, 0.0, 296.9297699654982],
                                       [0.0, 714.8576567892494, 194.88265037301576],
                                       [0.0, 0.0, 1.0]])
        self.dist_coeffs = np.array(
            [[-1.526832375685591], [2.217300696985744], [-0.00035517449407590306], [-0.013740460640726298], [0.0]])
    
    
        # Precompute undistortion maps
        h, w = 480, 640  # Adjust to your image size
        self.new_camera_matrix, self.roi = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix, self.dist_coeffs, (w, h), 1, (w, h))
        self.map1, self.map2 = cv2.initUndistortRectifyMap(
            self.camera_matrix, self.dist_coeffs, None, self.new_camera_matrix, (w, h), cv2.CV_16SC2)
    
    
        # Color detection parameters in HSV format
        self.lower_blue = np.array([100, 150, 50])
        self.upper_blue = np.array([140, 255, 255])
        self.lower_red = np.array([0, 150, 50])
        self.upper_red = np.array([10, 255, 255])
        self.lower_green = np.array([35, 50, 50])
        self.upper_green = np.array([90, 200, 255])
    
    
        # Initialize bridge and subscribe to camera feed
        self._vehicle_name = os.environ['VEHICLE_NAME']
        self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"
        self._bridge = CvBridge()
    
        # Publisher for processed image
        self.pub = rospy.Publisher(f"/{self._vehicle_name}/processed_image", Image, queue_size=10)
    
        # Publisher for lane detection results (color)
        self.lane_hehavior_service = None

        try:
            rospy.wait_for_service('behavior_service', timeout=1)
            self.lane_hehavior_service = rospy.ServiceProxy('behavior_service', LaneBehaviorCMD)
        except rospy.ROSException:
            self.lane_hehavior_service = None

        # self.color_pub = rospy.Publisher('detected_color', String, queue_size=1)
    
        # Subscribe to camera feed
        self.sub = rospy.Subscriber(self._camera_topic, CompressedImage, self.callback)
    
        # Other variables
        self.rate = rospy.Rate(3)  # Increase rate to 10 Hz
        self.last_color = None
    
    def undistort_image(self, image):
        return cv2.remap(image, self.map1, self.map2, cv2.INTER_LINEAR)
    
    
    def preprocess_image(self, image):
        # Downscale the image
        image = cv2.resize(image, (320, 240))  # Adjust resolution as needed
        return cv2.GaussianBlur(image, (5, 5), 0)
    
    
    def detect_lane_color(self, image):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        masks = {
            "blue": cv2.inRange(hsv_image, self.lower_blue, self.upper_blue),
            "red": cv2.inRange(hsv_image, self.lower_red, self.upper_red),
            "green": cv2.inRange(hsv_image, self.lower_green, self.upper_green)
        }
        return masks
    
    
    def detect_lane(self, image, masks):
        colors = {"blue": (255, 0, 0), "red": (0, 0, 255), "green": (0, 255, 0)}
        detected_colors = []
    
    
        for color_name, mask in masks.items():
            masked_color = cv2.bitwise_and(image, image, mask=mask)
            gray = cv2.cvtColor(masked_color, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    
            for contour in contours:
                if cv2.contourArea(contour) > 200:  # Filter small contours
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(image, (x, y), (x + w, y + h), colors[color_name], 2)
                    detected_colors.append(color_name)
        return image, detected_colors
    
    
    def callback(self, msg):
        # Convert compressed image to CV2
        image = self._bridge.compressed_imgmsg_to_cv2(msg)
    
    
        # Undistort image
        undistorted_image = self.undistort_image(image)
    
    
        # Preprocess image
        preprocessed_image = self.preprocess_image(undistorted_image)
    
    
        # Detect lanes and colors
        masks = self.detect_lane_color(preprocessed_image)
        lane_detected_image, detected_colors = self.detect_lane(preprocessed_image.copy(), masks)
    
    
        # Publish processed image (optional)
        self.pub.publish(self._bridge.cv2_to_imgmsg(lane_detected_image, encoding="bgr8"))
    
    
        # Publish lane detection results (color)
        if detected_colors:
            detected_color = detected_colors[0]  # Publish the first detected color
    
            if detected_color != self.last_color:
                if self.lane_hehavior_service is not None:
                    self.lane_hehavior_service(detected_color)
                # self.color_pub.publish(detected_color)
                self.last_color = detected_color
                rospy.loginfo(f"Detected lane color: {detected_color}")
                
                try:
                    if self.lane_hehavior_service is not None:
                        self.lane_hehavior_service("shutdown")
                except rospy.service.ServiceException:
                    rospy.signal_shutdown("Task completed.")
        else:
            if self.last_color != "None":
                if self.lane_hehavior_service is not None:
                    self.lane_hehavior_service("None")
                # self.color_pub.publish("None")
                self.last_color = "None"
                rospy.loginfo(f"No color detected")
    
        # self.rate.sleep()


if __name__ == '__main__':
   node = LaneDetectionNode(node_name='lane_detection_node')
   rospy.spin()
