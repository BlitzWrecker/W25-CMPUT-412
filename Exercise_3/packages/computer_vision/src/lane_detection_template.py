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

class LaneDetectionNode(DTROS):
    def __init__(self, node_name):
        super(LaneDetectionNode, self).__init__(node_name=node_name, node_type=NodeType.PERCEPTION)
        
        # Camera calibration parameters
        self.camera_matrix = np.array([[729.3017308196419, 0.0, 296.9297699654982],
                                       [0.0, 714.8576567892494, 194.88265037301576],
                                       [0.0, 0.0, 1.0]])
        self.dist_coeffs = np.array(
            [[-1.526832375685591], [2.217300696985744], [-0.00035517449407590306], [-0.013740460640726298], [0.0]])

        # Color detection parameters in HSV format
        self.yellow_lower = np.array([20, 100, 100])
        self.yellow_upper = np.array([30, 255, 255])
        self.white_lower = np.array([0, 0, 200])
        self.white_upper = np.array([255, 30, 255])
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
        self.pub = rospy.Publisher(f"/{self._vehicle_name}/processed_image", Image, queue_size=1)

        # Publisher for lane detection results (color)
        self.lane_color_pub = rospy.Publisher(f"/{self._vehicle_name}/lane_color", String, queue_size=1)

        # Subscribe to camera feed
        self.sub = rospy.Subscriber(self._camera_topic, CompressedImage, self.callback)

        # Other variables
        self.rate = rospy.Rate(5)

    def undistort_image(self, image):
        h, w = image.shape[:2]
        new_camera_matrix, self.roi = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix, self.dist_coeffs, (w, h), 1, (w, h)
        )
        undistorted = cv2.undistort(image, self.camera_matrix, self.dist_coeffs, None, new_camera_matrix)
        return undistorted

    def preprocess_image(self, image):
        if self.roi is not None:
            x, y, w, h = self.roi
            resized = image[y:y+h, x:x+w]
            return cv2.GaussianBlur(resized, (5, 5), 0)
        else:
            return cv2.GaussianBlur(image, (5, 5), 0)

    def detect_lane_color(self, image):
        """
        Detect lane colors (blue, red, green) using HSV thresholds.
        """
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        blue_mask = cv2.inRange(hsv_image, self.lower_blue, self.upper_blue)
        red_mask = cv2.inRange(hsv_image, self.lower_red, self.upper_red)
        green_mask = cv2.inRange(hsv_image, self.lower_green, self.upper_green)
        return blue_mask, red_mask, green_mask

    def detect_lane(self, image, blue_mask, red_mask, green_mask):
        """
        Detect lanes using contour detection and draw colored rectangles.
        """
        masks = [blue_mask, red_mask, green_mask]
        colors = [(255, 0, 0), (0, 0, 255), (0, 255, 0)]  # BGR color codes
        detected_colors = []

        for mask, color in zip(masks, colors):
            masked_color = cv2.bitwise_and(image, image, mask=mask)
            gray = cv2.cvtColor(masked_color, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if w > 20 and h > 20:
                    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                    if color == (255, 0, 0):
                        detected_colors.append("blue")
                    elif color == (0, 0, 255):
                        detected_colors.append("red")
                    elif color == (0, 255, 0):
                        detected_colors.append("green")
        return image, detected_colors

    def callback(self, msg):
        # Convert compressed image to CV2
        image = self._bridge.compressed_imgmsg_to_cv2(msg)

        # Undistort image
        undistorted_image = self.undistort_image(image)

        # Preprocess image
        preprocessed_image = self.preprocess_image(undistorted_image)

        # Detect lanes and colors
        blue_mask, red_mask, green_mask = self.detect_lane_color(preprocessed_image)
        lane_detected_image, detected_colors = self.detect_lane(preprocessed_image.copy(), blue_mask, red_mask, green_mask)

        # Publish processed image
        self.pub.publish(self._bridge.cv2_to_imgmsg(lane_detected_image, encoding="bgr8"))

        # Publish lane detection results (color)
        if detected_colors:
            detected_color = detected_colors[0]  # Publish the first detected color
            self.lane_color_pub.publish(detected_color)
            rospy.loginfo(f"Detected lane color: {detected_color}")

        self.rate.sleep()

if __name__ == '__main__':
    node = LaneDetectionNode(node_name='lane_detection_node')
    rospy.spin()