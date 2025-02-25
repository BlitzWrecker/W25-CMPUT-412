#!/usr/bin/env python3

# import required libraries
import os
import rospy
import numpy as np
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import cv2
import time

# Import navigation functions
from navigation_control import NavigationControl  # Assuming navigation_control.py is in the same directory

class BehaviorController(DTROS):
    def __init__(self, node_name):
        super(BehaviorController, self).__init__(node_name=node_name, node_type=NodeType.CONTROL)
        
        # Initialize navigation control
        self.navigation = NavigationControl(node_name="navigation_control_node")

        # Define parameters
        self._vehicle_name = os.environ["VEHICLE_NAME"]
        self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"

        # Color ranges in HSV
        self.lower_blue = np.array([100, 150, 50])
        self.upper_blue = np.array([140, 255, 255])
        self.lower_red = np.array([0, 150, 50])
        self.upper_red = np.array([10, 255, 255])
        self.lower_green = np.array([35, 50, 50])
        self.upper_green = np.array([90, 200, 255])

        # Initialize bridge
        self._bridge = CvBridge()

        # Subscribers
        self.sub = rospy.Subscriber(self._camera_topic, CompressedImage, self.callback)

        # State variables
        self.current_color = None
        self.is_stopped = False
        self.start_time = None

        # Define other variables as needed
        self.rate = rospy.Rate(10)  # 10 Hz

    def detect_line(self, image):
        """
        Detect lines (blue, red, green) using HSV thresholds.
        """
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        blue_mask = cv2.inRange(hsv_image, self.lower_blue, self.upper_blue)
        red_mask = cv2.inRange(hsv_image, self.lower_red, self.upper_red)
        green_mask = cv2.inRange(hsv_image, self.lower_green, self.upper_green)
        
        # Check which color is detected
        if np.sum(blue_mask) > 1000:
            return "blue"
        elif np.sum(red_mask) > 1000:
            return "red"
        elif np.sum(green_mask) > 1000:
            return "green"
        else:
            return None

    def execute_blue_line_behavior(self):
        """
        Behavior for blue line:
        1. Stop for 3-5 seconds.
        2. Move in a curve through 90 degrees to the right.
        """
        rospy.loginfo("Executing blue line behavior")
        
        # Stop for 3-5 seconds
        self.navigation.stop(duration=4)
        
        # Move in a curve to the right
        self.navigation.turn_right()

    def execute_red_line_behavior(self):
        """
        Behavior for red line:
        1. Stop for 3-5 seconds.
        2. Move straight for at least 30 cm.
        """
        rospy.loginfo("Executing red line behavior")
        
        # Stop for 3-5 seconds
        self.navigation.stop(duration=4)
        
        # Move straight for 30 cm
        self.navigation.move_straight(0.3)

    def execute_green_line_behavior(self):
        """
        Behavior for green line:
        1. Stop for 3-5 seconds.
        2. Move in a curve through 90 degrees to the left.
        """
        rospy.loginfo("Executing green line behavior")
        
        # Stop for 3-5 seconds
        self.navigation.stop(duration=4)
        
        # Move in a curve to the left
        self.navigation.turn_left()

    def callback(self, msg):
        """
        Callback for processing camera images.
        """
        # Convert compressed image to CV2
        image = self._bridge.compressed_imgmsg_to_cv2(msg)
        
        # Detect line color
        detected_color = self.detect_line(image)
        
        if detected_color and detected_color != self.current_color:
            self.current_color = detected_color
            rospy.loginfo(f"Detected line color: {detected_color}")
            
            # Execute behavior based on detected color
            if detected_color == "blue":
                self.execute_blue_line_behavior()
            elif detected_color == "red":
                self.execute_red_line_behavior()
            elif detected_color == "green":
                self.execute_green_line_behavior()
        
        # If no color is detected, keep moving forward
        elif not detected_color:
            self.navigation.move_straight(0.1)  # Move forward slowly
        
        self.rate.sleep()

if __name__ == '__main__':
    node = BehaviorController(node_name='behavior_controller_node')
    rospy.spin()