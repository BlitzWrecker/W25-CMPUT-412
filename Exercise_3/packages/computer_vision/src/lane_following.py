#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
import time
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage
from duckietown_msgs.msg import WheelsCmdStamped
from std_msgs.msg import Float32
from cv_bridge import CvBridge


class LaneFollowingNode(DTROS):
    def __init__(self, node_name):
        super(LaneFollowingNode, self).__init__(node_name=node_name, node_type=NodeType.CONTROL)

        # Select controller type ('P', 'PD', or 'PID')
        self.controller_type = 'PID'  # Change as needed ('P', 'PD', 'PID')

        # PID Gains
        self.kp = 0.5  # Proportional gain
        self.kd = 0.1  # Derivative gain
        self.ki = 0.01  # Integral gain

        # Control variables
        self.prev_error = 0
        self.integral = 0
        self.last_time = time.time()

        # Movement parameters
        self.base_speed = 0.3  # Base wheel speed
        self.max_speed = 1.0  # Max wheel speed

        # Initialize bridge and publishers/subscribers
        self.bridge = CvBridge()
        self.pub_cmd = rospy.Publisher("/wheels_cmd", WheelsCmdStamped, queue_size=1)
        rospy.Subscriber("/camera_node/image/compressed", CompressedImage, self.image_callback)

    def preprocess_image(self, image):
        """Converts and preprocesses the image for lane detection."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        return hsv

    def calculate_error(self, image):
        """Detects lane and computes lateral offset from center."""
        hsv = self.preprocess_image(image)
        yellow_mask = cv2.inRange(hsv, (20, 100, 100), (30, 255, 255))

        contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest_contour)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                image_center = image.shape[1] // 2
                return (cx - image_center) / float(image.shape[1])
        return 0.0

    def p_control(self, error):
        return self.kp * error

    def pd_control(self, error):
        current_time = time.time()
        dt = current_time - self.last_time if self.last_time else 0.1
        d_term = self.kd * (error - self.prev_error) / dt if dt > 0 else 0
        self.prev_error = error
        self.last_time = current_time
        return self.p_control(error) + d_term

    def pid_control(self, error):
        current_time = time.time()
        dt = current_time - self.last_time if self.last_time else 0.1
        self.integral += error * dt
        i_term = self.ki * self.integral
        return self.pd_control(error) + i_term

    def publish_cmd(self, error):
        """Computes control output and publishes wheel commands."""
        if self.controller_type == 'P':
            control = self.p_control(error)
        elif self.controller_type == 'PD':
            control = self.pd_control(error)
        else:
            control = self.pid_control(error)

        left_speed = max(min(self.base_speed - control, self.max_speed), -self.max_speed)
        right_speed = max(min(self.base_speed + control, self.max_speed), -self.max_speed)

        cmd = WheelsCmdStamped()
        cmd.vel_left = left_speed
        cmd.vel_right = right_speed
        self.pub_cmd.publish(cmd)

    def image_callback(self, msg):
        """Processes camera image to detect lane and compute error."""
        image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
        error = self.calculate_error(image)
        self.publish_cmd(error)


if __name__ == '__main__':
    node = LaneFollowingNode(node_name='lane_following_node')
    rospy.spin()
