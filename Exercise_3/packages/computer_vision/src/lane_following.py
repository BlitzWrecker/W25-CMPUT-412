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
import os


class LaneFollowingNode(DTROS):
    def __init__(self, node_name):
        super(LaneFollowingNode, self).__init__(node_name=node_name, node_type=NodeType.CONTROL)

        self.camera_matrix = np.array([[729.3017308196419, 0.0, 296.9297699654982],
                                       [0.0, 714.8576567892494, 194.88265037301576],
                                       [0.0, 0.0, 1.0]])

        self.dist_coeffs = np.array(
            [[-1.526832375685591], [2.217300696985744], [-0.00035517449407590306], [-0.013740460640726298], [0.0]])

        # Precompute undistortion maps
        h, w = 480, 640  # Adjust to your image size
        self.new_camera_matrix, self.roi = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix, self.dist_coeffs, (w, h), 1, (w, h))
        # Select controller type ('P', 'PD', or 'PID')
        self.map1, self.map2 = cv2.initUndistortRectifyMap(
            self.camera_matrix, self.dist_coeffs, None, self.new_camera_matrix, (w, h), cv2.CV_16SC2)

        self.controller_type = 'P'  # Change as needed ('P', 'PD', 'PID')

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
        self._vehicle_name = os.environ['VEHICLE_NAME']
        self.pub_cmd = rospy.Publisher(f"/{self._vehicle_name}/wheels_driver_node/wheels_cmd", WheelsCmdStamped, queue_size=1)
        rospy.Subscriber(f"/{self._vehicle_name}/camera_node/image/compressed", CompressedImage, self.image_callback)

    def undistort_image(self, image):
        return cv2.remap(image, self.map1, self.map2, cv2.INTER_LINEAR)

    def preprocess_image(self, image):
        """Converts and preprocesses the image for lane detection."""
        image = cv2.resize(image, (320, 240))  # Adjust resolution as needed
        return cv2.GaussianBlur(image, (5, 5), 0)

    def calculate_error(self, image):
        """Detects lane and computes lateral offset from center."""
        undistorted_image = self.undistort_image(image)
        hsv = self.preprocess_image(undistorted_image)
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
        rate = rospy.Rate(10)
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
        # self.pub_cmd.publish(cmd)
        rate.sleep()

    def image_callback(self, msg):
        """Processes camera image to detect lane and compute error."""
        image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
        error = self.calculate_error(image)
        rospy.loginfo(error)
        self.publish_cmd(error)


if __name__ == '__main__':
    node = LaneFollowingNode(node_name='lane_following_node')
    rospy.spin()
