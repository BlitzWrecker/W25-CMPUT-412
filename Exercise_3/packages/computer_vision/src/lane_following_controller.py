#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
import time
import math
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage, Image
from duckietown_msgs.msg import WheelsCmdStamped, WheelEncoderStamped
from std_msgs.msg import Float32
from cv_bridge import CvBridge
import os

# Constants
TICKS_PER_ROTATION = 135  # For Duckietown wheels
WHEEL_RADIUS = 0.0318  # 3.18 cm


class LaneFollowingNode(DTROS):
    def __init__(self, node_name):
        super(LaneFollowingNode, self).__init__(node_name=node_name, node_type=NodeType.CONTROL)

        # Camera calibration parameters
        self.camera_matrix = np.array([[729.3017308196419, 0.0, 296.9297699654982],
                                       [0.0, 714.8576567892494, 194.88265037301576],
                                       [0.0, 0.0, 1.0]])
        self.dist_coeffs = np.array(
            [[-1.526832375685591], [2.217300696985744], [-0.00035517449407590306], [-0.013740460640726298], [0.0]])

        # Initialize undistortion maps
        h, w = 480, 640
        self.new_camera_matrix, self.roi = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix, self.dist_coeffs, (w, h), 1, (w, h))
        self.map1, self.map2 = cv2.initUndistortRectifyMap(
            self.camera_matrix, self.dist_coeffs, None, self.new_camera_matrix, (w, h), cv2.CV_16SC2)

        # Controller configuration
        self.controller_type = 'P'
        self.kp = 0.5
        self.kd = 0.1
        self.ki = 0.01
        self.prev_error = 0
        self.integral = 0
        self.last_time = time.time()

        # Movement parameters
        self.base_speed = 0.3
        self.max_speed = 0.5
        self.target_distance = 1.5  # meters
        self.distance_travelled = 0.0
        self._ticks_left = 0
        self._ticks_right = 0
        self._ticks_left_init = None
        self._ticks_right_init = None

        # ROS infrastructure
        self.bridge = CvBridge()
        self._vehicle_name = os.environ['VEHICLE_NAME']

        # Publishers
        self.pub_cmd = rospy.Publisher(f"/{self._vehicle_name}/wheels_driver_node/wheels_cmd",
                                       WheelsCmdStamped, queue_size=1)
        self.image_pub = rospy.Publisher(f"/{self._vehicle_name}/processed_image", Image, queue_size=10)

        # Subscribers
        self.image_sub = rospy.Subscriber(f"/{self._vehicle_name}/camera_node/image/compressed",
                                          CompressedImage, self.image_callback)
        rospy.Subscriber(f"/{self._vehicle_name}/left_wheel_encoder_node/tick",
                         WheelEncoderStamped, self.callback_left)
        rospy.Subscriber(f"/{self._vehicle_name}/right_wheel_encoder_node/tick",
                         WheelEncoderStamped, self.callback_right)

        # Color thresholds
        self.lower_yellow = np.array([20, 100, 100])
        self.upper_yellow = np.array([30, 255, 255])
        self.lower_white = np.array([0, 0, 150])
        self.upper_white = np.array([180, 60, 255])

    def callback_left(self, data):
        if self._ticks_left_init is None:
            self._ticks_left_init = data.data
        self._ticks_left = data.data - self._ticks_left_init

    def callback_right(self, data):
        if self._ticks_right_init is None:
            self._ticks_right_init = data.data
        self._ticks_right = data.data - self._ticks_right_init

    def calculate_distance_travelled(self):
        avg_ticks = (self._ticks_left + self._ticks_right) / 2.0
        rotations = avg_ticks / TICKS_PER_ROTATION
        return rotations * (2 * math.pi * WHEEL_RADIUS)

    def undistort_image(self, image):
        return cv2.remap(image, self.map1, self.map2, cv2.INTER_LINEAR)

    def preprocess_image(self, image):
        image = cv2.resize(image, (320, 240))
        return cv2.GaussianBlur(image, (5, 5), 0)

    def detect_lane_color(self, image):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        return {
            "yellow": cv2.inRange(hsv_image, self.lower_yellow, self.upper_yellow),
            "white": cv2.inRange(hsv_image, self.lower_white, self.upper_white)
        }

    def detect_lane(self, image, masks):
        yellow_max_x = 0
        white_min_x = image.shape[1]
        colors = {"yellow": (0, 255, 255), "white": (255, 255, 255)}

        for color_name, mask in masks.items():
            masked = cv2.bitwise_and(image, image, mask=mask)
            gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                if cv2.contourArea(contour) > 200:
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(image, (x, y), (x + w, y + h), colors[color_name], 2)
                    if color_name == "yellow":
                        yellow_max_x = max(yellow_max_x, x + w // 2)
                    else:
                        white_min_x = min(white_min_x, x + w // 2)

        return image, yellow_max_x, white_min_x

    def calculate_error(self, image):
        undistorted = self.undistort_image(image)
        processed = self.preprocess_image(undistorted)
        masks = self.detect_lane_color(processed)
        lane_img, yellow_x, white_x = self.detect_lane(processed, masks)

        self.image_pub.publish(self.bridge.cv2_to_imgmsg(lane_img, "bgr8"))

        mid = image.shape[1] // 2
        error = (mid - yellow_x) - (white_x - mid)
        return error

    def p_control(self, error):
        return self.kp * error

    def pd_control(self, error):
        dt = time.time() - self.last_time
        dt = max(dt, 1e-5)
        derivative = (error - self.prev_error) / dt
        self.prev_error = error
        self.last_time = time.time()
        return self.kp * error + self.kd * derivative

    def pid_control(self, error):
        dt = time.time() - self.last_time
        dt = max(dt, 1e-5)

        self.integral += error * dt
        self.integral = np.clip(self.integral, -1.0, 1.0)

        derivative = (error - self.prev_error) / dt

        self.prev_error = error
        self.last_time = time.time()

        return self.kp * error + self.ki * self.integral + self.kd * derivative

    def publish_cmd(self, error):
        rospy.loginfo(self.calculate_distance_travelled())
        if self.calculate_distance_travelled() >= self.target_distance:
            rospy.loginfo("Reached target distance. Stopping.")
            self.pub_cmd.publish(WheelsCmdStamped(vel_left=0, vel_right=0))
            rospy.signal_shutdown("Target distance reached")
            return

        if self.controller_type == 'P':
            control = self.p_control(error)
        elif self.controller_type == 'PD':
            control = self.pd_control(error)
        else:
            control = self.pid_control(error)

        left = max(min(self.base_speed - control, self.max_speed), 0)
        right = max(min(self.base_speed + control, self.max_speed), 0)

        cmd = WheelsCmdStamped()
        cmd.vel_left = left
        cmd.vel_right = right
        self.pub_cmd.publish(cmd)

    def image_callback(self, msg):
        try:
            image = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
            height, width = image.shape[:2]
            cropped = image[height // 2:height, :]
            error = self.calculate_error(cropped)
            self.publish_cmd(error)
        except Exception as e:
            rospy.logerr(f"Image processing error: {str(e)}")


if __name__ == '__main__':
    node = LaneFollowingNode(node_name='lane_following_node')
    rospy.spin()