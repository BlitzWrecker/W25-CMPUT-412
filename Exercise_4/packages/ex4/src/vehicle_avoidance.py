#!/usr/bin/env python3

import cv2
import numpy as np
import os
import rospy
import time
from cv_bridge import CvBridge
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import BoolStamped, WheelsCmdStamped
from geometry_msgs.msg import Point32
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Float32, Int32

class LaneFollowingNode(DTROS):
    def __init__(self, node_name):
        super(LaneFollowingNode, self).__init__(node_name=node_name, node_type=NodeType.CONTROL)
        self.camera_matrix = np.array([
            [729.3017308196419, 0.0, 296.9297699654982],
            [0.0, 714.8576567892494, 194.88265037301576],
            [0.0, 0.0, 1.0]
        ])
        self.dist_coeffs = np.array([[-1.526832375685591], [2.217300696985744], [-0.00035517449407590306], [-0.013740460640726298], [0.0]])
        h, w = 480, 640
        self.new_camera_matrix, self.roi = cv2.getOptimalNewCameraMatrix(self.camera_matrix, self.dist_coeffs, (w, h), 1, (w, h))
        self.map1, self.map2 = cv2.initUndistortRectifyMap(self.camera_matrix, self.dist_coeffs, None, self.new_camera_matrix, (w, h), cv2.CV_16SC2)
        self.controller_type = 'PID'
        self.kp, self.kd, self.ki = 1.0, 0.1, 0.01
        self.prev_error, self.integral, self.last_time = 0, 0, time.time()
        self.base_speed, self.max_speed = 0.3, 0.5
        self.bridge = CvBridge()
        self._vehicle_name = os.environ['VEHICLE_NAME']
        self.pub_cmd = rospy.Publisher(f"/{self._vehicle_name}/wheels_driver_node/wheels_cmd", WheelsCmdStamped, queue_size=1)
        self.image_sub = rospy.Subscriber(f"/{self._vehicle_name}/camera_node/image/compressed", CompressedImage, self.image_callback)
        self.image_pub = rospy.Publisher(f"/{self._vehicle_name}/lane_following_processed_image", Image, queue_size=10)
        self.tag_id_sub = rospy.Subscriber(f"/{self._vehicle_name}/detected_tag_id", Int32, self.tag_id_callback)
        self.vehicle_detection_sub = rospy.Subscriber(f"/{self._vehicle_name}/vehicle_detection/detection", BoolStamped, self.vehicle_detection_callback)
        self.tag_stop_times = {21: 3.0, 133: 2.0, 94: 1.0, -1: 0.5}
        self.last_tag_id = -1
        self.lower_red, self.upper_red = np.array([0, 150, 50]), np.array([10, 255, 255])
        self.lower_yellow, self.upper_yellow = np.array([20, 100, 100]), np.array([30, 255, 255])
        self.lower_white, self.upper_white = np.array([0, 0, 150]), np.array([180, 60, 255])
        self.stopped_for_red, self.red_line_cooldown, self.last_red_line_time = False, 4.0, 0.0

        self.circlepattern_dims = [4, 3]  # Columns, rows in circle pattern
        self.blobdetector_min_area = 25
        self.blobdetector_min_dist_between_blobs = 5

        # Initialize blob detector
        params = cv2.SimpleBlobDetector_Params()
        params.minArea = self.blobdetector_min_area
        params.minDistBetweenBlobs = self.blobdetector_min_dist_between_blobs
        self.blob_detector = cv2.SimpleBlobDetector_create(params)

        # State machine variables
        self.state = "LANE_FOLLOWING"
        self.detection_time = 0.0
        self.maneuver_start_time = 0.0

    def detect_vehicle(self, image):
        """Direct vehicle detection using circle grid pattern"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        (detection, centers) = cv2.findCirclesGrid(
            gray,
            patternSize=tuple(self.circlepattern_dims),
            flags=cv2.CALIB_CB_SYMMETRIC_GRID,
            blobDetector=self.blob_detector
        )

        if detection:
            return True, centers
        return False, None

    def tag_id_callback(self, msg):
        self.last_tag_id = msg.data

    def undistort_image(self, image):
        return cv2.remap(image, self.map1, self.map2, cv2.INTER_LINEAR)

    def preprocess_image(self, image):
        image = cv2.resize(image, (320, 240))
        return cv2.GaussianBlur(image, (5, 5), 0)

    def detect_red_line(self, image):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        red_mask = cv2.inRange(hsv_image, self.lower_red, self.upper_red)
        return cv2.countNonZero(red_mask) > 100

    def stop_for_duration(self, duration):
        cmd = WheelsCmdStamped()
        cmd.vel_left, cmd.vel_right = 0, 0
        self.pub_cmd.publish(cmd)
        rospy.sleep(duration)

    def detect_lane_color(self, image):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        return {
            "yellow": cv2.inRange(hsv_image, self.lower_yellow, self.upper_yellow),
            "white": cv2.inRange(hsv_image, self.lower_white, self.upper_white),
            "red": cv2.inRange(hsv_image, self.lower_red, self.upper_red)
        }

    def detect_lane(self, image, masks):
        colors = {"yellow": (0, 255, 255), "white": (255, 255, 255)}
        yellow_max_x, white_min_x = 0, 1000
        for color_name, mask in masks.items():
            if color_name not in ["yellow", "white"]:
                continue
            masked_color = cv2.bitwise_and(image, image, mask=mask)
            gray = cv2.cvtColor(masked_color, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if color_name == "white":
                rightmost_x, rightmost_contour = -1, None
                for contour in contours:
                    if cv2.contourArea(contour) > 200:
                        x, y, w, h = cv2.boundingRect(contour)
                        if x + w / 2 > rightmost_x:
                            rightmost_x = x + w / 2
                            rightmost_contour = contour
                if rightmost_contour is not None:
                    x, y, w, h = cv2.boundingRect(rightmost_contour)
                    white_min_x = x + w / 2
                    cv2.rectangle(image, (x, y), (x + w, y + h), colors[color_name], 2)
            elif color_name == "yellow":
                for contour in contours:
                    if cv2.contourArea(contour) > 200:
                        x, y, w, h = cv2.boundingRect(contour)
                        yellow_max_x = x + w / 2
                        cv2.rectangle(image, (x, y), (x + w, y + h), colors[color_name], 2)
        return image, yellow_max_x, white_min_x

    def calculate_error(self, image):
        undistorted_image = self.undistort_image(image)
        preprocessed_image = self.preprocess_image(undistorted_image)
        masks = self.detect_lane_color(preprocessed_image)
        lane_image, yellow_x, white_x = self.detect_lane(preprocessed_image, masks)
        self.image_pub.publish(self.bridge.cv2_to_imgmsg(lane_image, "bgr8"))
        return (yellow_x - (lane_image.shape[1] // 2)) * 0.01  # Simplified error calculation

    def pid_control(self, error):
        current_time = time.time()
        dt = current_time - self.last_time
        self.integral += error * dt
        self.integral = max(min(self.integral, 1.0), -1.0)
        derivative = (error - self.prev_error) / dt if dt > 0 else 0
        control = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error, self.last_time = error, current_time
        return control

    def publish_cmd(self, error):
        control = self.pid_control(error)
        left = max(min(self.base_speed - control, self.max_speed), 0)
        right = max(min(self.base_speed + control, self.max_speed), 0)
        cmd = WheelsCmdStamped()
        cmd.vel_left, cmd.vel_right = left, right
        self.pub_cmd.publish(cmd)

    def publish_avoidance_command(self, direction):
        cmd = WheelsCmdStamped()
        if direction == "right":
            cmd.vel_left, cmd.vel_right = 0.4, 0.0
        elif direction == "forward":
            cmd.vel_left, cmd.vel_right = 0.4, 0.4
        elif direction == "left":
            cmd.vel_left, cmd.vel_right = 0.0, 0.4
        self.pub_cmd.publish(cmd)

    def image_callback(self, msg):
        try:
            image = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
            cropped_image = image[image.shape[0]//2:, :]
            current_time = rospy.get_time()

            vehicle_detected, _ = self.detect_vehicle(cropped_image)

            if vehicle_detected and self.state == "LANE_FOLLOWING":
                self.state = "STOPPING"
                self.detection_time = current_time
                rospy.loginfo("Vehicle detected! Initiating avoidance maneuver")

            if self.state == "LANE_FOLLOWING":
                if self.detect_red_line(cropped_image) and (current_time - self.last_red_line_time > self.red_line_cooldown):
                    stop_duration = self.tag_stop_times.get(self.last_tag_id, 0.5)
                    self.stop_for_duration(stop_duration)
                    self.last_red_line_time = current_time
                error = self.calculate_error(cropped_image)
                self.publish_cmd(error)
            elif self.state == "STOPPING":
                cmd = WheelsCmdStamped()
                cmd.vel_left, cmd.vel_right = 0, 0
                self.pub_cmd.publish(cmd)
                if current_time - self.detection_time > 1.0:
                    self.state = "AVOIDING"
                    self.maneuver_start_time = current_time
            elif self.state == "AVOIDING":
                elapsed = current_time - self.maneuver_start_time
                if elapsed < 1.0:
                    self.publish_avoidance_command("right")
                elif elapsed < 3.0:
                    self.publish_avoidance_command("forward")
                elif elapsed < 4.0:
                    self.publish_avoidance_command("left")
                else:
                    self.state = "LANE_FOLLOWING"
        except Exception as e:
            rospy.logerr(e)

    def on_shutdown(self):
        cmd = WheelsCmdStamped()
        cmd.vel_left, cmd.vel_right = 0, 0
        self.pub_cmd.publish(cmd)

if __name__ == '__main__':
    node = LaneFollowingNode('lane_following_node')
    rospy.on_shutdown(node.on_shutdown)
    rospy.spin()