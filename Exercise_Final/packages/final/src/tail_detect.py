#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
import time
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import Image
from duckietown_msgs.msg import WheelsCmdStamped, WheelEncoderStamped
from cv_bridge import CvBridge
import os
import math
from final.msg import LaneFollowCMD
from final.srv import MiscCtrlCMD

SAFE_DISTANCE = 0.5  # meters (safe following distance)
FOLLOW_DISTANCE = 0.6  # meters (desired following distance)


class DuckiebotFollowerNode(DTROS):
    def __init__(self, node_name):
        super(DuckiebotFollowerNode, self).__init__(node_name=node_name, node_type=NodeType.CONTROL)

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

        h, w = 480, 640  # Adjust to your image size
        self.new_camera_matrix, self.roi = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix, self.dist_coeffs, (w, h), 1, (w, h))
        self.map1, self.map2 = cv2.initUndistortRectifyMap(
            self.camera_matrix, self.dist_coeffs, None, self.new_camera_matrix, (w, h), cv2.CV_16SC2)

        rospy.wait_for_service("misc_ctrl_srv", timeout=1)
        self.misc_ctrl = rospy.ServiceProxy("misc_ctrl_srv", MiscCtrlCMD)

        # Control parameters
        self.base_speed = 0.3
        self.max_speed = 0.5
        self.min_speed = 0.1
        self.kp_distance = 0.5  # Proportional gain for distance control

        # Lane following parameters
        self.controller_type = 'PID'
        self.kp = 1.75  # Proportional gain
        self.kd = 0.1  # Derivative gain
        self.ki = 0.01  # Integral gain
        self.prev_error = 0
        self.integral = 0
        self.last_time = time.time()

        # Lane detection thresholds
        self.lower_yellow = np.array([20, 85, 100])
        self.upper_yellow = np.array([30, 255, 255])
        self.lower_white = np.array([0, 0, 150])
        self.upper_white = np.array([180, 40, 255])

        # Vehicle tracking variables
        self.last_detection_time = 0
        self.detection_timeout = 1.0  # seconds
        self.target_distance = FOLLOW_DISTANCE
        self.current_distance = SAFE_DISTANCE * 2  # Initialize to "far"

        # Circle pattern detection parameters
        self.circlepattern_dims = [4, 3]  # Columns, rows in circle pattern
        self.blobdetector_min_area = 25
        self.blobdetector_min_dist_between_blobs = 5

        # Initialize blob detector
        params = cv2.SimpleBlobDetector_Params()
        params.minArea = self.blobdetector_min_area
        params.minDistBetweenBlobs = self.blobdetector_min_dist_between_blobs
        self.blob_detector = cv2.SimpleBlobDetector_create(params)

        # Initialize bridge and publishers/subscribers
        self.bridge = CvBridge()
        self._vehicle_name = os.environ['VEHICLE_NAME']
        self.pub_cmd = rospy.Publisher(f"/{self._vehicle_name}/wheels_driver_node/wheels_cmd",
                                       WheelsCmdStamped, queue_size=1)
        self.image_sub = rospy.Subscriber(f"/{self._vehicle_name}/tailing_input",
                                         LaneFollowCMD, self.image_callback)
        self.image_pub = rospy.Publisher(f"/{self._vehicle_name}/duckiebot_follower_processed_image",
                                         Image, queue_size=1)
        self.lane_following_image_pub = rospy.Publisher(f"/{self._vehicle_name}/lane_following_proccessed_image",
                                                        Image, queue_size=1)

        rospy.on_shutdown(self.on_shutdown)
        rospy.loginfo("Duckiebot Follower Node Initialized")

    def undistort_image(self, image):
        return cv2.remap(image, self.map1, self.map2, cv2.INTER_LINEAR)

    def preprocess_image(self, image):
        # Downscale the image
        image = cv2.resize(image, (320, 240))  # Adjust resolution as needed
        return cv2.GaussianBlur(image, (5, 5), 0)

    def detect_vehicle(self, image):
        """Detects the rear pattern of another Duckiebot and returns distance"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        (detection, centers) = cv2.findCirclesGrid(
            gray,
            patternSize=tuple(self.circlepattern_dims),
            flags=cv2.CALIB_CB_SYMMETRIC_GRID,
            blobDetector=self.blob_detector
        )

        if not detection:
            return False, None

        # Estimate distance based on pattern size (empirical calibration needed)
        pattern_width_pixels = np.max(centers[:, :, 0]) - np.min(centers[:, :, 0])
        distance = (0.1 * 320) / pattern_width_pixels  # 0.1 is a scaling factor to adjust

        return True, distance

    def calculate_base_speed(self, distance_error):
        """Calculate base speed based on distance error"""
        # Speed decreases as we get closer to the target distance
        speed_factor = min(1.0, max(0.2, distance_error / self.target_distance))
        base_speed = self.base_speed * speed_factor
        return np.clip(base_speed, self.min_speed, self.max_speed)

    def detect_lane_color(self, image):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        masks = {
            "yellow": cv2.inRange(hsv_image, self.lower_yellow, self.upper_yellow),
            "white": cv2.inRange(hsv_image, self.lower_white, self.upper_white)
        }
        return masks

    def detect_lane(self, image, masks):
        colors = {"yellow": (0, 255, 255), "white": (255, 255, 255)}
        detected_white, detected_yellow = False, False
        yellow_max_x = 0
        white_min_x = 1000

        for color_name, mask in masks.items():
            if color_name == "white":
                detected_white = True
            elif color_name == "yellow":
                detected_yellow = True
            else:
                continue

            masked_color = cv2.bitwise_and(image, image, mask=mask)
            gray = cv2.cvtColor(masked_color, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                if cv2.contourArea(contour) > 200:  # Filter small contours
                    x, y, w, h = cv2.boundingRect(contour)

                    if color_name == "yellow":
                        yellow_max_x = min(max(yellow_max_x, x + w / 2), image.shape[1] // 2)
                    elif color_name == "white":
                        white_min_x = max(min(white_min_x, x + w / 2), image.shape[1] // 2)
                    else:
                        raise ValueError

                    cv2.rectangle(image, (x, y), (x + w, y + h), colors[color_name], 2)

        final_yellow_x = yellow_max_x if detected_yellow else 0
        final_white_x = white_min_x if detected_white else image.shape[1]
        # rospy.loginfo(f"{final_yellow_x}, {final_white_x}")
        return image, final_yellow_x, final_white_x

    def calculate_lane_error(self, image):
        """Detects lane and computes lateral offset from center."""
        masks = self.detect_lane_color(image)
        lane_detected_image, yellow_x, white_x = self.detect_lane(image, masks)
        self.lane_following_image_pub.publish(self.bridge.cv2_to_imgmsg(lane_detected_image, encoding="bgr8"))

        v_mid_line = self.extrinsic_transform(image.shape[1] // 2, 0)
        yellow_line = self.extrinsic_transform(yellow_x, 0)
        white_line = self.extrinsic_transform(white_x, 0)
        yellow_line_displacement = max(float(self.calculate_distance(yellow_line, v_mid_line)), 0.0)
        white_line_displacement = max(float(self.calculate_distance(v_mid_line, white_line)), 0)

        error = yellow_line_displacement - white_line_displacement
        return error

    def extrinsic_transform(self, u, v):
        pixel_coord = np.array([u, v, 1]).reshape(3, 1)
        world_coord = np.dot(self.homography, pixel_coord)
        world_coord /= world_coord[2]
        return world_coord[:2].flatten()

    def calculate_distance(self, l1, l2):
        return np.linalg.norm(l2 - l1)

    def pid_control(self, error):
        current_time = time.time()
        dt = current_time - self.last_time
        if dt <= 0:
            dt = 1e-5  # Avoid division by zero

        # Calculate integral term with windup protection
        self.integral += error * dt
        integral_max = 1.0  # Tune this value based on your system
        self.integral = np.clip(self.integral, -integral_max, integral_max)

        # Calculate derivative term
        derivative = (error - self.prev_error) / dt

        control = self.kp * error + self.ki * self.integral + self.kd * derivative

        # Update previous values for next iteration
        self.prev_error = error
        self.last_time = current_time

        return control

    def calculate_wheel_speeds(self, base_speed, lane_error):
        """Calculate wheel speeds combining base speed and lane following"""
        control = self.pid_control(lane_error)
        left_speed = max(min(base_speed - control, self.max_speed), 0)
        right_speed = max(min(base_speed + control, self.max_speed), 0)
        return left_speed, right_speed

    def image_callback(self, msg):
        shutdown, imgmsg = msg.shutdown, msg.image

        if shutdown:
            rospy.signal_shutdown('Shutting down Duckiebot tailing node.')

        try:
            # Convert compressed image to OpenCV format
            image = self.bridge.imgmsg_to_cv2(imgmsg, desired_encoding="bgr8")
            image = cv2.resize(image, (640, 480))  # Work with full resolution for better detection

            undistorted_image = self.undistort_image(image)
            preprocessed_image = self.preprocess_image(undistorted_image)
            height = preprocessed_image.shape[0]
            cropped_image = preprocessed_image[height//2:, :]

            # Always calculate lane error (we use it regardless of following mode)
            lane_error = self.calculate_lane_error(cropped_image)

            # Detect vehicle
            detected, distance = self.detect_vehicle(image)
            current_time = rospy.get_time()

            if detected:
                self.last_detection_time = current_time
                self.current_distance = distance

                # Calculate distance error
                distance_error = distance - self.target_distance

                # Calculate base speed based on distance
                base_speed = self.calculate_base_speed(distance_error)

                # If we're too close, stop
                if distance < SAFE_DISTANCE:
                    left_speed = 0
                    right_speed = 0
                    rospy.loginfo("Too close! Stopping.")
                else:
                    # Calculate wheel speeds with lane following
                    left_speed, right_speed = self.calculate_wheel_speeds(base_speed, lane_error)

                cmd = WheelsCmdStamped()
                cmd.vel_left = left_speed
                cmd.vel_right = right_speed
                self.pub_cmd.publish(cmd)
                self.misc_ctrl("set_led", 1)

                # Draw detection info on image
                cv2.putText(image, "MODE: FOLLOWING + LANE KEEPING", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(image, f"Distance: {distance:.2f}m", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(image, f"Speed: L={left_speed:.2f}, R={right_speed:.2f}", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                # If no detection recently, switch to pure lane following
                if current_time - self.last_detection_time > self.detection_timeout:
                    # Use default base speed for lane following
                    left_speed, right_speed = self.calculate_wheel_speeds(self.base_speed, lane_error)

                    cmd = WheelsCmdStamped()
                    cmd.vel_left = left_speed
                    cmd.vel_right = right_speed
                    self.pub_cmd.publish(cmd)
                    self.misc_ctrl("set_led", 0)

                    cv2.putText(image, "MODE: LANE FOLLOWING", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255),
                                2)
                    cv2.putText(image, f"Speed: L={left_speed:.2f}, R={right_speed:.2f}", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                else:
                    # If we just lost detection but within timeout, maintain last command
                    cv2.putText(image, "MODE: SEARCHING", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # Publish processed image
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(image, encoding="bgr8"))
        except Exception as e:
            rospy.loginfo(e)

    def on_shutdown(self):
        cmd = WheelsCmdStamped()
        cmd.vel_left = 0
        cmd.vel_right = 0
        self.pub_cmd.publish(cmd)
        rospy.loginfo("Shutting down, stopping motors")


if __name__ == '__main__':
    node = DuckiebotFollowerNode(node_name='duckiebot_follower_node')
    rospy.spin()