#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
import time
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage, Image
from duckietown_msgs.msg import WheelsCmdStamped
from std_msgs.msg import Float32, Int32
from cv_bridge import CvBridge
import os
from ex4.srv import MiscCtrlCMD


class PeduckstrianNode(DTROS):
    def __init__(self, node_name):
        super(PeduckstrianNode, self).__init__(node_name=node_name, node_type=NodeType.CONTROL)

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

        self.controller_type = 'PID'  # Change as needed ('P', 'PD', 'PID')

        # PID Gains
        self.kp = 1.0  # Proportional gain
        self.kd = 0.1  # Derivative gain
        self.ki = 0.01  # Integral gain

        # Control variables
        self.prev_error = 0
        self.integral = 0
        self.last_time = time.time()

        # Movement parameters
        self.base_speed = 0.3  # Base wheel speed
        self.max_speed = 0.5  # Max wheel speed

        # Initialize bridge and publishers/subscribers
        self.bridge = CvBridge()
        self._vehicle_name = os.environ['VEHICLE_NAME']
        self.pub_cmd = rospy.Publisher(f"/{self._vehicle_name}/wheels_driver_node/wheels_cmd", WheelsCmdStamped, queue_size=1)
        self.image_sub = rospy.Subscriber(f"/{self._vehicle_name}/camera_node/image/compressed", CompressedImage, self.image_callback)
        self.image_pub = rospy.Publisher(f"/{self._vehicle_name}/peduckstrian_processed_image", Image, queue_size=10)

        # Blue line detection thresholds
        self.lower_blue = np.array([100, 150, 50])
        self.upper_blue = np.array([140, 255, 255])

        # Orangish-yellow block detection thresholds
        self.lower_orangish_yellow = np.array([15, 100, 100])
        self.upper_orangish_yellow = np.array([20, 255, 255])

        # Lane detection thresholds
        self.lower_yellow = np.array([20, 100, 100])
        self.upper_yellow = np.array([30, 255, 255])
        self.lower_white = np.array([0, 0, 150])
        self.upper_white = np.array([180, 60, 255])


        self.stopped_for_blue = False
        self.blue_line_cooldown = 4.0
        self.last_blue_line_time = 0.0

        # Blue line and orangish-yellow detection flags
        self.blue_line_detected = False
        self.orangish_yellow_detected = False

        rospy.wait_for_service("misc_ctrl_srv")
        self.misc_ctrl = rospy.ServiceProxy("misc_ctrl_src", MiscCtrlCMD)
        self.misc_ctrl("set_fr", 3)

    def undistort_image(self, image):
        return cv2.remap(image, self.map1, self.map2, cv2.INTER_LINEAR)

    def preprocess_image(self, image):
        """Converts and preprocesses the image for lane detection."""
        image = cv2.resize(image, (320, 240))  # Adjust resolution as needed
        return cv2.GaussianBlur(image, (5, 5), 0)

    def detect_blue_line(self, image):
        """Detects blue lines in the image."""
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        blue_mask = cv2.inRange(hsv_image, self.lower_blue, self.upper_blue)
        blue_pixels = cv2.countNonZero(blue_mask)
        return blue_pixels > 100  # Threshold for blue line detection

    def detect_orangish_yellow(self, image):
        """Detects orangish-yellow blocks in the image."""
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        orangish_yellow_mask = cv2.inRange(hsv_image, self.lower_orangish_yellow, self.upper_orangish_yellow)
        orangish_yellow_pixels = cv2.countNonZero(orangish_yellow_mask)
        return orangish_yellow_pixels > 100  # Threshold for orangish-yellow detection

    def detect_lane_color(self, image):
        """Detects lane colors and returns masks for yellow, white, blue, and orangish-yellow."""
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        masks = {
            "yellow": cv2.inRange(hsv_image, self.lower_yellow, self.upper_yellow),
            "white": cv2.inRange(hsv_image, self.lower_white, self.upper_white),
            "blue": cv2.inRange(hsv_image, self.lower_blue, self.upper_blue),
            "orangish_yellow": cv2.inRange(hsv_image, self.lower_orangish_yellow, self.upper_orangish_yellow)
        }
        return masks

    def draw_contours(self, image, masks):
        """Draws contours around detected regions for each color."""
        colors = {
            "yellow": (0, 255, 255),  # Yellow in BGR
            "white": (255, 255, 255),  # White in BGR
            "blue": (255, 0, 0),  # Blue in BGR
            "orangish_yellow": (0, 165, 255)  # Orangish-yellow in BGR
        }

        for color_name, mask in masks.items():
            if color_name not in colors:
                continue

            # Find contours in the mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Draw contours on the image
            for contour in contours:
                if cv2.contourArea(contour) > 100:  # Filter small contours
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(image, (x, y), (x + w, y + h), colors[color_name], 2)

        return image

    def image_callback(self, msg):
        """Processes camera image to detect lane and compute error."""
        image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")

        image = self.undistort_image(image)
        image = self.preprocess_image(image)

        # Crop the image to only include the lower half for lane following
        height, width = image.shape[:2]
        cropped_image = image[height // 2:height, :]

        # Detect colors and get masks
        masks = self.detect_lane_color(cropped_image)  # Use the entire image for color detection

        # Draw contours on the original image
        processed_image = self.draw_contours(cropped_image, masks)

        # Publish the processed image
        self.image_pub.publish(self.bridge.cv2_to_imgmsg(processed_image, encoding="bgr8"))

        # Detect blue line (using the lower half of the image)
        current_time = rospy.get_time()
        if self.detect_blue_line(cropped_image):
            if not self.stopped_for_blue or (current_time - self.last_blue_line_time > self.blue_line_cooldown):
                rospy.loginfo("Blue line detected. Stopping.")
                self.blue_line_detected = True
                self.stop_for_duration(0)  # Stop indefinitely until further logic
        else:
            self.stopped_for_blue = False

        # If blue line is detected, check for orangish-yellow (using the entire image)
        if self.blue_line_detected:
            if self.detect_orangish_yellow(cropped_image):  # Use the entire image for orangish-yellow detection
                if not self.orangish_yellow_detected:
                    rospy.loginfo("Orangish-yellow detected. Waiting...")
                    self.orangish_yellow_detected = True
            else:
                if self.orangish_yellow_detected:
                    rospy.loginfo("Orangish-yellow no longer detected. Waiting for 2 seconds.")
                    self.stop_for_duration(2.0)
                    self.blue_line_detected = False
                    self.orangish_yellow_detected = False
        else:
            # Continue lane following (using the lower half of the image)
            error = self.calculate_error(cropped_image)
            rospy.loginfo(error)
            self.publish_cmd(error)



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
            _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # For white blocks, find the rightmost one
            if color_name == "white":
                rightmost_x = -1
                rightmost_contour = None

                for contour in contours:
                    if cv2.contourArea(contour) > 200:  # Filter small contours
                        x, y, w, h = cv2.boundingRect(contour)
                        if x + w / 2 > rightmost_x:  # Check if this is the rightmost block
                            rightmost_x = x + w / 2
                            rightmost_contour = contour

                # Only process the rightmost white block
                if rightmost_contour is not None:
                    x, y, w, h = cv2.boundingRect(rightmost_contour)
                    white_min_x = max(min(white_min_x, x + w / 2), image.shape[1] // 2)
                    cv2.rectangle(image, (x, y), (x + w, y + h), colors[color_name], 2)

            # For yellow blocks, process all of them
            elif color_name == "yellow":
                for contour in contours:
                    if cv2.contourArea(contour) > 200:  # Filter small contours
                        x, y, w, h = cv2.boundingRect(contour)
                        yellow_max_x = min(max(yellow_max_x, x + w / 2), image.shape[1] // 2)
                        cv2.rectangle(image, (x, y), (x + w, y + h), colors[color_name], 2)

        final_yellow_x = yellow_max_x if detected_yellow else 0
        final_white_x = white_min_x if detected_white else image.shape[1]
        return image, final_yellow_x, final_white_x



    def calculate_error(self, image):
        """Detects lane and computes lateral offset from center."""
        undistorted_image = self.undistort_image(image)
        preprocessed_image = self.preprocess_image(undistorted_image)
        masks = self.detect_lane_color(preprocessed_image)
        lane_detected_image, yellow_x, white_x = self.detect_lane(preprocessed_image, masks)

        # self.image_pub.publish(self.bridge.cv2_to_imgmsg(lane_detected_image, encoding="bgr8"))

        v_mid_line = self.extrinsic_transform(preprocessed_image.shape[1] // 2, 0)
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
    
    def stop_for_duration(self, duration):
        """Stops the robot for the specified duration."""
        cmd = WheelsCmdStamped()
        cmd.vel_left = 0
        cmd.vel_right = 0
        self.pub_cmd.publish(cmd)
        rospy.sleep(duration)

    def publish_cmd(self, error):
        """Computes control output and publishes wheel commands."""
        if self.controller_type == 'P':
            control = self.p_control(error)
        elif self.controller_type == 'PD':
            control = self.pd_control(error)
        else:
            control = self.pid_control(error)

        left_speed = max(min(self.base_speed - control, self.max_speed), 0)
        right_speed = max(min(self.base_speed + control, self.max_speed), 0)

        cmd = WheelsCmdStamped()
        cmd.vel_left = left_speed
        cmd.vel_right = right_speed
        self.pub_cmd.publish(cmd)

    def p_control(self, error):
        return self.kp * error

    def pd_control(self, error):
        current_time = time.time()
        dt = current_time - self.last_time
        if dt <= 0:
            dt = 1e-5  # Avoid division by zero

        derivative = (error - self.prev_error) / dt
        control = self.kp * error + self.kd * derivative

        # Update previous values for next iteration
        self.prev_error = error
        self.last_time = current_time

        return control

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

    def on_shutdown(self):
        cmd = WheelsCmdStamped()
        cmd.vel_left = 0
        cmd.vel_right = 0
        self.pub_cmd.publish(cmd)


if __name__ == '__main__':
    node = PeduckstrianNode(node_name='peduckstrian_node')
    rospy.on_shutdown(node.on_shutdown)
    rospy.spin()