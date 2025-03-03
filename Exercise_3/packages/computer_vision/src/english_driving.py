#!/usr/bin/env python3


import rospy
import cv2
import numpy as np
import time
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage, Image
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
        # Select controller type ('P', 'PD', or 'PID')
        self.map1, self.map2 = cv2.initUndistortRectifyMap(
            self.camera_matrix, self.dist_coeffs, None, self.new_camera_matrix, (w, h), cv2.CV_16SC2)


        self.controller_type = 'P'  # Change as needed ('P', 'PD', 'PID')


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
        self.max_speed = 0.5 # Max wheel speed


        # Initialize bridge and publishers/subscribers
        self.bridge = CvBridge()
        self._vehicle_name = os.environ['VEHICLE_NAME']
        self.pub_cmd = rospy.Publisher(f"/{self._vehicle_name}/wheels_driver_node/wheels_cmd", WheelsCmdStamped, queue_size=1)
        self.image_sub = rospy.Subscriber(f"/{self._vehicle_name}/camera_node/image/compressed", CompressedImage, self.image_callback)
        self.image_pub = rospy.Publisher(f"/{self._vehicle_name}/processed_image", Image, queue_size=10)

        self.lower_yellow = np.array([20, 100, 100])
        self.upper_yellow = np.array([30, 255, 255])
        self.lower_white = np.array([0, 0, 150])
        self.upper_white = np.array([180, 60, 255])


    def undistort_image(self, image):
        return cv2.remap(image, self.map1, self.map2, cv2.INTER_LINEAR)

    def preprocess_image(self, image):
        """Converts and preprocesses the image for lane detection."""
        image = cv2.resize(image, (320, 240))  # Adjust resolution as needed
        return cv2.GaussianBlur(image, (5, 5), 0)

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
        yellow_min_x = 1000
        white_max_x = 0

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

            for contour in contours:
                if cv2.contourArea(contour) > 200:  # Filter small contours
                    x, y, w, h = cv2.boundingRect(contour)


                    if color_name == "yellow":
                        yellow_min_x = max(min(yellow_min_x, x + w / 2), image.shape[1] // 2)
                    elif color_name == "white":
                        white_max_x = min(max(white_max_x, x + w / 2), image.shape[1] // 2)
                    else:
                        raise ValueError


                    cv2.rectangle(image, (x, y), (x + w, y + h), colors[color_name], 2)


        final_yellow_x = yellow_min_x if detected_yellow else image.shape[1]
        final_white_x = white_max_x if detected_white else 0
        # rospy.loginfo(f"{final_yellow_x}, {final_white_x}")
        return image, final_yellow_x, final_white_x


    # Asked ChatGPT "how to use extrinsic parameters to calculate distance between two objects in an image"
    # ChatGPT answered with a very generic computation method using a rotation matrix and a translation vector
    # Then followed up with "I am working with a duckiebot".
    # ChatGPT answered with an algorithm using the homography matrx
    def extrinsic_transform(self, u, v):
        pixel_coord = np.array([u, v, 1]).reshape(3, 1)
        world_coord = np.dot(self.homography, pixel_coord)
        world_coord /= world_coord[2]
        return world_coord[:2].flatten()
    

    def calculate_distance(self, l1, l2):
        return np.linalg.norm(l1 - l2)


    def calculate_error(self, image):
        """Detects lane and computes lateral offset from center."""
        undistorted_image = self.undistort_image(image)
        preprocessed_image = self.preprocess_image(undistorted_image)
        masks = self.detect_lane_color(preprocessed_image)
        lane_detected_image, yellow_x, white_x = self.detect_lane(preprocessed_image, masks)


        self.image_pub.publish(self.bridge.cv2_to_imgmsg(lane_detected_image, encoding="bgr8"))

        v_mid_line = self.extrinsic_transform(preprocessed_image.shape[1] // 2, 0) 
        yellow_line = self.extrinsic_transform(yellow_x, 0)
        white_line = self.extrinsic_transform(white_x, 0)
        yellow_line_displacement = max(float(self.calculate_distance(yellow_line, v_mid_line)), 0.0)
        white_line_displacement = max(float(self.calculate_distance(v_mid_line, white_line)), 0)
        # rospy.loginfo(f"{image.shape}, {yellow_line_displacement}, {white_line_displacement}")

        error = -(yellow_line_displacement - white_line_displacement)
        return error


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
        # rate.sleep()


    def image_callback(self, msg):
        """Processes camera image to detect lane and compute error."""
        image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")

        # Crop the image to only include the lower half
        height, width = image.shape[:2]
        cropped_image = image[height//2:height, :]

        error = self.calculate_error(cropped_image)
        rospy.loginfo(error)
        self.publish_cmd(error)


    def on_shutdown(self):
        cmd = WheelsCmdStamped()
        cmd.vel_left = 0
        cmd.vel_right = 0
        self.pub_cmd.publish(cmd)


if __name__ == '__main__':
    node = LaneFollowingNode(node_name='lane_following_node')
    rospy.on_shutdown(node.on_shutdown)
    rospy.spin()
