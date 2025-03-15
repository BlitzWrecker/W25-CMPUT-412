#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
import time
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage, Image
from duckietown_msgs.msg import WheelsCmdStamped, WheelEncoderStamped
from std_msgs.msg import Float32, Int32
from cv_bridge import CvBridge
import os
import math

WHEEL_RADIUS = 0.0318  # meters (Duckiebot wheel radius)
WHEEL_BASE = 0.05  # meters (distance between left and right wheels)
TICKS_PER_ROTATION = 135  # Encoder ticks per full wheel rotation
TURN_SPEED = 0.3  # Adjust speed for accuracy

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

        self._ticks_left_init = None
        self._ticks_right_init = None
        self._ticks_left = None
        self._ticks_right = None

        # Initialize bridge and publishers/subscribers
        self.bridge = CvBridge()
        self._vehicle_name = os.environ['VEHICLE_NAME']
        self.pub_cmd = rospy.Publisher(f"/{self._vehicle_name}/wheels_driver_node/wheels_cmd", WheelsCmdStamped, queue_size=1)
        self.image_sub = rospy.Subscriber(f"/{self._vehicle_name}/camera_node/image/compressed", CompressedImage, self.image_callback)
        self.image_pub = rospy.Publisher(f"/{self._vehicle_name}/lane_following_processed_image", Image, queue_size=10)

        # Subscribe to the detected_tag_id topic
        self.tag_id_sub = rospy.Subscriber(f"/{self._vehicle_name}/detected_tag_id", Int32, self.tag_id_callback)

        self._left_encoder_topic = f"/{self._vehicle_name}/left_wheel_encoder_node/tick"
        self._right_encoder_topic = f"/{self._vehicle_name}/right_wheel_encoder_node/tick"

        self.sub_left = rospy.Subscriber(self._left_encoder_topic, WheelEncoderStamped, self.callback_left)
        self.sub_right = rospy.Subscriber(self._right_encoder_topic, WheelEncoderStamped, self.callback_right)
        wheels_topic = f"/{self._vehicle_name}/wheels_driver_node/wheels_cmd"
        self._publisher = rospy.Publisher(wheels_topic, WheelsCmdStamped, queue_size=1)
        # AprilTag stop times
        self.tag_stop_times = {21: 3.0, 133: 2.0, 94: 1.0, -1: 0.5}
        self.last_tag_id = -1

        # Red line detection thresholds
        self.lower_red = np.array([0, 150, 50])
        self.upper_red = np.array([10, 255, 255])

        # Lane detection thresholds
        self.lower_yellow = np.array([20, 100, 100])
        self.upper_yellow = np.array([30, 255, 255])
        self.lower_white = np.array([0, 0, 150])
        self.upper_white = np.array([180, 60, 255])

        # Red line detection cooldown
        self.stopped_for_red = False  # Flag to track if the bot has already stopped for the red line
        self.red_line_cooldown = 4.0  # Cooldown time in seconds (adjust as needed)
        self.last_red_line_time = 0.0  # Timestamp of the last red line detection

        # Vehicle detection parameters
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
        self.pub_wheels = rospy.Publisher(f'/{self._vehicle_name}/wheels_driver_node/wheels_cmd', WheelsCmdStamped,
                                          queue_size=1)
        self.happened = False
    def move_wheels(self, left_vel, right_vel, duration):
        """Actively publishes movement commands at a controlled rate."""
        rospy.loginfo(f"Moving: Left = {left_vel}, Right = {right_vel} for {duration} seconds")
        self.reset_encoders()

        cmd = WheelsCmdStamped()
        cmd.vel_left = left_vel
        cmd.vel_right = right_vel

        distance_traveled = (2 * math.pi * 0.0318 * (self._ticks_left + self._ticks_right)/2 ) / 135
        distance = duration
        message = WheelsCmdStamped(vel_left=left_vel, vel_right=right_vel)
        self._publisher.publish(message)
        rate = rospy.Rate(100)
        while not rospy.is_shutdown():
            if self._ticks_right is not None and self._ticks_left is not None:
                distance_traveled = (2 * math.pi * 0.0318 * (self._ticks_left + self._ticks_right)/2 ) / 135

            if distance_traveled >= distance:

                message = WheelsCmdStamped(vel_left=0, vel_right=0)
                self._publisher.publish(message)
                break

            rospy.loginfo(distance_traveled)
            rospy.loginfo(self._ticks_left)

            self._publisher.publish(message)
            rate.sleep()

        # Stop the robot after moving
        rospy.loginfo("Stopping robot")
        stop_command = WheelsCmdStamped(vel_left=0, vel_right=0)  # ✅ Correct Stop Command
        self.pub_wheels.publish(stop_command)
        rospy.sleep(0.5)  # Ensure stop command is received

    def turn_90_degrees(self, direction=1):
        """
        Turns the Duckiebot 90 degrees in place.
        :param direction: 1 for left, -1 for right
        """
        # Reset encoder counters before each turn
        self.reset_encoders()

        # Compute required encoder ticks for 90-degree turn
        ticks_needed = round((WHEEL_BASE / (8 * WHEEL_RADIUS)) * TICKS_PER_ROTATION) + 11
        rospy.loginfo(f"Ticks needed for 90-degree turn: {ticks_needed}")

        # Command wheels to rotate in opposite directions
        turn_command = WheelsCmdStamped(
            vel_left=TURN_SPEED * direction,
            vel_right=-TURN_SPEED * direction
        )
        self._publisher.publish(turn_command)

        # Wait until the required ticks are reached
        rate = rospy.Rate(100)  # 10 Hz loop
        while not rospy.is_shutdown():
            if self._ticks_left is not None and self._ticks_right is not None:
                avg_ticks = (abs(self._ticks_left) + abs(self._ticks_right)) / 2
                rospy.loginfo(f"Current ticks: {avg_ticks}")

                if avg_ticks >= ticks_needed:
                    rospy.loginfo("90-degree turn complete.")
                    break

            self._publisher.publish(turn_command)
            rate.sleep()

        # Stop the robot
        stop_command = WheelsCmdStamped(vel_left=0, vel_right=0)
        self._publisher.publish(stop_command)
        rospy.sleep(1)  # Small delay to stabilize
    def callback_left(self, data):
        # log general information once at the beginning
        rospy.loginfo_once(f"Left encoder resolution: {data.resolution}")
        rospy.loginfo_once(f"Left encoder type: {data.type}")
        # store data value
        if self._ticks_left_init is None:
            self._ticks_left_init = data.data
            self._ticks_left = 0
        else:
            self._ticks_left = data.data - self._ticks_left_init

    def callback_right(self, data):
        # log general information once at the beginning
        rospy.loginfo_once(f"Right encoder resolution: {data.resolution}")
        rospy.loginfo_once(f"Right encoder type: {data.type}")
        # store data value
        if self._ticks_right_init is None:
            self._ticks_right_init = data.data
            self._ticks_right = 0
        else:
            self._ticks_right = data.data - self._ticks_right_init
    def reset_encoders(self):
        """ Reset encoder counters to track new movements """
        self._ticks_left_init = None
        self._ticks_right_init = None
        self._ticks_left = None
        self._ticks_right = None
        # Wait for encoder data to reinitialize
        rospy.loginfo("Resetting encoders...")
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if self._ticks_left is not None and self._ticks_right is not None:
                break
            rate.sleep()
        rospy.loginfo("Encoders reset complete.")

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
        """Callback for the detected_tag_id topic."""
        self.last_tag_id = msg.data

    def undistort_image(self, image):
        return cv2.remap(image, self.map1, self.map2, cv2.INTER_LINEAR)

    def preprocess_image(self, image):
        """Converts and preprocesses the image for lane detection."""
        image = cv2.resize(image, (320, 240))  # Adjust resolution as needed
        return cv2.GaussianBlur(image, (5, 5), 0)

    def detect_red_line(self, image):
        """Detects red lines in the image."""
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        red_mask = cv2.inRange(hsv_image, self.lower_red, self.upper_red)
        red_pixels = cv2.countNonZero(red_mask)
        return red_pixels > 100  # Threshold for red line detection

    def stop_for_duration(self, duration):
        """Stops the robot for the specified duration."""
        cmd = WheelsCmdStamped()
        cmd.vel_left = 0
        cmd.vel_right = 0
        self.pub_cmd.publish(cmd)
        rospy.sleep(duration)

    def detect_lane_color(self, image):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        masks = {
            "yellow": cv2.inRange(hsv_image, self.lower_yellow, self.upper_yellow),
            "white": cv2.inRange(hsv_image, self.lower_white, self.upper_white),
            "red": cv2.inRange(hsv_image, self.lower_red, self.upper_red)
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

    def extrinsic_transform(self, u, v):
        pixel_coord = np.array([u, v, 1]).reshape(3, 1)
        world_coord = np.dot(self.homography, pixel_coord)
        world_coord /= world_coord[2]
        return world_coord[:2].flatten()

    def calculate_distance(self, l1, l2):
        return np.linalg.norm(l2 - l1)

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

        error = yellow_line_displacement - white_line_displacement
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

    def publish_avoidance_command(self, direction):
        cmd = WheelsCmdStamped()
        if direction == "right":
            cmd.vel_left, cmd.vel_right = 0.4, 0.0
        elif direction == "forward":
            cmd.vel_left, cmd.vel_right = 0.4, 0.4
        elif direction == "left":
            cmd.vel_left, cmd.vel_right = 0.0, 0.4
        elif direction == "stop":
            cmd.vel_left, cmd.vel_right = 0.0, 0.0
        self.pub_cmd.publish(cmd)

    def image_callback(self, msg):


        image = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")

        cropped_image = image
        current_time = rospy.get_time()

        # Add vehicle detection directly here
        vehicle_detected, _ = self.detect_vehicle(cropped_image)


        
        if vehicle_detected and self.state == "LANE_FOLLOWING":
            self.state = "STOPPING"
            self.detection_time = current_time
            rospy.loginfo("Vehicle detected! Initiating avoidance maneuver")

        if self.state == "LANE_FOLLOWING":
            self.happened = False
            if self.detect_red_line(cropped_image) and (
                    current_time - self.last_red_line_time > self.red_line_cooldown):
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

            self.image_sub.unregister()

            rospy.sleep(3)
            self.turn_90_degrees(-1)
            self.move_wheels(0.5,0.5,0.10)
            self.turn_90_degrees(1)
            self.move_wheels(0.5, 0.5, 0.60)
            self.turn_90_degrees(1)
            self.move_wheels(0.5, 0.5, 0.10)
            self.turn_90_degrees(-1)

            self.image_sub = rospy.Subscriber(f"/{self._vehicle_name}/camera_node/image/compressed", CompressedImage,
                                              self.image_callback)

            rospy.loginfo("Back to lane following")
            self.state = "LANE_FOLLOWING"


    def on_shutdown(self):
        cmd = WheelsCmdStamped()
        cmd.vel_left = 0
        cmd.vel_right = 0
        self.pub_cmd.publish(cmd)


if __name__ == '__main__':
    node = LaneFollowingNode(node_name='lane_following_node')
    rospy.on_shutdown(node.on_shutdown)
    rospy.spin()