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
from ex4.srv import MiscCtrlCMD

WHEEL_RADIUS = 0.0318  # meters (Duckiebot wheel radius)
WHEEL_BASE = 0.05  # meters (distance between left and right wheels)
TICKS_PER_ROTATION = 135  # Encoder ticks per full wheel rotation
TURN_SPEED = 0.35  # Adjust speed for accuracy
SAFE_DISTANCE = 0.3  # meters (safe following distance)
FOLLOW_DISTANCE = 0.5  # meters (desired following distance)

class DuckiebotFollowerNode(DTROS):
    def __init__(self, node_name):
        super(DuckiebotFollowerNode, self).__init__(node_name=node_name, node_type=NodeType.CONTROL)

        # Camera calibration parameters
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
        h, w = 480, 640
        self.new_camera_matrix, self.roi = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix, self.dist_coeffs, (w, h), 1, (w, h))
        self.map1, self.map2 = cv2.initUndistortRectifyMap(
            self.camera_matrix, self.dist_coeffs, None, self.new_camera_matrix, (w, h), cv2.CV_16SC2)

        # Control parameters
        self.base_speed = 0.3
        self.max_speed = 0.5
        self.min_speed = 0.1
        self.kp_distance = 0.5  # Proportional gain for distance control
        self.kp_centering = 0.3  # Proportional gain for centering
        
        # Lane following parameters
        self.controller_type = 'PID'
        self.kp = 1.0  # Proportional gain
        self.kd = 0.1  # Derivative gain
        self.ki = 0.01  # Integral gain
        self.prev_error = 0
        self.integral = 0
        self.last_time = time.time()
        
        # Lane detection thresholds
        self.lower_yellow = np.array([20, 100, 100])
        self.upper_yellow = np.array([30, 255, 255])
        self.lower_white = np.array([0, 0, 150])
        self.upper_white = np.array([180, 60, 255])
        
        # Vehicle tracking variables
        self.last_detection_time = 0
        self.detection_timeout = 1.0  # seconds
        self.target_distance = FOLLOW_DISTANCE
        self.current_distance = SAFE_DISTANCE * 2  # Initialize to "far"
        self.target_centroid_x = 0
        self.image_center_x = 320  # Assuming 640x480 image resized to 320x240
        
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
        
        # Camera subscriber with queue size and buffering options
        self.image_sub = rospy.Subscriber(
            f"/{self._vehicle_name}/camera_node/image/compressed", 
            CompressedImage, 
            self.image_callback,
            queue_size=1,  # Small queue size to prevent buildup
            buff_size=2*640*480*3  # Adjust buffer size based on image size
        )
        
        self.image_pub = rospy.Publisher(
            f"/{self._vehicle_name}/duckiebot_follower_processed_image", 
            Image, 
            queue_size=1  # Smaller queue for processed images
        )

        self.lane_detection_image_pub = rospy.Publisher(
            f"/{self._vehicle_name}/lane_detection_image",
            Image,
            queue_size=1
        )
        
        # Encoder subscribers
        self._left_encoder_topic = f"/{self._vehicle_name}/left_wheel_encoder_node/tick"
        self._right_encoder_topic = f"/{self._vehicle_name}/right_wheel_encoder_node/tick"
        self.sub_left = rospy.Subscriber(self._left_encoder_topic, WheelEncoderStamped, self.callback_left)
        self.sub_right = rospy.Subscriber(self._right_encoder_topic, WheelEncoderStamped, self.callback_right)
        
        # Encoder tracking
        self._ticks_left_init = None
        self._ticks_right_init = None
        self._ticks_left = None
        self._ticks_right = None

        # Frame rate control
        self.last_processed_time = 0
        self.min_processing_interval = 0.05  # 20 Hz max processing rate
        
        # Limiting the camera frame rate to 3 so we don't lag the hell out
        rospy.wait_for_service("misc_ctrl_srv", timeout=1)
        self.misc_ctrl = rospy.ServiceProxy("misc_ctrl_srv", MiscCtrlCMD)
        self.misc_ctrl("set_fr", 3)
        
        rospy.on_shutdown(self.on_shutdown)
        rospy.loginfo("Duckiebot Follower Node Initialized")

    def callback_left(self, data):
        if self._ticks_left_init is None:
            self._ticks_left_init = data.data
            self._ticks_left = 0
        else:
            self._ticks_left = data.data - self._ticks_left_init

    def callback_right(self, data):
        if self._ticks_right_init is None:
            self._ticks_right_init = data.data
            self._ticks_right = 0
        else:
            self._ticks_right = data.data - self._ticks_right_init

    def detect_vehicle(self, image):
        """Detects the rear pattern of another Duckiebot and returns distance and centroid"""
        # Use a smaller region of interest for faster processing
        h, w = image.shape[:2]
        roi = image[int(h*0.3):int(h*0.8), int(w*0.2):int(w*0.8)]
        
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        (detection, centers) = cv2.findCirclesGrid(
            gray,
            patternSize=tuple(self.circlepattern_dims),
            flags=cv2.CALIB_CB_SYMMETRIC_GRID,
            blobDetector=self.blob_detector
        )
        
        if not detection:
            return False, None, None
        
        # Calculate centroid of the pattern (adjust for ROI offset)
        centroid = np.mean(centers, axis=0)[0]
        centroid_x = centroid[0] + w*0.2  # Add back the ROI x offset
        
        # Estimate distance based on pattern size (empirical calibration needed)
        pattern_width_pixels = np.max(centers[:,:,0]) - np.min(centers[:,:,0])
        distance = (0.1 * 320) / pattern_width_pixels  # 0.1 is a scaling factor to adjust
        
        return True, distance, centroid_x

    def calculate_wheel_speeds_follow(self, distance_error, centroid_error):
        """Calculate wheel speeds for following another Duckiebot"""
        # Distance control term - adjust speed based on distance error
        distance_term = self.kp_distance * (distance_error)
        
        # Centering control term - adjust steering based on centroid error
        centering_term = self.kp_centering * (centroid_error)
        
        # Base speed decreases as we get closer to the target distance
        speed_factor = min(1.0, max(0.2, distance_error / self.target_distance))
        base_speed = self.base_speed * speed_factor
        
        # Calculate left and right wheel speeds
        left_speed = base_speed - distance_term - centering_term
        right_speed = base_speed - distance_term + centering_term
        
        # Clip speeds to valid range
        left_speed = np.clip(left_speed, self.min_speed, self.max_speed)
        right_speed = np.clip(right_speed, self.min_speed, self.max_speed)
        
        return left_speed, right_speed

    def detect_lane_color(self, image):
        # Resize image first for faster processing
        small_img = cv2.resize(image, (320, 240))
        hsv_image = cv2.cvtColor(small_img, cv2.COLOR_BGR2HSV)
        masks = {
            "yellow": cv2.inRange(hsv_image, self.lower_yellow, self.upper_yellow),
            "white": cv2.inRange(hsv_image, self.lower_white, self.upper_white)
        }
        return masks, small_img

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

    def calculate_lane_error(self, image):
        """Detects lane and computes lateral offset from center."""
        masks, small_img = self.detect_lane_color(image)
        lane_detected_image, yellow_x, white_x = self.detect_lane(small_img, masks)

        self.lane_detection_image_pub.publish(self.bridge.cv2_to_imgmsg(lane_detected_image, encoding="bgr8"))

        v_mid_line = self.extrinsic_transform(small_img.shape[1] // 2, 0)
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

    def calculate_wheel_speeds_lane(self, error):
        """Calculate wheel speeds for lane following"""
        control = self.pid_control(error)
        left_speed = max(min(self.base_speed - control, self.max_speed), 0)
        right_speed = max(min(self.base_speed + control, self.max_speed), 0)
        return left_speed, right_speed

    def image_callback(self, msg):
        # Skip processing if we're still processing the last frame
        current_time = time.time()
        if current_time - self.last_processed_time < self.min_processing_interval:
            return
            
        try:
            # Convert compressed image to OpenCV format
            image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
            
            # Detect vehicle
            detected, distance, centroid_x = self.detect_vehicle(image)
            current_ros_time = rospy.get_time()
            
            if detected:
                self.last_detection_time = current_ros_time
                self.current_distance = distance
                self.target_centroid_x = centroid_x
                
                # Calculate errors
                distance_error = distance - self.target_distance
                centroid_error = (self.target_centroid_x - self.image_center_x) / self.image_center_x
                
                # Calculate and publish wheel commands
                left_speed, right_speed = self.calculate_wheel_speeds_follow(distance_error, centroid_error)
                
                # If we're too close, stop
                if distance < SAFE_DISTANCE:
                    left_speed, right_speed = 0, 0
                    rospy.logwarn("Too close! Stopping.")
                
                cmd = WheelsCmdStamped()
                cmd.vel_left = left_speed
                cmd.vel_right = right_speed
                self.pub_cmd.publish(cmd)
                
                # Draw detection info on image
                cv2.putText(image, "MODE: FOLLOWING", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(image, f"Distance: {distance:.2f}m", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(image, f"Speed: L={left_speed:.2f}, R={right_speed:.2f}", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                # If no detection recently, switch to lane following
                if current_ros_time - self.last_detection_time > self.detection_timeout:
                    # Calculate lane following error
                    error = self.calculate_lane_error(image)
                    left_speed, right_speed = self.calculate_wheel_speeds_lane(error)
                    
                    cmd = WheelsCmdStamped()
                    cmd.vel_left = left_speed
                    cmd.vel_right = right_speed
                    self.pub_cmd.publish(cmd)
                    
                    cv2.putText(image, "MODE: LANE FOLLOWING", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.putText(image, f"Speed: L={left_speed:.2f}, R={right_speed:.2f}", (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                else:
                    # If we just lost detection but within timeout, maintain last command
                    cv2.putText(image, "MODE: SEARCHING", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Publish processed image (if needed)
            try:
                self.image_pub.publish(self.bridge.cv2_to_imgmsg(image, encoding="bgr8"))
            except:
                pass
                
            self.last_processed_time = current_time
            
        except Exception as e:
            rospy.logerr(f"Error in image processing: {str(e)}")
            self.last_processed_time = current_time

    def on_shutdown(self):
        cmd = WheelsCmdStamped()
        cmd.vel_left = 0
        cmd.vel_right = 0
        self.pub_cmd.publish(cmd)
        rospy.loginfo("Shutting down, stopping motors")

if __name__ == '__main__':
    node = DuckiebotFollowerNode(node_name='duckiebot_follower_node')
    rospy.spin()