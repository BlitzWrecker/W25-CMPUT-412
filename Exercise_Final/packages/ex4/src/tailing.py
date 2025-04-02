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
        self.image_sub = rospy.Subscriber(f"/{self._vehicle_name}/camera_node/image/compressed", 
                                         CompressedImage, self.image_callback)
        self.image_pub = rospy.Publisher(f"/{self._vehicle_name}/duckiebot_follower_processed_image", 
                                        Image, queue_size=10)
        
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
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        (detection, centers) = cv2.findCirclesGrid(
            gray,
            patternSize=tuple(self.circlepattern_dims),
            flags=cv2.CALIB_CB_SYMMETRIC_GRID,
            blobDetector=self.blob_detector
        )
        
        if not detection:
            return False, None, None
        
        # Calculate centroid of the pattern
        centroid = np.mean(centers, axis=0)[0]
        centroid_x = centroid[0]
        
        # Estimate distance based on pattern size (empirical calibration needed)
        # This is a simplified approach - you should calibrate this for your setup
        pattern_width_pixels = np.max(centers[:,:,0]) - np.min(centers[:,:,0])
        distance = (0.1 * 320) / pattern_width_pixels  # 0.1 is a scaling factor to adjust
        
        return True, distance, centroid_x

    def calculate_wheel_speeds(self, distance_error, centroid_error):
        """Calculate wheel speeds based on distance and centering errors"""
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

    def image_callback(self, msg):
        try:
            # Convert compressed image to OpenCV format
            image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
            image = cv2.resize(image, (640, 480))  # Work with full resolution for better detection
            
            # Detect vehicle
            detected, distance, centroid_x = self.detect_vehicle(image)
            current_time = rospy.get_time()
            
            if detected:
                self.last_detection_time = current_time
                self.current_distance = distance
                self.target_centroid_x = centroid_x
                
                # Calculate errors
                distance_error = distance - self.target_distance
                centroid_error = (self.target_centroid_x - self.image_center_x) / self.image_center_x
                
                # Calculate and publish wheel commands
                left_speed, right_speed = self.calculate_wheel_speeds(distance_error, centroid_error)
                
                # If we're too close, stop
                if distance < SAFE_DISTANCE:
                    left_speed, right_speed = 0, 0
                    rospy.logwarn("Too close! Stopping.")
                
                cmd = WheelsCmdStamped()
                cmd.vel_left = left_speed
                cmd.vel_right = right_speed
                self.pub_cmd.publish(cmd)
                
                # Draw detection info on image
                cv2.putText(image, f"Distance: {distance:.2f}m", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(image, f"Speed: L={left_speed:.2f}, R={right_speed:.2f}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                # If no detection recently, stop
                if current_time - self.last_detection_time > self.detection_timeout:
                    cmd = WheelsCmdStamped()
                    cmd.vel_left = 0
                    cmd.vel_right = 0
                    self.pub_cmd.publish(cmd)
                    rospy.logwarn("No Duckiebot detected - stopping")
                
                cv2.putText(image, "No Duckiebot detected", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Publish processed image
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(image, encoding="bgr8"))
            
        except Exception as e:
            rospy.logerr(f"Error in image processing: {str(e)}")

    def on_shutdown(self):
        cmd = WheelsCmdStamped()
        cmd.vel_left = 0
        cmd.vel_right = 0
        self.pub_cmd.publish(cmd)
        rospy.loginfo("Shutting down, stopping motors")

if __name__ == '__main__':
    node = DuckiebotFollowerNode(node_name='duckiebot_follower_node')
    rospy.spin()