#!/usr/bin/env python3

# import required libraries
import os
import rospy
import numpy as np
from duckietown.dtros import DTROS, NodeType
import cv2
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
from duckietown_msgs.msg import WheelsCmdStamped, WheelEncoderStamped
import math

# Constants
WHEEL_RADIUS = 0.0318  # meters (Duckiebot wheel radius)
WHEEL_BASE = 0.05  # meters (distance between left and right wheels)
TICKS_PER_ROTATION = 135  # Encoder ticks per full wheel rotation
CURVE_SPEED = 0.5  # Base speed for curved movement

class BehaviorController(DTROS):
    def __init__(self, node_name):
        super(BehaviorController, self).__init__(node_name=node_name, node_type=NodeType.CONTROL)
        
        # Define parameters
        self._vehicle_name = os.environ["VEHICLE_NAME"]
        self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"

        # Get Duckiebot's name
        self._vehicle_name = os.environ["VEHICLE_NAME"]

        # Publisher for wheel commands
        self._wheels_topic = f"/{self._vehicle_name}/wheels_driver_node/wheels_cmd"
        self._publisher = rospy.Publisher(self._wheels_topic, WheelsCmdStamped, queue_size=1)

        # Encoder topics
        self._left_encoder_topic = f"/{self._vehicle_name}/left_wheel_encoder_node/tick"
        self._right_encoder_topic = f"/{self._vehicle_name}/right_wheel_encoder_node/tick"

        # Encoder tick tracking
        self._ticks_left_init = None
        self._ticks_right_init = None
        self._ticks_left = None
        self._ticks_right = None

        # Subscribers to wheel encoders
        self.sub_left = rospy.Subscriber(self._left_encoder_topic, WheelEncoderStamped, self.callback_left)
        self.sub_right = rospy.Subscriber(self._right_encoder_topic, WheelEncoderStamped, self.callback_right)


        # Color ranges in HSV
        self.lower_blue = np.array([100, 150, 50])
        self.upper_blue = np.array([140, 255, 255])
        self.lower_red = np.array([0, 150, 50])
        self.upper_red = np.array([10, 255, 255])
        self.lower_green = np.array([35, 50, 50])
        self.upper_green = np.array([90, 200, 255])

        # Initialize bridge
        self._bridge = CvBridge()

        # Subscribers
        self.sub = rospy.Subscriber(self._camera_topic, CompressedImage, self.callback)

        # State variables
        self.current_color = None
        self.is_stopped = False
        self.start_time = None

        # Define other variables as needed
        self.rate = rospy.Rate(10)  # 10 Hz

    def detect_line(self, image):
        """
        Detect lines (blue, red, green) using HSV thresholds.
        """
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        blue_mask = cv2.inRange(hsv_image, self.lower_blue, self.upper_blue)
        red_mask = cv2.inRange(hsv_image, self.lower_red, self.upper_red)
        green_mask = cv2.inRange(hsv_image, self.lower_green, self.upper_green)
        
        # Check which color is detected
        if np.sum(blue_mask) > 1000:
            return "blue"
        elif np.sum(red_mask) > 1000:
            return "red"
        elif np.sum(green_mask) > 1000:
            return "green"
        else:
            return None

    def execute_blue_line_behavior(self):
        """
        Behavior for blue line:
        1. Stop for 3-5 seconds.
        2. Move in a curve through 90 degrees to the right.
        """
        rospy.loginfo("Executing blue line behavior")
        
        # Stop for 3-5 seconds
        self.navigation.stop(duration=4)
        
        # Move in a curve to the right
        self.navigation.turn_right()

    def execute_red_line_behavior(self):
        """
        Behavior for red line:
        1. Stop for 3-5 seconds.
        2. Move straight for at least 30 cm.
        """
        rospy.loginfo("Executing red line behavior")
        
        # Stop for 3-5 seconds
        self.navigation.stop(duration=4)
        
        # Move straight for 30 cm
        self.navigation.move_straight(0.3)

    def execute_green_line_behavior(self):
        """
        Behavior for green line:
        1. Stop for 3-5 seconds.
        2. Move in a curve through 90 degrees to the left.
        """
        rospy.loginfo("Executing green line behavior")
        
        # Stop for 3-5 seconds
        self.navigation.stop(duration=4)
        
        # Move in a curve to the left
        self.navigation.turn_left()

    def callback(self, msg):
        """
        Callback for processing camera images.
        """
        # Convert compressed image to CV2
        image = self._bridge.compressed_imgmsg_to_cv2(msg)
        
        # Detect line color
        detected_color = self.detect_line(image)
        
        if detected_color and detected_color != self.current_color:
            self.current_color = detected_color
            rospy.loginfo(f"Detected line color: {detected_color}")
            
            # Execute behavior based on detected color
            if detected_color == "blue":
                self.execute_blue_line_behavior()
            elif detected_color == "red":
                self.execute_red_line_behavior()
            elif detected_color == "green":
                self.execute_green_line_behavior()
        
        # If no color is detected, keep moving forward
        elif not detected_color:
            self.navigation.move_straight(0.1)  # Move forward slowly
        
        self.rate.sleep()

    def callback_right(self, data):
        """Callback for right encoder ticks."""
        if self._ticks_right_init is None:
            self._ticks_right_init = data.data
            self._ticks_right = 0
        else:
            self._ticks_right = data.data - self._ticks_right_init

    def reset_encoders(self):
        """Reset encoder counters to track new movements."""
        self._ticks_left_init = None
        self._ticks_right_init = None
        self._ticks_left = None
        self._ticks_right = None
        rospy.loginfo("Resetting encoders...")
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if self._ticks_left is not None and self._ticks_right is not None:
                break
            rate.sleep()
        rospy.loginfo("Encoders reset complete.")

    def publish_velocity(self, left_vel, right_vel):
        """
        Publish wheel velocities to move the Duckiebot.
        :param left_vel: Left wheel velocity.
        :param right_vel: Right wheel velocity.
        """
        cmd = WheelsCmdStamped(vel_left=left_vel, vel_right=right_vel)
        self._publisher.publish(cmd)

    def stop(self, duration=0):
        """
        Stop the Duckiebot for a specified duration.
        :param duration: Duration to stop (in seconds).
        """
        rospy.loginfo(f"Stopping for {duration} seconds...")
        self.publish_velocity(0, 0)
        rospy.sleep(duration)

    def move_straight(self, distance):
        """
        Move the Duckiebot in a straight line for a specified distance.
        :param distance: Distance to move (in meters).
        """
        rospy.loginfo(f"Moving straight for {distance} meters...")
        self.reset_encoders()

        # Compute required encoder ticks for the distance
        ticks_needed = (distance / (2 * math.pi * WHEEL_RADIUS)) * TICKS_PER_ROTATION

        # Command wheels to move forward
        self.publish_velocity(CURVE_SPEED, CURVE_SPEED)

        # Wait until the required ticks are reached
        rate = rospy.Rate(100)  # 10 Hz loop
        while not rospy.is_shutdown():
            if self._ticks_left is not None and self._ticks_right is not None:
                avg_ticks = (self._ticks_left + self._ticks_right) / 2
                rospy.loginfo(f"Current ticks: {avg_ticks}")

                if avg_ticks >= ticks_needed:
                    rospy.loginfo("Straight movement complete.")
                    break

            self.publish_velocity(CURVE_SPEED, CURVE_SPEED)
            rate.sleep()

        # Stop the robot
        self.stop()

    def turn_right(self):
        """
        Move the Duckiebot in a curve through 90 degrees to the right.
        """
        rospy.loginfo("Moving in a curve through 90 degrees to the right...")
        self.reset_encoders()

        # Define curve radius (increase for a wider turn)
        curve_radius = 0.4  # meters (adjust as needed)
        arc_length = (math.pi / 2) * curve_radius  # Arc length for 90 degrees

        # Compute required encoder ticks for the arc length
        ticks_needed = (arc_length / (2 * math.pi * WHEEL_RADIUS)) * TICKS_PER_ROTATION

        # Command wheels to move in a curve (right wheel slower)
        self.publish_velocity(CURVE_SPEED, CURVE_SPEED * 0.35)  # Adjust ratio as needed

        # Wait until the required ticks are reached
        rate = rospy.Rate(100)  # 10 Hz loop
        while not rospy.is_shutdown():
            if self._ticks_left is not None and self._ticks_right is not None:
                avg_ticks = (self._ticks_left + self._ticks_right) / 2
                rospy.loginfo(f"Current ticks: {avg_ticks}")

                if avg_ticks >= ticks_needed:
                    rospy.loginfo("90-degree curve to the right complete.")
                    break

            self.publish_velocity(CURVE_SPEED, CURVE_SPEED * 0.35)
            rate.sleep()

        # Stop the robot
        self.stop()

    def turn_left(self):
        """
        Move the Duckiebot in a curve through 90 degrees to the left.
        """
        rospy.loginfo("Moving in a curve through 90 degrees to the left...")
        self.reset_encoders()

        # Define curve radius (increase for a wider turn)
        curve_radius = 0.4  # meters (adjust as needed)
        arc_length = (math.pi / 2) * curve_radius  # Arc length for 90 degrees

        # Compute required encoder ticks for the arc length
        ticks_needed = (arc_length / (2 * math.pi * WHEEL_RADIUS)) * TICKS_PER_ROTATION

        # Command wheels to move in a curve (left wheel slower)
        self.publish_velocity(CURVE_SPEED * 0.3, CURVE_SPEED)  # Adjust ratio as needed

        # Wait until the required ticks are reached
        rate = rospy.Rate(100)  # 10 Hz loop
        while not rospy.is_shutdown():
            if self._ticks_left is not None and self._ticks_right is not None:
                avg_ticks = (self._ticks_left + self._ticks_right) / 2
                rospy.loginfo(f"Current ticks: {avg_ticks}")

                if avg_ticks >= ticks_needed:
                    rospy.loginfo("90-degree curve to the left complete.")
                    break

            self.publish_velocity(CURVE_SPEED * 0.3, CURVE_SPEED)
            rate.sleep()

        # Stop the robot
        self.stop()

if __name__ == '__main__':
    node = BehaviorController(node_name='behavior_controller_node')
    rospy.spin()