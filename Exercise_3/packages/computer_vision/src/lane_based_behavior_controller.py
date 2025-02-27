#!/usr/bin/env python3

# import required libraries
import os
import rospy
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import WheelsCmdStamped, WheelEncoderStamped
from std_msgs.msg import String  # For publishing the lane color
import math
from duckietown_msgs.msg import LEDPattern
from std_msgs.msg import ColorRGBA
from computer_vision.srv import LaneBehaviorCMD, LaneBehaviorCMDResponse

# Constants
WHEEL_RADIUS = 0.0318  # meters (Duckiebot wheel radius)
WHEEL_BASE = 0.05  # meters (distance between left and right wheels)
TICKS_PER_ROTATION = 135  # Encoder ticks per full wheel rotation
CURVE_SPEED = 0.5  # Base speed for curved movement

class BehaviorController(DTROS):
    def __init__(self, node_name):
        super(BehaviorController, self).__init__(node_name=node_name, node_type=NodeType.CONTROL)
        
        # Get Duckiebot's name
        self._vehicle_name = os.environ["VEHICLE_NAME"]

        # Publisher for wheel commands
        self._wheels_topic = f"/{self._vehicle_name}/wheels_driver_node/wheels_cmd"
        self._publisher = rospy.Publisher(self._wheels_topic, WheelsCmdStamped, queue_size=1)

        # LED Publisher
        self.led_pub = rospy.Publisher(f"{self._vehicle_name}/led_emitter_node/led_pattern", LEDPattern, queue_size=10)

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

        # Subscribers
        self.color_sub = rospy.Subscriber("detected_color", String, self.callback, queue_size=1)

        # State variables
        self.current_color = None
        self.is_stopped = False
        self.start_time = None

        # Define other variables as needed
        self.rate = rospy.Rate(10)  # 10 Hz

    def execute_blue_line_behavior(self):
        """
        Behavior for blue line:
        1. Stop for 3-5 seconds.
        2. Move in a curve through 90 degrees to the right.
        """
        rospy.loginfo("Executing blue line behavior")
        
        # Stop for 3-5 seconds
        self.stop(duration=4)
        
        pattern = LEDPattern()
        pattern.rgb_vals = [ColorRGBA(r=1, g=0, b=0, a=0.5)] * 5
        pattern.frequency = 2.0
        pattern.frequency_mask = [0, 1, 0, 1, 0]
        self.led_pub.publish(pattern)

        # Move in a curve to the right
        self.turn_right()

        pattern.frequency_mask = [0, 0, 0, 0, 0]
        self.led_pub.publish(pattern)

    def execute_red_line_behavior(self):
        """
        Behavior for red line:
        1. Stop for 3-5 seconds.
        2. Move straight for at least 30 cm.
        """
        rospy.loginfo("Executing red line behavior")
        
        # Stop for 3-5 seconds
        self.stop(duration=4)
        
        # Move straight for 30 cm
        self.move_straight(0.3)

    def execute_green_line_behavior(self):
        """
        Behavior for green line:
        1. Stop for 3-5 seconds.
        2. Move in a curve through 90 degrees to the left.
        """
        rospy.loginfo("Executing green line behavior")
        
        # Stop for 3-5 seconds
        self.stop(duration=4)
        
        pattern = LEDPattern()
        pattern.rgb_vals = [ColorRGBA(r=1, g=0, b=0, a=0.5)] * 5
        pattern.frequency = 2.0
        pattern.frequency_mask = [1, 0, 0, 0, 1]
        self.led_pub.publish(pattern)
        
        # Move in a curve to the left
        self.turn_left()

        pattern.frequency_mask = [0, 0, 0, 0, 0]
        self.led_pub.publish(pattern)

    def callback(self, msg):
        """
        Callback for processing camera images.
        """
        # Detect line color
        # detected_color = msg.cmd
        detected_color = msg
        
        if detected_color:
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
            else:
                self.move_straight(0.1)  # Move forward slowly
        
        self.rate.sleep()
        return LaneBehaviorCMDResponse(True)

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

    def callback_left(self, data):
        """Callback for left encoder ticks."""
        if self._ticks_left_init is None:
            self._ticks_left_init = data.data
            self._ticks_left = 0
        else:
            self._ticks_left = data.data - self._ticks_left_init

    def callback_right(self, data):
        """Callback for right encoder ticks."""
        if self._ticks_right_init is None:
            self._ticks_right_init = data.data
            self._ticks_right = 0
        else:
            self._ticks_right = data.data - self._ticks_right_init

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
    # s = rospy.Service('behavior_service', LaneBehaviorCMD, node.callback)
    rospy.spin()
