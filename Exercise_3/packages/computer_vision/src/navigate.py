#!/usr/bin/env python3

# potentially useful for question - 1.5

# import required libraries
import os
import rospy
import math
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import WheelsCmdStamped, WheelEncoderStamped

# Constants
WHEEL_RADIUS = 0.0318  # meters (Duckiebot wheel radius)
WHEEL_BASE = 0.05  # meters (distance between left and right wheels)
TICKS_PER_ROTATION = 135  # Encoder ticks per full wheel rotation
CURVE_SPEED = 0.5  # Base speed for curved movement

class NavigationControl(DTROS):
    def __init__(self, node_name):
        super(NavigationControl, self).__init__(node_name=node_name, node_type=NodeType.CONTROL)
        
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

        # Wait for encoders to initialize
        rospy.sleep(2)

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
        curve_radius = 0.2  # meters (adjust as needed)
        arc_length = (math.pi / 2) * curve_radius  # Arc length for 90 degrees

        # Compute required encoder ticks for the arc length
        ticks_needed = (arc_length / (2 * math.pi * WHEEL_RADIUS)) * TICKS_PER_ROTATION

        # Command wheels to move in a curve (right wheel slower)
        self.publish_velocity(CURVE_SPEED, CURVE_SPEED * 0.3)  # Adjust ratio as needed

        # Wait until the required ticks are reached
        rate = rospy.Rate(100)  # 10 Hz loop
        while not rospy.is_shutdown():
            if self._ticks_left is not None and self._ticks_right is not None:
                avg_ticks = (self._ticks_left + self._ticks_right) / 2
                rospy.loginfo(f"Current ticks: {avg_ticks}")

                if avg_ticks >= ticks_needed:
                    rospy.loginfo("90-degree curve to the right complete.")
                    break

            self.publish_velocity(CURVE_SPEED, CURVE_SPEED * 0.3)
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
        curve_radius = 0.2  # meters (adjust as needed)
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
    node = NavigationControl(node_name='navigation_control_node')
    # Move forward 1 meter
    node.move_straight(0.1)
    
    # Curve 90 degrees to the right
    node.turn_right()

     # Stop for 3 seconds
    node.stop(duration=3)
    
    # Curve 90 degrees to the left
    node.turn_left()
    
    rospy.spin()