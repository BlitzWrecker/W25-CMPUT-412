#!/usr/bin/env python3
import rospy
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import WheelsCmdStamped, WheelEncoderStamped
from final.srv import NavigateCMD, NavigateCMDResponse
import math
import os

WHEEL_RADIUS = 0.0318  # meters (Duckiebot wheel radius)
WHEEL_BASE = 0.05  # meters (distance between left and right wheels)
TICKS_PER_ROTATION = 135  # Encoder ticks per full wheel rotation
TURN_SPEED = 0.40  # Adjust speed for accuracy

class NavigationNode(DTROS):
    def __init__(self, node_name):
        super(NavigationNode, self).__init__(node_name=node_name, node_type=NodeType.CONTROL)

        self._vehicle_name = os.environ["VEHICLE_NAME"]

        # Movement parameters
        self.base_speed = 0.3  # Base wheel speed
        self.max_speed = 0.5  # Max wheel speed

        self._ticks_left_init = None
        self._ticks_right_init = None
        self._ticks_left = None
        self._ticks_right = None

        self._left_encoder_topic = f"/{self._vehicle_name}/left_wheel_encoder_node/tick"
        self._right_encoder_topic = f"/{self._vehicle_name}/right_wheel_encoder_node/tick"

        self.sub_left = rospy.Subscriber(self._left_encoder_topic, WheelEncoderStamped, self.callback_left)
        self.sub_right = rospy.Subscriber(self._right_encoder_topic, WheelEncoderStamped, self.callback_right)
        wheels_topic = f"/{self._vehicle_name}/wheels_driver_node/wheels_cmd"
        self._publisher = rospy.Publisher(wheels_topic, WheelsCmdStamped, queue_size=1)

        rospy.on_shutdown(self.on_shutdown)

    def move_wheels(self, left_vel, right_vel, duration):
        """Actively publishes movement commands at a controlled rate."""
        rospy.loginfo(f"Moving: Left = {left_vel}, Right = {right_vel} for {duration} seconds")
        self.reset_encoders()

        cmd = WheelsCmdStamped()
        cmd.vel_left = left_vel
        cmd.vel_right = right_vel

        # distance_traveled = (2 * math.pi * 0.0318 * (self._ticks_left + self._ticks_right)/2 ) / 135
        distance = duration
        message = WheelsCmdStamped(vel_left=left_vel, vel_right=right_vel)
        # self._publisher.publish(message)
        rate = rospy.Rate(100)
        while not rospy.is_shutdown():
            if self._ticks_right is not None and self._ticks_left is not None:
                distance_traveled = (2 * math.pi * 0.0318 * (self._ticks_left + self._ticks_right)/2 ) / 135

                if distance_traveled >= distance:

                    message = WheelsCmdStamped(vel_left=0, vel_right=0)
                    self._publisher.publish(message)
                    break

                # rospy.loginfo(distance_traveled)
                # rospy.loginfo(self._ticks_left)

                self._publisher.publish(message)
                rate.sleep()

        # Stop the robot after moving
        rospy.loginfo("Stopping robot")
        stop_command = WheelsCmdStamped(vel_left=0, vel_right=0)  # ✅ Correct Stop Command
        self._publisher.publish(stop_command)
        rospy.sleep(0.5)  # Ensure stop command is received

    def turn_90_degrees(self, direction=1):
        """
        Turns the Duckiebot 90 degrees in place.
        :param direction: 1 to turn right, -1 to turn left
        """
        # Reset encoder counters before each turn
        self.reset_encoders()

        # Compute required encoder ticks for 90-degree turn
        ticks_needed = round((WHEEL_BASE / (8 * WHEEL_RADIUS)) * TICKS_PER_ROTATION) + 11
        if direction == -1:
            ticks_needed = ticks_needed * 1.3
        else:
            ticks_needed = ticks_needed * 1.1
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
                # rospy.loginfo(f"Current ticks: {avg_ticks}")

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

    def stop_for_duration(self, duration):
        """Stops the robot for the specified duration."""
        cmd = WheelsCmdStamped()
        cmd.vel_left = 0
        cmd.vel_right = 0
        self._publisher.publish(cmd)
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
        self._publisher.publish(cmd)

    def execute_cmd(self, msg):
        cmd, val1, val2, duration = msg.cmd, msg.val1, msg.val2, msg.duration

        if cmd == 255:
            self.on_shutdown()
            s.shutdown("Shutting down navigation service")
            rospy.signal_shutdown("Shutting down navigation service node")
            return NavigateCMDResponse(False)

        if cmd == 0:
            self.stop_for_duration(duration)

        elif cmd == 1:
            self.move_wheels(val1, val2, duration)

        elif cmd == 2:
            self.turn_90_degrees(val1)

        return NavigateCMDResponse(True)

    def on_shutdown(self):
        cmd = WheelsCmdStamped()
        cmd.vel_left = 0
        cmd.vel_right = 0
        self._publisher.publish(cmd)


if __name__ == '__main__':
    node = NavigationNode(node_name='navigation_node')
    s = rospy.Service("nav_srv", NavigateCMD, node.execute_cmd)
    rospy.spin()
