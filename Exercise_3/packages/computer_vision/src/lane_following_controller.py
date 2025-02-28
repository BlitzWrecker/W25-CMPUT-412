#!/usr/bin/env python3

import rospy
import time
import math
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import WheelsCmdStamped, WheelEncoderStamped
from std_msgs.msg import Float32
import os

# Constants
WHEEL_RADIUS = 0.0318  # meters (Duckiebot wheel radius)
WHEEL_BASE = 0.05  # meters (distance between left and right wheels)
TICKS_PER_ROTATION = 135  # Encoder ticks per full wheel rotation
CURVE_SPEED = 0.5  # Base speed for curved movement

class LaneControllerNode(DTROS):
    def __init__(self, node_name):
        super(LaneControllerNode, self).__init__(node_name=node_name, node_type=NodeType.CONTROL)

        # Controller type ('P', 'PD', or 'PID')
        self.controller_type = 'PID'  # Change as needed ('P', 'PD', 'PID')

        # PID Gains
        self.kp = 0.5  # Proportional gain
        self.kd = 0.1  # Derivative gain
        self.ki = 0.01  # Integral gain

        # Control variables
        self.prev_error = 0
        self.integral = 0
        self.last_time = time.time()

        # Movement parameters
        self.base_speed = 0.3  # Base wheel speed
        self.max_speed = 1.0  # Max wheel speed

        # Distance tracking
        self.distance_travelled = 0
        self.target_distance = 1.5  # meters
        self._ticks_left_init = None
        self._ticks_right_init = None
        self._ticks_left = None
        self._ticks_right = None

        # Initialize publisher/subscribers
        self._vehicle_name = os.environ['VEHICLE_NAME']
        self.pub_cmd = rospy.Publisher(f"/{self._vehicle_name}/wheels_cmd", WheelsCmdStamped, queue_size=1)
        rospy.Subscriber(f"/{self._vehicle_name}/lane_error", Float32, self.yellow_lane_callback)
        rospy.Subscriber(f"/{self._vehicle_name}/left_wheel_encoder_node/tick", WheelEncoderStamped, self.callback_left)
        rospy.Subscriber(f"/{self._vehicle_name}/right_wheel_encoder_node/tick", WheelEncoderStamped, self.callback_right)

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

    def calculate_distance_travelled(self):
        if self._ticks_left is not None and self._ticks_right is not None:
            avg_ticks = (self._ticks_left + self._ticks_right) / 2
            self.distance_travelled = (avg_ticks / TICKS_PER_ROTATION) * (2 * math.pi * WHEEL_RADIUS)
        return self.distance_travelled

    def calculate_p_control(self, error):
        return self.kp * error

    def calculate_pd_control(self, error):
        current_time = time.time()
        dt = current_time - self.last_time if self.last_time else 0.1
        d_term = self.kd * (error - self.prev_error) / dt if dt > 0 else 0
        self.prev_error = error
        self.last_time = current_time
        return self.calculate_p_control(error) + d_term

    def calculate_pid_control(self, error):
        current_time = time.time()
        dt = current_time - self.last_time if self.last_time else 0.1
        self.integral += error * dt
        i_term = self.ki * self.integral
        return self.calculate_pd_control(error) + i_term

    def get_control_output(self, error):
        if self.controller_type == 'P':
            return self.calculate_p_control(error)
        elif self.controller_type == 'PD':
            return self.calculate_pd_control(error)
        else:
            return self.calculate_pid_control(error)

    def publish_cmd(self, control):
        if self.calculate_distance_travelled() >= self.target_distance:
            rospy.loginfo("Reached target distance. Stopping.")
            self.pub_cmd.publish(WheelsCmdStamped(vel_left=0, vel_right=0))
            rospy.signal_shutdown("Target distance reached")
            return

        left_speed = max(min(self.base_speed - control, self.max_speed), -self.max_speed)
        right_speed = max(min(self.base_speed + control, self.max_speed), -self.max_speed)

        cmd = WheelsCmdStamped()
        cmd.vel_left = left_speed
        cmd.vel_right = right_speed
        self.pub_cmd.publish(cmd)

    def yellow_lane_callback(self, msg):
        error = msg.data
        control = self.get_control_output(error)
        self.publish_cmd(control)


if __name__ == '__main__':
    node = LaneControllerNode(node_name='lane_controller_node')
    rospy.spin()
