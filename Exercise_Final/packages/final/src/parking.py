#!/usr/bin/env python3
from time import sleep
import rospy
import sys
import os
import math
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import WheelsCmdStamped, WheelEncoderStamped


WHEEL_RADIUS = 0.0318  # meters (Duckiebot wheel radius)
WHEEL_BASE = 0.05  # meters (distance between left and right wheels)
TICKS_PER_ROTATION = 135  # Encoder ticks per full wheel rotation
TURN_SPEED = 0.6  # Adjust speed for accuracy


class ParkingNode(DTROS):
   def __init__(self, parking_stall):
       super(ParkingNode, self).__init__(node_name="parking_node", node_type=NodeType.GENERIC)
       self.parking_stall = parking_stall
      
       # Get Duckiebot's name
       self._vehicle_name = os.environ["VEHICLE_NAME"]


       # Publisher for wheel movement
       self._publisher = rospy.Publisher(
           f'/{self._vehicle_name}/wheels_driver_node/wheels_cmd',
           WheelsCmdStamped,
           queue_size=1
       )


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


       self.rate = rospy.Rate(10)  # 10 Hz loop rate
       rospy.sleep(2)  # Wait for everything to initialize


       # Start parking maneuver after initialization
       self.execute_parking_maneuver()


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


   def move_wheels(self, left_vel, right_vel, duration):
       """Move wheels for a specific distance."""
       rospy.loginfo(f"Moving: Left = {left_vel}, Right = {right_vel} for {duration} meters")
       self.reset_encoders()


       cmd = WheelsCmdStamped()
       cmd.vel_left = left_vel
       cmd.vel_right = right_vel


       distance_traveled = 0
       start_time = rospy.Time.now().to_sec()
      
       while not rospy.is_shutdown():
           current_time = rospy.Time.now().to_sec()
           elapsed_time = current_time - start_time
          
           if elapsed_time >= duration:
               break
              
           self._publisher.publish(cmd)
           self.rate.sleep()


       # Stop the robot
       stop_command = WheelsCmdStamped(vel_left=0, vel_right=0)
       self._publisher.publish(stop_command)
       rospy.sleep(0.5)


   def turn_90_degrees(self, direction=1):
       """
       Turns the Duckiebot 90 degrees in place.
       :param direction: 1 for left, -1 for right
       """
       self.reset_encoders()
       ticks_needed = round((WHEEL_BASE / (8 * WHEEL_RADIUS)) * TICKS_PER_ROTATION) + 11
       rospy.loginfo(f"Ticks needed for 90-degree turn: {ticks_needed}")


       turn_command = WheelsCmdStamped(
           vel_left=TURN_SPEED * direction,
           vel_right=-TURN_SPEED * direction
       )
       self._publisher.publish(turn_command)


       rate = rospy.Rate(10)
       while not rospy.is_shutdown():
           if self._ticks_left is not None and self._ticks_right is not None:
               avg_ticks = (abs(self._ticks_left) + abs(self._ticks_right)) / 2
               if avg_ticks >= ticks_needed:
                   break
           self._publisher.publish(turn_command)
           rate.sleep()


       stop_command = WheelsCmdStamped(vel_left=0, vel_right=0)
       self._publisher.publish(stop_command)
       rospy.sleep(1)


   def execute_parking_maneuver(self):
        """Execute the parking maneuver based on the stall number"""

        rospy.loginfo(f"Starting parking in stall {self.parking_stall}")

        self.move_wheels(-0.2, 0.6, 0.25)  # Forward

        # Different turns based on parking stall
        if self.parking_stall == 1:
            self.move_wheels(0.5, 0.5, 1)  # Forward
            self.turn_90_degrees(-1)  # Left turn
            self.move_wheels(-0.5, -0.5, 1)  # Reverse
        elif self.parking_stall == 2:
            self.move_wheels(0.5, 0.5, 2)  # Forward
            self.turn_90_degrees(-1)  # Left turn
            self.move_wheels(-0.5, -0.5, 1)  # Reverse
        elif self.parking_stall == 3:
            self.move_wheels(0.5, 0.5, 1)  # Forward
            self.turn_90_degrees(1)  # Right turn
            self.move_wheels(-0.5, -0.5, 2.5)  # Reverse
        elif self.parking_stall == 4:
            self.move_wheels(0.5, 0.5, 2)  # Forward
            self.turn_90_degrees(1)  # Right turn
            self.move_wheels(-0.5, -0.5, 2.5)  # Reverse
        else:
            rospy.logerr(f"Invalid parking stall number: {self.parking_stall}")
            
        rospy.loginfo("Parking completed!")
        rospy.signal_shutdown("Task completed")


if __name__ == '__main__':
   # Check if parking stall number is provided
   if len(sys.argv) < 2:
       rospy.logerr("Usage: rosrun ex4 parking.py <parking_stall_number>")
       sys.exit(1)
  
   try:
       parking_stall = int(sys.argv[1])
       if parking_stall < 1 or parking_stall > 4:
           raise ValueError
   except ValueError:
       rospy.logerr("Parking stall must be an integer between 1 and 4")
       sys.exit(1)
   rospy.loginfo("Command line argument:")
   rospy.loginfo(parking_stall)
   # Initialize the node with the parking stall number
   node = ParkingNode(parking_stall)
   rospy.spin()