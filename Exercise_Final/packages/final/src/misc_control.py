#!/usr/bin/env python3

# import required libraries
import os
import rospy
from duckietown.dtros import DTROS, NodeType
from final.srv import MiscCtrlCMD, MiscCtrlCMDResponse
from duckietown_msgs.msg import LEDPattern
from std_msgs.msg import ColorRGBA


class MiscellaneousControl(DTROS):
    def __init__(self, node_name):
        super(MiscellaneousControl, self).__init__(node_name=node_name, node_type=NodeType.CONTROL)
        # add your code here

        # robot params
        self._vehicle_name = os.environ['VEHICLE_NAME']

        # LED publisher
        self.led_pub = rospy.Publisher(f"/{self._vehicle_name}/led_emitter_node/led_pattern", LEDPattern, queue_size=10)

        # Param address for the camera framerate
        self._camera_framerate_param = f"/{self._vehicle_name}/camera_node/framerate"

        # Save the original value to restore later 
        self.original_framerate = rospy.get_param(self._camera_framerate_param)

        # define other variables as needed
        self.cmd = ["set_fr", "reset_fr", "set_led"]
        self.colors = [ColorRGBA(r=1, g=0, b=0, a=0.5),  # red
                       ColorRGBA(r=0, g=0, b=1, a=0.5),  # blue
                       ColorRGBA(r=0, g=1, b=0, a=0.5),  # green
                       ColorRGBA(r=1, g=1, b=1, a=0.5)]  # white
        self.color_str = ['red', 'blue', 'green', 'white']

        # Store the last set LED color
        self.last_led_color = ""

        # Set shutdown callback to restore framerate to the original value
        rospy.on_shutdown(self.on_shutdown)
        
    def set_framerate(self, fr):
        rospy.set_param(self._camera_framerate_param, fr)
        rospy.loginfo(f"Original camera framerate: {self.original_framerate}. Setting it to {fr}")

    def reset_framerate(self):
        rospy.set_param(self._camera_framerate_param, self.original_framerate)
        rospy.loginfo(f"Reset framerate to {self.original_framerate}")

    def set_led(self, idx):
        # Only update the LED color if it's different from the last set color
        # rospy.loginfo(idx)
        if self.color_str[idx] == "white" and self.last_led_color == "white":
            pass
        elif self.color_str[idx] == "white" and self.last_led_color != "white": # if the last color is not None, then do not update to white
            pattern = LEDPattern()
            pattern.rgb_vals = [self.colors[idx]] * 5
            self.led_pub.publish(pattern)
            self.last_led_color = self.color_str[idx]  # Update the last set LED color
            rospy.loginfo(f"LED color changed to {self.color_str[idx]}")
        elif self.last_led_color != self.color_str[idx]:
            pattern = LEDPattern()
            pattern.rgb_vals = [self.colors[idx]] * 5
            self.led_pub.publish(pattern)
            self.last_led_color = self.color_str[idx]  # Update the last set LED color
            rospy.loginfo(f"LED color changed to {self.color_str[idx]}")            

    def callback(self, msg):
        cmd, value = msg.cmd, msg.value

        if cmd not in self.cmd:
            return MiscCtrlCMDResponse(False, f"Invalid command. Accepted commands are: {self.cmd}.")
        
        if cmd == self.cmd[0]:
            if not 0 < value <= 30:
                return MiscCtrlCMDResponse(False, "Framerate must be between 0 and 31 exclusive.")
            
            self.set_framerate(value)
            return MiscCtrlCMDResponse(True, f"Successfully set framerate to {value}")

        elif cmd == self.cmd[1]:
            self.reset_framerate()
            return MiscCtrlCMDResponse(True, f"Successfully reset framerate to {self.original_framerate}")
        
        elif cmd == self.cmd[2]:
            if not 0 <= value < len(self.colors):
                return MiscCtrlCMDResponse(False, "Index of LED color must be between 0 and 3 inclusive.")

            self.set_led(value)
            return MiscCtrlCMDResponse(True, f"Successfully set LED color to {self.color_str[value]}")
        
    def on_shutdown(self):
        self.reset_framerate()
        return MiscCtrlCMDResponse(True, f"Successfully reset framerate to {self.original_framerate}")


if __name__ == '__main__':
    node = MiscellaneousControl(node_name='miscellaneous_control_node')
    s = rospy.Service('misc_ctrl_srv', MiscCtrlCMD, node.callback)
    rospy.spin()