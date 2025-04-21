#!/usr/bin/env python3
from time import sleep

# import required libraries
import rospy
import os
import sys
import cv2
import subprocess
import time
import numpy as np
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge
from final.msg import LaneFollowCMD
from final.srv import MiscCtrlCMD, NavigateCMD, ImageDetect

class MasterNode(DTROS):

    def __init__(self, node_name):
        super(MasterNode, self).__init__(node_name=node_name, node_type=NodeType.LOCALIZATION)

        # add your code here
        self._vehicle_name = os.environ['VEHICLE_NAME']

        rospy.wait_for_service("misc_ctrl_srv", timeout=1)
        self.misc_ctrl = rospy.ServiceProxy("misc_ctrl_srv", MiscCtrlCMD)
        self.misc_ctrl("set_fr", 1)
        self.misc_ctrl("set_led", 3)


        # define other variables as needed
        self.lane_follow_pub = None
        self.apriltag_srv = None
        self.crosswalk_srv = None
        self.nav_srv = None
        self.bot_detect_srv = None

        # Camera calibration parameters extracted from the file manager on the dashboard
        # Hard coding is a bad practice; We will have to hard code these parameters again if we switch to another Duckiebot
        # We found a ROS topic that gives us the intrinsic parameters, but not the extrinsict parameters (i.e. the homography matrix)
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

        self.lower_red = np.array([0, 150, 50])
        self.upper_red = np.array([10, 255, 255])

        # Initialize bridge and subscribe to camera feed
        self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"
        self._bridge = CvBridge()
        self.sub = rospy.Subscriber(self._camera_topic, CompressedImage, self.image_callback, queue_size=1)

        self.redline_pub = rospy.Publisher(f"/{self._vehicle_name}/redline_image", Image, queue_size=1)

        self.apriltag_id = None

        # Used to wait for camera initialization
        self.start_time = rospy.get_time()

        self.stage = 0
        self.num_stage1_red_lines = 0
        self.s1p0_stop_frames = 0
        self.s1p1_stop_frames = 0
        self.s1p2_stop_frames = 0
        self.last_bot_pos = 0

        self.stage1_left = False
        self.stage1_right = False

        self.stage2_left = False
        self.stage2_right = False

        self.stage3_event = 1

        # Red line detection cooldown
        self.red_line_cooldown = 10.0  # Cooldown time in seconds (adjust as needed)
        self.last_red_line_time = -11.0  # Timestamp of the last red line detection

    def undistort_image(self, image):
        return cv2.remap(image, self.map1, self.map2, cv2.INTER_LINEAR)

    def preprocess_image(self, image):
        # Downscale the image
        image = cv2.resize(image, (320, 240))  # Adjust resolution as needed
        return cv2.GaussianBlur(image, (5, 5), 0)

    def detect_red_line(self, image):
        """Detects red lines in the image."""
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        red_mask = cv2.inRange(hsv_image, self.lower_red, self.upper_red)
        red_pixels = cv2.countNonZero(red_mask)
        self.redline_pub.publish(self._bridge.cv2_to_imgmsg(red_mask, encoding="mono8"))
        return red_pixels > 100  # Threshold for red line detection

    def stop(self, duration):
        self.nav_srv(0, 0, 0, duration)

    def turn_left(self):
        self.nav_srv(1, 0.3, 0.28, 0.1)
        self.nav_srv(1, 0.3, 0.52, 0.60)

    def turn_right(self):
        self.nav_srv(1, 0.3, 0.28, 0.1)
        self.nav_srv(1, 0.75, 0.3, 0.4555)

    def drive_straight(self, speed, duration):
        self.nav_srv(1, speed, speed, duration)

    def lane_follow(self, image):
        cmd = LaneFollowCMD()
        cmd.shutdown = False
        cmd.image = self._bridge.cv2_to_imgmsg(image.copy(), encoding="bgr8")
        cmd.state = 0
        self.lane_follow_pub.publish(cmd)

    def image_callback(self, msg):
        current_time = rospy.get_time()
        if current_time - self.start_time <= 10:
            rospy.loginfo("Waiting for camera initialization")
            return

        # Convert compressed image to CV2
        image = self._bridge.compressed_imgmsg_to_cv2(msg)

        # Undistort image
        undistorted_image = self.undistort_image(image)

        # Preprocess image
        preprocessed_image = self.preprocess_image(undistorted_image)

        if self.stage == 0:
            self.sub.unregister()
            self.stage += 1
            # self.stage = 3

            subprocess.Popen(['rosrun', 'final', 'navigation.py'])
            rospy.wait_for_service("nav_srv", timeout=30)
            self.nav_srv = rospy.ServiceProxy("nav_srv", NavigateCMD)

            subprocess.Popen(['rosrun', 'final', 'soft_bot_detect.py'])
            rospy.wait_for_service("bot_detect_srv", timeout=30)
            self.bot_detect_srv = rospy.ServiceProxy("bot_detect_srv", ImageDetect)

            subprocess.Popen(['rosrun', 'final', 'tail_detect.py'])
            self.lane_follow_pub = rospy.Publisher(f"/{self._vehicle_name}/tailing_input", LaneFollowCMD, queue_size=1)
            time.sleep(5)


            # subprocess.Popen(['rosrun', 'final', 'bot_detect.py'])
            # rospy.wait_for_service("bot_detect_srv", timeout=30)
            # self.bot_detect_srv = rospy.ServiceProxy("bot_detect_srv", ImageDetect)

            # subprocess.Popen(['rosrun', 'final', 'crosswalk.py'])
            # rospy.wait_for_service("crosswalk_detect_srv", timeout=30)
            # self.crosswalk_srv = rospy.ServiceProxy("crosswalk_detect_srv", ImageDetect)

            # self.misc_ctrl("set_led", 3)

            # subprocess.Popen(['rosrun', 'final', 'lane_follow.py'])
            # self.lane_follow_pub = rospy.Publisher(f"/{self._vehicle_name}/lane_follow_input", LaneFollowCMD,
            #                                         queue_size=1)
            # time.sleep(5)

            rospy.loginfo("Entering stage 1.")
            self.sub = rospy.Subscriber(self._camera_topic, CompressedImage, self.image_callback)
        elif self.stage == 1:
            height = image.shape[0]
            cropped_image_redline = image[height * 2 // 3:, :]
            detected_red = self.detect_red_line(cropped_image_redline)

            imgmsg = self._bridge.cv2_to_imgmsg(undistorted_image.copy(), encoding="bgr8")

            if self.num_stage1_red_lines == 0:
                bot_pos = self.bot_detect_srv(False, imgmsg).res
                if bot_pos > 0:
                    rospy.loginfo(f"Detected bot on {'left' if bot_pos == 1 else 'right'}.")
                    self.last_bot_pos = bot_pos
                    self.s1p0_stop_frames = 0

            current_time = rospy.get_time()
            if detected_red:
                rospy.loginfo("Detected red line.")
                bot_pos = self.bot_detect_srv(False, imgmsg).res

                if self.num_stage1_red_lines == 0:
                    self.stop(0)
                    if bot_pos > 0:
                        rospy.loginfo("Lead bot too close.")
                        self.s1p0_stop_frames = 0
                    else:
                        self.s1p0_stop_frames += 1

                    if self.s1p0_stop_frames >= 4:
                        self.sub.unregister()

                        self.num_stage1_red_lines += 1
                        if self.last_bot_pos == 1:
                            self.stage1_left = True
                            self.turn_left()
                        elif self.last_bot_pos == 2:
                            self.stage1_right = True
                            self.turn_right()

                        self.last_red_line_time = current_time

                        self.sub = rospy.Subscriber(self._camera_topic, CompressedImage, self.image_callback, queue_size=1)

                elif self.num_stage1_red_lines == 1:
                    self.stop(0)

                    if current_time - self.last_red_line_time > self.red_line_cooldown:
                        if bot_pos > 0:
                            rospy.loginfo("Lead bot too close.")
                            self.s1p1_stop_frames = 0
                        else:
                            self.s1p1_stop_frames += 1

                        if self.s1p1_stop_frames >= 4:
                            self.sub.unregister()
                            self.num_stage1_red_lines += 1
                            rospy.loginfo("Driving straight at the intersection")
                            self.drive_straight(0.3, 0.8)
                            self.sub = rospy.Subscriber(self._camera_topic, CompressedImage, self.image_callback, queue_size=1)

                            self.last_red_line_time = current_time
                    else:
                        self.lane_follow(image)
                elif self.num_stage1_red_lines == 2:
                    self.stop(0)

                    if current_time - self.last_red_line_time > self.red_line_cooldown:
                        if bot_pos > 0:
                            rospy.loginfo("Lead bot too close.")
                            self.s1p2_stop_frames = 0
                        else:
                            self.s1p2_stop_frames += 1

                        if self.s1p2_stop_frames >= 4:
                            self.sub.unregister()
                            self.num_stage1_red_lines += 1
                            if self.stage1_left:
                                self.turn_right()

                            if self.stage1_right:
                                self.turn_left()

                            self.sub = rospy.Subscriber(self._camera_topic, CompressedImage, self.image_callback, queue_size=1)
                            self.last_red_line_time = current_time
                    else:
                        self.lane_follow(image)

            else:
                self.lane_follow(image)

            if self.num_stage1_red_lines == 3:
                self.stage += 1

                self.sub.unregister()

                shutdown_cmd = LaneFollowCMD()
                shutdown_cmd.shutdown = True
                shutdown_cmd.image = self._bridge.cv2_to_imgmsg(image.copy(), encoding="bgr8")
                shutdown_cmd.state = 0
                self.lane_follow_pub.publish(shutdown_cmd)

                try:
                    self.bot_detect_srv(True, self._bridge.cv2_to_imgmsg(image.copy(), encoding="bgr8"))
                except rospy.ServiceException:
                    self.bot_detect_srv = None

                subprocess.Popen(['rosrun', 'final', 'lane_follow.py'])
                self.lane_follow_pub = rospy.Publisher(f"/{self._vehicle_name}/lane_follow_input", LaneFollowCMD,
                                                       queue_size=1)
                time.sleep(5)

                subprocess.Popen(['rosrun', 'final', 'apriltag_detection.py'])
                rospy.wait_for_service('apriltag_detection_srv', timeout=30)
                self.apriltag_srv = rospy.ServiceProxy('apriltag_detection_srv', ImageDetect)

                self.sub = rospy.Subscriber(self._camera_topic, CompressedImage, self.image_callback, queue_size=1)

                rospy.loginfo("Entering stage 2.")

        elif self.stage == 2:
            imgmsg = self._bridge.cv2_to_imgmsg(undistorted_image.copy(), encoding="bgr8")
            apriltag_id = self.apriltag_srv(False, imgmsg).res

            if apriltag_id != 255:
                self.apriltag_id = apriltag_id

            height = image.shape[0]
            cropped_image_redline = image[height * 2 // 3:, :]
            detected_red = self.detect_red_line(cropped_image_redline)

            if detected_red:
                self.sub.unregister()

                self.nav_srv(0, 0, 0, 2)

                if self.apriltag_id == 48:
                    self.stage2_left = True

                    self.turn_left()

                elif self.apriltag_id == 50:
                    self.stage2_right = True

                    self.turn_right()

                else:
                    rospy.loginfo("No valid Apriltag detected.")

                self.sub = rospy.Subscriber(self._camera_topic, CompressedImage, self.image_callback, queue_size=1)

            if self.stage2_left and self.stage2_right:
                self.stage += 1

                self.sub.unregister()

                try:
                    self.apriltag_srv(True, imgmsg)
                except rospy.service.ServiceException:
                    self.apriltag_srv = None

                subprocess.Popen(['rosrun', 'final', 'bot_detect.py'])
                rospy.wait_for_service("bot_detect_srv", timeout=30)
                self.bot_detect_srv = rospy.ServiceProxy("bot_detect_srv", ImageDetect)

                subprocess.Popen(['rosrun', 'final', 'crosswalk.py'])
                rospy.wait_for_service("crosswalk_detect_srv", timeout=30)
                self.crosswalk_srv = rospy.ServiceProxy("crosswalk_detect_srv", ImageDetect)

                self.misc_ctrl("set_led", 3)

                self.sub = rospy.Subscriber(self._camera_topic, CompressedImage, self.image_callback, queue_size=1)

                rospy.loginfo("Entering stage 3.")
                return

            self.lane_follow(preprocessed_image)
        elif self.stage == 3:
            assert self.nav_srv is not None, "Stage 3: Navigation service is not available."
            assert self.crosswalk_srv is not None, "Stage 3: Crosswalk detection service is not available."
            assert self.bot_detect_srv is not None, "Stage 3: Bot detection service is not available."

            imgmsg = self._bridge.cv2_to_imgmsg(preprocessed_image.copy(), encoding="bgr8")

            if self.stage3_event == 2:

                bot_res = self.bot_detect_srv(False, imgmsg)

                if bot_res.res > 0:
                    self.stage3_event += 1
                    self.sub.unregister()

                    self.nav_srv(0, 0, 0, 2)
                    self.nav_srv(2, -1, 0, 0)
                    self.nav_srv(1, 0.5, 0.49, 0.15)
                    self.nav_srv(2, 1, 0, 0)
                    self.nav_srv(1, 0.5, 0.48, 0.45)
                    self.nav_srv(2, 1, 0, 0)
                    self.nav_srv(1, 0.5, 0.49, 0.15)
                    self.nav_srv(2, -1, 0, 0)

                    self.sub = rospy.Subscriber(self._camera_topic, CompressedImage, self.image_callback, queue_size=1)
                else:
                    self.lane_follow(preprocessed_image)
            elif self.stage3_event == 1 or self.stage3_event == 3:
                self.sub.unregister()
                crosswalk_res = self.crosswalk_srv(False, imgmsg).res
                self.sub = rospy.Subscriber(self._camera_topic, CompressedImage, self.image_callback, queue_size=1)
                self.stage3_event += crosswalk_res
                rospy.loginfo(f"{crosswalk_res}")

            height = image.shape[0]
            cropped_image_redline = image[height * 2 // 3:, :]
            detected_red = self.detect_red_line(cropped_image_redline)

            if detected_red and self.stage3_event > 1:
                self.stage += 1
                self.stop(0)

                try:
                    self.crosswalk_srv(True, imgmsg)
                except rospy.service.ServiceException:
                    self.crosswalk_srv = None

                try:
                    self.bot_detect_srv(True, imgmsg)
                except rospy.service.ServiceException:
                    self.bot_detect_srv = None

                try:
                    self.nav_srv(255, 0, 0, 0)
                except rospy.service.ServiceException:
                    self.nav_srv = None

                try:
                    self.misc_ctrl("shutdown", 0)
                except rospy.service.ServiceException:
                    self.misc_ctrl = None

                rospy.signal_shutdown("Reason")
                rospy.loginfo("Entering stage 4.")
        elif self.stage == 4:
            pass
        else:
            raise NotImplementedError("Something went wrong: we have entered an unknown stage.")


if __name__ == '__main__':
    # create the node
    node = MasterNode(node_name='master')
    rospy.spin()
