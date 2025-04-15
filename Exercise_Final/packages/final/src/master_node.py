#!/usr/bin/env python3

# import required libraries
import rospy
import os
import cv2
import numpy as np
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
from final.srv import MiscCtrlCMD, NavigateCMD, ImageDetect

class MasterNode(DTROS):

    def __init__(self, node_name):
        super(MasterNode, self).__init__(node_name=node_name, node_type=NodeType.LOCALIZATION)

        # add your code here
        self._vehicle_name = os.environ['VEHICLE_NAME']

        rospy.wait_for_service("misc_ctrl_srv", timeout=1)
        self.misc_ctrl = rospy.ServiceProxy("misc_ctrl_srv", MiscCtrlCMD)
        self.misc_ctrl("set_fr", 3)

        # define other variables as needed
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

        # self.lane_follow_pub = rospy.Publisher(f"/{self._vehicle_name}/lane_follow_input", LaneFollowCMD, queue_size=1)

        # Initialize bridge and subscribe to camera feed
        self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"
        self._bridge = CvBridge()
        self.sub = rospy.Subscriber(self._camera_topic, CompressedImage, self.image_callback, queue_size=1)

        self.prev_state = None

        # Used to wait for camera initialization
        self.start_time = rospy.get_time()

        self.stage = 1

        self.num_crosswalks = 0
        self.broken_bot = 0

    def undistort_image(self, image):
        return cv2.remap(image, self.map1, self.map2, cv2.INTER_LINEAR)

    def preprocess_image(self, image):
        # Downscale the image
        image = cv2.resize(image, (320, 240))  # Adjust resolution as needed
        return cv2.GaussianBlur(image, (5, 5), 0)

    def detect_lane_color(self, image):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        masks = {
            "blue": cv2.inRange(hsv_image, self.lower_blue, self.upper_blue),
            "orange": cv2.inRange(hsv_image, self.lower_orange, self.upper_orange)
        }
        return masks

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

        if self.stage == 1:
            self.stage += 1
        elif self.stage == 2:
            self.stage += 1
            rospy.wait_for_service("crosswalk_detect_srv", timeout=1)
            self.crosswalk_srv = rospy.ServiceProxy("crosswalk_detect_srv", ImageDetect)

            rospy.wait_for_service("nav_srv", timeout=1)
            self.nav_srv = rospy.ServiceProxy("nav_srv", NavigateCMD)

            rospy.wait_for_service("bot_detect_srv", timeout=1)
            self.bot_detect_srv = rospy.ServiceProxy("bot_detect_srv", ImageDetect)

        elif self.stage == 3:
            assert self.nav_srv is not None, "Stage 3: Navigation service is not available."
            assert self.crosswalk_srv is not None, "Stage 3: Crosswalk detection service is not available."
            assert self.bot_detect_srv is not None, "Stage 3: Bot detection service is not available."

            compressed_image = self._bridge.cv2_to_compressed_imgmsg(preprocessed_image)
            msg = ImageDetect()
            msg.shutdown = False
            msg.image = compressed_image
            crosswalk_res = self.crosswalk_srv(msg)
            bot_res = self.bot_detect_srv(msg)

            self.crosswalk_srv += crosswalk_res.res
            self.broken_bot += bot_res.res

            if bot_res.res == 1:
                self.sub.unregister()

                self.nav_srv(NavigateCMD(cmd=2, val1=-1, val2=0, duration=0))
                self.nav_srv(NavigateCMD(cmd=1, val1=0.5, val2=0.5, duration=0.15))
                self.nav_srv(NavigateCMD(cmd=2, val1=1, val2=0, duration=0))
                self.nav_srv(NavigateCMD(cmd=1, val1=0.5, val2=0.5, duration=0.60))
                self.nav_srv(NavigateCMD(cmd=2, val1=1, val2=0, duration=0))
                self.nav_srv(NavigateCMD(cmd=1, val1=0.5, val2=0.5, duration=0.15))
                self.nav_srv(NavigateCMD(cmd=2, val1=-1, val2=0, duration=0))

                self.sub = rospy.Subscriber(self._camera_topic, CompressedImage, self.image_callback, queue_size=1)

            if self.broken_bot == 1 and self.num_crosswalks == 2:
                self.stage += 1

                shutdown_cmd = ImageDetect()
                shutdown_cmd.shutdown = True

                try:
                    self.crosswalk_srv(shutdown_cmd)
                except rospy.service.ServiceException:
                    self.crosswalk_srv = None

                try:
                    self.bot_detect_srv(shutdown_cmd)
                except rospy.service.ServiceException:
                    self.bot_detect_srv = None

                shutdown_cmd = NavigateCMD()
                shutdown_cmd.cmd = 255

                try:
                    self.nav_srv(shutdown_cmd)
                except rospy.service.ServiceException:
                    self.nav_srv = None


if __name__ == '__main__':
    # create the node
    node = MasterNode(node_name='master')
    rospy.spin()
