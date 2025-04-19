#!/usr/bin/env python3

# potentially useful for part 2 of exercise 4

# import required libraries
import rospy
import os
import cv2
import time
import math
import numpy as np
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from final.srv import MiscCtrlCMD, ImageDetect, ImageDetectResponse
from final.msg import LaneFollowCMD

class CrossWalkNode(DTROS):

    def __init__(self, node_name):
        super(CrossWalkNode, self).__init__(node_name=node_name, node_type=NodeType.LOCALIZATION)

        # add your code here
        self._vehicle_name = os.environ['VEHICLE_NAME']

        # define other variables as needed
        
        # Camera calibration parameters extracted from the file manager on the dashboard
        # Hard coding is a bad practice; We will have to hard code these parameters again if we switch to another Duckiebot
        # We found a ROS topic that gives us the intrinsic parameters, but not the extrinsict parameters (i.e. the homography matrix)
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
    
        # Color detection parameters in HSV format
        self.lower_blue = np.array([100, 150, 50])
        self.upper_blue = np.array([140, 255, 255])
        self.lower_orange = np.array([15, 100, 100])
        self.upper_orange = np.array([20, 255, 255])

        # Set a distance threshhold for detecting lines so we don't detect lines that are too far away
        self.dist_thresh = 5
        
        self.img_pub = rospy.Publisher(f"/{self._vehicle_name}/crosswalk_processed_image", Image, queue_size=10)
        self.res_pub = rospy.Publisher(f"/{self._vehicle_name}/lane_follow_input", LaneFollowCMD,queue_size=1)

        # Initialize bridge
        self._bridge = CvBridge()

        self.prev_state = None
    
    def detect_lane_color(self, image):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        masks = {
            "blue": cv2.inRange(hsv_image, self.lower_blue, self.upper_blue),
            "orange": cv2.inRange(hsv_image, self.lower_orange, self.upper_orange)
        }
        return masks

    # Asked ChatGPT "how to use extrinsic parameters to calculate distance between two objects in an image"
    # ChatGPT answered with a very generic computation method using a rotation matrix and a translation vector
    # Then followed up with "I am working with a duckiebot".
    # ChatGPT answered with an algorithm using the homography matrx
    def extrinsic_transform(self, u, v):
        pixel_coord = np.array([u, v, 1]).reshape(3, 1)
        world_coord = np.dot(self.homography, pixel_coord)
        world_coord /= world_coord[2]
        return world_coord[:2].flatten()

    def calculate_dist(self, l1, l2):
        return np.linalg.norm(l2 - l1)

    def detect_crosswalk(self, image, masks):
        colors = {"blue": (255, 0, 0), "orange": (0, 165, 255)}
        contour_dists = {}

        for color_name, mask in masks.items():
            contour_dists[color_name] = []
            masked_color = cv2.bitwise_and(image, image, mask=mask)
            gray = cv2.cvtColor(masked_color, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
            for contour in contours:
                if cv2.contourArea(contour) > 200:  # Filter small contours
                    x, y, w, h = cv2.boundingRect(contour)
                    box_rep = self.extrinsic_transform(x + w // 2, y + h)
                    screen_bot = self.extrinsic_transform(x + w // 2, image.shape[0])

                    # Estimate the distance of the line from the robot using the distance of the line from the bottom of the screen
                    dist = self.calculate_dist(box_rep, screen_bot)

                    if (dist <= self.dist_thresh):
                        contour_dists[color_name].append((dist, x, y, w, h))

            contour_dists[color_name] = sorted(contour_dists[color_name], key=lambda elem: elem[0])

        if "blue" in contour_dists.keys():
            if "orange" in contour_dists.keys() and len(contour_dists["blue"]) >= 1 and len(contour_dists["orange"]) > 0:
                for i in contour_dists.keys():
                    for dist, x, y, w, h in contour_dists[i]:
                        cv2.rectangle(image, (x, y), (x + w, y + h), colors[i], 2)
                        cv2.putText(image, f"Dist: {dist*30:.2f} cm", (x, y + h + 10), cv2.FONT_HERSHEY_PLAIN, 1, colors[i])
 
                rospy.loginfo("Detected crosswalk with ducks.")
                return image, 3

            if len(contour_dists["blue"]) >= 2:
                for dist, x, y, w, h in contour_dists["blue"]:
                    cv2.rectangle(image, (x, y), (x + w, y + h), colors["blue"], 2)
                    cv2.putText(image, f"Dist: {dist*30:.2f} cm", (x, y + h + 10), cv2.FONT_HERSHEY_PLAIN, 1, colors["blue"])
 
                rospy.loginfo("Detected empty crosswalk.")
                return image, 2

        rospy.loginfo("Nothing detected yet")
        return image, 0

    def image_callback(self, msg):
        shutdown, image = msg.shutdown, msg.image
        if shutdown:
            s.shutdown("Shutting down crosswalk detection service.")
            rospy.signal_shutdown('Shutting down crosswalk detection node.')
            return ImageDetectResponse(255)

        # Convert compressed image to CV2
        preprocessed_image = self._bridge.imgmsg_to_cv2(image, desired_encoding="bgr8")

        # Crop the bottom of the image before detecting crosswalks because the bottom of the image is warped even after
        # undistortion. Another way to achieve the same purpose is to apply a minimum distance threshold, i.e. the blue
        # lines have to be at least x units away.
        height, _ = preprocessed_image.shape[:2]
        cropped_image = preprocessed_image[:math.ceil(height * 0.7), :]

        # Detect lanes and colors
        masks = self.detect_lane_color(cropped_image)
        lane_detected_image, detected_crosswalk = self.detect_crosswalk(cropped_image.copy(), masks)

        # Publish processed image (optional)
        self.img_pub.publish(self._bridge.cv2_to_imgmsg(lane_detected_image, encoding="bgr8"))

        lane_follow_message = LaneFollowCMD()

        if self.prev_state == 1:
            if detected_crosswalk >= 2:
                lane_follow_message.shutdown = False
                lane_follow_message.image = self._bridge.cv2_to_imgmsg(preprocessed_image.copy(), encoding='bgr8')
                lane_follow_message.state = 1
                self.res_pub.publish(lane_follow_message)
                return ImageDetectResponse(0)

        # We have come across an empty crosswalk. Either (1) there were ducks on this crosswalk before, but now they
        # have crossed, or (2) there were no ducks on this crosswalk when we first approached it.
        if detected_crosswalk == 2:
            # Case (2): we have to stop for one second before moving on
            if self.prev_state is None or (self.prev_state is not None and self.prev_state < 1):
                # Signal the lane following node to stop the vehicle
                lane_follow_message.shutdown = False
                lane_follow_message.image = self._bridge.cv2_to_imgmsg(preprocessed_image.copy(), encoding='bgr8')
                lane_follow_message.state = detected_crosswalk
                self.res_pub.publish(lane_follow_message)

                # We must stop the vehicle for at least one second. We implement this using the Python built-in function
                # time.sleep().
                time.sleep(1)

            self.prev_state = 1
            return ImageDetectResponse(0)

        lane_follow_message.shutdown = False
        lane_follow_message.image = self._bridge.cv2_to_imgmsg(preprocessed_image.copy(), encoding='bgr8')
        lane_follow_message.state = detected_crosswalk
        self.res_pub.publish(lane_follow_message)
        self.prev_state = detected_crosswalk

        if self.prev_state == 1 and detected_crosswalk == 0:
            return ImageDetectResponse(1)

        return ImageDetectResponse(0)


if __name__ == '__main__':
    # create the node
    node = CrossWalkNode(node_name='crosswalk_detect')
    s = rospy.Service("crosswalk_detect_srv", ImageDetect, node.image_callback)
    rospy.spin()
