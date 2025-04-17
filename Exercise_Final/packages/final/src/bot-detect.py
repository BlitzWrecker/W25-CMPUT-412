#!/usr/bin/env python3

import rospy
import os
import cv2
import math
import numpy as np
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from final.srv import ImageDetect, ImageDetectResponse


class BotDetectNode(DTROS):
    def __init__(self, node_name):
        super(BotDetectNode, self).__init__(node_name=node_name, node_type=NodeType.LOCALIZATION)

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
        self.lower_light_blue = np.array([105, 150, 50])
        self.upper_light_blue = np.array([108, 255, 255])
        self.lower_blue = np.array([115, 150, 50])
        self.upper_blue = np.array([125, 255, 255])

        # Set a distance threshhold for detecting lines so we don't detect lines that are too far away
        self.dist_thresh = 5

        # Initialize bridge
        self._bridge = CvBridge()

        self.img_pub = rospy.Publisher(f"/{self._vehicle_name}/bot_detection_processed_image", Image, queue_size=10)

    def detect_lane_color(self, image):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        masks = {
            "blue": cv2.inRange(hsv_image, self.lower_blue, self.upper_blue),
            "light_blue": cv2.inRange(hsv_image, self.lower_light_blue, self.upper_light_blue),
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

    def detect_broken_bot(self, image, masks):
        colors = {"blue": (255, 0, 0), "light_blue": (135, 206, 235)}
        detected = {"blue": False, "light_blue": False}

        for color_name, mask in masks.items():
            masked_color = cv2.bitwise_and(image, image, mask=mask)
            gray = cv2.cvtColor(masked_color, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                area = cv2.contourArea(contour)
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h

                box_rep = self.extrinsic_transform(x + w // 2, y + h)
                screen_bot = self.extrinsic_transform(x + w // 2, image.shape[0])

                # Estimate the distance of the line from the robot using the distance of the line from the bottom of the screen
                dist = self.calculate_dist(box_rep, screen_bot)

                if (color_name == 'blue' and area > 400 and aspect_ratio < 5 and dist <= self.dist_thresh) \
                            or (color_name == 'light_blue' and area > 50 and dist <= self.dist_thresh):  # Filter small contours
                    detected[color_name] = True
                    cv2.rectangle(image, (x, y), (x + w, y + h), colors[color_name], 2)
                    cv2.putText(image, f"Dist: {dist * 30:.2f} cm", (x, y + h + 10), cv2.FONT_HERSHEY_PLAIN, 1,
                                colors[color_name])

        detected_bot = detected["blue"] and detected['light_blue']
        rospy.loginfo(f"Potentially broken bot: {detected_bot}")
        return image, detected_bot

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

        masks = self.detect_lane_color(cropped_image)
        processed_image, is_detected = self.detect_broken_bot(cropped_image.copy(), masks)

        self.img_pub.publish(self._bridge.cv2_to_imgmsg(processed_image, "bgr8"))

        return ImageDetectResponse(1 if is_detected else 0)


if __name__ == '__main__':
    # create the node
    node = BotDetectNode(node_name='bot_detect')
    s = rospy.Service("bot_detect_srv", ImageDetect, node.image_callback)
    rospy.spin()
