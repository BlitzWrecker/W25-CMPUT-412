#!/usr/bin/env python3

# potentially useful for part 2 of exercise 4

# import required libraries
import rospy
import os
import cv2
import numpy as np
from duckietown.dtros import DTROS, NodeType
from ex4.srv import NavigateCMD
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge

class CrossWalkNode(DTROS):

    def __init__(self, node_name):
        super(CrossWalkNode, self).__init__(node_name=node_name, node_type=NodeType.LOCALIZATION)

        # add your code here
        self._vehicle_name = os.environ['VEHICLE_NAME']

        # call navigation control node
        self.naviagte_service = None

        try:
            rospy.wait_for_service('navigate_service', timeout=1)
            self.lane_hehavior_service = rospy.ServiceProxy('navigate_service', NavigateCMD)
        except rospy.ROSException:
            self.naviagte_service = None

        # define other variables as needed
        
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
    
        # Color detection parameters in HSV format
        self.lower_blue = np.array([100, 150, 50])
        self.upper_blue = np.array([140, 255, 255])
    
        # Initialize bridge and subscribe to camera feed
        self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"
        self._bridge = CvBridge()
        self.sub = rospy.Subscriber(self._camera_topic, CompressedImage, self.image_callback)

        # Color detection parameters in HSV format
        self.lower_blue = np.array([100, 150, 50])
        self.upper_blue = np.array([140, 255, 255])

        # Remember the last detected color. We only have to execute a different navigation control when there is a color
        # change
        self.last_color = None

        # Set a distance threshhold for detecting lines so we don't detect lines that are too far away
        self.dist_thresh = 0.1  # 10 cm

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

    def detect_line(self, image, masks):
        colors = {"blue": (255, 0, 0)}
    
        for color_name, mask in masks.items():
            masked_color = cv2.bitwise_and(image, image, mask=mask)
            gray = cv2.cvtColor(masked_color, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contour_dists = []
    
            for contour in contours:
                if cv2.contourArea(contour) > 200:  # Filter small contours
                    x, y, w, h = cv2.boundingRect(contour)
                    box_rep = self.extrinsic_transform(x + w //2, y + h)

                    # Estimate the distance of the line from the robot using the distance of the line from the bottom of the screen
                    dist = self.calculate_dist(box_rep, np.array([x + w //2], image.shape[0]))

                    if (dist <= self.dist_thresh):
                        contour_dists.append((dist, x, y, w, h))


            coutour_dists = sorted(contour_dists, key=lambda x: x[0])

        if len(contour_dists) != 2:
            return image, False

        for dist, x, y, w, h in coutour_dists:
            cv2.rectangle(image, (x, y), (x + w, y + h), colors[color_name], 2)

        dist, x, y, _, h = coutour_dists[0]
        cv2.putText(image, f"Dist: {dist*30:.2f} cm", (x, y + h + 10), cv2.FONT_HERSHEY_PLAIN, 1, colors[color_name])
        return image, True

    def detect_ducks(self, **kwargs):
        pass

    def image_callback(self, msg):
        # Convert compressed image to CV2
        image = self._bridge.compressed_imgmsg_to_cv2(msg)
    
        # Undistort image
        undistorted_image = self.undistort_image(image)
    
        # Preprocess image
        preprocessed_image = self.preprocess_image(undistorted_image)
    
        # Detect lanes and colors
        masks = self.detect_lane_color(preprocessed_image)
        lane_detected_image, detected_crosswalk = self.detect_line(preprocessed_image.copy(), masks)
    
        # Publish processed image (optional)
        self.pub.publish(self._bridge.cv2_to_imgmsg(lane_detected_image, encoding="bgr8"))


if __name__ == '__main__':
    # create the node
    node = CrossWalkNode(node_name='april_tag_detector')
    rospy.spin()
