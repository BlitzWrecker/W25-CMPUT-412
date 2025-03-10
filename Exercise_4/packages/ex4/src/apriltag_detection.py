#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import BoolStamped
from sensor_msgs.msg import CompressedImage, CameraInfo
from cv_bridge import CvBridge
from dt_apriltags import Detector
from geometry_msgs.msg import Twist
from std_msgs.msg import String

class ApriltagNode(DTROS):

    def __init__(self, node_name):
        super(ApriltagNode, self).__init__(node_name=node_name, node_type=NodeType.CONTROL)

        # Initialize variables
        self.bridge = CvBridge()
        self.camera_matrix = None
        self.distortion_coeffs = None
        self.tag_size = 0.065  # Default tag size in meters
        self.tag_family = 'tag36h11'
        self.detector = Detector(families=self.tag_family, nthreads=1)
        self.last_tag_id = None
        self.stop_time = 0.5  # Default stop time (seconds)
        self.led_color = "white"  # Default LED color

        # Subscribe to camera feed
        self.image_sub = rospy.Subscriber(
            "~image/compressed", CompressedImage, self.camera_callback, queue_size=1
        )
        self.camera_info_sub = rospy.Subscriber(
            "~camera_info", CameraInfo, self.camera_info_callback, queue_size=1
        )

        # Publish augmented image
        self.augmented_img_pub = rospy.Publisher(
            "~augmented_image/compressed", CompressedImage, queue_size=1
        )

        # Publish LED commands
        self.led_pub = rospy.Publisher(
            "~led_command", String, queue_size=1
        )

        # Publish stop commands
        self.stop_pub = rospy.Publisher(
            "~stop_command", Twist, queue_size=1
        )

        rospy.loginfo(f"[{node_name}] Node initialized.")

    def camera_info_callback(self, msg):
        """Callback for camera info to get camera matrix and distortion coefficients."""
        self.camera_matrix = np.array(msg.K).reshape(3, 3)
        self.distortion_coeffs = np.array(msg.D)
        self.camera_info_sub.unregister()  # Unsubscribe after getting the info

    def camera_callback(self, msg):
        """Callback for processing incoming camera images."""
        try:
            # Convert compressed image to OpenCV format
            cv_image = self.bridge.compressed_imgmsg_to_cv2(msg)
        except Exception as e:
            rospy.logerr(f"Error converting image: {e}")
            return

        # Undistort the image
        if self.camera_matrix is not None and self.distortion_coeffs is not None:
            cv_image = cv2.undistort(cv_image, self.camera_matrix, self.distortion_coeffs)

        # Preprocess the image
        processed_image = self.process_image(cv_image)

        # Detect AprilTags
        tags = self.detect_tag(processed_image)

        # Draw bounding boxes and tag IDs
        augmented_image = self.publish_augmented_img(cv_image, tags)

        # Publish augmented image
        try:
            augmented_msg = self.bridge.cv2_to_compressed_imgmsg(augmented_image)
            self.augmented_img_pub.publish(augmented_msg)
        except Exception as e:
            rospy.logerr(f"Error publishing augmented image: {e}")

        # Change LEDs based on detection
        self.publish_leds(tags)

        # Stop the robot based on detection
        self.stop_robot(tags)

    def process_image(self, image):
        """Preprocess the image for AprilTag detection."""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return gray

    def detect_tag(self, image):
        """Detect AprilTags in the image."""
        if self.camera_matrix is None:
            return []

        # Detect tags
        tags = self.detector.detect(
            image,
            estimate_tag_pose=True,
            camera_params=[
                self.camera_matrix[0, 0],  # fx
                self.camera_matrix[1, 1],  # fy
                self.camera_matrix[0, 2],  # cx
                self.camera_matrix[1, 2],  # cy
            ],
            tag_size=self.tag_size,
        )
        return tags

    def publish_augmented_img(self, image, tags):
        """Draw bounding boxes and tag IDs on the image."""
        for tag in tags:
            # Draw bounding box
            for idx in range(len(tag.corners)):
                pt1 = tuple(tag.corners[idx - 1].astype(int))
                pt2 = tuple(tag.corners[idx].astype(int))
                cv2.line(image, pt1, pt2, (0, 255, 0), 2)

            # Draw tag ID
            cv2.putText(
                image,
                str(tag.tag_id),
                org=(int(tag.center[0]), int(tag.center[1])),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.8,
                color=(0, 0, 255),
                thickness=2,
            )
        return image

    def publish_leds(self, tags):
        """Change LEDs based on detected tags."""
        if not tags:
            self.led_color = "white"  # Default color
        else:
            tag_id = tags[0].tag_id
            if tag_id == 0:  # Stop sign
                self.led_color = "red"
            elif tag_id == 1:  # T-Intersection
                self.led_color = "blue"
            elif tag_id == 2:  # UofA Tag
                self.led_color = "green"
            else:
                self.led_color = "white"

        # Publish LED command
        self.led_pub.publish(String(data=self.led_color))

    def stop_robot(self, tags):
        """Stop the robot based on detected tags."""
        if not tags:
            self.stop_time = 0.5  # Default stop time
        else:
            tag_id = tags[0].tag_id
            if tag_id == 0:  # Stop sign
                self.stop_time = 3.0
            elif tag_id == 1:  # T-Intersection
                self.stop_time = 2.0
            elif tag_id == 2:  # UofA Tag
                self.stop_time = 1.0
            else:
                self.stop_time = 0.5

        # Publish stop command
        twist_msg = Twist()
        twist_msg.linear.x = 0.0  # Stop the robot
        self.stop_pub.publish(twist_msg)
        rospy.sleep(self.stop_time)  # Stop for the specified time

if __name__ == '__main__':
    # Create the node
    node = ApriltagNode(node_name='apriltag_detector_node')
    rospy.spin()