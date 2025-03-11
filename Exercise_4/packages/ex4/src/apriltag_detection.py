#!/usr/bin/env python3

import rospy
import cv2
import os
import numpy as np
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import BoolStamped
from sensor_msgs.msg import CompressedImage, CameraInfo, Image
from cv_bridge import CvBridge
from dt_apriltags import Detector
from ex4.srv import MiscCtrlCMD, MiscCtrlCMDResponse
from std_msgs.msg import Int32  # Import the Int32 message type for the tag ID

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

        self._vehicle_name = os.environ['VEHICLE_NAME']
        self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"

        # Subscribe to camera feed
        self.sub = rospy.Subscriber(self._camera_topic, CompressedImage, self.camera_callback)

        self.camera_info_sub = rospy.Subscriber(
            f"/{self._vehicle_name}/camera_node/camera_info", CameraInfo, self.camera_info_callback, queue_size=1
        )

        # Publish augmented image
        self.augmented_img_pub = rospy.Publisher(f"/{self._vehicle_name}/apriltag_processed_image", Image, queue_size=10)

        # Publish tag ID to a custom topic
        self.tag_id_pub = rospy.Publisher(f"/{self._vehicle_name}/detected_tag_id", Int32, queue_size=1)

        # Set the new camera framerate to 3
        self.new_framerate = 3
        rospy.wait_for_service("misc_ctrl_srv", timeout=1)
        self.misc_ctrl_srv = rospy.ServiceProxy("misc_ctrl_srv", MiscCtrlCMD)
        self.misc_ctrl_srv("set_fr", self.new_framerate)

        self.tag_mapping = {21: 0, 133: 1, 94: 2, -1: 3}
        self.prev_tag = None

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

        # Find the closest tag
        closest_tag = self.find_closest_tag(tags)

        # Publish the tag ID to the custom topic
        if closest_tag is not None:
            tag_id_msg = Int32()
            tag_id_msg.data = closest_tag.tag_id
            self.tag_id_pub.publish(tag_id_msg)
        else:
            # Publish -1 if no tag is detected
            tag_id_msg = Int32()
            tag_id_msg.data = -1
            self.tag_id_pub.publish(tag_id_msg)

        # Publish to LED
        if closest_tag is None:
            tag_id = -1
        else:
            tag_id = closest_tag.tag_id

        if tag_id != self.prev_tag:
            self.misc_ctrl_srv("set_led", self.tag_mapping[tag_id])
            self.prev_tag = tag_id

        # Draw bounding boxes and tag IDs for the closest tag only
        augmented_image = self.publish_augmented_img(cv_image, closest_tag)

        # Publish augmented image
        try:
            self.augmented_img_pub.publish(self.bridge.cv2_to_imgmsg(augmented_image, encoding="bgr8"))
        except Exception as e:
            rospy.logerr(f"Error publishing augmented image: {e}")

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

    def find_closest_tag(self, tags):
        """Find the closest AprilTag based on the translation vector."""
        if not tags:
            return None

        # Calculate the distance of each tag from the camera
        closest_tag = None
        min_distance = float('inf')

        for tag in tags:
            # Calculate the Euclidean distance from the translation vector
            distance = np.linalg.norm(tag.pose_t)
            if distance < min_distance:
                min_distance = distance
                closest_tag = tag
        return closest_tag

    def publish_augmented_img(self, image, tag):
        if tag:
            """Draw bounding boxes and tag IDs on the image."""
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
            rospy.loginfo(tag.tag_id)
        return image

if __name__ == '__main__':
    # Create the node
    node = ApriltagNode(node_name='apriltag_detector_node')
    rospy.spin()