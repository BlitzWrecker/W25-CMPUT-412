#!/usr/bin/env python3

import rospy
import cv2
import os
import numpy as np
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Int32
from cv_bridge import CvBridge
from dt_apriltags import Detector
from final.srv import ImageDetect, ImageDetectResponse


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

        self.camera_info_sub = rospy.Subscriber(
            f"/{self._vehicle_name}/camera_node/camera_info", CameraInfo, self.camera_info_callback, queue_size=1
        )

        # Publish augmented image
        self.augmented_img_pub = rospy.Publisher(f"/{self._vehicle_name}/apriltag_processed_image", Image, queue_size=10)

        self.tag_id_pub = rospy.Publisher(f"/{self._vehicle_name}/detected_tag_id", Int32, queue_size=1)

        self.tag_mapping = {48: 0, 50: 1}
        self.prev_tag = None

        rospy.loginfo(f"[{node_name}] Node initialized.")

    def camera_info_callback(self, msg):
        """Callback for camera info to get camera matrix and distortion coefficients."""
        self.camera_matrix = np.array(msg.K).reshape(3, 3)
        self.distortion_coeffs = np.array(msg.D)
        self.camera_info_sub.unregister()  # Unsubscribe after getting the info

    def camera_callback(self, msg):
        """Callback for processing incoming camera images."""
        shutdown, image = msg.shutdown, msg.image
        if shutdown:
            s.shutdown("Shutting down Apriltag detection service.")
            rospy.signal_shutdown('Shutting down Apriltag detection node.')
            return ImageDetectResponse(255)

        cv_image = self.bridge.imgmsg_to_cv2(image)

        # Preprocess the image
        processed_image = self.process_image(cv_image)

        # Detect AprilTags
        tags = self.detect_tag(processed_image)

        # Find the closest tag
        closest_tag = self.find_closest_tag(tags)

        # Update LED color only if a new tag is detected
        if closest_tag:
            tag_id = closest_tag.tag_id

            if tag_id != self.prev_tag:
                try:
                    self.prev_tag = tag_id
                    self.tag_id_pub.publish(tag_id)
                    rospy.loginfo("Published tag_id:")
                    rospy.loginfo(tag_id)
                except:
                    rospy.loginfo("Apriltag not recognized:")
                    rospy.loginfo(tag_id)
            # Draw bounding boxes and tag IDs for the closest tag only

        augmented_image = self.publish_augmented_img(cv_image, closest_tag)
        try:
            self.augmented_img_pub.publish(self.bridge.cv2_to_imgmsg(augmented_image, encoding="bgr8"))
        except Exception as e:
            rospy.logerr(f"Error publishing augmented image: {e}")

        return ImageDetectResponse(closest_tag.tag_id if closest_tag is not None else 255)

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
    s = rospy.Service('apriltag_detection_srv', ImageDetect, node.camera_callback)
    rospy.spin()