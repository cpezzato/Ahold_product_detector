#!/usr/bin/env python3
import rospy
from ahold_product_detection.msg import Detection, ProductPose, ProductPoseArray
import tf
from tf.transformations import quaternion_from_euler, euler_from_quaternion
from geometry_msgs.msg import PoseStamped, Point, Quaternion, Pose, Vector3
import numpy as np
from cv_bridge import CvBridge
from copy import copy
import tf
from tf.transformations import quaternion_from_euler, euler_from_quaternion
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from sensor_msgs.msg import CameraInfo
from rotation_compensation import rotate_image, rotate_bounding_boxes
from jsk_recognition_msgs.msg import BoundingBoxArray, BoundingBox
from orientation_estimator import OrientationEstimator

# Helpers
def _it(self):
    yield self.x
    yield self.y
    yield self.z


Point.__iter__ = _it


def _it(self):
    yield self.x
    yield self.y
    yield self.z
    yield self.w


Quaternion.__iter__ = _it


class DetectorData:
    def __init__(self) -> None:
        self.subscriber = rospy.Subscriber(
            "/detection_results", Detection, self.callback
        )
        self.data = None
        self.bridge = CvBridge()
        self.previous_stamp = rospy.Time.now()

    def callback(self, data):
        self.data = data


product_mapping  = {
0: 8710400280507,
1: 8718907400718,
2: 8718907457583,
3: 8718907400701,
4: 8718906697560,
5: 8718907056274,
6: 8710400035862,
7: 8710400687122,
8: 8718907056311,
9: 8710400145981,
10: 8718907306744,
11: 8718907306775,
12: 8718907306737,
13: 8718907306751,
14: 8718907056298,
15: 8710400011668,
16: 3083681149630,
17: 3083681025484,
18: 3083681068146,
19: 3083681126471,
20: 3083681068122,
21: 8712800147008,
22: 8712800147770,
23: 8714100795699,
24: 8714100795774,
25: 8720600612848,
26: 8720600609893,
27: 8720600606731,
28: 8717662264382,
29: 8717662264368,
30: 87343267,
31: 8710400514107,
32: 8718906872844,
33: 8718907039987,
34: 8710400416395,
35: 8718907039963,
36: 5414359921711,
37: 8718906948631,
38: 8718265082151,
39: 8718906536265,
40: 8718951065826,
41: 3574661734712,
42: 8006540896778,
43: 8720181388774,
44: 90453656,
45: 90453533,
46: 5410013114697,
47: 5410013136149,
48: 80042556,
49: 80042563,
50: 8005110170324,
51: 8001250310415,
52: 8004690052044,
53: 8718906124066,
54: 8718906124073,
55: 9999,
}

class PoseEstimator:
    def __init__(self) -> None:
        self.detection = DetectorData()
        self.rate = rospy.Rate(30)
        self.tf_listener = tf.TransformListener()
        self.pub_results = rospy.Publisher(
            "/pose_estimation_results", ProductPoseArray, queue_size=10
        )
        self.pub_detections = rospy.Publisher(
            "/3d_detections", BoundingBoxArray, queue_size=10
        )

        self.intrinsics_subscriber = rospy.Subscriber(
            "/camera/color/camera_info", CameraInfo, self.intrinsics_callback
        )
        rospy.sleep(1.0)

    def intrinsics_callback(self, data):
        self.intrinsics = np.array(data.K).reshape((3, 3))

    def transform_poses(self, product_poses):
        transformed_poses = ProductPoseArray()
        transformed_poses.header = product_poses.header
        for pose in product_poses.poses:
            ros_pose = PoseStamped()
            ros_pose.header.stamp = product_poses.header.stamp
            ros_pose.header.frame_id = "camera_color_optical_frame"
            ros_pose.pose.position = Point(x=pose.x, y=pose.y, z=pose.z)
            ros_pose.pose.orientation = Quaternion(
                *quaternion_from_euler(pose.theta, pose.phi, pose.psi)
            )
            try:
                ros_pose_transformed = self.tf_listener.transformPose(
                    "base_link", ros_pose
                )
            except Exception as e:
                rospy.logerr("couldn't transform correctly ", e)

            new_pose = ProductPose()
            new_pose.x = ros_pose_transformed.pose.position.x
            new_pose.y = ros_pose_transformed.pose.position.y
            new_pose.z = ros_pose_transformed.pose.position.z
            new_pose.theta, pose.phi, pose.psi = euler_from_quaternion(
                list(ros_pose_transformed.pose.orientation)
            )
            new_pose.header = pose.header
            new_pose.header.frame_id = "base_link"
            new_pose.label = pose.label
            new_pose.score = pose.score
            transformed_poses.poses.append(new_pose)

        return transformed_poses

    def estimate_bbox_depth(self, boxes, depth_image):
        rotated_depth_image = rotate_image(image=depth_image, angle=boxes[0].angle)
        boxes_xywh = np.array([[box.x, box.y, box.w, box.h] for box in boxes])
        rotated_bounding_boxes = rotate_bounding_boxes(
            boxes_xywh, depth_image, angle=boxes[0].angle
        )

        estimated_z = []
        for x, y, w, h in rotated_bounding_boxes:
            depth_path = rotated_depth_image[
                int(y - (w / 2)) : int(y + (w / 2)), int(x - (h / 2)) : int(x + (h / 2))
            ]
            estimated_z.append(np.median(depth_path[depth_path != 0])/1000)

        return estimated_z

    def estimate_pose_bounding_box(self, z, bbox):
        pixel_vector = np.array([bbox.x, bbox.y, 1])
        scaled_xyz_vector = np.linalg.inv(self.intrinsics) @ pixel_vector.T
        bboxxyxy = [int(bbox.x - bbox.w/2), int(bbox.y - bbox.h/2), int(bbox.x + bbox.w/2), int(bbox.y + bbox.h/2)]
        orientation = self.orientation_estimation.get_object_orientation(bboxxyxy)
        
        return list(z * scaled_xyz_vector) + orientation

    def run(self):
        # read data when available
        try:
            
            depth_image = self.detection.bridge.imgmsg_to_cv2(
                self.detection.data.depth_image, desired_encoding="passthrough"
            )
            stamp = copy(self.detection.data.header.stamp)
            boxes = self.detection.data.detections

            
            if self.detection.previous_stamp.to_sec() == stamp.to_sec():
                raise ValueError("New data has not been received... stopping pose estimation")
            self.detection.previous_stamp = stamp

        except Exception as e:
            return

        if len(boxes) == 0:
            return

        # Convert to 3D
        product_poses = ProductPoseArray()
        product_poses.header.stamp = stamp
        self.orientation_estimation = OrientationEstimator(depth_image=depth_image, camera_intrinsics=self.intrinsics)

        boxes_estimated_z = self.estimate_bbox_depth(boxes, depth_image)
        
        for z, bbox in zip(boxes_estimated_z, boxes):
            if not np.isnan(z):
                # depth exists and non-zero
                xyz_detection = self.estimate_pose_bounding_box(z, bbox)
                product_pose = ProductPose()

                (
                    product_pose.x,
                    product_pose.y,
                    product_pose.z,
                    product_pose.theta,
                    product_pose.phi,
                    product_pose.psi,
                ) = xyz_detection
                product_pose.score = bbox.score
                # Change label to barcode
                product_pose.label = product_mapping[bbox.label]

                product_poses.poses.append(product_pose)

        # Transform to non-moving frame
        transformed_product_poses = self.transform_poses(product_poses)
        # Create jsk boundingboxes
        detections = BoundingBoxArray()
        detections.header.frame_id = "base_link"
        for product_pose in transformed_product_poses.poses:
            detection = BoundingBox()
            detection.header.frame_id = "base_link"
            detection.pose = Pose(
                position=Point(x=product_pose.x, y=product_pose.y, z=product_pose.z),
                orientation=Quaternion(
                    *quaternion_from_euler(
                        product_pose.theta, product_pose.phi, product_pose.psi
                    )
                )
            )
            detection.dimensions = Vector3(x=0.05, y=0.05, z=0.05)
            detection.value = product_pose.score
            detection.label = product_pose.label
            detections.boxes.append(detection)


        self.pub_results.publish(transformed_product_poses)
        self.pub_detections.publish(detections)


import time

if __name__ == "__main__":
    rospy.init_node("product_pose_estimator")
    pose_estimator = PoseEstimator()

    t0 = time.time()
    while not rospy.is_shutdown():
        pose_estimator.run()
        pose_estimator.rate.sleep()
        # print(f"product pose estimation rate: {1/(time.time() - t0)}")
        t0 = time.time()
