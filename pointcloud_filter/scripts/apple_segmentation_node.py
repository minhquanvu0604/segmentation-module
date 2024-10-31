# PYTHON VERSION, UNDONE

import os, sys
top_level_package = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')) # segmentation_module
sys.path.insert(0, top_level_package)

import rospy
from sensor_msgs.msg import PointCloud2, Image
from sensor_msgs import point_cloud2
from cv_bridge import CvBridge, CvBridgeError
import cv2
from segmentation_module.model_inference.scripts.model_inference import RealTimeInference
import numpy as np


class ROSRealTimeInference:
    def __init__(self, config):
        # Initialize the ROS node
        rospy.init_node('real_time_inference_node')

        # Initialize the inference engine
        model_path = config['model_path']
        input_size = config['input_size']
        num_classes = config['num_classes']
        
        self.inference_ = RealTimeInference(model_path, input_size, num_classes)

        # For converting ROS image messages
        self.bridge_ = CvBridge()

        self.pointcloud_sub = rospy.Subscriber('/points', PointCloud2, self.pointcloud_callback)
        self.image_sub = rospy.Subscriber('/camera/rgb/image_raw', Image, self.image_callback)

        self.filtered_pc_pub = rospy.Publisher('/inference/filtered_points', PointCloud2, queue_size=10)

        rospy.loginfo("ROS Real-Time Inference node initialized.")

        self.current_mask = None

    def image_callback(self, msg):
        """
        Callback function that processes incoming images.
        Generates a mask using the inference model.
        """
        try:
            # Convert the ROS image message to a PIL image
            cv_image = self.bridge_.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            pil_image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))

            # Perform inference on the image and update the current mask
            self.current_mask = self.inference_.infer_single_image(pil_image)
            rospy.loginfo("Mask updated from image.")

        except CvBridgeError as e:
            rospy.logerr(f"Error converting ROS Image to OpenCV: {e}")
        except Exception as e:
            rospy.logerr(f"Error during inference: {e}")

    def pointcloud_callback(self, msg):
        """
        Callback function that processes incoming point clouds.
        Filters the point cloud based on the current mask.
        """
        if self.current_mask is None:
            rospy.logwarn("No mask available yet, skipping point cloud processing.")
            return

        # Convert PointCloud2 message to numpy array
        pc_array = self.pointcloud2_to_array(msg)
        if pc_array is None:
            rospy.logwarn("Failed to convert point cloud, skipping.")
            return

        # Filter point cloud using the mask
        filtered_pc = self.filter_pointcloud_by_mask(pc_array, self.current_mask)

        # Publish the filtered point cloud
        filtered_pc_msg = self.array_to_pointcloud2(filtered_pc, msg.header)
        self.filtered_pc_pub.publish(filtered_pc_msg)
        rospy.loginfo("Filtered point cloud published.")

    def filter_pointcloud_by_mask(self, pc_array, mask):
        """
        Filter the point cloud based on the mask. Points corresponding to 0 in the mask are removed.
        """
        # Ensure mask dimensions match the point cloud's projection
        if pc_array.shape[0] != mask.shape[0] or pc_array.shape[1] != mask.shape[1]:
            rospy.logwarn("Point cloud and mask dimensions do not match. Skipping filtering.")
            return pc_array

        # Apply the mask: retain only points where the mask is positive
        mask_flat = mask.flatten()  # Assuming mask is binary or probability
        filtered_pc = pc_array[mask_flat > 0]  # Filter points

        return filtered_pc

    def pointcloud2_to_array(self, msg):
        """
        Convert a PointCloud2 message to a numpy array.
        """
        try:
            pc = np.array(list(point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)))
            return pc
        except Exception as e:
            rospy.logerr(f"Error converting PointCloud2 to array: {e}")
            return None

    def array_to_pointcloud2(self, points_array, header):
        """
        Convert a numpy array back to a PointCloud2 message.
        """
        try:
            # Create PointCloud2 message from numpy array
            cloud_msg = point_cloud2.create_cloud_xyz32(header, points_array.tolist())
            return cloud_msg
        except Exception as e:
            rospy.logerr(f"Error converting numpy array to PointCloud2: {e}")
            return None


if __name__ == '__main__':

    try:
        # Config for the ROSRealTimeInference node
        config = {
            'model_path': '/home/quanvu/git/segmentation-module/deeplabv3_apples/output/2024_10_02_23_37_35/model.pth',
            'input_size': (512, 512),
            'num_classes': 2
        }

        # Run the ROS inference node
        node = ROSRealTimeInference(config)
        rospy.spin()

    except rospy.ROSInterruptException:
        pass