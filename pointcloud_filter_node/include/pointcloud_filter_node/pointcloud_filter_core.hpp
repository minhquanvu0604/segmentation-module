#ifndef POINTCLOUD_FILTER_CORE_H
#define POINTCLOUD_FILTER_CORE_H

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <cv_bridge/cv_bridge.h>

#include "predict.hpp"  // Include the ModelInference class for inference

class PointCloudFilterCore {
public:
    // Constructor with ROS node handles and model path
    PointCloudFilterCore(ros::NodeHandle& nh, ros::NodeHandle& private_nh, const std::string& model_path, const cv::Size& input_size);

private:
    ros::Subscriber image_sub_;
    ros::Subscriber pointcloud_sub_;
    ros::Publisher filtered_cloud_pub_;
    ModelInference model_inference_;  // The object that handles image inference
    cv::Mat current_image_;  // Stores the current image
    cv::Mat segmentation_mask_;  // Stores the segmentation mask
    bool image_received_ = false;

    // Camera intrinsic parameters
    double fx_, fy_, cx_, cy_;

    // Callback function for image topic
    void imageCallback(const sensor_msgs::ImageConstPtr& img_msg);

    // Callback function for point cloud topic
    void pointCloudCallback(const sensor_msgs::PointCloud2ConstPtr& cloud_msg);
};

#endif // POINTCLOUD_FILTER_CORE_H
