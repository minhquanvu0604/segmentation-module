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
    // Callback function for image topic
    void imageCallback(const sensor_msgs::ImageConstPtr& img_msg);

    // Callback function for point cloud topic
    /*
    * @brief Callback function for the point cloud topic
    * 
    * Filters the point cloud based on the segmentation mask obtained from the image inference.
    * Assumes every pixel in the image is corresponded to a point in the point cloud
    * The image size is 480x640 and the point cloud has 307200 (480x640) points
    */
    void pointCloudCallback(const sensor_msgs::PointCloud2ConstPtr& cloud_msg);
    
private:
    ros::Subscriber image_sub_;
    ros::Subscriber pointcloud_sub_;
    ros::Publisher filtered_cloud_pub_;
    ros::Publisher filtered_image_pub_;
    ModelInference model_inference_;  // The object that handles image inference
    cv::Mat current_image_;  // Stores the current image

    cv::Mat segmentation_mask_;  // Stores the segmentation mask
    double mask_threshold_ = 0.5;  // Threshold for the segmentation mask

    bool image_received_ = false;
    long long image_num_ = 0;
    long long pointcloud_num_ = 0;

};

#endif // POINTCLOUD_FILTER_CORE_H
