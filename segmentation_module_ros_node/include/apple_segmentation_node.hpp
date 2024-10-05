#ifndef SEGMENTATION_MODULE_ROS_SEGMENTATIONNODE_HPP
#define SEGMENTATION_MODULE_ROS_SEGMENTATIONNODE_HPP

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>

// Include the model inference module
#include <model_inference/model_inference.hpp>

class SegmentationNode
{
public:
    SegmentationNode(ros::NodeHandle& nh);
    ~SegmentationNode();

private:
    void pointCloudCallback(const sensor_msgs::PointCloud2ConstPtr& cloud_msg);

    ros::NodeHandle nh_;
    ros::Subscriber pointcloud_sub_;
    ros::Publisher filtered_pointcloud_pub_;

    // Inference model
    model_inference::ModelInference model_inference_;

    // Threshold for filtering (e.g., 50%)
    float threshold_;
};

#endif // SEGMENTATION_MODULE_ROS_SEGMENTATIONNODE_HPP
