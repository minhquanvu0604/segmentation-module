// segmentation_node.hpp

#ifndef SEGMENTATION_MODULE_ROS_SEGMENTATION_NODE_HPP
#define SEGMENTATION_MODULE_ROS_SEGMENTATION_NODE_HPP

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

// Include the model inference header
#include <model_inference/model_inference.hpp> // Adjust the path based on your package structure

#include <memory>    // For std::shared_ptr
#include <vector>    // For std::vector

namespace segmentation_module_ros
{

class SegmentationNode
{
public:
    /**
     * @brief Constructor for SegmentationNode.
     * 
     * Initializes the ROS subscriber and publisher, and sets up the model inference module.
     * 
     * @param nh ROS NodeHandle
     */
    SegmentationNode(ros::NodeHandle& nh);

    /**
     * @brief Destructor for SegmentationNode.
     * 
     * Cleans up resources if necessary.
     */
    ~SegmentationNode();

private:
    /**
     * @brief Callback function for incoming point cloud messages.
     * 
     * Processes the point cloud, applies the detection mask, and publishes the filtered point cloud.
     * 
     * @param cloud_msg Pointer to the incoming PointCloud2 message
     */
    void pointCloudCallback(const sensor_msgs::PointCloud2ConstPtr& cloud_msg);

    ros::Subscriber pointcloud_sub_;                       ///< Subscriber for input point cloud
    ros::Publisher filtered_pointcloud_pub_;               ///< Publisher for filtered point cloud
    std::shared_ptr<ModelInference> model_inference_;      ///< Shared pointer to the model inference module
};

} // namespace segmentation_module_ros

#endif // SEGMENTATION_MODULE_ROS_SEGMENTATION_NODE_HPP
