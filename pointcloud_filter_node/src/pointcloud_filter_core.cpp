#include "pointcloud_filter_node/pointcloud_filter_core.hpp"

// Constructor implementation
PointCloudFilterCore::PointCloudFilterCore(ros::NodeHandle& nh, ros::NodeHandle& private_nh, const std::string& model_path, const cv::Size& input_size)
    : model_inference_(model_path, input_size) {
    // Initialize the subscribers and publishers
    image_sub_ = nh.subscribe("/camera/rgb/image_raw", 1, &PointCloudFilterCore::imageCallback, this);
    pointcloud_sub_ = nh.subscribe("/camera/depth/points", 1, &PointCloudFilterCore::pointCloudCallback, this);
    filtered_cloud_pub_ = nh.advertise<sensor_msgs::PointCloud2>("/filtered_points", 1);

    // Get camera intrinsic parameters from private parameters
    private_nh.getParam("fx", fx_);
    private_nh.getParam("fy", fy_);
    private_nh.getParam("cx", cx_);
    private_nh.getParam("cy", cy_);
}

// Image callback function implementation
void PointCloudFilterCore::imageCallback(const sensor_msgs::ImageConstPtr& img_msg) {
    try {
        // Convert ROS image to OpenCV format
        cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::BGR8);
        current_image_ = cv_ptr->image;

        // Perform inference and get the segmentation mask
        segmentation_mask_ = model_inference_.infer_single_image(current_image_);

        image_received_ = true;
    } catch (cv_bridge::Exception& e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }
}

// Point cloud callback function implementation
void PointCloudFilterCore::pointCloudCallback(const sensor_msgs::PointCloud2ConstPtr& cloud_msg) {
    if (!image_received_) {
        ROS_WARN("No image received yet. Waiting...");
        return;
    }

    // Convert ROS point cloud to PCL point cloud
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::fromROSMsg(*cloud_msg, *cloud);

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    filtered_cloud->header = cloud->header;

    // Filter the point cloud based on the segmentation mask
    for (const auto& point : cloud->points) {
        // Project 3D point into the 2D image plane using camera intrinsics
        int u = static_cast<int>(fx_ * point.x / point.z + cx_);
        int v = static_cast<int>(fy_ * point.y / point.z + cy_);

        if (u >= 0 && u < segmentation_mask_.cols && v >= 0 && v < segmentation_mask_.rows) {
            float mask_value = segmentation_mask_.at<float>(v, u);
            if (mask_value > 0.5) {  // Filter based on the mask (binary mask: 0 = background, 1 = object)
                filtered_cloud->points.push_back(point);
            }
        }
    }

    // Publish the filtered point cloud
    sensor_msgs::PointCloud2 output_msg;
    pcl::toROSMsg(*filtered_cloud, output_msg);
    filtered_cloud_pub_.publish(output_msg);
}
