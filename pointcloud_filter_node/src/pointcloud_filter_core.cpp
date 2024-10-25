#include "pointcloud_filter_node/pointcloud_filter_core.hpp"

// Constructor implementation
PointCloudFilterCore::PointCloudFilterCore(ros::NodeHandle& nh, ros::NodeHandle& private_nh, const std::string& model_path, const cv::Size& input_size)
    : model_inference_(model_path, input_size) {
    // Initialize the subscribers and publishers
    image_sub_ = nh.subscribe("/camera/color/image_raw", 1, &PointCloudFilterCore::imageCallback, this);
    pointcloud_sub_ = nh.subscribe("/camera/depth_registered/points", 1, &PointCloudFilterCore::pointCloudCallback, this);
    
    filtered_image_pub_ = nh.advertise<sensor_msgs::Image>("/camera/color/filtered_image", 1);
    filtered_cloud_pub_ = nh.advertise<sensor_msgs::PointCloud2>("/filtered_points", 1);
}

// Image callback function implementation
void PointCloudFilterCore::imageCallback(const sensor_msgs::ImageConstPtr& img_msg) {

    // image_num_++;
    // std::cout << "Image " << image_num_ << " received" << std::endl;
    // std::cout << "Image dimensions: " << current_image_.rows << "x" << current_image_.cols << std::endl;

    try {
        cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::BGR8);
        current_image_ = cv_ptr->image;
        image_received_ = true;

    } catch (cv_bridge::Exception& e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }
}

// Point cloud callback function implementation
void PointCloudFilterCore::pointCloudCallback(const sensor_msgs::PointCloud2ConstPtr& cloud_msg) {

    pointcloud_num_++;
    std::cout << "Pointcloud " << pointcloud_num_ << " received" << std::endl;
    // ROS_INFO("Point cloud received");

    if (!image_received_) {
        ROS_WARN("No image received yet. Waiting...");
        return;
    }

    // Convert ROS point cloud to PCL point cloud
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::fromROSMsg(*cloud_msg, *cloud);

    // std::cout << "Point cloud size: " << cloud->points.size() << std::endl;




    // Range mask for Z-axis filtering --------------------------------------------
    // Define the range for filtering (e.g., minimum and maximum Z-axis distance)
    float min_z = 0.2;  // Minimum Z distance in meters
    float max_z = 2.0;  // Maximum Z distance in meters

    // Create a binary mask (CV_8UC1) to indicate whether a pixel is within the Z-axis range
    cv::Mat range_mask = cv::Mat::zeros(current_image_.size(), CV_8UC1);
    // Loop over the points in the point cloud
    for (size_t i = 0; i < cloud->points.size(); ++i) {
        const auto& point = cloud->points[i];
        // std::cout << "Point " << i << ": " << point.x << ", " << point.y << ", " << point.z << std::endl;
        // Check if the Z value is within the specified range
        
        // if (!std::isnan(point.z) && point.z >= min_z && point.z <= max_z) {
        if (!std::isnan(point.z) && point.z >= min_z && point.z <= max_z) {

            // Calculate the corresponding row and column in the image
            int row = i / current_image_.cols;
            int col = i % current_image_.cols;

            // Mark the corresponding pixel in the range mask
            range_mask.at<uchar>(row, col) = 255;  // Set the pixel to white (within Z range)
        }
    }
    // Apply the mask to filter the image
    current_image_.setTo(cv::Scalar(0, 0, 0), range_mask == 0);



    // -----------------------------------------------------------------------------
    // Perform inference and get the segmentation mask
    segmentation_mask_ = model_inference_.infer_single_image(current_image_);

    // 1
    // // Filter the image to retain only the object pixels
    // // Create a black image of the same size as the original
    // cv::Mat filtered_image = cv::Mat::zeros(current_image_.size(), current_image_.type());

    // // Filter the image based on the segmentation mask
    // int detected_pixels = 0;
    // for (int v = 0; v < segmentation_mask_.rows; ++v) {
    //     for (int u = 0; u < segmentation_mask_.cols; ++u) {
    //         float mask_value = segmentation_mask_.at<float>(v, u);
    //         if (mask_value > mask_threshold_) {  // Keep only the pixels where the mask indicates the object
    //             filtered_image.at<cv::Vec3b>(v, u) = current_image_.at<cv::Vec3b>(v, u);
    //             detected_pixels++;
    //         }
    //     }
    // }

    // 2
    // // Create an overlay mask (colored) to visualize the detected object
    // cv::Mat overlay_image = current_image_.clone();
    // cv::Vec3b overlay_color = cv::Vec3b(0, 255, 0);  // Green color for the overlay

    // int detected_pixels = 0;
    // for (int v = 0; v < segmentation_mask_.rows; ++v) {
    //     for (int u = 0; u < segmentation_mask_.cols; ++u) {
    //         float mask_value = segmentation_mask_.at<float>(v, u);
    //         if (mask_value > mask_threshold_) {  // Mark detected object pixels
    //             overlay_image.at<cv::Vec3b>(v, u) = overlay_color;
    //             detected_pixels++;
    //         }
    //     }
    // }
    // // Blend the overlay with the original image
    // double alpha = 0.5;  // Transparency factor (0.5 for 50% transparency)
    // cv::addWeighted(overlay_image, alpha, current_image_, 1.0 - alpha, 0, overlay_image);

    // 3
    // Create a binary mask for detected regions
    cv::Mat detection_mask = cv::Mat::zeros(segmentation_mask_.size(), CV_8UC1);

    int detected_pixels = 0;
    for (int v = 0; v < segmentation_mask_.rows; ++v) {
        for (int u = 0; u < segmentation_mask_.cols; ++u) {
            float mask_value = segmentation_mask_.at<float>(v, u);
            if (mask_value > mask_threshold_) {  // Mark detected object pixels
                detection_mask.at<uchar>(v, u) = 255;  // Set to white
                detected_pixels++;
            }
        }
    }



    // std::cout << "Detected pixels: " << detected_pixels << std::endl;   
    // Publish the filtered image
    // sensor_msgs::ImagePtr filtered_image_msg = cv_bridge::CvImage(cloud_msg->header, sensor_msgs::image_encodings::BGR8, filtered_image).toImageMsg();
    // sensor_msgs::ImagePtr overlay_image_msg = cv_bridge::CvImage(cloud_msg->header, sensor_msgs::image_encodings::BGR8, overlay_image).toImageMsg();
    // filtered_image_pub_.publish(overlay_image_msg);

    sensor_msgs::ImagePtr detection_mask_msg = cv_bridge::CvImage(cloud_msg->header, sensor_msgs::image_encodings::MONO8, detection_mask).toImageMsg();
    filtered_image_pub_.publish(detection_mask_msg);
    // std::cout << "Filtered image " << image_num_ << " published" << std::endl;
    // -----------------------------------------------------------------------------------------------


    // DEBUG
    // sensor_msgs::ImagePtr filtered_image_msg = cv_bridge::CvImage(cloud_msg->header, sensor_msgs::image_encodings::BGR8, current_image_).toImageMsg();
    // filtered_image_pub_.publish(filtered_image_msg);


    // Filter the point cloud based on the segmentation mask --------------------------------
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    filtered_cloud->header = cloud->header;

    // Iterate over the points in the point cloud
    for (int v = 0; v < segmentation_mask_.rows; ++v) {
        for (int u = 0; u < segmentation_mask_.cols; ++u) {
            int index = v * segmentation_mask_.cols + u;
            const auto& point = cloud->points[index];

            // Check if the point passes the segmentation mask threshold
            float mask_value = segmentation_mask_.at<float>(v, u);
            if (mask_value > mask_threshold_) {  // Retain points where the mask indicates the object
                filtered_cloud->points.push_back(point);
            }
        }
    }

    // Publish the filtered point cloud
    sensor_msgs::PointCloud2 output_msg;
    pcl::toROSMsg(*filtered_cloud, output_msg);
    filtered_cloud_pub_.publish(output_msg);

    std::cout << "Filtered point cloud "<< pointcloud_num_ << " published" << std::endl;
}
