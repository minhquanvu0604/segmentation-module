#include <thread>
#include <yaml-cpp/yaml.h>
#include <ros/package.h>
#include <pluginlib/class_list_macros.h>

#include "pointcloud_filter/pointcloud_filter_core.hpp"

PLUGINLIB_EXPORT_CLASS(segmentation_module::PointCloudFilterCore, nodelet::Nodelet)


void saveFlattenedPointCloud(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& pointcloud, 
                             const cv::Mat& original_image, 
                             const cv::Size& image_size, 
                             const std::string& base_path);


namespace segmentation_module {

    PointCloudFilterCore::PointCloudFilterCore(ros::NodeHandle& nh, ros::NodeHandle& private_nh, const std::string& model_path, const cv::Size& input_size) {
        init(nh, private_nh, model_path, input_size);
    }

    void PointCloudFilterCore::onInit(){
        std::string package_path = ros::package::getPath("pointcloud_filter");
        std::string yaml_file = package_path + "/config/pointcloud_filter.yaml";
        YAML::Node config = YAML::LoadFile(yaml_file);

        std::string model_path = config["model_path"].as<std::string>();
        int input_width = config["input_size"]["width"].as<int>();
        int input_height = config["input_size"]["height"].as<int>();
        cv::Size input_size(input_width, input_height);

        init(getNodeHandle(), getPrivateNodeHandle(), model_path, input_size);
    }

    // Constructor implementation
    void PointCloudFilterCore::init(ros::NodeHandle& nh, ros::NodeHandle& private_nh, 
                                    const std::string& model_path, const cv::Size& input_size) {
        // Load the model for inference
        model_inference_ = std::make_unique<ModelInference>(model_path, input_size);

        // Allocate memory 
        current_pointcloud_ = pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>);
        data_received_.store(false);


        // Initialize the subscribers and publishers
        // image_sub_ = nh.subscribe("/camera/color/image_raw", 1, &PointCloudFilterCore::imageCallback, this);
        // pointcloud_sub_ = nh.subscribe("/camera/depth_registered/points", 1, &PointCloudFilterCore::pointCloudCallback, this);

        // Use time synchronizer to synchronize the image and point cloud callbacks
        image_sub_.subscribe(nh, "/camera/color/image_raw", 1);
        pointcloud_sub_.subscribe(nh, "/camera/depth_registered/points", 1);
        
        // Approximate time synchronizer with a queue size of 10
        sync_ = std::make_shared<Sync>(MySyncPolicy(10), image_sub_, pointcloud_sub_);
        sync_->registerCallback(std::bind(&PointCloudFilterCore::callback, this, std::placeholders::_1, std::placeholders::_2));

        filtered_image_pub_ = nh.advertise<sensor_msgs::Image>("/camera/color/filtered_image", 1);
        filtered_cloud_pub_ = nh.advertise<sensor_msgs::PointCloud2>("/filtered_points", 1);

        filtered_pointcloud_srv_ = nh.advertiseService("/mvps/camera_module/filtered_pointcloud", &PointCloudFilterCore::filterPointCloudService, this);
    }

    void PointCloudFilterCore::callback(const sensor_msgs::ImageConstPtr& image_msg,
                    const sensor_msgs::PointCloud2ConstPtr& pointcloud_msg) {
        std::unique_lock<std::mutex> lock(input_mutex_);
        try {
            // Convert ROS image message to OpenCV image
            cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::BGR8);

            current_image_ = cv_ptr->image;
            current_image_header_ = image_msg->header;
            data_received_.store(true); // Mark data as received

        } catch (cv_bridge::Exception& e) {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }
        pcl::fromROSMsg(*pointcloud_msg, *current_pointcloud_);
    }

    bool PointCloudFilterCore::filterPointCloudService(hydra_utils::CloudService::Request &req, hydra_utils::CloudService::Response &res) {
        ROS_INFO("Received service request to filter the point cloud");
        
        if (!data_received_.load()) {
            ROS_WARN("No data received yet. Waiting...");
            return false;
        }

        std::unique_lock<std::mutex> lock(input_mutex_);
        // TODO Investigate copy or move or pointer would be better
        auto current_image = current_image_.clone(); ;
        auto current_image_header = current_image_header_;
        auto current_pointcloud = pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>());
        pcl::copyPointCloud(*current_pointcloud_, *current_pointcloud);
        lock.unlock();


        std::cout << "Inferring image..." << std::endl;
        const cv::Mat probability_map = model_inference_->infer_single_image(current_image);
        CV_Assert(probability_map.type() == CV_32F);


        // // Print the mask
        // const float tolerance = 0.5;
        // int d = 0, p = 0;
        // std::cout << "Segmentation mask (probabilities):\n";
        // for (int v = 0; v < probability_map.rows; ++v) {
        //     for (int u = 0; u < probability_map.cols; ++u) {
        //         p++;
        //         float pixel_value = probability_map.at<float>(v, u);
        //         if (std::abs(pixel_value) < tolerance) {
        //                 // std::cout << "0 ";  // Print 0 if the value is close to zero
        //             } else {
        //                 // std::cout << pixel_value << " ";
        //                 d++;
        //             }
        //     }
        //     std::cout << std::endl;
        // }
        // std::cout << "Detected pixels: " << d << " out of " << p << " processed pixels" << std::endl;


        // [DEBUG] Printing
        // ROS_INFO("Segmentation mask dimensions: width = %d, height = %d", probability_map.cols, probability_map.rows);
        
        // ROS_INFO("Current Image dimensions: width = %d, height = %d", current_image.cols, current_image.rows);
        // std::cout << "Type: " << current_image_.type() << ", Channels: " << current_image_.channels() << std::endl;

        // ROS_INFO("Loaded Image dimensions: width = %d, height = %d", original_image.cols, original_image.rows);
        // std::cout << "Type: " << original_image.type() << ", Channels: " << original_image.channels() << std::endl;

        // ROS_INFO("probability_map type: %d, which should be %d", probability_map.type(), CV_32F);  // Should be CV_32F
        // ROS_INFO("current_image_ type: %d, which should be %d", current_image_.type(), CV_8UC3);          // Should be CV_8UC3



        CV_Assert(probability_map.cols == current_image.cols && probability_map.rows == current_image.rows);



        //////////////////
        // [DEBUG] Save images for debugging

        // // Define paths to save the mask and filtered image
        // std::string desktop_path = "/home/quanvu/Desktop/apple_testing/";
        // std::string current_image_path      = desktop_path + "current_image.png";
        // std::string mask_path               = desktop_path + "probability_map.png";
        // std::string filtered_image_path     = desktop_path + "filtered_image.png";


        // // [DEBUG 1] Save the current image
        // if (cv::imwrite(current_image_path, current_image)) 
        //     ROS_INFO("Current image saved to %s", current_image_path.c_str());
        // else 
        //     ROS_ERROR("Failed to save segmentation mask");


        // [DEBUG 2] Filter the image to retain only the object pixels, using the probability map
        cv::Mat filtered_image = cv::Mat::zeros(current_image.size(), current_image.type());
        int detected_pixels = 0;
        int processed_pixels = 0;
        for (int v = 0; v < probability_map.rows; ++v) {
            for (int u = 0; u < probability_map.cols; ++u) {
                float mask_value = probability_map.at<float>(v, u);
                processed_pixels++;
                if (mask_value > mask_threshold_) {  // Keep only the pixels where the mask indicates the object
                    filtered_image.at<cv::Vec3b>(v, u) = current_image.at<cv::Vec3b>(v, u);
                    detected_pixels++;
                }
            }
        }
        // ROS_INFO("[1] Detected pixels: %d out of %d processed pixels", detected_pixels, processed_pixels);
        // // Save the filtered image
        // if (cv::imwrite(filtered_image_path, filtered_image)) 
        //     ROS_INFO("Filtered image saved to %s", filtered_image_path.c_str());
        // else 
        //     ROS_ERROR("Failed to save filtered image");

        filtered_image_pub_.publish(cv_bridge::CvImage(current_image_header, sensor_msgs::image_encodings::BGR8, filtered_image).toImageMsg());
        ROS_INFO("Filtered image published");

        // // [DEBUG 3] Save the probability map as an image
        // cv::Mat mask_for_save;
        // probability_map.convertTo(mask_for_save, CV_8U, 255.0);  // Scale to 0-255 range    
        // int white_below_800 = 0;
        // for (int u = 0; u < mask_for_save.cols; ++u) {
        //     for (int v = 0; v < mask_for_save.rows; ++v) {
        //         if (v < 800){
        //             // mask_for_save.at<uchar>(v, u) = 255;
        //             if (mask_for_save.at<uchar>(v, u) > 230) white_below_800++;
        //         } 
                    
        //     }
        // }
        // std::cout << "White pixels below 800: " << white_below_800 << std::endl;

        // double minVal, maxVal;
        // cv::minMaxLoc(mask_for_save, &minVal, &maxVal);
        // std::cout << "\nMin value in segmentation mask: " << minVal << std::endl;
        // std::cout << "Max value in segmentation mask: " << maxVal << std::endl;
        // if (cv::imwrite(mask_path, mask_for_save)) 
        //     ROS_INFO("Segmentation mask saved to %s", mask_path.c_str());
        // else 
        //     ROS_ERROR("Failed to save segmentation mask");
        

        // MAIN
        // Filter the point cloud based on the segmentation mask 
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
        filtered_cloud->width = probability_map.cols;
        filtered_cloud->height = probability_map.rows;
        filtered_cloud->is_dense = false;  // Allows NaN values for "empty" points
        filtered_cloud->points.resize(filtered_cloud->width * filtered_cloud->height);

        detected_pixels = 0;
        processed_pixels = 0;
        for (int v = 0; v < probability_map.rows; ++v) {
            for (int u = 0; u < probability_map.cols; ++u) {
                processed_pixels++;
                int index = v * probability_map.cols + u;
                const auto& point = current_pointcloud->points[index];

                float mask_value = probability_map.at<float>(v, u);
                if (mask_value > mask_threshold_) {
                    filtered_cloud->points[index] = point;  // Retain the point
                    detected_pixels++;
                } else {
                    // Set a default point with NaN values to represent an "empty" pixel
                    filtered_cloud->points[index].x = std::numeric_limits<float>::quiet_NaN();
                    filtered_cloud->points[index].y = std::numeric_limits<float>::quiet_NaN();
                    filtered_cloud->points[index].z = std::numeric_limits<float>::quiet_NaN();
                }
            }
        }
        // ROS_INFO("[2] Detected points: %d out of %d processed points", detected_pixels, processed_pixels);
        // std::cout << "Filtered cloud size: " << filtered_cloud->points.size() << std::endl;

        auto filtered_cloud_msg = std::make_shared<sensor_msgs::PointCloud2>();
        pcl::toROSMsg(*filtered_cloud, *filtered_cloud_msg);
        filtered_cloud_msg->header = current_image_header;
        res.cloud = *filtered_cloud_msg;
        
        // cv::Size image_size(1920, 1080);
        // saveFlattenedPointCloud(filtered_cloud, current_image, image_size, "/home/quanvu/Desktop/apple_testing/poindcloud_projected");

        // DEBUG
        // Publish the filtered point cloud
        filtered_cloud_pub_.publish(*filtered_cloud_msg);
        ROS_INFO("Filtered point cloud published");

        return true;
    }
}

void saveFlattenedPointCloud(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& pointcloud, 
                             const cv::Mat& original_image, 
                             const cv::Size& image_size, 
                             const std::string& base_path) {
    // Initialize empty images
    cv::Mat depth_image = cv::Mat::zeros(image_size, CV_32FC1);  // Depth image with float depth values
    cv::Mat color_image = cv::Mat::zeros(image_size, CV_8UC3);   // Color image with BGR values
    cv::Mat original_with_purple = original_image.clone();       // Clone original for editing
    
    for (int v = 0; v < image_size.height; ++v) {
        for (int u = 0; u < image_size.width; ++u) {
            int index = v * image_size.width + u;

            // Check bounds and skip if point is invalid
            if (index >= pointcloud->points.size()) continue;
            const auto& point = pointcloud->points[index];
            
            if (!std::isnan(point.z) && point.z > 0) {
                // Set the depth value (e.g., z-coordinate)
                depth_image.at<float>(v, u) = point.z;

                // Set the color values
                color_image.at<cv::Vec3b>(v, u) = cv::Vec3b(point.b, point.g, point.r);
            } else {
                // Optional: set depth as NaN or black if the point is invalid
                depth_image.at<float>(v, u) = 0.0;  
                color_image.at<cv::Vec3b>(v, u) = cv::Vec3b(0, 0, 0);
            }
        }
    }

    // Normalize the depth image for visualization
    double minVal, maxVal;
    cv::minMaxLoc(depth_image, &minVal, &maxVal);
    cv::Mat depth_image_normalized;
    depth_image.convertTo(depth_image_normalized, CV_8UC1, 255.0 / (maxVal - minVal));

    // Apply purple color to the middle region for all three images
    cv::Rect central_region(image_size.width / 4, image_size.height / 4, image_size.width / 2, image_size.height / 2);
    cv::Vec3b purple_color(255, 0, 255);  // BGR for purple

    // for (int v = central_region.y; v < central_region.y + central_region.height; ++v) {
    //     for (int u = central_region.x; u < central_region.x + central_region.width; ++u) {
    //         depth_image_normalized.at<uchar>(v, u) = 128; // Purple as mid-gray in grayscale
    //         color_image.at<cv::Vec3b>(v, u) = purple_color;
    //         original_with_purple.at<cv::Vec3b>(v, u) = purple_color;
    //     }
    // }

    // Save depth, color, and original images
    std::string depth_path = base_path + "_depth.png";
    std::string color_path = base_path + "_color.png";
    std::string original_path = base_path + "_original.png";
    cv::imwrite(depth_path, depth_image_normalized);
    cv::imwrite(color_path, color_image);
    cv::imwrite(original_path, original_with_purple);

    std::cout << "Depth image saved at " << depth_path << std::endl;
    std::cout << "Color image saved at " << color_path << std::endl;
    std::cout << "Original image with purple region saved at " << original_path << std::endl;
}

