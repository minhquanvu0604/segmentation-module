#ifndef POINTCLOUD_FILTER_CORE_H
#define POINTCLOUD_FILTER_CORE_H

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <std_msgs/Header.h>
#include <pcl_ros/point_cloud.h>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <nodelet/nodelet.h>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <mutex>

#include <hydra_utils/CloudService.h>
#include "predict.hpp"  // Include the ModelInference class for inference

    
namespace segmentation_module {

    class PointCloudFilterCore : public nodelet::Nodelet {
    public:
        PointCloudFilterCore() = default;
        // Constructor with ROS node handles and model path
        PointCloudFilterCore(ros::NodeHandle& nh, ros::NodeHandle& private_nh, const std::string& model_path, const cv::Size& input_size);

    private:
        virtual void onInit() override;

        void init(ros::NodeHandle& nh, ros::NodeHandle& private_nh, const std::string& model_path, const cv::Size& input_size);

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

        // Callback function for synchronized image and point cloud topics
        void callback(const sensor_msgs::ImageConstPtr& img_msg, const sensor_msgs::PointCloud2ConstPtr& cloud_msg);

        // Service callback function for filtering the point cloud
        bool filterPointCloudService(hydra_utils::CloudService::Request &req, hydra_utils::CloudService::Response &res);
        
    private:
        // The object that handles image inference
        // ModelInference model_inference_;  
        std::unique_ptr<ModelInference> model_inference_;

        ros::Publisher filtered_cloud_pub_;
        ros::Publisher filtered_image_pub_;

        message_filters::Subscriber<sensor_msgs::Image> image_sub_;
        message_filters::Subscriber<sensor_msgs::PointCloud2> pointcloud_sub_;
        std::mutex input_mutex_;

        std::atomic<bool> data_received_;

        // Service server for filtering the point cloud
        ros::ServiceServer filtered_pointcloud_srv_;

        // Approx Sync
        typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::PointCloud2> MySyncPolicy;
        typedef message_filters::Synchronizer<MySyncPolicy> Sync;
        std::shared_ptr<Sync> sync_;

        // Uses OpenCV’s reference counting, so storing directly is efficient
        cv::Mat current_image_; 
        std_msgs::Header current_image_header_;

        // Stored as a shared pointer to avoid costly deep copies and to allow efficient memory management.
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr current_pointcloud_;  


        bool image_received_ = false;
        long long image_num_ = 0;
        long long pointcloud_num_ = 0;

        // CONSTANT
        double mask_threshold_ = 0.5;  // Threshold for the segmentation mask
    };
}

#endif // POINTCLOUD_FILTER_CORE_H
