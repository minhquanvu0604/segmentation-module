#include "pointcloud_filter_node/pointcloud_filter_core.hpp"

int main(int argc, char** argv) {
    ros::init(argc, argv, "pointcloud_filter_node");
    ros::NodeHandle nh;
    ros::NodeHandle private_nh("~");

    std::string model_path = "/home/quanvu/ros/apple_ws/src/segmentation_module/deeplabv3_apples/output/2024_10_02_23_37_35/model.pt";
    cv::Size input_size(800, 800);  // Model input size

    // Create the PointCloudFilterCore object
    PointCloudFilterCore filter_core(nh, private_nh, model_path, input_size);

    // Keep the node running
    ros::spin();

    return 0;
}