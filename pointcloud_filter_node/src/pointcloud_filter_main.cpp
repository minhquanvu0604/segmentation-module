#include "pointcloud_filter_node/pointcloud_filter_core.hpp"

#include <yaml-cpp/yaml.h>
#include <ros/package.h> 

int main(int argc, char** argv) {

    ros::init(argc, argv, "pointcloud_filter_node");
    ros::NodeHandle nh;
    ros::NodeHandle private_nh("~");

    // Load the YAML file parameters into the parameter server
    std::string package_path = ros::package::getPath("pointcloud_filter_node");
    std::string yaml_file = package_path + "/config/pointcloud_filter_node.yaml";

    // Load the YAML file
    YAML::Node config = YAML::LoadFile(yaml_file);
    std::string model_path = config["model_path"].as<std::string>();
    int input_width = config["input_size"]["width"].as<int>();
    int input_height = config["input_size"]["height"].as<int>();

    cv::Size input_size(input_width, input_height);

    PointCloudFilterCore filter_core(nh, private_nh, model_path, input_size);

    ros::spin();

    return 0;
}