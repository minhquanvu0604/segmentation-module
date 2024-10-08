// ./predict_demo \
        /home/quanvu/ros/apple_ws/src/segmentation_module/deeplabv3_apples/output/2024_10_02_23_37_35/model.pt \
        /home/quanvu/uts/APPLE_DATA/few_test_images 

#include <iostream>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include "predict.cpp"  // Assuming you saved the previous ModelInference class in a header file

int main(int argc, const char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: ./predict_demo <path-to-exported-script-module> <image-folder>" << std::endl;
        return -1;
    }

    std::string model_path = argv[1];
    std::string image_folder = argv[2];
    // std::string output_folder = argv[3];

    cv::Size input_size(800, 800);
    ModelInference model_inference(model_path, input_size);

    // // Ensure the output folder exists
    // if (!std::filesystem::exists(output_folder)) {
    //     std::filesystem::create_directory(output_folder);
    // }

    // Process each image in the folder
    for (const auto& entry : std::filesystem::directory_iterator(image_folder)) {
        if (entry.path().extension() == ".jpg" || entry.path().extension() == ".png") {
            std::string image_path = entry.path().string();
            std::string image_name = entry.path().filename().string();
            std::cout << "\nProcessing: " << image_name << std::endl;

            // Load the image
            cv::Mat original_image = cv::imread(image_path);
            if (original_image.empty()) {
                std::cerr << "Error loading image: " << image_path << std::endl;
                continue; // Skip if image loading failed
            }

            // Perform inference
            std::cout << "Inferring image..." << std::endl;
            cv::Mat probability_map = model_inference.infer_single_image(original_image);

            if (probability_map.empty()) {
                std::cerr << "Inference failed for image: " << image_name << std::endl;
                continue;  // Skip if inference fails
            }

            // Save the result
            cv::Mat raw_prob_8u;
            probability_map.convertTo(raw_prob_8u, CV_8U, 255.0);

            // Save the raw probability map as an image
            cv::imwrite("../raw_probability_map.png", raw_prob_8u);
            std::cout << "Saved raw probability map as raw_probability_map.png" << std::endl;
        }
    }

    return 0;
}
