#ifndef PREDICT_H
#define PREDICT_H

#include <torch/script.h>  // LibTorch
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

class ModelInference {
public:
    // Constructor to initialize the model and input size
    ModelInference(const std::string& model_path, const cv::Size& input_size);

    // Function to infer a single image
    cv::Mat infer_single_image(const cv::Mat& image);

private:
    torch::jit::script::Module model_;  // TorchScript model
    cv::Size input_size_;  // Input size for image preprocessing

    // Preprocess the image
    torch::Tensor preprocess_image(cv::Mat image, const cv::Size& input_size);

    // Run inference and get probability map
    cv::Mat infer(torch::jit::script::Module& model, torch::Tensor& image_tensor, const cv::Size& original_size);
};

#endif  // PREDICT_H
