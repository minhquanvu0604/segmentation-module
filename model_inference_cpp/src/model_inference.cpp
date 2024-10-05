#include "model_inference.hpp"
#include <iostream>

namespace model_inference_cpp {

    ModelInference::ModelInference(const std::string& model_path) {
        try {
            // Load the PyTorch model
            model_ = torch::jit::load(model_path);
            model_.eval();
        } catch (const c10::Error& e) {
            std::cerr << "Error loading the model: " << e.what() << std::endl;
            throw e;
        }
    }

    cv::Mat ModelInference::infer(const cv::Mat& input_image) {
        // Convert the input image to float and normalize
        cv::Mat img_float;
        input_image.convertTo(img_float, CV_32F, 1.0 / 255);

        // Convert BGR to RGB if needed
        cv::cvtColor(img_float, img_float, cv::COLOR_BGR2RGB);

        // Create a tensor from the image
        auto tensor_image = torch::from_blob(
            img_float.data,
            {1, img_float.rows, img_float.cols, 3},
            torch::kFloat32);

        // Permute dimensions to match [N, C, H, W]
        tensor_image = tensor_image.permute({0, 3, 1, 2});

        // Move the tensor to the appropriate device (CPU)
        tensor_image = tensor_image.to(torch::kCPU);

        // Perform the forward pass
        torch::NoGradGuard no_grad; // Disable gradient computation
        auto output = model_.forward({tensor_image}).toTensor();

        // Process the output tensor
        output = output.squeeze().detach().cpu();

        // Convert the output tensor to a cv::Mat
        cv::Mat output_mat(
            output.sizes()[1],
            output.sizes()[0],
            CV_32F,
            output.data_ptr<float>());

        // Clone the matrix to ensure data safety
        return output_mat.clone();
    }

} // namespace model_inference_cpp
