#ifndef MODEL_INFERENCE_CPP_MODELINFERENCE_H
#define MODEL_INFERENCE_CPP_MODELINFERENCE_H

#include <torch/script.h> // LibTorch header
#include <opencv2/opencv.hpp>
#include <string>



#include <torch/torch.h>

namespace model_inference_cpp {

    class ModelInference {
    public:
        // Constructor that loads the model
        explicit ModelInference(const std::string& model_path);

        // Method to perform inference on an input image
        cv::Mat infer(const cv::Mat& input_image);

    private:
        torch::jit::script::Module model_;
    };

} // namespace model_inference_cpp

#endif // MODEL_INFERENCE_CPP_MODELINFERENCE_H
