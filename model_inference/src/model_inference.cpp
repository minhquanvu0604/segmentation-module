// model_inference.cpp

#include "model_inference/model_inference.hpp"
#include <iostream>
#include <vector>

namespace model_inference {

ModelInference::ModelInference(const std::string& model_path) {
    try {
        // Load the TorchScript model
        module_ = torch::jit::load(model_path);
        module_.eval(); // Set the module to evaluation mode
    }
    catch (const c10::Error& e) {
        std::cerr << "Error loading the model from " << model_path << "\n";
        std::cerr << e.what() << "\n";
        throw std::runtime_error("Failed to load the model.");
    }
}

ModelInference::~ModelInference() {
    // Destructor can be used for cleanup if necessary
}

std::vector<float> ModelInference::getDetectionMask(const torch::Tensor& input_tensor, float threshold) {
    // Ensure the input tensor is on the same device as the model
    torch::Tensor input = input_tensor.to(module_.device());

    // Perform forward pass
    torch::Tensor output;
    try {
        std::vector<torch::jit::IValue> inputs;
        inputs.emplace_back(input);
        output = module_.forward(inputs).toTensor();
    }
    catch (const c10::Error& e) {
        std::cerr << "Error during model inference.\n";
        std::cerr << e.what() << "\n";
        throw std::runtime_error("Model inference failed.");
    }

    // Apply sigmoid if necessary (assuming binary classification)
    // Modify based on your model's output activation
    output = torch::sigmoid(output);

    // Move tensor to CPU and ensure it's contiguous
    torch::Tensor output_cpu = output.to(torch::kCPU).contiguous();

    // Get the number of elements
    size_t size = output_cpu.numel();

    // Initialize the detection mask vector with the tensor data
    std::vector<float> detection_mask(output_cpu.data_ptr<float>(), output_cpu.data_ptr<float>() + size);

    // Apply threshold
    for (auto& prob : detection_mask) {
        prob = (prob >= threshold) ? 1.0f : 0.0f;
    }

    return detection_mask;
}

} // namespace model_inference
