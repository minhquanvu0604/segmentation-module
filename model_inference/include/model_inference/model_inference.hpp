#ifndef model_inference_MODEL_INFERENCE_HPP
#define model_inference_MODEL_INFERENCE_HPP

#include <torch/script.h> // LibTorch
#include <string>
#include <vector>

namespace model_inference {

    /**
     * @brief A class for performing model inference using LibTorch.
     */
    class ModelInference {
    public:
        /**
         * @brief Constructor that loads the TorchScript model.
         * 
         * @param model_path Path to the TorchScript model file.
         */
        ModelInference(const std::string& model_path);

        /**
         * @brief Destructor.
         */
        ~ModelInference();

        /**
         * @brief Performs inference on the input data and returns a detection mask.
         * 
         * @param input_tensor A Torch tensor containing input data.
         * @param threshold The threshold to apply on the detection probabilities.
         * @return std::vector<float> A vector representing the detection mask.
         */
        std::vector<float> getDetectionMask(const torch::Tensor& input_tensor, float threshold = 0.5);

    private:
        torch::jit::script::Module module_; ///< The loaded TorchScript model.
    };

} // namespace model_inference

#endif // model_inference_MODEL_INFERENCE_HPP
