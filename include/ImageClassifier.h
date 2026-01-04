#ifndef IMAGE_CLASSIFIER_H
#define IMAGE_CLASSIFIER_H

#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <memory>

/**
 * @class ImageClassifier
 * @brief A class for loading and running image classification models using LibTorch
 */
class ImageClassifier {
public:
    /**
     * @brief Constructor
     * @param model_path Path to the TorchScript model file
     * @param labels_path Optional path to class labels file
     */
    explicit ImageClassifier(const std::string& model_path, 
                            const std::string& labels_path = "");

    /**
     * @brief Load a pre-trained model
     * @param model_path Path to the TorchScript model file
     * @return true if successful, false otherwise
     */
    bool loadModel(const std::string& model_path);

    /**
     * @brief Load class labels from file
     * @param labels_path Path to the labels file (one label per line)
     * @return true if successful, false otherwise
     */
    bool loadLabels(const std::string& labels_path);

    /**
     * @brief Classify an image
     * @param image_path Path to the image file
     * @param top_k Number of top predictions to return
     * @return Vector of pairs (class_index, confidence_score)
     */
    std::vector<std::pair<int, float>> classify(const std::string& image_path, 
                                                 int top_k = 5);

    /**
     * @brief Get the label for a class index
     * @param class_idx Class index
     * @return Label string or empty if not loaded
     */
    std::string getLabel(int class_idx) const;

    /**
     * @brief Check if model is loaded
     * @return true if model is loaded
     */
    bool isModelLoaded() const { return model_loaded_; }

private:
    /**
     * @brief Preprocess image for model input
     * @param image OpenCV Mat image
     * @return Preprocessed torch tensor
     */
    torch::Tensor preprocessImage(const cv::Mat& image);

    torch::jit::script::Module module_;
    std::vector<std::string> labels_;
    bool model_loaded_;
    
    // Model input parameters
    static constexpr int INPUT_SIZE = 224;
    static constexpr float MEAN[3] = {0.485f, 0.456f, 0.406f};
    static constexpr float STD[3] = {0.229f, 0.224f, 0.225f};
};

#endif // IMAGE_CLASSIFIER_H
