#include "ImageClassifier.h"
#include <iostream>
#include <fstream>
#include <algorithm>

ImageClassifier::ImageClassifier(const std::string& model_path, 
                                 const std::string& labels_path)
    : model_loaded_(false) {
    loadModel(model_path);
    if (!labels_path.empty()) {
        loadLabels(labels_path);
    }
}

bool ImageClassifier::loadModel(const std::string& model_path) {
    try {
        std::cout << "Loading model from: " << model_path << std::endl;
        module_ = torch::jit::load(model_path);
        module_.eval();
        model_loaded_ = true;
        std::cout << "Model loaded successfully!" << std::endl;
        return true;
    } catch (const c10::Error& e) {
        std::cerr << "Error loading model: " << e.what() << std::endl;
        model_loaded_ = false;
        return false;
    }
}

bool ImageClassifier::loadLabels(const std::string& labels_path) {
    std::ifstream file(labels_path);
    if (!file.is_open()) {
        std::cerr << "Error opening labels file: " << labels_path << std::endl;
        return false;
    }

    labels_.clear();
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty()) {
            labels_.push_back(line);
        }
    }

    std::cout << "Loaded " << labels_.size() << " labels" << std::endl;
    return true;
}

torch::Tensor ImageClassifier::preprocessImage(const cv::Mat& image) {
    // Resize image to model input size
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(INPUT_SIZE, INPUT_SIZE));

    // Convert BGR to RGB
    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

    // Convert to float and normalize to [0, 1]
    rgb.convertTo(rgb, CV_32FC3, 1.0 / 255.0);

    // Create tensor from image
    torch::Tensor tensor = torch::from_blob(
        rgb.data, 
        {1, INPUT_SIZE, INPUT_SIZE, 3}, 
        torch::kFloat32
    ).clone();

    // Permute dimensions from HWC to CHW
    tensor = tensor.permute({0, 3, 1, 2});

    // Normalize with ImageNet mean and std
    for (int c = 0; c < 3; ++c) {
        tensor[0][c] = (tensor[0][c] - MEAN[c]) / STD[c];
    }

    return tensor;
}

std::vector<std::pair<int, float>> ImageClassifier::classify(
    const std::string& image_path, int top_k) {
    
    std::vector<std::pair<int, float>> results;

    if (!model_loaded_) {
        std::cerr << "Model is not loaded!" << std::endl;
        return results;
    }

    // Load image
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        std::cerr << "Error loading image: " << image_path << std::endl;
        return results;
    }

    std::cout << "Processing image: " << image_path << std::endl;
    std::cout << "Image size: " << image.cols << "x" << image.rows << std::endl;

    // Preprocess image
    torch::Tensor input_tensor = preprocessImage(image);

    // Create input vector
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input_tensor);

    // Run inference
    torch::Tensor output;
    try {
        output = module_.forward(inputs).toTensor();
    } catch (const c10::Error& e) {
        std::cerr << "Error during inference: " << e.what() << std::endl;
        return results;
    }

    // Apply softmax to get probabilities
    output = torch::softmax(output, 1);

    // Get top k predictions
    auto [values, indices] = torch::topk(output, top_k, 1);

    // Convert to CPU and access data
    values = values.to(torch::kCPU);
    indices = indices.to(torch::kCPU);

    auto values_accessor = values.accessor<float, 2>();
    auto indices_accessor = indices.accessor<int64_t, 2>();

    for (int i = 0; i < top_k; ++i) {
        int class_idx = indices_accessor[0][i];
        float confidence = values_accessor[0][i];
        results.push_back({class_idx, confidence});
    }

    return results;
}

std::string ImageClassifier::getLabel(int class_idx) const {
    if (class_idx >= 0 && class_idx < static_cast<int>(labels_.size())) {
        return labels_[class_idx];
    }
    return "";
}
