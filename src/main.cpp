#include "ImageClassifier.h"
#include <iostream>
#include <iomanip>
#include <string>

void printUsage(const char* program_name) {
    std::cout << "\n=== AI Image Classifier ===" << std::endl;
    std::cout << "\nUsage: " << program_name << " [OPTIONS]" << std::endl;
    std::cout << "\nOptions:" << std::endl;
    std::cout << "  -m, --model <path>     Path to TorchScript model file (required)" << std::endl;
    std::cout << "  -i, --image <path>     Path to image file to classify (required)" << std::endl;
    std::cout << "  -l, --labels <path>    Path to labels file (optional)" << std::endl;
    std::cout << "  -k, --topk <number>    Number of top predictions to show (default: 5)" << std::endl;
    std::cout << "  -h, --help             Show this help message" << std::endl;
    std::cout << "\nExample:" << std::endl;
    std::cout << "  " << program_name << " -m model.pt -i image.jpg -l labels.txt -k 3" << std::endl;
    std::cout << std::endl;
}

struct ProgramArgs {
    std::string model_path;
    std::string image_path;
    std::string labels_path;
    int top_k = 5;
    bool show_help = false;
};

ProgramArgs parseArguments(int argc, char* argv[]) {
    ProgramArgs args;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            args.show_help = true;
            return args;
        } else if ((arg == "-m" || arg == "--model") && i + 1 < argc) {
            args.model_path = argv[++i];
        } else if ((arg == "-i" || arg == "--image") && i + 1 < argc) {
            args.image_path = argv[++i];
        } else if ((arg == "-l" || arg == "--labels") && i + 1 < argc) {
            args.labels_path = argv[++i];
        } else if ((arg == "-k" || arg == "--topk") && i + 1 < argc) {
            args.top_k = std::stoi(argv[++i]);
        } else {
            std::cerr << "Unknown argument or missing value: " << arg << std::endl;
        }
    }

    return args;
}

void printPredictions(const std::vector<std::pair<int, float>>& predictions,
                     const ImageClassifier& classifier) {
    std::cout << "\n=== Classification Results ===" << std::endl;
    std::cout << std::fixed << std::setprecision(2);

    for (size_t i = 0; i < predictions.size(); ++i) {
        int class_idx = predictions[i].first;
        float confidence = predictions[i].second * 100.0f;

        std::string label = classifier.getLabel(class_idx);
        std::cout << "  " << (i + 1) << ". ";

        if (!label.empty()) {
            std::cout << label << " (class " << class_idx << ")";
        } else {
            std::cout << "Class " << class_idx;
        }

        std::cout << " - Confidence: " << confidence << "%" << std::endl;
    }
    std::cout << std::endl;
}

int main(int argc, char* argv[]) {
    std::cout << "=== AI Image Classifier Application ===" << std::endl;
    std::cout << "Built with LibTorch and OpenCV\n" << std::endl;

    // Parse command line arguments
    ProgramArgs args = parseArguments(argc, argv);

    if (args.show_help || argc == 1) {
        printUsage(argv[0]);
        return 0;
    }

    // Validate required arguments
    if (args.model_path.empty() || args.image_path.empty()) {
        std::cerr << "Error: Model path and image path are required!" << std::endl;
        printUsage(argv[0]);
        return 1;
    }

    try {
        // Create classifier
        ImageClassifier classifier(args.model_path, args.labels_path);

        if (!classifier.isModelLoaded()) {
            std::cerr << "Failed to load model. Exiting." << std::endl;
            return 1;
        }

        // Classify image
        std::cout << "\nClassifying image..." << std::endl;
        auto predictions = classifier.classify(args.image_path, args.top_k);

        if (predictions.empty()) {
            std::cerr << "Classification failed!" << std::endl;
            return 1;
        }

        // Print results
        printPredictions(predictions, classifier);

        std::cout << "Classification completed successfully!" << std::endl;
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
