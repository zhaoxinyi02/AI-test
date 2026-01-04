# AI Image Classifier

A simple C++ image classification application using LibTorch (PyTorch's C++ API) and OpenCV. This application loads a pre-trained TorchScript model and classifies images through a command-line interface.

## Features

- **Modular Design**: Clean class-based architecture with separation of concerns
- **LibTorch Integration**: Uses PyTorch's C++ API for model inference
- **OpenCV Support**: Efficient image loading and preprocessing
- **Command-Line Interface**: Easy-to-use CLI with multiple options
- **Top-K Predictions**: Shows multiple prediction results with confidence scores
- **Label Support**: Optional label file support for human-readable class names

## Project Structure

```
.
├── CMakeLists.txt              # CMake build configuration
├── README.md                   # This file
├── include/
│   └── ImageClassifier.h       # ImageClassifier class header
└── src/
    ├── ImageClassifier.cpp     # ImageClassifier implementation
    └── main.cpp                # Main application with CLI
```

## Prerequisites

### Required Dependencies

1. **CMake** (version 3.14 or higher)
   ```bash
   # Ubuntu/Debian
   sudo apt-get install cmake
   
   # macOS
   brew install cmake
   ```

2. **LibTorch** (PyTorch C++ API)
   - Download from: https://pytorch.org/get-started/locally/
   - Choose the version appropriate for your system (CPU or CUDA)
   - Extract to a location on your system
   
   Example for Linux (CPU):
   ```bash
   wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.0.0%2Bcpu.zip
   unzip libtorch-cxx11-abi-shared-with-deps-2.0.0+cpu.zip
   ```

3. **OpenCV** (version 4.x recommended)
   ```bash
   # Ubuntu/Debian
   sudo apt-get install libopencv-dev
   
   # macOS
   brew install opencv
   ```

## Building the Application

### 1. Clone the Repository
```bash
git clone https://github.com/zhaoxinyi02/AI-test.git
cd AI-test
```

### 2. Create Build Directory
```bash
mkdir build
cd build
```

### 3. Configure with CMake
```bash
# Set CMAKE_PREFIX_PATH to your LibTorch installation directory
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..

# Example:
# cmake -DCMAKE_PREFIX_PATH=/home/user/libtorch ..
```

### 4. Build
```bash
cmake --build . --config Release
```

After successful build, the executable `image_classifier` will be in the `build` directory.

## Usage

### Basic Usage
```bash
./image_classifier -m <model_path> -i <image_path>
```

### Command-Line Options
```
Options:
  -m, --model <path>     Path to TorchScript model file (required)
  -i, --image <path>     Path to image file to classify (required)
  -l, --labels <path>    Path to labels file (optional)
  -k, --topk <number>    Number of top predictions to show (default: 5)
  -h, --help             Show help message
```

### Examples

1. **Basic classification with a model:**
   ```bash
   ./image_classifier -m model.pt -i cat.jpg
   ```

2. **Classification with labels and top-3 predictions:**
   ```bash
   ./image_classifier -m resnet18.pt -i dog.jpg -l imagenet_labels.txt -k 3
   ```

3. **Show help:**
   ```bash
   ./image_classifier --help
   ```

## Preparing Models

### Converting a PyTorch Model to TorchScript

To use a model with this application, you need to convert it to TorchScript format (`.pt` file).

Example Python script to convert a pre-trained ResNet18 model:

```python
import torch
import torchvision.models as models

# Load a pre-trained model
model = models.resnet18(pretrained=True)
model.eval()

# Create a sample input
example_input = torch.rand(1, 3, 224, 224)

# Trace the model
traced_model = torch.jit.trace(model, example_input)

# Save the traced model
traced_model.save("resnet18.pt")
print("Model saved as resnet18.pt")
```

### Creating a Labels File

For ImageNet-based models, create a text file with one label per line (1000 lines for ImageNet):

```text
tench
goldfish
great white shark
tiger shark
hammerhead
...
```

You can download ImageNet labels from various sources online.

## Model Input Requirements

The application preprocesses images with the following specifications:
- **Input size**: 224x224 pixels (configurable in code)
- **Color format**: RGB
- **Normalization**: ImageNet mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`
- **Tensor format**: NCHW (batch, channels, height, width)

If your model uses different preprocessing, modify the constants in `ImageClassifier.h`.

## Example Output

```
=== AI Image Classifier Application ===
Built with LibTorch and OpenCV

Loading model from: resnet18.pt
Model loaded successfully!
Loaded 1000 labels

Classifying image...
Processing image: cat.jpg
Image size: 800x600

=== Classification Results ===
  1. tabby cat (class 281) - Confidence: 45.23%
  2. Egyptian cat (class 285) - Confidence: 32.15%
  3. tiger cat (class 282) - Confidence: 18.67%
  4. Persian cat (class 283) - Confidence: 2.89%
  5. lynx (class 287) - Confidence: 0.54%

Classification completed successfully!
```

## Troubleshooting

### LibTorch Not Found
- Ensure `CMAKE_PREFIX_PATH` points to your LibTorch directory
- The path should contain `share/cmake/Torch/TorchConfig.cmake`

### OpenCV Not Found
- Install OpenCV development package for your system
- On Ubuntu: `sudo apt-get install libopencv-dev`
- On macOS: `brew install opencv`

### Runtime Errors
- Ensure model file is in TorchScript format (`.pt`)
- Verify image file exists and is in a supported format (JPG, PNG, etc.)
- Check that the model expects 224x224 input size

### ABI Compatibility Issues (Linux)
- Use the LibTorch version matching your compiler's C++11 ABI
- Download either `cxx11-abi` or `pre-cxx11-abi` version accordingly

## Architecture

### ImageClassifier Class
The main class responsible for:
- Loading TorchScript models
- Image preprocessing
- Running inference
- Managing class labels

### Main Application
- Command-line argument parsing
- User interaction
- Result formatting and display

## License

This project is provided as-is for educational and development purposes.

## Contributing

Feel free to submit issues and enhancement requests!

## Acknowledgments

- Built with [LibTorch](https://pytorch.org/cppdocs/) (PyTorch C++ API)
- Uses [OpenCV](https://opencv.org/) for image processing
- Designed for educational purposes and demonstration
