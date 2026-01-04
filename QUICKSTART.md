# Quick Start Guide

This guide will help you quickly get started with the AI Image Classifier application.

## Step 1: Install Dependencies

### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install cmake g++ libopencv-dev python3 python3-pip
```

### macOS
```bash
brew install cmake opencv python3
```

## Step 2: Download LibTorch

```bash
# For Linux CPU version
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.0.0%2Bcpu.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.0.0+cpu.zip -d ~/

# For macOS
# Download from https://pytorch.org/get-started/locally/ and extract
```

## Step 3: Build the Application

```bash
cd AI-test
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=~/libtorch ..
cmake --build . --config Release
```

## Step 4: Prepare a Model

### Option A: Use the provided Python script

```bash
# Install PyTorch (if not already installed)
pip3 install torch torchvision

# Convert a pre-trained ResNet18 model
cd ..
python3 convert_model.py -m resnet18 -o resnet18.pt

# Also download ImageNet labels
python3 convert_model.py --labels
```

### Option B: Manual conversion

Create a file `export_model.py`:
```python
import torch
import torchvision.models as models

model = models.resnet18(pretrained=True)
model.eval()
example = torch.rand(1, 3, 224, 224)
traced = torch.jit.trace(model, example)
traced.save("resnet18.pt")
```

Run it:
```bash
python3 export_model.py
```

## Step 5: Test with an Image

```bash
# Download a test image
wget https://upload.wikimedia.org/wikipedia/commons/3/3a/Cat03.jpg -O test_cat.jpg

# Run classification
./build/image_classifier -m resnet18.pt -i test_cat.jpg -l imagenet_labels.txt -k 5
```

## Expected Output

```
=== AI Image Classifier Application ===
Built with LibTorch and OpenCV

Loading model from: resnet18.pt
Model loaded successfully!
Loaded 1000 labels

Classifying image...
Processing image: test_cat.jpg
Image size: 1200x900

=== Classification Results ===
  1. tabby cat (class 281) - Confidence: 42.35%
  2. Egyptian cat (class 285) - Confidence: 28.91%
  3. tiger cat (class 282) - Confidence: 15.23%
  4. Persian cat (class 283) - Confidence: 8.67%
  5. lynx (class 287) - Confidence: 3.12%

Classification completed successfully!
```

## Troubleshooting

### "Could NOT find Torch" during CMake
- Verify LibTorch path: `ls ~/libtorch/share/cmake/Torch/`
- Use full path: `cmake -DCMAKE_PREFIX_PATH=/full/path/to/libtorch ..`

### "Could NOT find OpenCV"
- Install OpenCV: `sudo apt-get install libopencv-dev` (Ubuntu)
- Or: `brew install opencv` (macOS)

### "Error loading image"
- Check image file exists
- Supported formats: JPG, PNG, BMP, etc.
- Verify file permissions

### "Error loading model"
- Ensure model is in TorchScript format (.pt file)
- Check file path is correct
- Verify LibTorch version compatibility

## Next Steps

1. Try different models: `python3 convert_model.py --list-models`
2. Test with your own images
3. Modify preprocessing in `ImageClassifier.h` for custom models
4. Integrate into your own C++ application

## Support

For issues and questions:
- Check the main [README.md](README.md)
- Review the code in `src/` and `include/` directories
- Open an issue on GitHub

Happy classifying! 🎉
