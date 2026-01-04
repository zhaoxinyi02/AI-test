# Examples

This directory contains examples and additional resources for using the AI Image Classifier.

## Example: Using ResNet18 for Image Classification

### Step 1: Convert a Pre-trained Model

```bash
# Make sure you have PyTorch installed
pip3 install torch torchvision

# Convert ResNet18 to TorchScript format
python3 convert_model.py -m resnet18 -o resnet18.pt

# Download ImageNet labels
python3 convert_model.py --labels
```

### Step 2: Test with Sample Images

```bash
# Download a test image
wget https://upload.wikimedia.org/wikipedia/commons/3/3a/Cat03.jpg -O cat.jpg

# Run the classifier
./build/image_classifier -m resnet18.pt -i cat.jpg -l imagenet_labels.txt -k 5
```

## Example: Custom Model Integration

If you have a custom PyTorch model, you can convert it to TorchScript:

```python
import torch

# Load your custom model
model = YourCustomModel()
model.load_state_dict(torch.load('your_model.pth'))
model.eval()

# Create example input (adjust dimensions as needed)
example_input = torch.rand(1, 3, 224, 224)

# Convert to TorchScript
traced_model = torch.jit.trace(model, example_input)
traced_model.save('custom_model.pt')
```

Then use it with the application:

```bash
./build/image_classifier -m custom_model.pt -i your_image.jpg
```

## Example: Batch Processing

You can create a simple shell script to process multiple images:

```bash
#!/bin/bash
# batch_classify.sh

MODEL="resnet18.pt"
LABELS="imagenet_labels.txt"

for image in images/*.jpg; do
    echo "Processing: $image"
    ./build/image_classifier -m "$MODEL" -i "$image" -l "$LABELS" -k 3
    echo "---"
done
```

## Example: Different Model Types

### MobileNetV2 (Lightweight)
```bash
python3 convert_model.py -m mobilenet_v2 -o mobilenet_v2.pt
./build/image_classifier -m mobilenet_v2.pt -i image.jpg -l imagenet_labels.txt
```

### EfficientNet (High Accuracy)
```bash
python3 convert_model.py -m efficientnet_b0 -o efficientnet_b0.pt
./build/image_classifier -m efficientnet_b0.pt -i image.jpg -l imagenet_labels.txt
```

### VGG16 (Classic Architecture)
```bash
python3 convert_model.py -m vgg16 -o vgg16.pt
./build/image_classifier -m vgg16.pt -i image.jpg -l imagenet_labels.txt
```

## Performance Comparison

Different models have different characteristics:

| Model | Size | Speed | Accuracy | Use Case |
|-------|------|-------|----------|----------|
| MobileNetV2 | ~14MB | Fast | Good | Mobile/Embedded |
| ResNet18 | ~45MB | Medium | Good | General Purpose |
| ResNet50 | ~98MB | Medium | Very Good | General Purpose |
| EfficientNet-B0 | ~20MB | Medium | Very Good | Balanced |
| VGG16 | ~528MB | Slow | Good | Classic Baseline |

## Common Use Cases

### 1. Object Recognition in Photos
Perfect for identifying objects, animals, and scenes in photographs.

### 2. Quality Control
Can be trained to detect defects or categorize products in manufacturing.

### 3. Medical Imaging
With proper training, can assist in medical image analysis.

### 4. Document Classification
Can categorize different types of documents or forms.

### 5. Wildlife Monitoring
Useful for camera trap image analysis and species identification.

## Tips for Best Results

1. **Use the right model size**: Smaller models (MobileNet) for speed, larger models (ResNet50+) for accuracy
2. **Proper preprocessing**: Ensure your images are properly preprocessed (resize, normalize)
3. **Good training data**: If using custom models, ensure diverse and representative training data
4. **GPU acceleration**: For production use, consider using the CUDA version of LibTorch
5. **Batch processing**: Process multiple images efficiently with scripting

## Troubleshooting Examples

### Low Confidence Scores
- Image might be blurry or low quality
- Object not well represented in training data
- Try different preprocessing parameters

### Wrong Classifications
- Model might not be suitable for the image type
- Consider using a specialized model
- Check if preprocessing matches training

### Slow Performance
- Use a lighter model (MobileNet)
- Enable GPU support
- Reduce image resolution

## Additional Resources

- [PyTorch Hub Models](https://pytorch.org/hub/)
- [TorchVision Models](https://pytorch.org/vision/stable/models.html)
- [LibTorch Documentation](https://pytorch.org/cppdocs/)
- [OpenCV Documentation](https://docs.opencv.org/)
