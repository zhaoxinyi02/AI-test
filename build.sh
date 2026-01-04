#!/bin/bash

# Build script for AI Image Classifier
# This script simplifies the build process

set -e  # Exit on error

echo "=== AI Image Classifier Build Script ==="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default LibTorch path (user can override)
LIBTORCH_PATH=${LIBTORCH_PATH:-"$HOME/libtorch"}

# Check if LibTorch path is provided as argument
if [ $# -ge 1 ]; then
    LIBTORCH_PATH=$1
fi

echo -e "${YELLOW}Configuration:${NC}"
echo "  LibTorch Path: $LIBTORCH_PATH"
echo ""

# Check if LibTorch exists
if [ ! -d "$LIBTORCH_PATH" ]; then
    echo -e "${RED}Error: LibTorch not found at $LIBTORCH_PATH${NC}"
    echo ""
    echo "Please download LibTorch from:"
    echo "  https://pytorch.org/get-started/locally/"
    echo ""
    echo "Then either:"
    echo "  1. Extract to ~/libtorch (default location)"
    echo "  2. Set LIBTORCH_PATH environment variable"
    echo "  3. Pass the path as argument: ./build.sh /path/to/libtorch"
    echo ""
    exit 1
fi

# Check if CMake is installed
if ! command -v cmake &> /dev/null; then
    echo -e "${RED}Error: CMake is not installed${NC}"
    echo "Please install CMake:"
    echo "  Ubuntu/Debian: sudo apt-get install cmake"
    echo "  macOS: brew install cmake"
    exit 1
fi

# Check if OpenCV is installed (basic check)
if ! pkg-config --exists opencv4 2>/dev/null && ! pkg-config --exists opencv 2>/dev/null; then
    echo -e "${YELLOW}Warning: OpenCV might not be installed${NC}"
    echo "If the build fails, install OpenCV:"
    echo "  Ubuntu/Debian: sudo apt-get install libopencv-dev"
    echo "  macOS: brew install opencv"
    echo ""
fi

# Create build directory
echo -e "${GREEN}Creating build directory...${NC}"
mkdir -p build
cd build

# Run CMake
echo -e "${GREEN}Running CMake configuration...${NC}"
cmake -DCMAKE_PREFIX_PATH="$LIBTORCH_PATH" -DCMAKE_BUILD_TYPE=Release ..

if [ $? -ne 0 ]; then
    echo -e "${RED}CMake configuration failed!${NC}"
    exit 1
fi

# Build
echo ""
echo -e "${GREEN}Building the application...${NC}"
cmake --build . --config Release

if [ $? -ne 0 ]; then
    echo -e "${RED}Build failed!${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}=== Build Successful! ===${NC}"
echo ""
echo "The executable is located at: build/image_classifier"
echo ""
echo "To test it, you can:"
echo "  1. Convert a model: python3 convert_model.py -m resnet18"
echo "  2. Download labels: python3 convert_model.py --labels"
echo "  3. Run classifier: ./build/image_classifier -m resnet18.pt -i image.jpg -l imagenet_labels.txt"
echo ""
