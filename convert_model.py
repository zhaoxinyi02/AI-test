#!/usr/bin/env python3
"""
Script to convert a pre-trained PyTorch model to TorchScript format.
This creates a model file that can be loaded by the C++ application.
"""

import torch
import torchvision.models as models
import argparse


def convert_model(model_name, output_path):
    """
    Convert a pre-trained torchvision model to TorchScript format.
    
    Args:
        model_name: Name of the model (e.g., 'resnet18', 'mobilenet_v2')
        output_path: Path to save the TorchScript model
    """
    print(f"Loading pre-trained {model_name} model...")
    
    # Dictionary of available models
    available_models = {
        'resnet18': models.resnet18,
        'resnet34': models.resnet34,
        'resnet50': models.resnet50,
        'resnet101': models.resnet101,
        'resnet152': models.resnet152,
        'mobilenet_v2': models.mobilenet_v2,
        'mobilenet_v3_small': models.mobilenet_v3_small,
        'mobilenet_v3_large': models.mobilenet_v3_large,
        'vgg16': models.vgg16,
        'vgg19': models.vgg19,
        'squeezenet1_0': models.squeezenet1_0,
        'squeezenet1_1': models.squeezenet1_1,
        'densenet121': models.densenet121,
        'densenet161': models.densenet161,
        'densenet169': models.densenet169,
        'densenet201': models.densenet201,
        'efficientnet_b0': models.efficientnet_b0,
        'efficientnet_b1': models.efficientnet_b1,
        'efficientnet_b2': models.efficientnet_b2,
        'efficientnet_b3': models.efficientnet_b3,
        'efficientnet_b4': models.efficientnet_b4,
    }
    
    if model_name not in available_models:
        print(f"Error: Model '{model_name}' is not supported.")
        print(f"Available models: {', '.join(available_models.keys())}")
        return False
    
    # Load the model
    try:
        model = available_models[model_name](pretrained=True)
        model.eval()
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return False
    
    # Create example input
    print("Creating example input...")
    example_input = torch.rand(1, 3, 224, 224)
    
    # Trace the model
    print("Tracing model...")
    try:
        traced_model = torch.jit.trace(model, example_input)
    except Exception as e:
        print(f"Error tracing model: {e}")
        return False
    
    # Save the traced model
    print(f"Saving model to {output_path}...")
    try:
        traced_model.save(output_path)
        print(f"Model successfully saved as {output_path}")
        print(f"\nYou can now use this model with the C++ application:")
        print(f"  ./image_classifier -m {output_path} -i your_image.jpg")
        return True
    except Exception as e:
        print(f"Error saving model: {e}")
        return False


def download_imagenet_labels(output_path="imagenet_labels.txt"):
    """
    Download ImageNet class labels.
    
    Args:
        output_path: Path to save the labels file
    """
    import urllib.request
    
    print(f"Downloading ImageNet labels...")
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    
    try:
        urllib.request.urlretrieve(url, output_path)
        print(f"Labels downloaded successfully to {output_path}")
        return True
    except Exception as e:
        print(f"Error downloading labels: {e}")
        print("You can manually download from:")
        print(url)
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Convert PyTorch models to TorchScript format for C++ inference"
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        default="resnet18",
        help="Model name (default: resnet18)"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output path for TorchScript model (default: <model_name>.pt)"
    )
    parser.add_argument(
        "-l", "--labels",
        action="store_true",
        help="Download ImageNet labels file"
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List all available models"
    )
    
    args = parser.parse_args()
    
    # List models if requested
    if args.list_models:
        print("Available models:")
        models_list = [
            'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
            'mobilenet_v2', 'mobilenet_v3_small', 'mobilenet_v3_large',
            'vgg16', 'vgg19',
            'squeezenet1_0', 'squeezenet1_1',
            'densenet121', 'densenet161', 'densenet169', 'densenet201',
            'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2',
            'efficientnet_b3', 'efficientnet_b4',
        ]
        for model in models_list:
            print(f"  - {model}")
        return
    
    # Download labels if requested
    if args.labels:
        download_imagenet_labels()
    
    # Set output path
    output_path = args.output if args.output else f"{args.model}.pt"
    
    # Convert model
    print("\n=== Converting Model to TorchScript ===\n")
    convert_model(args.model, output_path)
    print("\n=== Conversion Complete ===\n")


if __name__ == "__main__":
    main()
