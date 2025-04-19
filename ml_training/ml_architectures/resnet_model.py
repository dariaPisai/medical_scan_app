# ml_training/model_architectures/resnet_model.py
import torch
import torch.nn as nn
import torchvision.models as models

def build_adapted_resnet(num_classes, num_input_channels=4, pretrained=True):
    """
    Builds a ResNet model adapted for medical image classification.

    Args:
        num_classes (int): Number of output classes (e.g., 2 for LGG vs HGG).
        num_input_channels (int): Number of input image channels
                                   (e.g., 4 for BraTS: T1, T1c, T2, FLAIR).
        pretrained (bool): Whether to load weights pre-trained on ImageNet.

    Returns:
        torch.nn.Module: The adapted ResNet model.
    """
    # Load a pre-trained ResNet (e.g., resnet50)
    # weights=models.ResNet50_Weights.IMAGENET1K_V1 for older torchvision
    # weights=models.ResNet50_Weights.DEFAULT for newer torchvision
    weights = models.ResNet50_Weights.DEFAULT if pretrained else None
    model = models.resnet50(weights=weights)

    # --- Adapt the first convolutional layer for different input channels ---
    # Original conv1 weights shape: [out_channels, 3, kernel_size, kernel_size]
    original_conv1 = model.conv1
    original_weights = original_conv1.weight.data

    # Create new conv1 layer with desired input channels
    new_conv1 = nn.Conv2d(num_input_channels,
                          original_conv1.out_channels,
                          kernel_size=original_conv1.kernel_size,
                          stride=original_conv1.stride,
                          padding=original_conv1.padding,
                          bias=(original_conv1.bias is not None))

    # Adapt weights (simple averaging if going from 3 to more/less channels)
    # This is a basic approach; more sophisticated methods exist.
    if pretrained:
        # Average weights across the original 3 channels
        avg_weights = original_weights.mean(dim=1, keepdim=True)
        # Repeat the averaged weights for the new number of input channels
        new_weights = avg_weights.repeat(1, num_input_channels, 1, 1)
        new_conv1.weight.data = new_weights
        # Initialize bias if it exists
        if original_conv1.bias is not None:
            new_conv1.bias.data = original_conv1.bias.data
    else:
        # If not pretrained, just let PyTorch initialize the new layer
        pass

    # Replace the original conv1 layer
    model.conv1 = new_conv1
    print(f"Adapted ResNet conv1 to accept {num_input_channels} input channels.")

    # --- Adapt the final fully connected layer ---
    num_ftrs = model.fc.in_features # Get the number of features input to the original fc layer
    # Replace the final layer with a new one matching num_classes
    model.fc = nn.Linear(num_ftrs, num_classes)
    print(f"Adapted ResNet final layer for {num_classes} output classes.")

    return model

# Example Usage (in train_resnet.py):
# NUM_CLASSES = 2 # LGG vs HGG
# INPUT_CHANNELS = 4 # T1, T1c, T2, FLAIR
# resnet_classifier = build_adapted_resnet(NUM_CLASSES, INPUT_CHANNELS)
# print(resnet_classifier)