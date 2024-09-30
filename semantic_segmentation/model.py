import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50


def get_model(num_classes, pretrained=True):
    """
    Initialize the DeepLabV3 model.

    Parameters:
        num_classes (int): Number of output classes for segmentation (e.g., 2 for binary segmentation).
        pretrained (bool): Whether to load pretrained weights on COCO dataset.

    Returns:
        model (nn.Module): The DeepLabV3 model.
    """
    # Load the pre-trained DeepLabV3 model
    model = deeplabv3_resnet50(pretrained=pretrained)

    # Replace the classifier head to match the number of classes
    model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
    return model
