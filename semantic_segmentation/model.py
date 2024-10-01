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

def print_model_info(logger, model, input_size, device):
    """
    Print detailed model information, including the number of parameters, layers, and architecture.

    Parameters:
        logger (Logger): Logger instance for logging model information.
        model (nn.Module): The model to analyze.
        input_size (tuple): The expected input size (channels, height, width).
        device (str): Device to place the model on ('cuda' or 'cpu').
    """
    from torchsummary import summary  # Requires torchsummary library
    
    # Move the model to the device
    model.to(device)

    # Log model architecture
    logger.info("\nModel Architecture:\n")
    logger.info(model)

    # Calculate total and trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info("\nModel Parameters:")
    logger.info(f"Total Parameters: {total_params:,}")
    logger.info(f"Trainable Parameters: {trainable_params:,}")

    # Provide a detailed summary (requires torchsummary)
    logger.info("\nModel Summary:")
    summary_str = summary(model, input_size=input_size, device=device)
    logger.info(summary_str)

