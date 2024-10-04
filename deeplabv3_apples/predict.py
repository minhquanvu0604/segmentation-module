import os, sys
top_level_package = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) # segmentation-module
sys.path.insert(0, top_level_package)

import torch
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd

from torchvision import transforms
import torchvision.transforms.functional as F

from deeplabv3_apples.config.config import INPUT_SIZE, CSV_PATH
from deeplabv3_apples.model import get_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('Test Predict')


# Main function for inference
def main_inference(split, model_path, image_folder, input_size, num_classes, output_folder=None):

    if output_folder:
        os.makedirs(output_folder, exist_ok=True)

    model = load_model(model_path, num_classes=num_classes)
    test_images = get_test_images(split)

    for image_name in os.listdir(image_folder):
        if image_name.endswith(('.jpg', '.png', '.jpeg')):
            
            image_path = os.path.join(image_folder, image_name)

            if image_name not in test_images:
                continue  # Skip images not in the test set

            logger.info(f"Processing    |    {image_name}")
            
            image_tensor, original_image = preprocess_image(image_path, input_size)
            if image_tensor is None:
                continue  # Skip the image if there's an error
            
            original_size = original_image.size
            predicted_mask = infer(model, image_tensor, original_size)

            if output_folder:
                save_path = os.path.join(output_folder, f'pred_{image_name}')
            else:
                save_path = None

            plot_result(original_image, predicted_mask, save_path=save_path)


def load_model(model_path, num_classes):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    logger.info(f"Loading model from {model_path}")
    model = get_model(num_classes=num_classes, pretrained=False)
    
    # Load the checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Load only the model weights, ignoring auxiliary classifier keys
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
    model.to(device)
    model.eval()
    logger.info("Model loaded successfully.")
    return model

def get_inference_transforms(input_size):
    return transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

def preprocess_image(image_path, input_size):
    try:
        image = Image.open(image_path).convert("RGB")
        transform = get_inference_transforms(input_size)
        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        return image_tensor, image
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {e}")
        return None, None

def infer(model, image_tensor, original_size):
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        output = model(image_tensor)['out']  # Shape: [N, C, H, W]
        probabilities = torch.softmax(output, dim=1)
        apple_prob = probabilities[:, 1, :, :].unsqueeze(1)  # Shape: [N, 1, H, W]
        
        # Resize probabilities to original size
        apple_prob_resized = torch.nn.functional.interpolate(
            apple_prob,
            size=original_size[::-1],  # (width, height)
            mode='bilinear',
            align_corners=False
        ).squeeze(0).squeeze(0)  # Remove batch and channel dimensions
        
        apple_prob_resized = apple_prob_resized.cpu().numpy()
    return apple_prob_resized

def plot_result(original_image, probability_map, save_path=None):
    plt.figure(figsize=(18, 6))
    
    # Original Image
    plt.subplot(1, 3, 1)
    plt.imshow(original_image)
    plt.title('Original Image')
    plt.axis('off')
    
    # Probability Map
    plt.subplot(1, 3, 2)
    plt.imshow(probability_map, cmap='viridis')
    plt.colorbar(label='Probability')
    plt.title('Apple Probability Map')
    plt.axis('off')
    
    # Overlay Probability Map on Original Image
    plt.subplot(1, 3, 3)
    plt.imshow(original_image)
    plt.imshow(probability_map, cmap='viridis', alpha=0.5)
    plt.colorbar(label='Probability')
    plt.title('Overlayed Probability Map')
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Result saved to {save_path}")
    
    plt.show(block=True)

def get_test_images(split_set):
    """
    Get the list of image names in the test set according to CSV file.
    """
    df = pd.read_csv(CSV_PATH)

    if split_set not in df.columns:
        raise ValueError(f"Split set '{split_set}' not found in the CSV file.")
    
    image_names = df[df[split_set].str.startswith('test', na=False)]['Image Name'].tolist()
    return image_names


if __name__ == "__main__":

    split = 'random_split'
    # model_path = '/root/segmentation-module/deeplabv3_apples/output/2024_10_02_23_37_35/model.pth'
    model_path = '/home/quanvu/git/segmentation-module/deeplabv3_apples/output/2024_10_02_23_37_35/model.pth'
    image_folder = '/home/quanvu/uts/APPLE_DATA/images-Fuji'
    output_folder = None
    # threshold = 0.5  # Threshold for classifying apples
    num_classes = 2

    main_inference(split, model_path, image_folder, INPUT_SIZE, num_classes, output_folder=output_folder)
