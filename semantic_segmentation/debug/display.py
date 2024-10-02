import cv2
import sys
import numpy as np


def resize_image(image, scale_percent):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

def display_mask_image(image_path, image_name=None):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Check if the image was successfully loaded
    if image is None:
        print(f"Failed to load image at '{image_path}'")
        sys.exit(1)

    # Amplify the intensities of the image
    amplify_factor = 20
    image = np.clip(image * amplify_factor, 0, 255).astype(np.uint8)

    # Resize the image
    scale_percent = 50
    image = resize_image(image, scale_percent)

    # Display the image
    cv2.imshow(f"{image_name}", image)