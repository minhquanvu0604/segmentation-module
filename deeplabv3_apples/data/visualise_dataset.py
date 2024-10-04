import os
import numpy as np

import cv2

from data import APPLE_LABEL

def amplify_intensities_opencv(input_dir, output_dir, amplification_factor):
    """
    Amplifies the intensities of images in the specified input directory using OpenCV.
    The amplified images are saved to the specified output directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"The specified input directory does not exist: {input_dir}")

    for filename in os.listdir(input_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(input_dir, filename)
            image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)

            # Unique intensities in the image
            unique_intensities = np.unique(image)
            print(f'Unique intensities in {filename}: {unique_intensities}')

            # Only show the apples
            image[image != APPLE_LABEL] = 0

            amplified_image = np.clip(image * amplification_factor, 0, 255).astype(np.uint8)
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, amplified_image)
            print(f'Processed {filename}')

# Usage
if __name__ == "__main__":

    input_directory = '/media/quanvu/ApplesQV/APPLE_DATA/labels-Fuji/Amplified2'
    output_directory = '/media/quanvu/ApplesQV/APPLE_DATA/labels-Fuji/Amplified2'
    amplification_factor = 20  # Adjust as needed
    amplify_intensities_opencv(input_directory, output_directory, amplification_factor)

    # APPLE_DATA 
    # Foliage Canopy of Apple Trees in Formal Architecture
    # https://rex.libraries.wsu.edu/esploro/outputs/dataset/Foliage-Canopy-of-Apple-Trees-in/99900501885201842?institution=01ALLIANCE_WSU
    # LABELS    
    # 1: Branches
    # 2: Apples
    # 3: Background
    # 4: Trunk
