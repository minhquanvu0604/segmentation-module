import sys, os
top_level_package = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, top_level_package)

wool_estimation_path = "/root/wool-estimation"
fibre_segmentation_path = "/root/wool-estimation/fibre_segmentation"

if wool_estimation_path not in sys.path:
    sys.path.insert(0, wool_estimation_path)
if fibre_segmentation_path not in sys.path:
    sys.path.insert(0, fibre_segmentation_path)

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import cv2

from fibre_width.utils.visualise_utils import display_mask_image
from fibre_width.utils.image_processing import resize_image

from fibre_segmentation.data_loader import data_loaders
import fibre_segmentation.transforms as T


def visualise_specific_pair(image_path):
    file_name = os.path.basename(image_path)

    mask_path = os.path.join(os.path.dirname(image_path), '../masks/instances', file_name)
    mask_path = os.path.normpath(mask_path)

    # Display the image 
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    scale_percent = 50
    image = resize_image(image, scale_percent)
    cv2.imshow(f"Mask", image)

    # Display the mask
    display_mask_image(mask_path)

    # Wait for a key press and close the image window
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def visualise_dataloader(data_loader, combined = True):
    """
    Visualize the data loader output by displaying a randomly picked image and its masks side by side.
    This helps in verifying that the data loader side, inspecting downstream to the dataset's synthesised images and their labels.

    Press q to move on to the next mask of the same image.
    """

    iterator = iter(data_loader)

    if combined:
        print("[visualise_dataloader] Visualise an image's all masks at once")
        display_all_masks_at_once(iterator)
    else:
        print("[visualise_dataloader] Visualise a single image's masks one after another")
        display_masks_one_by_one(iterator)

def display_all_masks_at_once(iterator):
    while True:
        try:
            # Get a batch of data
            img, target = next(iterator)
        except StopIteration:
            # If no more data, break the loop
            break

        img = img[0]  # First image in the batch
        # Convert tensor to numpy array for visualization
        img = img.permute(1, 2, 0).numpy()  # Convert from (C, H, W) to (H, W, C)

        # Initialize a combined mask with zeros
        combined_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

        # Loop through each mask and add it to the combined mask with varying intensities
        num_masks = len(target[0]['masks'])
        for i in range(num_masks):
            mask = target[0]['masks'][i].numpy()
            intensity = int(255 * (i + 1) / num_masks)  # Calculate intensity based on mask index
            combined_mask[mask > 0] = intensity  # Add mask to combined mask

        # Plot the image and combined mask side by side
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        # Display the image
        axs[0].imshow(img)
        axs[0].set_title('Image')
        axs[0].axis('off')
        # Display the combined mask
        axs[1].imshow(combined_mask, cmap='gray')
        axs[1].set_title('Combined Mask')
        axs[1].axis('off')
        plt.show()

def display_masks_one_by_one(iterator):
    while True:
        try:
            # Get a batch of data
            img, target = next(iterator)
        except StopIteration:
            # If no more data, break the loop
            break
        img = img[0]  # First image in the batch
        # Convert tensor to numpy array for visualization
        img = img.permute(1, 2, 0).numpy()  # Convert from (C, H, W) to (H, W, C)
        i = 0

        for i in range(len(target[0]['masks'])):
            mask = target[0]['masks'][i]  # First mask in the batch
            mask = mask.numpy()
            # Plot the image and mask side by side
            fig, axs = plt.subplots(1, 2, figsize=(10, 5))
            # Display the image
            axs[0].imshow(img)
            axs[0].set_title('Image')
            axs[0].axis('off')
            # Display the mask
            axs[1].imshow(mask, cmap='gray')
            axs[1].set_title('Mask')
            axs[1].axis('off')
            plt.show()


if __name__ == "__main__":

    config = {}
    config['dataset_dir'] = r"/root/wool_data/AWI"
    config['num_samples'] = 3 # Here it is the number of target-lable pair to visualise
    config['batch_size'] = 1 # display_all_masks_at_once() only shows the first image in the batch

    data_loader, data_loader_test = data_loaders(
        config, T.get_transform(train=True), T.get_transform(train=False))
    
    visualise_dataloader(data_loader)
    # visualise_specific_pair(sys.argv[1])