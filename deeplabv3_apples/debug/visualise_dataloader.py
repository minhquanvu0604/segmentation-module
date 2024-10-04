import sys, os
top_level_package = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, top_level_package)

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from deeplabv3_apples.data.dataloader import get_transforms, get_dataloaders


def visualise_dataloader(data_loader):
    """
    Visualize the data loader output by displaying a randomly picked image and its masks side by side.
    This helps in verifying that the data loader side, inspecting downstream to the dataset's synthesised images and their labels.

    Press q to move on to the next mask of the same image.
    """
    iterator = iter(data_loader)
    display_all_masks_at_once(iterator)

def display_all_masks_at_once(iterator, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], class_colors=None):
    """
    Displays the images and corresponding masks from an iterator.
    
    Parameters:
        iterator: DataLoader iterator that provides (image, mask) tuples.
        mean, std: Mean and standard deviation for denormalizing the image.
        class_colors: List of RGB tuples for coloring the segmentation masks by class.
    """
    def denormalize(img, mean, std):
        """Denormalizes the image for visualization."""
        img = img.clone()  # Clone the tensor to avoid modifying the original
        for i in range(3):  # Assuming the image has 3 channels (RGB)
            img[i] = img[i] * std[i] + mean[i]
        return img

    plt.ion()  # Turn on interactive mode
    try:
        while True:
            try:
                # Get a batch of data (assuming a single image per batch)
                img, target = next(iterator)
            except StopIteration:
                # If no more data, break the loop
                break

            img = img[0]  # First image in the batch
            target = target[0]  # First mask in the batch
            
            # Denormalize the image for visualization
            img = denormalize(img, mean, std)
            # Convert tensor to numpy array for visualization
            img = img.permute(1, 2, 0).cpu().numpy()  # Convert from (C, H, W) to (H, W, C)
            img = np.clip(img, 0, 1)  # Clip values to [0, 1] range for display

            # Convert the target tensor (mask) to numpy
            target = target.cpu().numpy()  # Shape: (H, W)
            
            if class_colors:
                # If class_colors is provided, map the mask to a colored version
                combined_mask = np.zeros((*target.shape, 3), dtype=np.uint8)  # Create an RGB image for the mask
                for class_idx, color in enumerate(class_colors):
                    combined_mask[target == class_idx] = color
            else:
                # Use grayscale mask if no colors are provided
                combined_mask = target  # Use the single mask directly as there are no multiple masks to combine

            # Plot the image and combined mask side by side
            fig, axs = plt.subplots(1, 2, figsize=(10, 5))
            # Display the image
            axs[0].imshow(img)
            axs[0].set_title('Image')
            axs[0].axis('off')
            # Display the mask
            if class_colors:
                axs[1].imshow(combined_mask)
            else:
                axs[1].imshow(combined_mask, cmap='gray')
            axs[1].set_title('Segmentation Mask')
            axs[1].axis('off')
            plt.draw()
            plt.pause(0.001)  # Pause to allow the figure to be drawn
            input("Press Enter to continue, or Ctrl+C to exit...")
            plt.close(fig)
    except KeyboardInterrupt:
        print("Interrupted by user.")
        plt.ioff()
        return


if __name__ == "__main__":

    config = {
        'split': 'random_split',
        'batch_size': 1
    }

    # Retrieve the transformations
    train_transforms, val_transforms = get_transforms(input_size=(520, 520))

    # Use them when creating DataLoader objects
    train_loader, val_loader = get_dataloaders(split_set=config['split'],
                                            batch_size=config['batch_size'],
                                            train_transforms=train_transforms,
                                            val_transforms=val_transforms)
    visualise_dataloader(train_loader)