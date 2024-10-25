import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch

from deeplabv3_apples.config.config import LABEL_COLOR_LIST


@torch.no_grad()
def validate(model, val_loader, device, criterion, epoch, save_dir):
    """
    Validate the model.

    Parameters:
        model (torch.nn.Module): The model to validate.
        val_loader (DataLoader): The validation data loader.
        device (torch.device): Device to perform validation on.
        criterion (torch.nn.Module): Loss function used.
        epoch (int): The current epoch number.
        save_dir (str): Directory to save validation results.

    Returns:
        tuple: (validation loss, pixel accuracy)
    """
    print('Validating...')
    model.eval()
    running_loss = 0.0
    running_correct, running_label = 0, 0

    num_batches = len(val_loader)

    # Create a progress bar
    prog_bar = tqdm(val_loader, total=num_batches, desc='Validating', bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')
    
    for i, (data, target) in enumerate(prog_bar):    
        labeled = (target != 255).sum() # Count the labeled pixels (excluding 255)
        if labeled == 0:  # Check if there are no labeled pixels
            raise ValueError("No labeled pixels in the validation batch. Check the labels.")

        data, target = data.to(device), target.to(device)

        # Perform inference
        outputs = model(data)['out']

        # Save the segmentation maps at the last batch
        if i == num_batches - 1:
            draw_translucent_seg_maps(
                data, 
                outputs, 
                epoch, 
                i, 
                save_dir)

        # Calculate loss
        loss = criterion(outputs, target)
        running_loss += loss.item()

        # Calculate pixel accuracy
        labeled, correct = pix_acc(target, outputs)
        running_label += labeled.sum()
        running_correct += correct

        # Update the progress bar with current loss and pixel accuracy
        valid_running_pixacc = 1.0 * correct / (np.spacing(1) + labeled.sum())
        prog_bar.set_description(desc=f"Loss: {loss.item():.4f} | PixAcc: {valid_running_pixacc.cpu().numpy()*100:.2f}%")
        
    # Calculate average loss and accuracy for the epoch
    valid_loss = running_loss / num_batches
    pixel_acc = ((1.0 * running_correct) / (np.spacing(1) + running_label)) * 100.

    return valid_loss, pixel_acc

def pix_acc(target, output):
    """
    Compute pixel accuracy.

    Parameters:
        target (torch.Tensor): Ground truth masks.
        output (torch.Tensor): Model's predicted masks.

    Returns:
        tuple: (total labeled pixels, correctly predicted pixels)
    """
    with torch.no_grad():
        # Extract the class with the highest probability for each pixel -> tensor of shape (B, H, W)
        preds = torch.argmax(output, dim=1) 
        
        correct = (preds == target).sum()  # Correctly classified pixels
        labeled = (target != 255).sum()  # Assuming 255 represents unlabeled pixels
    
    return labeled, correct

def draw_translucent_seg_maps(images, outputs, epoch, batch_idx, save_dir, alpha=0.6):
    """
    Draw translucent segmentation maps over the original images and save them.

    Parameters:
        images (torch.Tensor): Batch of input images, shape (batch_size, C, H, W).
        outputs (torch.Tensor): Model output segmentation maps, shape (batch_size, num_classes, H, W).
        epoch (int): Current epoch number.
        batch_idx (int): Index of the current batch.
        save_dir (str): Directory to save the images.
        alpha (float): Opacity value for the segmentation overlay (0 = fully transparent, 1 = fully opaque).
    """
    os.makedirs(save_dir, exist_ok=True)
    batch_size = images.shape[0]

    # Convert the output logits to predicted class labels
    predicted_masks = torch.argmax(outputs, dim=1).cpu().numpy()  # Shape: (batch_size, H, W)
    
    # Convert images to numpy
    images = images.permute(0, 2, 3, 1).cpu().numpy()  # Convert from (B, C, H, W) to (B, H, W, C)
    
    for idx in range(batch_size):
        # Get the original image and predicted mask for this sample
        img = (images[idx] * 255).astype(np.uint8)  # Rescale image values to [0, 255]
        pred_mask = predicted_masks[idx]  # Shape: (H, W)

        # Create a blank RGB image for the overlay
        overlay = np.zeros_like(img, dtype=np.uint8)

        # Apply color to the overlay based on the predicted mask
        for class_idx, color in enumerate(LABEL_COLOR_LIST):
            overlay[pred_mask == class_idx] = color

        # Create the translucent segmentation map by blending
        blended = cv2.addWeighted(img, 1 - alpha, overlay, alpha, 0)

        # Save the blended image
        save_path = os.path.join(save_dir, f"epoch_{epoch}_batch_{batch_idx}_sample_{idx}.png")
        plt.imsave(save_path, blended)
        print(f"Saved: {save_path}")

def save_plots(train_pix_acc, val_pix_acc, train_loss, val_loss, save_dir):
    """
    Function to save plots of training and validation accuracy and loss.

    Parameters:
        train_pix_acc (list): List of training pixel accuracies for each epoch.
        val_pix_acc (list): List of validation pixel accuracies for each epoch.
        train_loss (list): List of training losses for each epoch.
        val_loss (list): List of validation losses for each epoch.
        save_dir (str): The directory to save the plots.
    """
    # Plot loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Loss per Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'loss_plot.png'))
    plt.close()

    # Plot pixel accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(train_pix_acc, label='Train Pixel Accuracy')
    plt.plot(val_pix_acc, label='Validation Pixel Accuracy')
    plt.title('Pixel Accuracy per Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Pixel Accuracy (%)')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'pixel_accuracy_plot.png'))
    plt.close()
    print(f"Plots saved to {save_dir}")