import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch

from deeplabv3_apples.config.config import LABEL_COLOR_LIST

class MetricTracker:
    def __init__(self, save_dir, metrics=None):
        """
        Initializes the MetricTracker.

        Parameters:
            metrics (list): List of metric names to track (e.g., ['loss', 'pix_acc', 'iou', 'precision', 'recall', 'f1_score']).
            save_dir (str): Directory to save plots.
        """

        if metrics is None:
            metrics = ['loss', 'pix_acc', 'iou', 'precision', 'recall', 'f1_score']  # Default metrics

        self.metrics = {metric: {'train': [], 'val': []} for metric in metrics}
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def update(self, pred, target, loss, phase='train'):
        """
        Calculates and updates IoU, Precision, Recall, F1 score, Pixel Accuracy, and Average Loss.

        Parameters:
            pred (torch.Tensor): Predicted segmentation mask (binary or multiclass).
            target (torch.Tensor): Ground truth segmentation mask (binary or multiclass).
            loss (float): The current loss for the batch.
            phase (str): Phase of the process, either 'train' or 'val' (validation).
        """
        eps = np.spacing(1)  # Small epsilon to avoid division by zero
        pred = pred.int()  # Ensure binary integer predictions
        target = target.int()

        # True Positives (TP), False Positives (FP), and False Negatives (FN)
        TP = torch.sum((pred == 1) & (target == 1))
        FP = torch.sum((pred == 1) & (target == 0))
        FN = torch.sum((pred == 0) & (target == 1))

        # Calculate IoU
        intersection = TP
        union = torch.sum((pred == 1) | (target == 1))
        iou = (intersection + eps) / (union + eps)

        # Calculate Precision and Recall
        precision = (TP + eps) / (TP + FP + eps)
        recall = (TP + eps) / (TP + FN + eps)

        # Calculate F1 Score
        f1_score = 2 * (precision * recall) / (precision + recall + eps)

        # Pixel accuracy calculation
        correct = (pred == target).sum()  # Correctly classified pixels
        labeled = (target != 255).sum()  # Assuming 255 represents unlabeled pixels
        pix_acc = 100 * correct / (eps + labeled)

        # Update metrics for the specified phase (train or validation)
        metrics_dict = {
            "iou": iou.item(),
            "precision": precision.item(),
            "recall": recall.item(),
            "f1_score": f1_score.item(),
            "pix_acc": pix_acc.item(),
            "loss": loss.item()
        }

        for metric, value in metrics_dict.items():
            if metric in self.metrics:
                self.metrics[metric][phase].append(value)

    def reset(self):
        """
        Resets all tracked metrics.
        """
        for metric in self.metrics:
            self.metrics[metric] = {'train': [], 'val': []}

    def compute_epoch_average_metrics(self, phase):
        if phase not in ['train', 'val']:
            raise ValueError("Phase must be either 'train' or 'val'.")
        return {metric: np.mean(values[phase]) for metric, values in self.metrics.items()}

    def save_plots(self):
        """
        Saves plots for each metric after every epoch, overwriting the previous plots.
        """
        for metric, values in self.metrics.items():
            plt.figure(figsize=(10, 5))

            # Plot train and val metrics for each metric
            if 'train' in values and values['train']:
                plt.plot(values['train'], label=f'Train {metric.capitalize()}')
            if 'val' in values and values['val']:
                plt.plot(values['val'], label=f'Validation {metric.capitalize()}')

            plt.title(f'{metric.capitalize()} per Epoch')
            plt.xlabel('Epochs')
            plt.ylabel(f'{metric.capitalize()}')
            plt.legend()

            # Save and overwrite the plot in each epoch
            plot_path = os.path.join(self.save_dir, f'{metric}_plot.png')
            plt.savefig(plot_path)
            plt.close()
        print(f"Updated plots saved to {self.save_dir}")

@torch.no_grad()
def validate(model, val_loader, device, criterion, epoch, save_dir, metric_tracker):
    """
    Validate the model.

    Parameters:
        model (torch.nn.Module): The model to validate.
        val_loader (DataLoader): The validation data loader.
        device (torch.device): Device to perform validation on.
        criterion (torch.nn.Module): Loss function used.
        epoch (int): The current epoch number.
        save_dir (str): Directory to save validation results.
        metric_tracker (MetricTracker): Instance of MetricTracker to record validation metrics.

    Returns:
        dict: A dictionary with average validation metrics.
    """

    print('Validating...')
    model.eval()

    num_batches = len(val_loader)

    # Create a progress bar
    prog_bar = tqdm(val_loader, total=num_batches, desc='Validating', bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')
    
    for i, (data, target) in enumerate(prog_bar):
        labeled = (target != 255).sum()  # Count the labeled pixels (excluding 255)
        if labeled == 0:  # Check if there are no labeled pixels
            raise ValueError("No labeled pixels in the validation batch. Check the labels.")

        data, target = data.to(device), target.to(device)

        # Perform inference
        outputs = model(data)['out']

        # Save the segmentation maps at the last batch
        if i == num_batches - 1:
            draw_translucent_seg_maps(data, outputs, epoch, i, save_dir)

        # Calculate loss
        loss = criterion(outputs, target)

        # Calculate metrics
        metric_tracker.update(outputs, target, loss, phase='val')  # Update the validation metrics

        # prog_bar.set_description(desc=f"Loss: {loss.item():.4f} | PixAcc: {metrics['pix_acc']:.2f}% | IoU: {metrics['iou']:.2f}%")
        prog_bar.set_description(desc=f"Loss: {loss.item():.4f}")

    # Return average metrics for validation
    avg_val_metrics = metric_tracker.compute_epoch_average_metrics(phase='val')
    return avg_val_metrics


# def pix_acc(target, output):
#     """
#     Compute pixel accuracy.

#     Parameters:
#         target (torch.Tensor): Ground truth masks.
#         output (torch.Tensor): Model's predicted masks.

#     Returns:
#         tuple: (total labeled pixels, correctly predicted pixels)
#     """
#     with torch.no_grad():
#         preds = torch.argmax(output, dim=1)  # Predicted class per pixel
#         correct = (preds == target).sum()  # Correctly classified pixels
#         labeled = (target != 255).sum()  # Assuming 255 represents unlabeled pixels
    
#     return labeled, correct

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