import os

import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# Internal imports
from data import CSV_PATH, IMAGE_PATH, LABELS_PATH, APPLE_LABEL


class AppleDataset(Dataset):
    def __init__(self, split_set, split, transforms=None):
        """
        Initialize the AppleDataset class.

        Parameters:
            split (str): The dataset split to load
            transforms (callable, optional): A function/transform to apply to the images and masks.
        """
        self.images_dir = IMAGE_PATH
        self.labels_dir = LABELS_PATH        
        self.transforms = transforms
        
        # Load the Excel file
        self.df = pd.read_csv(CSV_PATH)

        # Filter rows based on the split argument
        self.image_names = self.df[self.df[split_set].str.startswith(split, na=False)]['Image Name'].tolist()
        print(f"Loaded {len(self.image_names)} images for the {split_set} split.")

    def __len__(self):
        """
        Returns the total number of images for the given split.
        """
        return len(self.image_names)

    def __getitem__(self, idx):
        """
        Retrieve the image and its corresponding label mask.

        Parameters:
            idx (int): Index of the item to retrieve.

        Returns:
            tuple: A tuple containing the image tensor and target mask tensor.
        """
        # Get the image name
        image_name = self.image_names[idx]

        # Load the image
        image_path = os.path.join(self.images_dir, image_name)
        raise ValueError(image_path)
        for ext in ['.jpg', '.png']:
            possible_path = image_path + ext
            if os.path.isfile(possible_path):
                image_path = possible_path
                break
        raise ValueError(possible_path)
        image = Image.open(image_path).convert("RGB")

        # Load the corresponding label mask
        label_path = os.path.join(self.labels_dir, image_name)
        label = Image.open(label_path)

        # Convert the label image to a binary mask
        label = (torch.tensor(label) == APPLE_LABEL).type(torch.long)  # Set apple pixels as 1, others as 0

        # Apply any transformations if provided
        if self.transforms:
            image, label = self.transforms(image, label)

        return image, label
    

def get_dataloaders(split_set, batch_size, num_workers=4, train_transforms=None, val_transforms=None):
    """
    Creates DataLoader objects for training and validation datasets.

    Parameters:
        batch_size (int): Number of samples per batch.
        num_workers (int): Number of subprocesses to use for data loading.
        train_transforms (callable, optional): Transformations to apply to the training data.
        val_transforms (callable, optional): Transformations to apply to the validation data.

    Returns:
        train_loader (DataLoader): DataLoader for the training dataset.
        val_loader (DataLoader): DataLoader for the validation dataset.
    """
    train_dataset = AppleDataset(split_set=split_set, split='train', transforms=train_transforms)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True  # Speeds up data transfer to GPU
    )

    # Create the validation dataset and DataLoader
    val_dataset = AppleDataset(split_set=split_set, split='val', transforms=val_transforms)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # No need to shuffle validation data
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader

def get_transforms(input_size):
    """
    Creates transformation pipelines for training and validation datasets.

    Parameters:
        input_size (tuple): Desired input size (height, width) for the model.

    Returns:
        train_transforms (callable): Transformations to apply to the training data.
        val_transforms (callable): Transformations to apply to the validation data.
    """
    # Define transformations for the training set
    # For Resize and Normalize, we use the same values as the ones used for training DeepLabV3 by PyTorch, refer to:
    # https://pytorch.org/vision/stable/models/generated/torchvision.models.segmentation.deeplabv3_resnet50.html#torchvision.models.segmentation.deeplabv3_resnet50
    train_transforms = transforms.Compose([
        transforms.Resize(input_size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.RandomHorizontalFlip(p=0.5),     # Randomly flip images horizontally
        transforms.RandomRotation(degrees=15),      # Randomly rotate images by +/-15 degrees
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Random changes in brightness, contrast, saturation, hue
        transforms.RandomGrayscale(p=0.1),          # Randomly convert images to grayscale with a probability of 0.1
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),  # Random Gaussian blur
        transforms.ToTensor(),                      # Convert to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], # Normalize with ImageNet statistics
                             std=[0.229, 0.224, 0.225])
    ])

    # Define transformations for the validation set
    val_transforms = transforms.Compose([
        transforms.Resize(input_size),              # Resize validation images to the consistent input size
        transforms.ToTensor(),                      # Convert to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])  # Normalize with ImageNet statistics
    ])
    return train_transforms, val_transforms


if __name__ == '__main__':
    # Retrieve the transformations
    train_transforms, val_transforms = get_transforms(input_size=(520, 520))

    # Use them when creating DataLoader objects
    train_loader, val_loader = get_dataloaders(
        batch_size=16, 
        num_workers=4, 
        train_transforms=train_transforms, 
        val_transforms=val_transforms
    )