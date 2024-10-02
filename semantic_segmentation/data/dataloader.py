import os
import random
import numpy as np
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
        This follows the specific structure of the dataset provided for the project, and its naming conventions.

        Parameters:
            split_set (str): The type of split (column in the CSV file).
            split (str): The dataset split to load ('train' or 'val').
            transforms (callable, optional): A function/transform to apply to the images and masks.
        """
        self.images_dir = IMAGE_PATH
        self.labels_dir = LABELS_PATH        
        self.transforms = transforms
        self.split = split

        # Load the Excel file
        self.df = pd.read_csv(CSV_PATH)

        # Filter rows based on the split argument
        self.image_names = self.df[self.df[split_set].str.startswith(self.split, na=False)]['Image Name'].tolist()
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
        image_name = self.image_names[idx] # Naming convention: Image_1.jpeg
        image_path = os.path.join(self.images_dir, image_name)
        image = Image.open(image_path).convert("RGB")

        # Load the corresponding label mask
        label_name = image_name.replace('Image', 'Label') # Naming convention: Label_1.png
        label_name = label_name.replace('.jpg', '.png')
        label_path = os.path.join(self.labels_dir, label_name)
        label = np.array(Image.open(label_path))
        label = (label == APPLE_LABEL).astype(np.uint8) # Set apple pixels as 1, others as 0
        label = Image.fromarray(label)

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
    print('Loading TRAINING data')
    train_dataset = AppleDataset(split_set=split_set, split='train', transforms=train_transforms)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True  # Speeds up data transfer to GPU
    )
    print("Loaded {} training images".format(len(train_dataset)))

    print('Loading VALIDATION data')
    val_dataset = AppleDataset(split_set=split_set, split='val', transforms=val_transforms)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # No need to shuffle validation data
        num_workers=num_workers,
        pin_memory=True
    )
    print("Loaded {} validation images".format(len(val_dataset)))
    return train_loader, val_loader

class JointTransform:
    """
    Custom transformation class that handles transformations for both the input image and label mask.
        
    Parameters:
        input_size (tuple): Desired input size (height, width) for resizing the image and mask.
        is_train (bool): Indicates whether the transformation is for training (includes augmentations) or validation/testing.
    """
    
    def __init__(self, input_size, is_train=True):
        self.input_size = input_size
        self.is_train = is_train
        
        # Augmentations that should apply only to training data
        self.augmentations = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),    # Randomly flip images horizontally
            transforms.RandomRotation(degrees=15),     # Randomly rotate images by +/-15 degrees
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Random changes in brightness, contrast, etc.
            transforms.RandomGrayscale(p=0.1),         # Randomly convert images to grayscale with a probability of 0.1
            transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),  # Apply Gaussian blur
        ])
        
        # Transformations to be applied to both train and validation data
        # For Resize and Normalize, we use the same values as the ones used for training DeepLabV3 by PyTorch, refer to:
        # https://pytorch.org/vision/stable/models/generated/torchvision.models.segmentation.deeplabv3_resnet50.html#torchvision.models.segmentation.deeplabv3_resnet50
        self.common_transforms = transforms.Compose([
            transforms.Resize(input_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, image, mask):
        # Apply augmentations only if training
        if self.is_train:
            seed = random.randint(0, 99999)  # Seed ensures same transformation is applied to both image and mask
            random.seed(seed)
            image = self.augmentations(image)
            random.seed(seed)
            mask = self.augmentations(mask)
        
        # Apply the common transformations to the image
        image = self.common_transforms(image)
        
        # Resize the mask and convert it to a tensor without normalization
        mask = transforms.Resize(self.input_size, interpolation=transforms.InterpolationMode.NEAREST)(mask)
        mask = torch.tensor(np.array(mask), dtype=torch.long)  # Keep mask as long type
        return image, mask

def get_transforms(input_size):
    """
    Creates transformation pipelines for training and validation datasets.

    Parameters:
        input_size (tuple): Desired input size (height, width) for the model.

    Returns:
        train_transforms (callable): Transformations to apply to the training data.
        val_transforms (callable): Transformations to apply to the validation data.
    """ 
    train_transforms = JointTransform(input_size, is_train=True)
    val_transforms = JointTransform(input_size, is_train=False)
    
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