import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image

from data import CSV_PATH, IMAGE_PATH, LABELS_PATH, APPLE_LABEL


class AppleDataset(Dataset):
    def __init__(self, split, transforms=None):
        """
        Initialize the AppleDataset class.

        Parameters:
            split (str): The dataset split to load
            transforms (callable, optional): A function/transform to apply to the images and masks.
        """
        self.images_dir = IMAGE_PATH
        self.labels_dir = LABELS_PATH        
        self.split = split
        self.transforms = transforms
        
        # Load the Excel file
        self.df = pd.read_excel(CSV_PATH)

        # Filter rows based on the split argument
        self.image_names = self.df[self.df[split].str.startswith('train', na=False)]['Image Name'].tolist()
        print(f"Loaded {len(self.image_names)} images for the {split} split.")

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
            tuple: A tuple containing the image tensor and target dictionary.
        """
        # Get the image name
        image_name = self.image_names[idx]

        # Load the image
        image_path = os.path.join(self.images_dir, image_name)
        image = Image.open(image_path).convert("RGB")

        # Load the corresponding label mask
        label_path = os.path.join(self.labels_dir, image_name)
        label = Image.open(label_path)

        # Convert the label image to a binary mask
        label = (torch.tensor(label) == APPLE_LABEL).type(torch.uint8)

        # Create the target dictionary
        target = {
            'image_id': torch.tensor([idx]),
            'masks': label.unsqueeze(0),  # Adding an extra dimension for compatibility
            'labels': torch.tensor([1]),  # Assuming one class of interest with label 1
            'boxes': torch.tensor([[0, 0, label.size(1), label.size(0)]], dtype=torch.float32),  # Bounding box coordinates
            'area': torch.tensor([(label.size(0) * label.size(1))]),
            'iscrowd': torch.tensor([0])
        }

        # Apply any transformations if provided
        if self.transforms:
            image, target = self.transforms(image, target)

        return image, target
