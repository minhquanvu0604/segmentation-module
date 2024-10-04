import pandas as pd
import os
import random

from deeplabv3_apples.config.config import CSV_PATH, IMAGE_PATH
RATIO = (0.7, 0.15, 0.15)


class DatasetSplitter:
    def __init__(self, default_ratio=None, csv_path=CSV_PATH, image_path=IMAGE_PATH):
        """
        Initialize the DatasetSplitter class.

        Parameters:
            CSV_PATH (str): Path to the CSV file.
            default_ratio (tuple): Default train/val/test ratio for random splitting.
        """
        self.csv_path = csv_path
        self.default_ratio = default_ratio if default_ratio is not None else RATIO

        # Check if the CSV file exists; if not, create a new one
        if not os.path.exists(self.csv_path):
            print(f"CSV file does not exist. Creating a new one at {self.csv_path}.")
            self.df = self._create_initial_csv(image_path)
        else:
            self.df = pd.read_csv(self.csv_path)
            if 'Image Name' not in self.df.columns:
                raise ValueError("The existing CSV file must contain an 'Image Name' column.")
            
            self._check_image_consistency(image_path)

    def _create_initial_csv(self, image_path):
        """ 
        Create a new CSV with all image names from the dataset folder. 
        """
        image_names = self._retrieve_image_names(image_path)
        df = pd.DataFrame(image_names, columns=['Image Name'])
        df.to_csv(self.csv_path, index=False)
        return df
    
    def _retrieve_image_names(self, folder_path):
        """
        Retrieve all image names from the dataset folder.

        Parameters:
            folder_path (str): Path to the dataset folder.

        Returns:
            list: A list of image file names.
        """
        # Assuming that your dataset contains images with extensions like .png, .jpg, etc.
        valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
        image_names = [f for f in os.listdir(folder_path)
                       if os.path.isfile(os.path.join(folder_path, f)) and os.path.splitext(f)[1].lower() in valid_extensions]
        
        # Images are named Image_1.jpg, Image_2.jpg, ..., Image_10.jpg, etc.
        def natural_sort_key(s):
            import re
            return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]
        
        return sorted(image_names, key=natural_sort_key)
    
    def _check_image_consistency(self, images_path):
        """
        Check for consistency between images in the directory and entries in the existing CSV.

        Parameters:
        - images_path (str): Path to the directory containing the images.
        """
        # Retrieve all image names from the directory
        actual_images = set(self._retrieve_image_names(images_path))
        
        # Retrieve image names from the CSV
        csv_images = set(self.df['Image Name'])

        # Find missing and extra images
        missing_images = csv_images - actual_images
        extra_images = actual_images - csv_images

        # Report the discrepancies
        if missing_images:
            raise ValueError(f"Missing images in the directory (present in CSV but not found): {missing_images}")
        if extra_images:
            raise ValueError(f"Extra images in the directory (found but not present in CSV): {extra_images}")
        
        print("CSV and image directory are consistent.")

    def _get_split_name(self, split_type: str):
        """ 
        Generate a unique column name based on existing split columns. 
        """
        existing_columns = [col for col in self.df.columns if col.startswith(split_type)]
        if not existing_columns:
            return split_type
        else:
            # Taking the substring of col starting from the len(split_type) character onwards
            numbers = [int(col[len(split_type):]) for col in existing_columns if col[len(split_type):].isdigit()]
            next_number = max(numbers) + 1 if numbers else 2
            return f"{split_type}{next_number}"

    def random_split(self, split_name=None, ratio=None):
        """
        Perform a random split of the dataset and add it to the CSV.

        Parameters:
            split_name (str): The base name for the split configuration.
            ratio (tuple): The train/val/test ratio to use for the split.
        """
        if ratio is None:
            ratio = self.default_ratio
        
        # Ensure the ratio sums to 1
        assert sum(ratio) == 1, "The split ratio must sum to 1."

        if split_name is None:
            split_name = 'random_split'

        split_name = self._get_split_name(split_name)

        # Shuffle the image names and split them according to the ratio
        image_names = self.df['Image Name'].tolist()
        random.shuffle(image_names)
        
        num_images = len(image_names)
        train_end = int(ratio[0] * num_images)
        val_end = train_end + int(ratio[1] * num_images)

        splits = ['train'] * train_end + ['val'] * (val_end - train_end) + ['test'] * (num_images - val_end)

        # Map splits to the image names
        split_map = dict(zip(image_names, splits))
        
        # Add the new split column to the DataFrame
        self.df[split_name] = self.df['Image Name'].map(split_map)
        self.df.to_csv(self.csv_path, index=False)
        print(f"Random split added to the CSV as column '{split_name}'.")

    def custom_split(self, split_name, split_mapping):
        """
        Add a custom split configuration to the CSV.

        Parameters:
            split_name (str): The base name for the split configuration.
            split_mapping (dict): Dictionary mapping image names to 'train', 'val', or 'test'.
        """
        split_name = self._get_split_name(split_name)
        
        # Map the split configuration to the DataFrame
        self.df[split_name] = self.df['Image Name'].map(split_mapping).fillna('unassigned')
        self.df.to_csv(self.csv_path, index=False)
        print(f"Custom split added to the CSV as column '{split_name}'.")

    def get_split(self, split_name):
        """ 
        Retrieve the list of image paths for a given split configuration. 
        """
        if split_name not in self.df.columns:
            raise ValueError(f"No split configuration found with name '{split_name}'.")
        
        return {
            'train': sorted(self.df[self.df[split_name] == 'train']['Image Name'].tolist()),
            'val': sorted(self.df[self.df[split_name] == 'val']['Image Name'].tolist()),
            'test': sorted(self.df[self.df[split_name] == 'test']['Image Name'].tolist()),
        }


if __name__ == "__main__":
    # Initialize the class with the path to the CSV file
    splitter = DatasetSplitter()

    # Create a random split with the default ratio
    splitter.random_split()

    # Retrieve the 'random_split' split configuration
    split_data = splitter.get_split('random_split')
    # print(split_data)
    print(f"Number of training images: {len(split_data['train'])}")
    print(f"Number of validation images: {len(split_data['val'])}")
    print(f"Number of test images: {len(split_data['test'])}")
