#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Functions for loading data and masks for the dataset. """

import os
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset as BaseDataset
from torch.utils.data import DataLoader

import utils


# Definition of the splits for training and testing for all datasets
DATA_GROUPS_ = {
    '0': [],
    '1': ['set01'],
    '2': ['set02'],
    '3': ['set03'],
    '4': ['set04'],
    '5': ['set05']
}

# Define images to reject
REJECT_LIST_ = {
    #'train': ['001141.png', '005117.png', '005462.png', '007572.png',
    #          '008156.png', '009440.png', '009937.png'],
    'train': None,
    'test': None,
    'validation': None
}

IMAGE_DIRECTORY_ = 'images'
MASK_DIRECTORY_ = 'masks/instances'


class FibreDataset(BaseDataset):
    def __init__(self, path_to_dataset, split_list=None, reject_list=None, transform=None, num_samples=-1, has_targets=True):
        self._path_to_dataset = path_to_dataset
        self._split_list = split_list
        self.reject_list_ = set(reject_list) if reject_list is not None else None
        self._transform = transform
        self._num_samples = num_samples
        self._has_targets = has_targets
        self._file_type = '.png'

        # Load all image files
        self._images, self._filenames = self._get_filenames()

    def _get_filenames(self):
        """ Get all the file names in the specified dataset directory.

            Returns
            -------
            list
                List of tuples (image, mask). Each tuple stores the absolute path to an image and the path to the
                mask image that has each pixel labelled with an integer representing the class id.
        """

        # Store the images in a list
        images = []
        filenames_stored = []

        # If split list is None, then assume the data directory stores images
        if not self._has_targets or self._split_list is None:
            self._has_targets = False
            path_to_images = os.path.join(self._path_to_dataset, IMAGE_DIRECTORY_)
            filenames = sorted(
                [f for f in os.listdir(path_to_images)
                 if os.path.isfile(os.path.join(path_to_images, f)) and f.endswith(self._file_type)]
            )
            for f in filenames:
                # Append to the list of images
                images.append((os.path.join(path_to_images, f), None))
                filenames_stored.append(f)
        # Otherwise, this is a dataset with ground truth masks
        else:
            # Get all images within the named directories
            for subdir in self._split_list:
                # Data assumed to be organised into "images" and "masks" directories, where each mask file has the same
                # file name as the RGB image
                path_to_images = os.path.join(self._path_to_dataset, subdir, IMAGE_DIRECTORY_)
                path_to_masks = os.path.join(self._path_to_dataset, subdir, MASK_DIRECTORY_)

                # Get all files in the directory
                filenames = sorted([f for f in os.listdir(path_to_images)
                                    if os.path.isfile(os.path.join(path_to_images, f)) and f.endswith(self._file_type)])
                # Add each filename to the list of images
                for f in filenames:
                    if self.reject_list_ is None or f not in self.reject_list_:
                        mask_path = os.path.join(path_to_masks, f)
                        image_path = os.path.join(path_to_images, f)
                        # Verify that a corresponding mask exists
                        if os.path.isfile(mask_path):
                            images.append((image_path, mask_path))
                            filenames_stored.append(f)
                        else:
                            raise FileNotFoundError(f'No matching mask file for image: {image_path}')

            # # Data assumed to be organised into "images" and "masks" directories, where each mask file has the same
            # # file name as the RGB image
            # path_to_images = os.path.join(self._path_to_dataset, IMAGE_DIRECTORY_)
            # path_to_masks = os.path.join(self._path_to_dataset, MASK_DIRECTORY_)
            # # Get all files in the directory
            # filenames = sorted([f for f in os.listdir(path_to_images)
            #                     if os.path.isfile(os.path.join(path_to_images, f)) and f.endswith(self._file_type)])
            # # Add each filename to the list of images
            # for f in filenames:
            #     if self.reject_list_ is None or f not in self.reject_list_:
            #         images.append((os.path.join(path_to_images, f), os.path.join(path_to_masks, f)))
            #         filenames_stored.append(f)

        # Select samples if less than the total is requested
        if self._num_samples > 0:
            assert self._num_samples < len(images), 'Requested number of samples is more than number of samples available'
            images = images[:self._num_samples]
            filenames_stored = filenames_stored[:self._num_samples]

        # Return the images and masks
        return images, filenames_stored

    def get_filename(self, i):
        return self._filenames[i]

    def __getitem__(self, i):
        """ Fetch data element i.

            Parameters
            ----------
            i : int
                Index of the element to retrieve.

            Returns
            -------
            tuple
                Tuple of two tensors for the image and the corresponding ground truth target (dictionary).
        """

        # Load image and mask
        img_path = self._images[i][0]
        mask_path = self._images[i][1]

        # Load image as RGB
        # Catch the error loading image because error encountered when using HPC
        # try:
        #     img = Image.open(img_path).convert('RGB')
        # except Exception as e:
        #     print(f"Error loading image {img_path}: {e}")
        #     exit(1)
        
        img = Image.open(img_path).convert('RGB')
        image_id = torch.tensor([i])

        # Create the target
        target = {}
        target['image_id'] = image_id

        # Load the mask if it is available
        if mask_path is not None:
            mask = Image.open(mask_path)
            # Not converted to RGB because each color corresponds to a different instance with 0 being background

            mask = np.array(mask)
            # instances are encoded as different colors
            obj_ids = np.unique(mask)
            # first id is the background, so remove it
            obj_ids = obj_ids[1:]

            # split the color-encoded mask into a set
            # of binary masks
            masks = mask == obj_ids[:, None, None]

            # get bounding box coordinates for each mask
            num_objs = len(obj_ids)
            boxes = []
            for i in range(num_objs):
                pos = np.where(masks[i])
                xmin = np.min(pos[1])
                xmax = np.max(pos[1])
                ymin = np.min(pos[0])
                ymax = np.max(pos[0])
                boxes.append([xmin, ymin, xmax, ymax])

            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            # there is only one class
            labels = torch.ones((num_objs,), dtype=torch.int64)
            masks = torch.as_tensor(masks, dtype=torch.uint8)

            if num_objs == 0:
                area = torch.zeros((num_objs,), dtype=torch.float32)
            else: 
                area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            # suppose all instances are not crowd
            iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

            target['boxes'] = boxes
            target['labels'] = labels
            target['masks'] = masks
            target['area'] = area
            target['iscrowd'] = iscrowd
            target['mask_image'] = torch.as_tensor(mask, dtype=torch.uint8)

        if self._transform is not None:
            try:
                img, target = self._transform(img, target)
            except:
                # TODO logging if can't transform
                print(self.get_filename(i))

        return img, target

    def __len__(self):
        return len(self._images)

    def has_targets(self):
        return self._has_targets


def data_loaders(config, tr_transform, val_transform):
    """ 
    Create the training and validation data loaders.

    Parameters
    ----------
    config : dict
        Dictionary of configuration parameters.
    tr_transform : transform
        Transform function for the training data loader.
    val_transform : transform
        Transform function for the validation data loader.

    Returns
    -------
    tuple
        Tuple with the training and validation data loaders.
    """
    
    # MODIFIED: adapt to new dataset structure --------------------------------
    # ----------------- IMPORTANT NOT TO MISTAKE ---------------###############
    dataset_folder = 'synthetic/full' # 'no_blending'
    print('Dataset folder: {}'.format(dataset_folder))

    # -------------------------------------------------------------------------
    # New datset structure of train, test, validation
    if config['fold'] == -1:
        # Load the datasets for training and validation
        print('Loading TRAINING data')
        train_set = FibreDataset(
            config['dataset_dir'], split_list=[dataset_folder], reject_list=REJECT_LIST_['train'],
            transform=tr_transform, num_samples=config['num_samples'])
        print('Loaded {} samples'.format(len(train_set)))

        print('Loading VALIDATION data')
        valid_set = FibreDataset(
            config['dataset_dir'], split_list=['testing_real_reorganised_with_location'], reject_list=REJECT_LIST_['validation'],
            transform=val_transform)
        print('Loaded {} samples'.format(len(valid_set)))
    else:
        # Load the datasets for training and validation
        print('Loading TRAINING data')
        train_set = FibreDataset(
            config['dataset_dir'], split_list=get_all_folds_except(str(config['fold']), DATA_GROUPS_),
            transform=tr_transform)
        print('Loaded {} samples'.format(len(train_set)))

        print('Loading VALIDATION data')
        if config['fold'] == 0:
            root_dir, sub_dir = os.path.split(os.path.normpath(config['test_dir']))
            valid_set = FibreDataset(root_dir, split_list=[sub_dir], transform=val_transform)
        else:
            valid_set = FibreDataset(
                config['dataset_dir'], split_list=DATA_GROUPS_[str(config['fold'])],
                transform=val_transform)
        print('Loaded {} samples'.format(len(valid_set)))

    # Data loaders
    train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True, collate_fn=utils.collate_fn)
    valid_loader = DataLoader(valid_set, batch_size=1, shuffle=False, collate_fn=utils.collate_fn)

    return train_loader, valid_loader


def test_loaders(config, transform):
    """ Create the training and validation data loaders.

        Parameters
        ----------
        config : dict
            Dictionary of configuration parameters.
        transform : transform
            Transform function for the test data loader.

        Returns
        -------
        tuple
            The test data loader.
    """

    # New dataset structure of train, test, validation
    print('Loading TESTING data')
    has_targets = True
    if config['fold'] == -1:
        # Load the dataset for new format
        root_dir = config['dataset_dir']
        split_list = ['test']
        reject_list = REJECT_LIST_['test']
        if not os.path.isdir(os.path.join(config['dataset_dir'], split_list[0], MASK_DIRECTORY_)):
            root_dir = config['dataset_dir']
            split_list = None
            reject_list = None
            has_targets = False
    else:
        # Load the dataset in original format
        root_dir = config['dataset_dir']
        if config['fold'] == 0:
            root_dir, sub_dir = os.path.split(os.path.normpath(root_dir))
            split_list = [sub_dir]
        else:
            image_dir = os.path.join(root_dir, DATA_GROUPS_[str(config['fold'])][0], IMAGE_DIRECTORY_)
            mask_dir = os.path.join(root_dir, DATA_GROUPS_[str(config['fold'])][0], MASK_DIRECTORY_)
            if os.path.exists(image_dir) and os.path.exists(mask_dir):
                split_list = DATA_GROUPS_[str(config['fold'])]
            else:
                split_list = None
        reject_list = None

    test_set = FibreDataset(root_dir, split_list=split_list, reject_list=reject_list, transform=transform,
                            has_targets=has_targets)
    print('Loaded {} samples'.format(len(test_set)))

    # Data loaders
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, collate_fn=utils.collate_fn)

    return test_loader


def get_all_folds_except(k, data_groups):
    """ 
    Get a flattened list from a group of lists (stored as a dictionary) except the kth list in that set.

    Parameters
    ----------
    k : int
        Index to exclude.
    data_groups : dict
        Dictionary of different lists.

    Returns
    -------
    list
        Flattened list of the elements in data_groups without the elements of the kth group.
    """
    fold_dirs = []
    for d in data_groups:
        if d != k:
            fold_dirs.extend(data_groups[d])
    return fold_dirs
