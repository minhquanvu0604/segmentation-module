import os
import yaml
train_config = 'train_config.yaml'
segmentation_module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

INPUT_SIZE = (800, 800) # (520, 520)
APPLE_LABEL = 2


TRAIN_CONFIG = os.path.join('config', train_config)
LABEL_COLOR_LIST = [
    (0, 0, 0),    # Background
    (255, 0, 0)   # Apples
]

# DATASET_PATH = '/data/minhqvu/APPLE_DATA'
# IMAGE_PATH = os.path.join(DATASET_PATH, 'synthesised_images')
# LABELS_PATH = os.path.join(DATASET_PATH, 'labels-Fuji', 'PixelLabelData_18')
# CSV_PATH = os.path.join(segmentation_module_path, 'data', 'dataset_splits_with_synth.csv')

DATASET_PATH = '/root/APPLE_DATA'
IMAGE_PATH = os.path.join(DATASET_PATH, 'images-Fuji')
LABELS_PATH = os.path.join(DATASET_PATH, 'labels-Fuji', 'PixelLabelData_18')
CSV_PATH = os.path.join(segmentation_module_path, 'data', 'dataset_splits.csv')