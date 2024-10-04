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
with open(os.path.join(os.path.dirname(__file__), train_config), 'r') as file:
    config = yaml.safe_load(file)
DATASET_PATH = config['dataset_path']

IMAGE_PATH = os.path.join(DATASET_PATH, 'images-Fuji')
LABELS_PATH = os.path.join(DATASET_PATH, 'labels-Fuji', 'PixelLabelData_18')
CSV_PATH = os.path.join(segmentation_module_path, 'data', 'dataset_splits.csv')