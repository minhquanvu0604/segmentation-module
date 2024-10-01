import os

DATASET_PATH = '/media/quanvu/ApplesQV/APPLE_DATA'
IMAGE_PATH = os.path.join(DATASET_PATH, 'images-Fuji')
LABELS_PATH = os.path.join(DATASET_PATH, 'labels-Fuji', 'PixelLabelData_18')

CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset_splits.csv')

APPLE_LABEL = 2