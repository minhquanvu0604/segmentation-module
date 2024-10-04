import os, sys
top_level_package = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) # segmentation-module
sys.path.insert(0, top_level_package)
import torch
from deeplabv3_apples.predict import load_model


def export_model(model_path, num_classes, output_path):
    model = load_model(model_path, num_classes)
    model.to('cpu')
    scripted_model = torch.jit.script(model)
    scripted_model.save(output_path)


if __name__ == '__main__':

    model_path = '/root/segmentation-module/deeplabv3_apples/output/2024_10_02_23_37_35/model.pth'
    output_path = '/root/segmentation-module/deeplabv3_apples/output/2024_10_02_23_37_35/model.pt'
    num_classes = 2

    export_model(model_path, num_classes, output_path)
