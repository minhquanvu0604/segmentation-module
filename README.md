# segmentation-module
Apple Picking Project - Segmentation module to filter relevant pointcloud for reconstruction


# Infer the model in Cpp
After training, model.pth is obtain, which needs to be converted to model.pt using `export.py`.

Include `segmentation_module_ros` in the ROS workspace. 

`model_inference` library will need to be built from source so it can be linked against by the ROS node.