# Segmentation Module
**A module of my capstone - Automation in Apple Picking**

Segmentation module to filter relevant pointcloud for reconstruction. Enhance the MVPS Segmentation Module by incorporating deep learning techniques for ROI detection, improving the accuracy
and efficiency of apple segmentation.

As we aimed to integrate the model into the nodelet system, an inference loop in C++ programming language is desired. LibTorch is selected as the C++ implementation for model inference - [set up guide](README.md#installing-libtorch).


Before and After Filtering:
<p align="center">
  <img height="400" alt="Before Filter" src="wiki\image\2024-11-01_15-44.png">
  <img height="400" alt="After Filter" src="wiki\image\2024-11-01_15-45.png">
</p>

# DeepLabV3 Instance Segmentation
<p align="center">
  <img width="65%" alt="Inference" src="wiki\image\inference_1.png">
</p>


# Installing LibTorch
Unzip and put the library inside `/opt` (should have path /opt/libtorch/share/cmake/Torch/TorchConfig.cmake)
```bash
export CMAKE_PREFIX_PATH=/opt/libtorch:$CMAKE_PREFIX_PATH
```
 
# Inference on Images
Use predict.py

# Inference Node in Cpp 
Inference in C++ so it can be integrated with MVPS nodelet system.

After training, model.pth is obtained, which needs to be converted to model.pt using `export.py`.

model_inference/src/predict and predict_demo.cpp: C++ equivalent of deeplabv3_apples/predict.py

The `model_inference` library needs to be compiled from source and installed separately because it is **not dependent 
on ROS**, hence not a catkin package (check CMakeLists.txt). 

It will be linked against by the ROS node that does model inference. `cmake/model_inferenceConfig.cmake.in` helps the package to be discovered by find_package(). 
Also it's path is set by setting CMake variables CMAKE_PREFIX_PATH equal to the virtual environment variable of the same name, which is set manually.
CMAKE_INSTALL_PREFIX is a CMake variable that specifies the directory prefix where make install will install your files.

```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export CMAKE_PREFIX_PATH=/opt/libtorch:$CMAKE_PREFIX_PATH
```
To build model_inference library and make it accessible to segmentation_module_ros 
```bash
cd segmentation-model/model_inference
mkdir build && cd build
cmake ..
make
make install
```
CMAKE_PREFIX_PATH can also be set in the cmake command: 
```bash
cmake -DCMAKE_INSTALL_PREFIX=/absolute/path/to/segmentation-model/install ..
```
Export the path so that model_inference can be found
```bash
export CMAKE_PREFIX_PATH=/absolute/path/to/segmentation-model/install:$CMAKE_PREFIX_PATH
```
Specifies additional paths to search for packages when using find_package() 
(if your library is installed in a directory that's not in CMake's default search paths (like /usr/local), you need to tell CMake where to find it)

# Inference Node in Python 
For fast prototyping - [Skipped]

# Test suite
Implement this tutorial: https://pytorch.org/tutorials/advanced/cpp_export.html

# Singularity 
```bash
singularity pull docker://minhquanvu0604/apple_image:latest
singularity shell --nv --bind /data/minhqvu:/data/minhqvu apple_image_latest.sif
```