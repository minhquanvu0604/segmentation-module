# segmentation-module
Apple Picking Project - Segmentation module to filter relevant pointcloud for reconstruction

# Installing LibTorch
Unzip and put the library inside `/opt` (should have path /opt/libtorch/share/cmake/Torch/TorchConfig.cmake)
```bash
export CMAKE_PREFIX_PATH=/opt/libtorch:$CMAKE_PREFIX_PATH
```
 

# Infer the model in Python
For fast prototyping



# Infer the model in Cpp 
Inference in Cpp so it can be integrated with MVPS nodelet system.

model_inference is a catkin package nly for easy discovery inside a catkin workspace, the package doesn't depend on ROS

model_inference/src/predict and predict_main.cpp: C++ equivalent of deeplabv3_apples/predict.py


## (POSTPONED) Making model_inference a cpp codebass without catkin - CMakeLists_non_catkin.txt
After training, model.pth is obtain, which needs to be converted to model.pt using `export.py`.

`segmentation_module_ros` is a ROS catkin package. Include only that directory in the catkin workspace. 

`model_inference` library will need to be built from source so it can be linked against by the ROS node. 
Check its CMakeLists.txt for notes

To build model_inference library and make it accessible to segmentation_module_ros 
```bash
cd segmentation-model/model_inference
mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=/absolute/path/to/segmentation-model/install ..
make
make install
```
CMAKE_INSTALL_PREFIX is a CMake variable that specifies the directory prefix where make install will install your files.

Export the path so that model_inference can be found
```bash
export CMAKE_PREFIX_PATH=/absolute/path/to/segmentation-model/install:$CMAKE_PREFIX_PATH
```
Specifies additional paths to search for packages when using find_package().

If your library is installed in a directory that's not in CMake's default search paths (like /usr/local), you need to tell CMake where to find it

# Test suite
Implement this tutorial: https://pytorch.org/tutorials/advanced/cpp_export.html

 