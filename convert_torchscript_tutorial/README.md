# Loading a TorchScript Model in C++
https://pytorch.org/tutorials/advanced/cpp_export.html

Building 
```bash
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=/opt/libtorch ..
cmake --build . --config Release
```

Loading model to C++ 
```bash
./example-app /home/quanvu/ros/apple_ws/src/segmentation_module/test/converted_resnet_model.pt
```