cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(model_inference)

# Find LibTorch (PyTorch C++ API)
find_package(Torch REQUIRED)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include directories
include_directories(include)

# Add the library
add_library(model_inference SHARED src/model_inference.cpp)

# Link against LibTorch
target_link_libraries(model_inference "${TORCH_LIBRARIES}")

# Ensure RPATH is set correctly (optional but recommended)
set_property(TARGET model_inference PROPERTY CXX_STANDARD 14)
set_property(TARGET model_inference PROPERTY CXX_STANDARD_REQUIRED ON)

# To ensure that the library is built with the same flags as LibTorch
set_property(TARGET model_inference PROPERTY POSITION_INDEPENDENT_CODE ON)

# Optional: Install targets
install(TARGETS model_inference
        LIBRARY DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)