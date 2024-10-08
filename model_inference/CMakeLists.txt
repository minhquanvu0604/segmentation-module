cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(model_inference)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Path to LibTorch
if(NOT CMAKE_PREFIX_PATH)
    set(CMAKE_PREFIX_PATH $ENV{CMAKE_PREFIX_PATH})
endif()
# message(STATUS "====== CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH} ======")

# Include directories
include_directories(
  include
  include/model_inference
  ${catkin_INCLUDE_DIRS}
  ${TORCH_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

# Add the library
# add_library(${PROJECT_NAME} SHARED src/model_inference.cpp)
add_library(${PROJECT_NAME} SHARED src/predict.cpp)

# Link against LibTorch
target_link_libraries(${PROJECT_NAME} 
                            ${TORCH_LIBRARIES} 
                            ${OpenCV_LIBS}
                            opencv_imgcodecs opencv_highgui)

# Ensure RPATH is set correctly (optional but recommended)
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD_REQUIRED ON)

# To ensure that the library is built with the same flags as LibTorch
set_property(TARGET ${PROJECT_NAME} PROPERTY POSITION_INDEPENDENT_CODE ON)

# Necessary to link LibTorch properly
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Ensure that the necessary libraries are included for the GPU support
if (TORCH_CUDA_ARCH_LIST)
  target_link_libraries(${PROJECT_NAME} ${TORCH_CUDA_LIBRARIES})
endif()

# Optional: Install targets
install(TARGETS ${PROJECT_NAME}
        LIBRARY DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)


# ==============================
# Add an executable for testing
# ==============================
add_executable(predict_main src/predict_main.cpp)

# Link the executable with the inference library and other dependencies
target_link_libraries(predict_main ${PROJECT_NAME} ${OpenCV_LIBS} "${TORCH_LIBRARIES}" opencv_imgcodecs opencv_highgui)

# Set properties for the executable
set_property(TARGET predict_main PROPERTY CXX_STANDARD 17)

# Optional: Install the executable
install(TARGETS predict_main RUNTIME DESTINATION bin)
