cmake_minimum_required(VERSION 3.0.2)
project(segmentation_module)

########### PARAMETER TO CHANGE ###########
# Used to making model_inference_cpp a ROS-agnostic package
# set(model_inference_DIR "/home/quanvu/git/segmentation-module/model_inference_cpp/cmake/model_inferenceConfig.cmake")

# include_directories(
#   include
#   ${catkin_INCLUDE_DIRS}
#   ${catkin_INCLUDE_DIRS}
#   ${OpenCV_INCLUDE_DIRS}
#   ${TORCH_INCLUDE_DIRS}
#   ${model_inference_INCLUDE_DIRS}
# )

# ## Specify libraries to link a library or executable target against
# target_link_libraries(${PROJECT_NAME}
#   ${catkin_LIBRARIES}
#   ${OpenCV_LIBRARIES}
#   ${TORCH_LIBRARIES}
#   model_inference::model_inference
# )

cmake_minimum_required(VERSION 3.0.2)
project(segmentation_module_ros)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  model_inference_cpp
)

find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS roscpp model_inference_cpp
  DEPENDS OpenCV Torch
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${TORCH_INCLUDE_DIRS}
)

add_executable(apple_segmentation_node
  src/main.cpp
  src/SegmentationNode.cpp
)

target_link_libraries(apple_segmentation_node
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${TORCH_LIBRARIES}
  model_inference_cpp
)

set_property(TARGET apple_segmentation_node PROPERTY CXX_STANDARD 14)
