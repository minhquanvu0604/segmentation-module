cmake_minimum_required(VERSION 3.0.2)
project(segmentation_module_ros_node)

########### PARAMETER TO CHANGE ###########
# Used to making model_inference a ROS-agnostic package
# set(model_inference_DIR "/home/quanvu/git/segmentation-module/model_inference/cmake/model_inferenceConfig.cmake")

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


## Compile as C++14, supported in ROS Noetic
add_compile_options(-std=c++14)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  sensor_msgs
  pcl_ros
  pcl_conversions
  model_inference
)

find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)
find_package(PCL REQUIRED)


catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS roscpp sensor_msgs pcl_ros pcl_conversions model_inference
  DEPENDS OpenCV Torch PCL
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${TORCH_INCLUDE_DIRS}
)

link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(apple_segmentation_node
  src/main.cpp
  src/apple_segmentation_node.cpp
)

target_link_libraries(apple_segmentation_node
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${TORCH_LIBRARIES}
  model_inference
)

install(TARGETS segmentation_node
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

set_target_properties(apple_segmentation_node PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
)