cmake_minimum_required(VERSION 3.0.2)
project(pointcloud_filter)

## Compile as C++14, supported in ROS Noetic
# add_compile_options(-std=c++14)
# add_compile_options(-std=c++17)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  sensor_msgs
  pcl_ros
  pcl_conversions
  cv_bridge
  nodelet
  pluginlib
)

# find_package(OpenCV REQUIRED)
find_package(OpenCV 4 REQUIRED) 
find_package(PCL REQUIRED)
find_package(model_inference REQUIRED)
find_package(yaml-cpp REQUIRED)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


# Check the variables - OK
# message(STATUS "====== model_inference_INCLUDE_DIRS: ${model_inference_INCLUDE_DIRS} ======")
# message(STATUS "====== model_inference_LIBRARIES: ${model_inference_LIBRARIES} ======")
# -- ====== model_inference_INCLUDE_DIRS: /home/quanvu/ros/apple_ws/devel/include/model_inference ======
# -- ====== model_inference_LIBRARIES: /home/quanvu/ros/apple_ws/devel/lib/libmodel_inference.so ======

# set(Torch_INCLUDE_DIRS ${TORCH_INCLUDE_DIRS}) 
# set(Torch_LIBRARIES ${TORCH_LIBRARIES})
catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS roscpp sensor_msgs pcl_ros pcl_conversions cv_bridge nodelet pluginlib
  DEPENDS OpenCV PCL
)

# if(NOT CMAKE_PREFIX_PATH) 
#     set(CMAKE_PREFIX_PATH $ENV{CMAKE_PREFIX_PATH})
# endif()

# message(STATUS "====== CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH} ======")
# message(STATUS "====== TORCH_INCLUDE_DIRS: ${TORCH_INCLUDE_DIRS} ======")
# message(STATUS "====== model_inference_LIBRARIES: ${model_inference_LIBRARIES} ======")

include_directories(
  include
  include/pointcloud_filter
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${model_inference_INCLUDE_DIRS}
)

# # Optional: Keep if PCL requires specific library directories
# link_directories(${PCL_LIBRARY_DIRS})

# # Optional: Keep if PCL requires specific definitions
# add_definitions(${PCL_DEFINITIONS})

# Pointcloud Filter node
add_executable(${PROJECT_NAME}_node 
  src/pointcloud_filter_main.cpp
  src/pointcloud_filter_core.cpp
)
target_link_libraries(${PROJECT_NAME}_node
  ${catkin_LIBRARIES}
  # ${OpenCV_LIBRARIES}
  ${model_inference_LIBRARIES}
  yaml-cpp
  ${OpenCV_LIBS} opencv_imgcodecs opencv_highgui
)
set_target_properties(${PROJECT_NAME}_node PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED YES
)
set_property(TARGET ${PROJECT_NAME}_node PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET ${PROJECT_NAME}_node PROPERTY CXX_STANDARD 17)


# Pointcloud Filter nodelet
add_library(${PROJECT_NAME}_nodelet
  src/pointcloud_filter_core.cpp
)
target_link_libraries(${PROJECT_NAME}_nodelet 
  ${catkin_LIBRARIES}
  ${model_inference_LIBRARIES}
  yaml-cpp
  ${OpenCV_LIBS} opencv_imgcodecs opencv_highgui
)


# Installing both node and nodelet
install(TARGETS ${PROJECT_NAME}_node 
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
install(TARGETS ${PROJECT_NAME}_nodelet
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
)
install(FILES nodelet_plugins.xml
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

# message(STATUS "====== CATKIN_PACKAGE_SHARE_DESTINATION: ${CATKIN_PACKAGE_SHARE_DESTINATION} ======")
message(STATUS "====== TORCH_INCLUDE_DIRS: ${TORCH_INCLUDE_DIRS} ======")
message(STATUS "====== TORCH_LIBRARIES: ${TORCH_LIBRARIES} ======")