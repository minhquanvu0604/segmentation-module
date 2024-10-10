cmake_minimum_required(VERSION 3.0.2)
project(pointcloud_filter_node)

## Compile as C++14, supported in ROS Noetic
# add_compile_options(-std=c++14)
# add_compile_options(-std=c++17)


find_package(catkin REQUIRED COMPONENTS
  roscpp
  sensor_msgs
  pcl_ros
  pcl_conversions
  cv_bridge
)

find_package(OpenCV REQUIRED)
# find_package(Torch REQUIRED)
find_package(PCL REQUIRED)
find_package(model_inference REQUIRED)

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
  CATKIN_DEPENDS roscpp sensor_msgs pcl_ros pcl_conversions cv_bridge
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
  include/pointcloud_filter_node
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  # ${TORCH_INCLUDE_DIRS}
  ${model_inference_INCLUDE_DIRS}
)

# # Optional: Keep if PCL requires specific library directories
# link_directories(${PCL_LIBRARY_DIRS})

# # Optional: Keep if PCL requires specific definitions
# add_definitions(${PCL_DEFINITIONS})

add_executable(${PROJECT_NAME} 
  src/pointcloud_filter_main.cpp
  src/pointcloud_filter_core.cpp
)

target_link_libraries(${PROJECT_NAME} 
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  # ${TORCH_LIBRARIES}
  ${model_inference_LIBRARIES}
)

set_target_properties(${PROJECT_NAME} PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED YES
)

set_property(TARGET ${PROJECT_NAME} PROPERTY POSITION_INDEPENDENT_CODE ON)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
# if (TORCH_CUDA_ARCH_LIST)
#   target_link_libraries(${PROJECT_NAME} ${TORCH_CUDA_LIBRARIES})
# endif()

install(TARGETS ${PROJECT_NAME} 
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

message(STATUS "====== TORCH_INCLUDE_DIRS: ${TORCH_INCLUDE_DIRS} ======")
message(STATUS "====== TORCH_LIBRARIES: ${TORCH_LIBRARIES} ======")