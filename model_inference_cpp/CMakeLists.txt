cmake_minimum_required(VERSION 3.0.2)
project(model_inference_cpp)

find_package(catkin REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")


catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS
  DEPENDS OpenCV Torch
)

include_directories(
  include
  include/${PROJECT_NAME}
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${TORCH_INCLUDE_DIRS}
)

add_library(${PROJECT_NAME}
  src/model_inference.cpp
)

set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)

target_link_libraries(${PROJECT_NAME}
  ${OpenCV_LIBRARIES}
  ${TORCH_LIBRARIES}
)