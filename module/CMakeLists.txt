cmake_minimum_required(VERSION 3.14)
project(module LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(MODULE_SOURCE_FILES
    src/detect.cpp
    src/homography.cpp
    src/RANSAC.cpp
)

find_package(OpenCV 4.5.5 REQUIRED)
find_package(Eigen3 3.4.0 REQUIRED)

add_library(${PROJECT_NAME}
    ${MODULE_SOURCE_FILES}
)

target_include_directories(${PROJECT_NAME} PUBLIC
    include
    ${OpenCV_INCLUDE_DIRS}
    ${Eigen_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME} PUBLIC
    ${OpenCV_LIBS}
    Eigen3::Eigen
)
