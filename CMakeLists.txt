cmake_minimum_required(VERSION 3.22)
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_TOOLCHAIN_FILE "/home/ctyler/.vcpkg/vcpkg/scripts/buildsystems/vcpkg.cmake")
project(cpp_inference)

find_package(OpenCV REQUIRED )
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(cpp_inference main.cpp)
target_link_libraries(cpp_inference ${OpenCV_LIBS})


