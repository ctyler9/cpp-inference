cmake_minimum_required(VERSION 3.22)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_TOOLCHAIN_FILE "/home/ctyler/.vcpkg/vcpkg/scripts/buildsystems/vcpkg.cmake")
project(cpp_inference)
add_executable(cpp_inference main.cpp)

# find/include packages
find_package(OpenCV REQUIRED )
include_directories(${OpenCV_INCLUDE_DIRS})

set(ONNXRUNTIME_PATH "/.vcpkg/vcpkg/packages/onnxruntime-gpu_x64-linux")
include_directories(${ONNXRUNTIME_PATH}/include)
link_directories(${ONNXRUNTIME_PATH}/lib)

# link libraries
target_link_libraries(cpp_inference onnx onnx_proto)
target_link_libraries(cpp_inference ${OpenCV_LIBS})

