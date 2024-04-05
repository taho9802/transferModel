cmake_minimum_required(VERSION 3.5)
project(main)

set(CMAKE_PREFIX_PATH /Users/taewoohong/libtorch)
find_package(Torch REQUIRED)

add_executable(main src/main.cpp)
target_link_libraries(main PRIVATE ${TORCH_LIBRARIES})
target_compile_features(main PRIVATE cxx_std_17)
target_compile_options(main PRIVATE ${TORCH_CXX_FLAGS})