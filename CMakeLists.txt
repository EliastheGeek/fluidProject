cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(DH2323Project LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-O2 -Wall")
set(CMAKE_CUDA_ARCHITECTURES 61)

IF(APPLE)
  SET(CMAKE_OSX_ARCHITECTURES "arm64" CACHE STRING "Build architectures for Mac OS X" FORCE)
ENDIF(APPLE)

set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
find_package (SDL2 REQUIRED)

message(STATUS "Lib: ${SDL2_LIBRARIES} , Include: ${SDL2_INCLUDE_DIRS}")

include_directories(
  ${SDL2_INCLUDE_DIRS}
  ${CMAKE_SOURCE_DIR}/glm
  ${CMAKE_SOURCE_DIR}/SDL2Auxiliary
)

set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDebug")

add_executable(DH2323SkeletonSDL2
  main.cpp
  ${CMAKE_SOURCE_DIR}/SDL2Auxiliary/SDL2Auxiliary.cpp
  fluid_simulator.cpp
  fluid_kernels.cu
)

target_link_libraries(DH2323SkeletonSDL2
  ${SDL2_LIBRARIES}
  cuda
  cudart
)