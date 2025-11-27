cmake_minimum_required(VERSION 3.18)
project(flash_attention CUDA CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find required packages
find_package(CUDA REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)

# CUDA settings
set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 89 90)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr --use_fast_math")

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/cpp)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/cuda)
include_directories(${Python3_INCLUDE_DIRS})
include_directories(${pybind11_INCLUDE_DIRS})

# Source files
set(CUDA_SOURCES
    src/cuda/flash.cu
)

set(CPP_SOURCES
    src/cpp/main.cpp
)

# Create CUDA library
add_library(flash_attention_cuda SHARED ${CUDA_SOURCES})
set_target_properties(flash_attention_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# Create Python extension
pybind11_add_module(minimal_attn ${CPP_SOURCES})
target_link_libraries(minimal_attn PRIVATE flash_attention_cuda)

# Installation
install(TARGETS minimal_attn
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages
)

# Build options
option(BUILD_TESTS "Build test suite" ON)
option(BUILD_EXAMPLES "Build examples" ON)
option(BUILD_BENCHMARKS "Build benchmarks" ON)

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

# Print configuration
message(STATUS "CUDA Version: ${CUDA_VERSION}")
message(STATUS "Python Version: ${Python3_VERSION}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
