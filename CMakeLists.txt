cmake_minimum_required(VERSION 3.20.0)
project(CellularAutomata VERSION 0.1.0 LANGUAGES CXX HOMEPAGE_URL https://github.com/29th-Day/CellularAutomata)

option(BUILD_EXAMPLE "Build example?" ON)
option(BUILD_TESTS "Build tests? (CTest & doctest)" OFF)
option(BUILD_BENCHMARKS "Build benchmarks? (Google benchmark)" OFF)

option(USE_OPENMP "Use OpenMP?" ON)
option(USE_CUDA "Use CUDA?" OFF)

set(CMAKE_CXX_STANDARD 11)

# https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
set(CUDA_ARCH "86" CACHE STRING "CUDA architecture to build for")

if(${USE_CUDA})
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
endif()

add_subdirectory(CellularAutomata)

if(${BUILD_TESTS})
    include(CTest)
    add_subdirectory(tests)
endif()

if (${BUILD_BENCHMARKS})
    add_subdirectory(benchmark)
endif()

if (${BUILD_EXAMPLE})
    add_subdirectory(example)
endif()
