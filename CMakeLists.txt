cmake_minimum_required(VERSION 3.0.0)
project(CellularAutomata VERSION 0.1.0 LANGUAGES CXX HOMEPAGE_URL https://github.com/29th-Day/CellularAutomata)

option(BUILD_EXAMPLE "Build example?" ON)
option(BUILD_TESTS "Build tests?" ON)
option(BUILD_BENCHMARKS "Build benchmarks?" ON)

option(USE_CUDA "Use CUDA in Engine?" ON)
option(USE_OPENMP "Use OpenMP in Engine?" ON)

set(CMAKE_CXX_STANDARD 17)

# https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
set(CUDA_ARCH "86-real" CACHE STRING "CUDA architecture to build for")

add_subdirectory(library)

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
