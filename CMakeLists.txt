cmake_minimum_required(VERSION 3.20.0)
project(CellularAutomata VERSION 0.1.0 LANGUAGES CXX HOMEPAGE_URL https://github.com/29th-Day/CellularAutomata)

option(BUILD_EXAMPLE "Build examples?" ON)
option(BUILD_TESTS "Build tests? (CTest & doctest)" OFF)
option(BUILD_BENCHMARKS "Build benchmarks? (Google benchmark)" OFF)

option(USE_OPENMP "Use OpenMP?" ON)
option(USE_CUDA "Use CUDA?" OFF)

set(CMAKE_CXX_STANDARD 11)

# development only, not actually nessecary but IntelliSense wouldn't find CUDA otherwise
if(USE_CUDA)
    find_package(CUDAToolkit REQUIRED)
    include_directories(${CUDAToolkit_INCLUDE_DIRS})
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
