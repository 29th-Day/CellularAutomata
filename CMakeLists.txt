cmake_minimum_required(VERSION 3.20.0)
project(CellularAutomata
    VERSION 0.3.0
    LANGUAGES CXX
    HOMEPAGE_URL "https://github.com/29th-Day/CellularAutomata"
)

option(BUILD_EXAMPLE "Build examples?" ON)
option(BUILD_TESTS "Build tests? (CTest & doctest)" OFF)
option(BUILD_BENCHMARKS "Build benchmarks? (Google benchmark)" OFF)


option(USE_OPENMP "Use OpenMP?" ON)
option(USE_CUDA "Use CUDA?" ON)

# development only, not actually nessecary but IntelliSense wouldn't find CUDA otherwise
find_package(CUDAToolkit)
include_directories(${CUDAToolkit_INCLUDE_DIRS})

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

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
set(CPACK_PACKAGE_CONTACT "29th-Day <https://github.com/29th-Day>")
set(CPACK_PACKAGE_DESCRIPTION "Simple C++ template library for simulating cellular automaton")
set(CPACK_PACKAGE_HOMEPAGE_URL "https://github.com/29th-Day/CellularAutomata")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
set(CPACK_PACKAGE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/install")
set(CPACK_PACKAGE_FILE_NAME "${PROJECT_NAME}-${PROJECT_VERSION}")
set(CPACK_GENERATOR "ZIP;NSIS;TGZ;External;")
set(CPACK_PACKAGE_CHECKSUM SHA256)
include(CPack)
