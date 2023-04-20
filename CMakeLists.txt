cmake_minimum_required(VERSION 3.0.0)
project(CellularAutomata VERSION 0.1.0 LANGUAGES CXX HOMEPAGE_URL https://github.com/29th-Day/CellularAutomata)

option(BUILD_EXAMPLE "Build example?" ON)
option(BUILD_TESTS "Run tests?" ON)

option(USE_OPENMP "Use OpenMP in Engine?" ON)

set(CMAKE_CXX_STANDARD 17)

add_subdirectory(library)

# sets release flag (useful for now?)
# set(CMAKE_CXX_FLAGS_RELEASE "-O3")

if(${BUILD_TESTS})
    include(CTest)
    add_subdirectory(tests)
endif()

if (${BUILD_EXAMPLE})
    add_subdirectory(example)
endif()
