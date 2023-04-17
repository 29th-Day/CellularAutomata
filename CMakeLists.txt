cmake_minimum_required(VERSION 3.0.0)
project(CellularAutomata VERSION 0.1.0 LANGUAGES CXX)

option(BUILD_EXAMPLE "Build example?" ON)
option(BUILD_TESTS "Run tests?" ON)

set(CMAKE_CXX_STANDARD 17)

add_subdirectory(library)

# sets release flag (useful for now?)
# set(CMAKE_CXX_FLAGS_RELEASE "-O3")

if(${BUILD_TESTS})
    add_subdirectory(tests)
endif()

if (${BUILD_EXAMPLE})
    add_subdirectory(example)
endif()

# ???
set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
