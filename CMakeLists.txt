cmake_minimum_required(VERSION 3.0.0)
project(CellularAutomata VERSION 0.1.0)

include(CTest)
enable_testing()

file(GLOB FILES
    "${PROJECT_SOURCE_DIR}/include/*.h"
    "${PROJECT_SOURCE_DIR}/include/*.hpp"
    "${PROJECT_SOURCE_DIR}/source/*.cpp"
    "${PROJECT_SOURCE_DIR}/source/*.c"
)

add_library(${PROJECT_NAME} STATIC ${FILES})
target_include_directories(${PROJECT_NAME} PUBLIC include)

set(CMAKE_CXX_FLAGS_RELEASE "-O3")

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

add_subdirectory(example)