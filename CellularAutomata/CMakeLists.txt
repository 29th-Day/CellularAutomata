set(TARGET ${PROJECT_NAME})

add_library(${TARGET} INTERFACE)
target_include_directories(${TARGET} INTERFACE .)

set_property(TARGET ${TARGET} PROPERTY CXX_STANDARD 11)

# configure OpenMP
if(USE_OPENMP)
    find_package(OpenMP REQUIRED)
    message(STATUS "Using OpenMP (${OpenMP_CXX_VERSION})")
    target_link_libraries(${TARGET} INTERFACE OpenMP::OpenMP_CXX)
endif()

# debug only
if(USE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    target_include_directories(${TARGET} INTERFACE ${CUDAToolkit_INCLUDE_DIRS})
endif()
