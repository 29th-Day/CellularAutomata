
if(USE_OPENMP)
    find_package(OpenMP REQUIRED)
    message(STATUS "Using OpenMP (${OpenMP_CXX_VERSION})")
endif()

add_subdirectory(CPU)

if (USE_CUDA)
    add_subdirectory(CUDA)
endif()
