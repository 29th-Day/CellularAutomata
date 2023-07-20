MESSAGE("BUILD EXAMPLE (CUDA)")

set(TARGET example_cuda)

enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

# source files
set(FILES
    source/main.cu
)

# build executable
add_executable(${TARGET} ${FILES})


message(STATUS "Using CUDA (${CUDAToolkit_VERSION})")

# https://cmake.org/cmake/help/latest/manual/cmake-properties.7.html
set_target_properties(${TARGET} PROPERTIES
    CUDA_ARCHITECTURES ${CUDA_ARCH}
    # CUDA_RUNTIME_LIBRARY Static
)

target_link_libraries(${TARGET}
    PRIVATE
    # CellularAutomata
    ${PROJECT_NAME}
)
