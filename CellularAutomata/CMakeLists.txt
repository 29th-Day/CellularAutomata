message("BUILD LIBRARY")

set(TARGET ${PROJECT_NAME})

set(FILES
    source/rng.cpp
    source/kernels.cpp
    source/states.cpp
    source/backend_base.cpp
    source/CellularAutomata.cpp
    source/activations.cu
    $<$<BOOL:${USE_CUDA}>:source/cuda/backend_cuda.cu>
)

if (NOT ${USE_CUDA})
    set_source_files_properties(${FILES} PROPERTIES LANGUAGE CXX)
endif()

# build static library
add_library(${TARGET} STATIC ${FILES})

# configure OpenMP
if(${USE_OPENMP})
    find_package(OpenMP REQUIRED)
    message(STATUS "Using OpenMP (${OpenMP_CXX_VERSION})")

    target_link_libraries(${TARGET} PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(${TARGET} PUBLIC ompEnabled=1)
endif()

# configure CUDA
if (${USE_CUDA})
    message(STATUS "Using CUDA (${CUDAToolkit_VERSION})")

    target_compile_definitions(${TARGET} PUBLIC cudaEnabled=1)

    target_compile_features(${TARGET} PUBLIC cuda_std_11)

    # https://cmake.org/cmake/help/latest/manual/cmake-properties.7.html
    set_target_properties(${TARGET} PROPERTIES
        CUDA_ARCHITECTURES ${CUDA_ARCH}
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        CUDA_RUNTIME_LIBRARY Static
    )
endif()

# include header files (public so the others can access it (?))
target_include_directories(${TARGET} PUBLIC include)
