message("BUILD LIBRARY")

set(TARGET ${PROJECT_NAME})

# USE_CUDA is always stronger

set(CPP_FILES
    source/rng.cpp
    source/kernels.cpp
    source/states.cpp
    source/activations.cu
    source/backend_base.cpp
    source/CellularAutomata.cpp
)

if (${USE_CUDA})
    # add CUDA files
    set(CUDA_FILES
        source/cuda/backend_cuda.cu
    )
else()
    set(CUDA_FILES "")
endif()

# build static library
add_library(${TARGET} STATIC ${CUDA_FILES} ${CPP_FILES})

# configure OpenMP
if(${USE_OPENMP})
    find_package(OpenMP REQUIRED)
    message(STATUS "Using OpenMP (${OpenMP_CXX_VERSION})")
    # target_compile_options(${TARGET} PRIVATE -openmp)
    target_link_libraries(${TARGET} PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(${TARGET} PRIVATE ompEnabled=1)
endif()

# configure CUDA
if (${USE_CUDA})
    message(STATUS "Using CUDA (${CUDAToolkit_VERSION})")


    target_compile_definitions(${TARGET} PUBLIC cudaEnabled=1)

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


target_compile_features(${TARGET} PUBLIC cuda_std_11)
