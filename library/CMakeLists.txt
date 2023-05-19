MESSAGE("BUILD LIBRARY")

set(TARGET ${PROJECT_NAME})

# USE_CUDA is always stronger

set(COMMON_FILES
    source/rng.cpp
    source/kernels.cpp
    source/states.cpp
)

if (${USE_CUDA})
    # CUDA build path

    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    MESSAGE(STATUS "Using CUDAToolkit (${CUDAToolkit_VERSION})")

    # source files
    set(FILES
        source/cuda/engine.cu
        source/cuda/activations.cu
    )

    # build static library
    add_library(${TARGET} STATIC ${FILES} ${COMMON_FILES})

    target_compile_definitions(${TARGET} PRIVATE CUDA=1)
    
    set_target_properties(${TARGET} PROPERTIES
        CUDA_ARCHITECTURES ${CUDA_ARCH} # defined in main CMake file
        CUDA_STANDARD 17
        CUDA_SEPARABLE_COMPILATION ON
    )
    
    target_compile_features(${TARGET} PUBLIC cxx_std_17)

    # include CUDA headers
    target_include_directories(${TARGET} PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
else()
    # Baseline / OMP build path

    # source files
    set(FILES
        source/base/engine.cpp
        source/base/activations.cpp
    )

    # build static library
    add_library(${TARGET} STATIC ${FILES} ${COMMON_FILES})

    target_compile_definitions(${TARGET} PRIVATE BASE=1)

    # link OpenMP
    if(${USE_OPENMP})
        find_package(OpenMP REQUIRED)
        MESSAGE(STATUS "Using OpenMP (CXX Version ${OpenMP_CXX_VERSION})")
        target_compile_definitions(${TARGET} PRIVATE OPENMP=1)
        target_compile_options(${TARGET} PRIVATE /openmp)
        target_link_libraries(${TARGET} PRIVATE OpenMP::OpenMP_CXX)
    endif()
endif()

# include header files (public so the others can access it (?))
target_include_directories(${TARGET} PUBLIC include)
