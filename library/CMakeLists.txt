MESSAGE("BUILD LIBRARY")

set(TARGET ${PROJECT_NAME})

set(ENGINE_STD 0)
set(ENGINE_OMP 1)
set(ENGINE_CUDA 2)

set(ENGINE ${ENGINE_STD})

# USE_CUDA is always stronger

set(COMMON_FILES
    source/rng.cpp
    source/kernels.cpp
    source/states.cpp
)

if (${USE_CUDA})
    # CUDA build path

    find_package(CUDAToolkit REQUIRED)
    MESSAGE(STATUS "Using CUDAToolkit (${CUDAToolkit_VERSION})")

    # source files
    set(FILES
        source/cuda/engine.cu
        source/cuda/activations.cu
    )

    # build static library
    add_library(${TARGET} STATIC ${FILES} ${COMMON_FILES})

    # target_compile_definitions(${TARGET} PRIVATE ENGINE=2)

    set(ENGINE ${ENGINE_CUDA})
    
    set_target_properties(${TARGET} PROPERTIES
        CUDA_ARCHITECTURES ${CUDA_ARCH} # defined in main CMake file
        CUDA_STANDARD 17
    )
else()
    # Baseline / OMP build path

    # source files
    set(FILES
        source/base/engine.cpp
        source/base/activations.cpp
    )

    # build static library
    add_library(${TARGET} STATIC ${FILES} ${COMMON_FILES})

    # link OpenMP
    if(${USE_OPENMP})
        find_package(OpenMP REQUIRED)
        MESSAGE(STATUS "Using OpenMP (CXX Version ${OpenMP_CXX_VERSION})")
        target_compile_options(${TARGET} PRIVATE /openmp)
        target_link_libraries(${TARGET} PRIVATE OpenMP::OpenMP_CXX)
        # target_compile_definitions(${TARGET} PRIVATE ENGINE=1)
        set(ENGINE ${ENGINE_OMP})
    endif()
endif()

target_compile_definitions(${TARGET}
    PRIVATE ENGINE_STD=${ENGINE_STD}
    PRIVATE ENGINE_OMP=${ENGINE_OMP}
    PRIVATE ENGINE_CUDA=${ENGINE_CUDA}
    PRIVATE ENGINE=${ENGINE}
)

# include header files (public so the others can access it (?))
target_include_directories(${TARGET} PUBLIC include)

target_compile_features(${TARGET} PUBLIC cxx_std_17)
