set(TARGET ${PROJECT_NAME})

set(FILES
    "include/core/common.hpp"
    "include/core/common.inl"
    "include/core/cpu.hpp"
    "include/core/cpu.inl"
    "include/core/epoch.hpp"
    "include/core/epoch.inl"
    "include/core/rng.hpp"
    "include/core/rng.inl"
    "include/cuda/cuda.cuh"
    "include/cuda/cuda.inl"
    "include/func/activations.hpp"
    "include/func/activations.inl"
    "include/func/kernels.hpp"
    "include/func/kernels.inl"
    "include/func/states.hpp"
    "include/func/states.inl"
)

add_library(${TARGET} INTERFACE)

target_sources(${TARGET} PUBLIC
    FILE_SET HEADERS
    BASE_DIRS include
    FILES ${FILES}
)
