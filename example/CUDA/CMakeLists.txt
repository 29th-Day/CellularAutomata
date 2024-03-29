MESSAGE("BUILD EXAMPLE (CUDA)")

set(TARGET example_cuda)

enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)
message(STATUS "Using CUDA (${CUDAToolkit_VERSION})")

add_executable(${TARGET} main.cu)

# https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
set_target_properties(${TARGET} PROPERTIES
    CUDA_ARCHITECTURES "86" 
)

if($<TARGET_EXISTS:OpenMP::OpenMP_CXX>)
    message(STATUS "openmp")
endif()

if(MSVC)
    target_link_options(${TARGET} PRIVATE /NODEFAULTLIB:LIBCMT)
endif()

target_link_libraries(${TARGET}
    PRIVATE
    # CellularAutomata
    CellularAutomata
    # OpenMP
    $<IF:$<TARGET_EXISTS:OpenMP::OpenMP_CXX>, OpenMP::OpenMP_CXX, >
)
