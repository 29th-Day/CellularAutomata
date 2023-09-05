MESSAGE("BUILD BENCHMARKS")

set(TARGET Benchmark)

find_package(benchmark CONFIG REQUIRED)

add_executable(${TARGET} benchmarks.cu)

set_target_properties(${TARGET} PROPERTIES
    CUDA_ARCHITECTURES ${CUDA_ARCH}
)

# link benchmark
target_link_libraries(${TARGET} PRIVATE
    CellularAutomata
    benchmark::benchmark benchmark::benchmark_main
)

# # Add the /NODEFAULTLIB:LIBCMT linker option
# if(${USE_CUDA} AND ${MSVC})
#   	target_link_options(${TARGET} PRIVATE /NODEFAULTLIB:LIBCMT)
# endif()
