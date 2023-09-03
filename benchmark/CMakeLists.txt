MESSAGE("BUILD BENCHMARKS")

set(TARGET Benchmark)

find_package(benchmark CONFIG REQUIRED)

add_executable(${TARGET} benchmarks.cpp)

set_target_properties(${TARGET} PROPERTIES
    CMAKE_BUILD_TYPE Release
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
