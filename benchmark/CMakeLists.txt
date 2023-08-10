MESSAGE("BUILD BENCHMARKS")

set(TARGET Benchmark)

find_package(benchmark CONFIG REQUIRED)

add_executable(${TARGET} benchmarks.cpp)

# link CellularAutomata
target_link_libraries(${TARGET} PRIVATE "CellularAutomata")

# link benchmark
target_link_libraries(${TARGET} PRIVATE benchmark::benchmark benchmark::benchmark_main)
# target_compile_options(-DCMAKE_BUILD_TYPE=Release)

# # Add the /NODEFAULTLIB:LIBCMT linker option
# if(${USE_CUDA} AND ${MSVC})
#   	target_link_options(${TARGET} PRIVATE /NODEFAULTLIB:LIBCMT)
# endif()
