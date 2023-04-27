MESSAGE("BUILD BENCHMARKS")

set(TARGET Benchmark)

find_package(benchmark CONFIG REQUIRED)

add_executable(${TARGET} benchmarks.cpp)

# link CellularAutomata
target_link_libraries(${TARGET} PRIVATE ${PROJECT_NAME})

# link benchmark
target_link_libraries(${TARGET} PRIVATE benchmark::benchmark benchmark::benchmark_main)
# target_compile_options(-DCMAKE_BUILD_TYPE=Release)
