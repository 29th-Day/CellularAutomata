
#include "CellularAutomata.h"

#include <benchmark/benchmark.h>

#include <vector>

static void Epoch(benchmark::State &state)
{
    CellularAutomata::InitRandom(42);

    unsigned int height = (unsigned int)state.range(0);
    unsigned int width = (unsigned int)state.range(0);
    bool recursive = true;

    state.SetComplexityN(state.range(0));

    State world;
    Kernel kernel;

    CellularAutomata::InitState(&world, height, width, States::randf);
    CellularAutomata::InitKernel(&kernel, 3, Kernels::rand);

    for (auto _ : state)
    {
        CellularAutomata::Epoch(&world, &kernel, Activations::clip, recursive);
    }

    CellularAutomata::DestroyState(&world);
    CellularAutomata::DestroyKernel(&kernel);
}

// Register the function as a benchmark
BENCHMARK(Epoch)->DenseRange(100, 2000, 100)->Complexity()->Unit(benchmark::kMillisecond)->Repetitions(10)->DisplayAggregatesOnly(true);

// Run the benchmark
BENCHMARK_MAIN();
