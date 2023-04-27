
#include "engine.h"

#include <benchmark/benchmark.h>

#include <vector>

static void Epoch(benchmark::State &state)
{
    Engine::InitRandom(42);

    unsigned int height = (unsigned int)state.range(0);
    unsigned int width = (unsigned int)state.range(0);
    bool recursive = true;

    state.SetComplexityN(state.range(0));

    State world;
    Kernel kernel;

    Engine::InitState(&world, height, width, States::randf);
    Engine::InitKernel(&kernel, Kernels::rand);

    for (auto _ : state)
    {
        Engine::Epoch(&world, &kernel, Activations::clip, recursive);
    }

    Engine::DestroyState(&world);
    Engine::DestroyKernel(&kernel);
}

// Register the function as a benchmark
BENCHMARK(Epoch)->DenseRange(100, 500, 100)->Complexity()->Unit(benchmark::kMillisecond)->Repetitions(10)->DisplayAggregatesOnly(true);

// Run the benchmark
BENCHMARK_MAIN();
