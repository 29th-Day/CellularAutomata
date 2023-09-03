
#include <CellularAutomata>

#include <benchmark/benchmark.h>

#include <vector>

using namespace CellularAutomata;

static void Epoch(benchmark::State& state)
{
    random::seed(42);

    unsigned int height = (unsigned int)state.range(0);
    unsigned int width = (unsigned int)state.range(0);
    bool recursive = true;

    state.SetComplexityN(state.range(0));

    State<int> world(height, width, States::binary, Device::CPU);
    Kernel<int> kernel(3, Kernels::life, Device::CPU);

    for (auto _ : state)
    {
        CellularAutomata::Epoch(&world, &kernel, Activations::life<int>(), recursive);
    }
}

// Register the function as a benchmark
BENCHMARK(Epoch)->DenseRange(100, 2000, 100)->Complexity()->Unit(benchmark::kMillisecond)->Repetitions(10)->DisplayAggregatesOnly(true);

// Run the benchmark
BENCHMARK_MAIN();
