#include <CellularAutomata>
#include <benchmark/benchmark.h>

using namespace CellularAutomata;

#define common RangeMultiplier(2)->Range(32, 2048)->Unit(benchmark::kMillisecond)->Repetitions(10)->ReportAggregatesOnly(true)

static void Epoch_CPU(benchmark::State& state)
{
    random::seed(42);

    unsigned int height = (unsigned int)state.range(0);
    unsigned int width = (unsigned int)state.range(0);
    bool recursive = true;

    State<int> world(height, width, States::binary, Device::CPU);
    Kernel<int> kernel(3, Kernels::life, Device::CPU);
    Activations::life<int> af;

    for (auto _ : state)
    {
        CellularAutomata::Epoch(&world, &kernel, af, recursive);
    }
}

static void Epoch_CUDA(benchmark::State& state)
{
    random::seed(42);

    unsigned int height = (unsigned int)state.range(0);
    unsigned int width = (unsigned int)state.range(0);
    bool recursive = true;

    State<int> world(height, width, States::binary, Device::CUDA);
    Kernel<int> kernel(3, Kernels::life, Device::CUDA);
    Activations::life<int> af;

    for (auto _ : state)
    {
        CellularAutomata::Epoch(&world, &kernel, af, recursive);
    }
}

BENCHMARK(Epoch_CPU)->Name("CPU")->common;
#ifdef __CUDACC__
BENCHMARK(Epoch_CUDA)->Name("CUDA")->common;
#endif


// Run the benchmark
BENCHMARK_MAIN();
